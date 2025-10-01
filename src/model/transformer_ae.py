import torch
import torch.nn as nn
from .positional import PositionalEncoding
from .layers import EncoderLayer, DecoderLayer
from ..config import ModelConfig
from ..vocab import PAD_ID, BOS_ID
from ..utils import subsequent_mask

class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

    def forward(self, x, src_key_padding_mask=None, need_attn=False):
        attns = []
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, src_key_padding_mask, need_attn=(need_attn and i == 0))
            if need_attn and attn is not None:
                attns.append(attn)  # keep layer 0 only
        return x, (attns[0] if attns else None)

class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

    def forward(self, y, memory, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, need_attn=False):
        self_attn_store = None
        cross_attn_store = None
        for i, layer in enumerate(self.layers):
            y, self_attn, cross_attn = layer(
                y, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask,
                need_attn=(need_attn and i == 0)
            )
            if need_attn and self_attn is not None and cross_attn is not None and i == 0:
                self_attn_store = self_attn
                cross_attn_store = cross_attn
        return y, self_attn_store, cross_attn_store

class TransformerAutoencoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.src_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.tgt_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.generator = nn.Linear(cfg.d_model, cfg.vocab_size)

    def _key_padding_mask(self, x):
        return (x == self.cfg.pad_id)

    def forward(self, src, tgt):
        src_key_pad = self._key_padding_mask(src)
        tgt_key_pad = self._key_padding_mask(tgt)
        src_emb = self.pos_enc(self.src_embed(src))
        memory, _ = self.encoder(src_emb, src_key_padding_mask=src_key_pad, need_attn=False)
        y_in = tgt[:, :-1]
        y_key_pad = self._key_padding_mask(y_in)
        y_emb = self.pos_enc(self.tgt_embed(y_in))
        tgt_mask = subsequent_mask(y_in.size(1), device=y_in.device)
        dec_out, _, _ = self.decoder(
            y_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=y_key_pad,
            memory_key_padding_mask=src_key_pad, need_attn=False
        )
        logits = self.generator(dec_out)
        return logits

    @torch.no_grad()
    def encode(self, src, need_attn=False):
        src_key_pad = self._key_padding_mask(src)
        emb = self.pos_enc(self.src_embed(src))
        memory, enc_attn = self.encoder(emb, src_key_padding_mask=src_key_pad, need_attn=need_attn)
        return memory, src_key_pad, enc_attn

    @torch.no_grad()
    def decode_with_external_memory(self, memory, src_key_pad, max_len=50, need_attn=False):
        B = memory.size(0)
        y = torch.full((B, 1), BOS_ID, dtype=torch.long, device=memory.device)
        for _ in range(max_len - 1):
            y_emb = self.pos_enc(self.tgt_embed(y))
            y_key_pad = self._key_padding_mask(y)
            tgt_mask = subsequent_mask(y.size(1), y.device)
            dec_out, _, _ = self.decoder(
                y_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=y_key_pad,
                memory_key_padding_mask=src_key_pad, need_attn=need_attn
            )
            logits = self.generator(dec_out[:, -1:, :])
            next_token = torch.argmax(logits, dim=-1)
            y = torch.cat([y, next_token], dim=1)
            if (next_token == self.cfg.eos_id).all():
                break
        return y

    @torch.no_grad()
    def greedy_decode_self(self, src, max_len=50):
        memory, src_key_pad, _ = self.encode(src, need_attn=False)
        return self.decode_with_external_memory(memory, src_key_pad, max_len=max_len, need_attn=False)
