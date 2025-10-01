import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None, need_attn=False):
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_attn, average_attn_weights=False
        )
        x = self.norm1(x + self.drop1(attn_out))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = self.norm2(x + self.drop2(ff))
        return x, attn_weights

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, y, memory, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, need_attn=False):
        self_out, self_attn = self.self_attn(
            y, y, y, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=need_attn, average_attn_weights=False
        )
        y = self.norm1(y + self.drop1(self_out))

        cross_out, cross_attn = self.cross_attn(
            y, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_attn, average_attn_weights=False
        )
        y = self.norm2(y + self.drop2(cross_out))

        ff = self.linear2(self.dropout(F.gelu(self.linear1(y))))
        y = self.norm3(y + self.drop3(ff))
        return y, self_attn, cross_attn
