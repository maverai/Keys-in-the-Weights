"""
Attention plotting utilities.

All functions are pure (no reliance on notebook globals) and save figures to disk.
They take an explicit `models` dict (e.g., {"M1": model1, ...}) and use encode_str
from src.vocab. Works headlessly (Agg backend) on servers/Kaggle.

Example:
    from src.viz.attn_plots import plot_attn_heatmaps_per_model
    plot_attn_heatmaps_per_model(models, "secure message", ["M1", "M2", "M3"],
                                 which="encoder", out_dir="plots", save_prefix="enc_heat")
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from matplotlib.lines import Line2D

from ..config import DEVICE
from ..vocab import encode_str, PAD_ID, BOS_ID, EOS_ID
from ..utils import subsequent_mask


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _mesh(Tq: int, Tk: int):
    x = np.arange(Tq); y = np.arange(Tk)
    return np.meshgrid(x, y, indexing="xy")


@torch.no_grad()
def _enc_attn_layer0_headavg(model, src_tensor):
    model.eval()
    memory, src_key_pad, enc_attn = model.encode(src_tensor, need_attn=True)
    if enc_attn is None:
        return None
    A = enc_attn.mean(dim=1)[0].detach().cpu().numpy()  # [T, T]
    A = (A + 1e-9) / (A.sum(axis=1, keepdims=True) + 1e-9)
    return A

@torch.no_grad()
def _dec_self_attn_final(model, src_tensor, max_len=50):
    model.eval()
    memory, src_key_pad, _ = model.encode(src_tensor, need_attn=False)
    B = memory.size(0)
    y = torch.full((B, 1), BOS_ID, dtype=torch.long, device=memory.device)
    last_self = None
    for _ in range(max_len - 1):
        y_emb = model.pos_enc(model.tgt_embed(y))
        y_pad = (y == PAD_ID)
        tgt_mask = subsequent_mask(y.size(1), y.device)
        dec_out, self_attn, cross_attn = model.decoder(
            y_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=y_pad,
            memory_key_padding_mask=src_key_pad, need_attn=True
        )
        if self_attn is not None:
            last_self = self_attn
        logits = model.generator(dec_out[:, -1:, :])
        next_token = torch.argmax(logits, dim=-1)
        y = torch.cat([y, next_token], dim=1)
        if (next_token == EOS_ID).all():
            break
    if last_self is None:
        return None
    S = last_self.mean(dim=1)[0].detach().cpu().numpy()
    S = (S + 1e-9) / (S.sum(axis=1, keepdims=True) + 1e-9)
    return S

@torch.no_grad()
def _dec_self_and_cross_final(model, memory, src_key_pad, max_len=50):
    model.eval()
    B = memory.size(0)
    y = torch.full((B, 1), BOS_ID, dtype=torch.long, device=memory.device)
    last_self, last_cross = None, None
    for _ in range(max_len - 1):
        y_emb = model.pos_enc(model.tgt_embed(y))
        y_pad = (y == PAD_ID)
        tgt_mask = subsequent_mask(y.size(1), y.device)
        dec_out, self_attn, cross_attn = model.decoder(
            y_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=y_pad,
            memory_key_padding_mask=src_key_pad, need_attn=True
        )
        if self_attn is not None:
            last_self = self_attn
        if cross_attn is not None:
            last_cross = cross_attn
        logits = model.generator(dec_out[:, -1:, :])
        next_token = torch.argmax(logits, dim=-1)
        y = torch.cat([y, next_token], dim=1)
        if (next_token == EOS_ID).all():
            break
    S = None if last_self is None else last_self.mean(dim=1)[0].detach().cpu().numpy()
    C = None if last_cross is None else last_cross.mean(dim=1)[0].detach().cpu().numpy()
    if S is not None:
        S = (S + 1e-9) / (S.sum(axis=1, keepdims=True) + 1e-9)
    if C is not None:
        C = (C + 1e-9) / (C.sum(axis=1, keepdims=True) + 1e-9)
    return S, C


# --------------------- Public plotting APIs ---------------------

@torch.no_grad()
def plot_attn_heatmaps_per_model(
    models: Dict[str, torch.nn.Module],
    input_text: str,
    model_keys: List[str],
    which: str = "encoder",
    out_dir: str = "plots",
    save_prefix: str = "attn_heat"
):
    """
    Save a separate 2D heatmap per model.
    which in {'encoder','decoder'}.
    """
    assert which in ("encoder", "decoder")
    assert isinstance(model_keys, (list, tuple)) and model_keys
    any_key = model_keys[0]
    assert any_key in models, f"Model key '{any_key}' not found."
    _ensure_dir(out_dir)

    max_len = models[any_key].cfg.max_len
    src = torch.tensor([encode_str(input_text, max_len)], dtype=torch.long, device=DEVICE)

    for k in model_keys:
        if k not in models: 
            print(f"[WARN] Missing model '{k}', skipping.")
            continue
        if which == "encoder":
            M = _enc_attn_layer0_headavg(models[k], src)
            title = f"[{k}] Encoder L0 Self-Attn\ninput='{input_text}'"
            xlab, ylab = "Key pos", "Query pos"
        else:
            M = _dec_self_attn_final(models[k], src, max_len=max_len)
            title = f"[{k}] Decoder L0 Self-Attn (final)\ninput='{input_text}'"
            xlab, ylab = "Key pos (tgt)", "Query pos (tgt)"
        if M is None:
            print(f"[WARN] No attention map for '{k}' in mode '{which}'.")
            continue
        fig = plt.figure(figsize=(7, 6))
        plt.imshow(M, aspect="auto", origin="lower")
        plt.title(title)
        plt.xlabel(xlab); plt.ylabel(ylab)
        plt.colorbar()
        fig.savefig(os.path.join(out_dir, f"{save_prefix}_{k}_{which}.png"),
                    dpi=220, bbox_inches="tight")
        plt.close(fig)


def _distinct_colors(n):
    base = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00",
            "#A65628","#F781BF","#999999","#66C2A5","#E6AB02",
            "#1B9E77","#D95F02","#7570B3","#66A61E","#E7298A"]
    return [base[i % len(base)] for i in range(n)]


@torch.no_grad()
def plot_overlaid_attn_3d(
    models: Dict[str, torch.nn.Module],
    input_text: str,
    model_keys: List[str],
    elev: int = 35,
    azim: int = -60,
    alpha: float = 0.9,
    out_dir: str = "plots",
    save_stem: str = "attn_overlay",
    style: str = "wire"
):
    """
    Overlay encoder & decoder final-step self-attention for multiple models in 3D.
    style: 'wire' or 'surface'
    Saves PNG and PDF.
    """
    assert isinstance(model_keys, (list, tuple)) and model_keys
    any_key = model_keys[0]
    assert any_key in models, f"Model key '{any_key}' not found."
    _ensure_dir(out_dir)

    max_len = models[any_key].cfg.max_len
    src = torch.tensor([encode_str(input_text, max_len)], dtype=torch.long, device=DEVICE)
    colors = {k: c for k, c in zip(model_keys, _distinct_colors(len(model_keys)))}

    # ----- Encoder overlay -----
    fig1 = plt.figure(figsize=(10, 7)); ax1 = fig1.add_subplot(111, projection="3d")
    handles = []; handled = False
    for idx, k in enumerate(model_keys):
        if k not in models: 
            continue
        A = _enc_attn_layer0_headavg(models[k], src)
        if A is None: 
            continue
        A = A + (0.004 * idx)  # z-offset to disambiguate
        T = A.shape[0]; X, Y = _mesh(T, T)
        if style == "surface":
            ax1.plot_surface(X, Y, A, rstride=1, cstride=1, linewidth=0.6,
                             antialiased=True, shade=False, alpha=0.45, color=colors[k])
        else:
            r = max(1, T // 18)
            ax1.plot_wireframe(X, Y, A, rstride=r, cstride=r,
                               color=colors[k], linewidth=1.2, alpha=alpha)
        handles.append(Line2D([0], [0], color=colors[k], lw=2, label=k))
        handled = True
    if handled:
        ax1.set_title(f"Encoder L0 Self-Attn (overlaid)\ninput='{input_text}'")
        ax1.set_xlabel("Query pos"); ax1.set_ylabel("Key pos"); ax1.set_zlabel("Attention")
        ax1.view_init(elev=elev, azim=azim); ax1.legend(handles=handles, loc="upper left")
        fig1.savefig(os.path.join(out_dir, f"{save_stem}_encoder.png"),
                     dpi=220, bbox_inches="tight")
        fig1.savefig(os.path.join(out_dir, f"{save_stem}_encoder.pdf"),
                     bbox_inches="tight")
    plt.close(fig1)

    # ----- Decoder overlay -----
    fig2 = plt.figure(figsize=(10, 7)); ax2 = fig2.add_subplot(111, projection="3d")
    handles = []; handled = False
    for idx, k in enumerate(model_keys):
        if k not in models: 
            continue
        S = _dec_self_attn_final(models[k], src, max_len=max_len)
        if S is None: 
            continue
        S = S + (0.004 * idx)
        Tt = S.shape[0]; Xd, Yd = _mesh(Tt, Tt)
        if style == "surface":
            ax2.plot_surface(Xd, Yd, S, rstride=1, cstride=1, linewidth=0.6,
                             antialiased=True, shade=False, alpha=0.45, color=colors[k])
        else:
            r = max(1, Tt // 18)
            ax2.plot_wireframe(Xd, Yd, S, rstride=r, cstride=r,
                               color=colors[k], linewidth=1.2, alpha=alpha)
        handles.append(Line2D([0], [0], color=colors[k], lw=2, label=k))
        handled = True
    if handled:
        ax2.set_title(f"Decoder L0 Self-Attn (final, overlaid)\ninput='{input_text}'")
        ax2.set_xlabel("Query pos (tgt)"); ax2.set_ylabel("Key pos (tgt)"); ax2.set_zlabel("Attention")
        ax2.view_init(elev=elev, azim=azim); ax2.legend(handles=handles, loc="upper left")
        fig2.savefig(os.path.join(out_dir, f"{save_stem}_decoder_self.png"),
                     dpi=220, bbox_inches="tight")
        fig2.savefig(os.path.join(out_dir, f"{save_stem}_decoder_self.pdf"),
                     bbox_inches="tight")
    plt.close(fig2)


@torch.no_grad()
def plot_crossdecode_attn_3d(
    models: Dict[str, torch.nn.Module],
    input_text: str,
    encoder_key: str,
    decoder_keys: List[str],
    elev: int = 35,
    azim: int = -60,
    alpha: float = 0.6,
    out_dir: str = "plots",
    save_stem: str = "crossdecode"
):
    """
    3D overlays of decoder self- and cross-attn when decoding memory produced by a specific encoder.
    Saves PNG/PDF.
    """
    assert encoder_key in models, f"Encoder key '{encoder_key}' not found."
    assert isinstance(decoder_keys, (list, tuple)) and decoder_keys
    _ensure_dir(out_dir)

    max_len = models[encoder_key].cfg.max_len
    src = torch.tensor([encode_str(input_text, max_len)], dtype=torch.long, device=DEVICE)
    memory, src_key_pad, _ = models[encoder_key].encode(src, need_attn=False)

    # Decoder SELF
    fig1 = plt.figure(figsize=(10, 7)); ax1 = fig1.add_subplot(111, projection="3d")
    handled = False
    for dk in decoder_keys:
        if dk not in models:
            continue
        S, C = _dec_self_and_cross_final(models[dk], memory, src_key_pad, max_len=max_len)
        if S is None:
            continue
        Tt = S.shape[0]; X, Y = _mesh(Tt, Tt)
        ax1.plot_surface(X, Y, S, rstride=1, cstride=1, linewidth=0.5,
                         antialiased=True, alpha=alpha)
        handled = True
    if handled:
        ax1.set_title(f"Cross-Decoding: Decoder SELF L0 (final)\nenc='{encoder_key}', input='{input_text}'")
        ax1.set_xlabel("Query pos (tgt)"); ax1.set_ylabel("Key pos (tgt)"); ax1.set_zlabel("Attention")
        ax1.view_init(elev=elev, azim=azim)
        fig1.savefig(os.path.join(out_dir, f"{save_stem}_self.png"), dpi=220, bbox_inches="tight")
        fig1.savefig(os.path.join(out_dir, f"{save_stem}_self.pdf"), bbox_inches="tight")
    plt.close(fig1)

    # Decoder CROSS
    fig2 = plt.figure(figsize=(10, 7)); ax2 = fig2.add_subplot(111, projection="3d")
    handled = False
    for dk in decoder_keys:
        if dk not in models:
            continue
        S, C = _dec_self_and_cross_final(models[dk], memory, src_key_pad, max_len=max_len)
        if C is None:
            continue
        Tt, Ts = C.shape; Xc, Yc = _mesh(Tt, Ts)
        ax2.plot_surface(Xc, Yc, C, rstride=1, cstride=1, linewidth=0.5,
                         antialiased=True, alpha=alpha)
        handled = True
    if handled:
        ax2.set_title(f"Cross-Decoding: Decoder CROSS L0 (final)\nenc='{encoder_key}', input='{input_text}'")
        ax2.set_xlabel("Decoder query (tgt)"); ax2.set_ylabel("Encoder key (src)"); ax2.set_zlabel("Attention")
        ax2.view_init(elev=elev, azim=azim)
        fig2.savefig(os.path.join(out_dir, f"{save_stem}_cross.png"), dpi=220, bbox_inches="tight")
        fig2.savefig(os.path.join(out_dir, f"{save_stem}_cross.pdf"), bbox_inches="tight")
    plt.close(fig2)


@torch.no_grad()
def plot_cross_attn_heatmaps(
    models: Dict[str, torch.nn.Module],
    input_text: str,
    encoder_key: str,
    decoder_keys: List[str],
    out_dir: str = "plots",
    save_prefix: str = "cross_heat"
):
    """
    Per-decoder, per-head cross-attn heatmaps (final step).
    Saves one multi-panel PNG per decoder.
    """
    assert encoder_key in models
    _ensure_dir(out_dir)

    max_len = models[encoder_key].cfg.max_len
    src = torch.tensor([encode_str(input_text, max_len)], dtype=torch.long, device=DEVICE)
    memory, src_key_pad, _ = models[encoder_key].encode(src, need_attn=False)

    for dk in decoder_keys:
        if dk not in models:
            print(f"[WARN] {dk} missing")
            continue

        # Collect final cross-attn (per head)
        B = memory.size(0)
        y = torch.full((B, 1), BOS_ID, dtype=torch.long, device=memory.device)
        last_cross = None
        for _ in range(max_len - 1):
            y_emb = models[dk].pos_enc(models[dk].tgt_embed(y))
            y_pad = (y == PAD_ID)
            tgt_mask = subsequent_mask(y.size(1), y.device)
            dec_out, self_attn, cross_attn = models[dk].decoder(
                y_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=y_pad,
                memory_key_padding_mask=src_key_pad, need_attn=True
            )
            if cross_attn is not None:
                last_cross = cross_attn  # [B, heads, T_tgt, T_src]
            next_token = models[dk].generator(dec_out[:, -1:, :]).argmax(-1)
            y = torch.cat([y, next_token], dim=1)
            if (next_token == EOS_ID).all():
                break

        if last_cross is None:
            print(f"[WARN] no cross-attn for {dk}")
            continue

        X = last_cross[0].detach().cpu().numpy()  # [heads, T_tgt, T_src]
        H = X.shape[0]
        fig, axes = plt.subplots(1, H, figsize=(4*H, 3), squeeze=False)
        for h in range(H):
            axes[0, h].imshow(X[h], aspect='auto', origin='lower')
            axes[0, h].set_title(f"{dk} cross-attn | head {h}")
            axes[0, h].set_xlabel("src (encoder) pos"); axes[0, h].set_ylabel("tgt (decoder) pos")
        fig.suptitle(f"Encode={encoder_key}  |  input='{input_text}'")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{save_prefix}_{encoder_key}_to_{dk}.png"),
                    dpi=220, bbox_inches="tight")
        plt.close(fig)


@torch.no_grad()
def plot_decoder_self_per_head(
    models: Dict[str, torch.nn.Module],
    input_text: str,
    model_key: str,
    out_dir: str = "plots",
    save_prefix: str = "dec_self_heads"
):
    """
    Per-head decoder self-attn (final step) for a single model.
    """
    assert model_key in models
    _ensure_dir(out_dir)

    max_len = models[model_key].cfg.max_len
    src = torch.tensor([encode_str(input_text, max_len)], dtype=torch.long, device=DEVICE)

    memory, src_key_pad, _ = models[model_key].encode(src, need_attn=False)
    B = memory.size(0)
    y = torch.full((B, 1), BOS_ID, dtype=torch.long, device=memory.device)
    last_self = None
    for _ in range(max_len - 1):
        y_emb = models[model_key].pos_enc(models[model_key].tgt_embed(y))
        y_pad = (y == PAD_ID)
        tgt_mask = subsequent_mask(y.size(1), y.device)
        dec_out, self_attn, cross_attn = models[model_key].decoder(
            y_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=y_pad,
            memory_key_padding_mask=src_key_pad, need_attn=True
        )
        if self_attn is not None:
            last_self = self_attn  # [B, heads, T, T]
        y = torch.cat([y, models[model_key].generator(dec_out[:, -1:, :]).argmax(-1)], dim=1)

    if last_self is None:
        print("[WARN] no self-attn captured"); 
        return

    A = last_self[0].detach().cpu().numpy()  # [heads, T, T]
    H = A.shape[0]
    fig, axes = plt.subplots(1, H, figsize=(4*H, 3), squeeze=False)
    for h in range(H):
        axes[0, h].imshow(A[h], aspect='auto', origin='lower')
        axes[0, h].set_title(f"{model_key} self-attn | head {h}")
        axes[0, h].set_xlabel("key (tgt) pos"); axes[0, h].set_ylabel("query (tgt) pos")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{save_prefix}_{model_key}.png"),
                dpi=220, bbox_inches="tight")
    plt.close(fig)
