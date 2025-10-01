import math
import random
from typing import List, Dict
import numpy as np
import torch
import torch.nn.functional as F
from .vocab import PAD_ID, EOS_ID
from .config import DEVICE

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        if deterministic:
            torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def subsequent_mask(sz: int, device=None):
    return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

def trim_to_eos(ids: List[int]) -> List[int]:
    out = []
    for t in ids:
        if t == EOS_ID or t == PAD_ID:
            break
        out.append(t)
    return out

def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [list(range(m+1))]
    for i in range(1, n+1):
        dp.append([i] + [0]*m)
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = 0 if ai == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[n][m]

@torch.no_grad()
def weight_l2_distance(m1: torch.nn.Module, m2: torch.nn.Module) -> float:
    s = 0.0
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        s += (p1.detach() - p2.detach()).pow(2).sum().item()
    return math.sqrt(s)

@torch.no_grad()
def attention_divergence(model_a, model_b, loader, n_batches: int = 3) -> Dict[str, float]:
    model_a.eval(); model_b.eval()
    kls, coss = [], []
    batches = []
    for i, (src, _) in enumerate(loader):
        batches.append(src.to(DEVICE))
        if i+1 >= n_batches: break

    for src in batches:
        mem_a, pad_a, attn_a = model_a.encode(src, need_attn=True)   # [B, heads, T, T]
        mem_b, pad_b, attn_b = model_b.encode(src, need_attn=True)
        if attn_a is None or attn_b is None:
            continue
        A = attn_a.mean(dim=1)  # [B, T, T]
        B = attn_b.mean(dim=1)
        A = (A + 1e-9) / (A.sum(dim=-1, keepdim=True) + 1e-9)
        B = (B + 1e-9) / (B.sum(dim=-1, keepdim=True) + 1e-9)
        kl = (A * (A.clamp_min(1e-9).log() - B.clamp_min(1e-9).log())).sum(dim=-1).mean().item()
        a_flat = A.reshape(A.size(0), -1)
        b_flat = B.reshape(B.size(0), -1)
        cos = F.cosine_similarity(a_flat, b_flat, dim=-1).mean().item()
        kls.append(kl); coss.append(cos)

    return {
        "encoder_layer0_headavg_KL_A||B": float(np.mean(kls) if kls else float('nan')),
        "encoder_layer0_headavg_cosine": float(np.mean(coss) if coss else float('nan')),
    }
