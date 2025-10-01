import time
import torch
import torch.nn as nn
from .vocab import PAD_ID
from .config import TrainConfig

def train_one(model: torch.nn.Module, loader, cfg_t: TrainConfig, max_len: int):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_t.lr, weight_decay=cfg_t.wd)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    for epoch in range(cfg_t.epochs):
        t0 = time.time()
        total, n_tok = 0.0, 0
        for src, tgt in loader:
            src = src.to(next(model.parameters()).device)
            tgt = tgt.to(next(model.parameters()).device)
            logits = model(src, tgt)
            gold = tgt[:, 1:].contiguous()
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * gold.numel()
            n_tok += gold.numel()
        print(f"Epoch {epoch+1}/{cfg_t.epochs} | nll={total/n_tok:.4f} | time={time.time()-t0:.1f}s")
