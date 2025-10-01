import random
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from .vocab import ITOS, PAD_ID, encode_str

class IdentityDataset(Dataset):
    def __init__(self, n_samples=6000, min_len=8, max_len=30, seed=1234):
        rng = random.Random(seed)
        pool = ITOS[3:]  # exclude specials
        self.samples = []
        for _ in range(n_samples):
            L = rng.randint(min_len, max_len)
            s = ''.join(rng.choice(pool) for _ in range(L))
            self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s, s  # identity mapping

def collate_fn(batch, max_len=50) -> Tuple[torch.Tensor, torch.Tensor]:
    src_ids = [encode_str(s, max_len) for s, _ in batch]
    tgt_ids = [encode_str(t, max_len) for _, t in batch]
    max_src = max(len(x) for x in src_ids)
    max_tgt = max(len(x) for x in tgt_ids)

    def pad_to(x, L):
        return x + [PAD_ID] * (L - len(x))

    src = torch.tensor([pad_to(x, max_src) for x in src_ids], dtype=torch.long)
    tgt = torch.tensor([pad_to(x, max_tgt) for x in tgt_ids], dtype=torch.long)
    return src, tgt

def make_loaders(batch_size: int, max_len: int):
    train_ds = IdentityDataset(n_samples=6000, min_len=8, max_len=30, seed=2025)
    test_ds  = IdentityDataset(n_samples=800,  min_len=8, max_len=30, seed=9090)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, max_len))
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda b: collate_fn(b, max_len))
    return train_loader, test_loader
