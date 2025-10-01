from dataclasses import dataclass
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 50
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 8
    lr: float = 3e-4
    wd: float = 0.0
