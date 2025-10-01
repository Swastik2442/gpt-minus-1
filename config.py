import os
import json
from dataclasses import dataclass, field
from typing import Literal

import torch

@dataclass
class Config:
    batch_size: int = 1024
    vocab_size: int = 50257 # GPT-2 Vocab
    n_ctx: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    weight_decay: float = 0.1
    num_warmup_steps: int = 1000
    num_lr_decay_steps: int = 2000
    min_lr: float = 6e-5
    max_lr: float = 6e-4
    iters: int = 1000
    device: Literal["cuda", "mps", "xpu", "cpu", "meta"] = "cuda"
    dataset: Literal["hiwiki", "shakespeare"] = "shakespeare"

    # derived attributes
    data_dir: str = field(init=False, repr=False)

    def __post_init__(self):
        self.data_dir = os.path.join("data", self.dataset)

        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. Switching to CPU.")
            self.device = "cpu"
        elif self.device == "mps" and not torch.backends.mps.is_available():
            print("MPS is not available. Switching to CPU.")
            self.device = "cpu"
        elif self.device == "xpu" and not torch.xpu.is_available():
            print("XPU is not available. Switching to CPU.")
            self.device = "cpu"

    def save(self, path: str = "out/config.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding="utf-8") as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def load(path: str = "out/config.json") -> 'Config':
        with open(path, 'r', encoding="utf-8") as f:
            config_dict = json.load(f)
        return Config(**config_dict)
