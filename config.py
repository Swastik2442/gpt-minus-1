import os
import torch

batch_size = 8
vocab_size = 50257 # GPT-2 Vocab
n_ctx = 256
d_model = 256
n_heads = 4
n_layers = 2
weight_decay = 0.1
num_warmup_steps = 1000
num_lr_decay_steps = 2000
min_lr = 6e-5
max_lr = 6e-4
iters = 1000
device_type = "cuda" # or cpu
device = "cuda" if torch.cuda.is_available() and device_type == "cuda" else "cpu"
dataset = "shakespeare" # or hiwiki
data_dir = os.path.join('data', dataset)
