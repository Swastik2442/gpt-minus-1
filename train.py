import os

import torch
import numpy as np

from config import *
from model import GPTMinus1, create_optimizer, create_scheduler, save_model

def get_batch(split: str):
    "From https://github.com/karpathy/nanoGPT/blob/master/train.py"
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - n_ctx, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+n_ctx]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+n_ctx]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def save_iter(iter_data: tuple[float, float, float | None], file_name: str = "iter_data.csv"):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'a', encoding="utf-8") as f:
        f.write(','.join([f"{(i) if i is not None else ''}" for i in iter_data]) + '\n')

def train_step(
    model: GPTMinus1,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: torch.nn.CrossEntropyLoss,
    inputs: torch.Tensor,
    targets: torch.Tensor
):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item()

def eval_step(
    model: GPTMinus1,
    criterion: torch.nn.CrossEntropyLoss,
    inputs: torch.Tensor,
    targets: torch.Tensor
):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    return loss.item()

if __name__ == "__main__":
    model = GPTMinus1(vocab_size, n_ctx, d_model, n_heads, n_layers, device=device).to(device)

    optimizer = create_optimizer(model, max_lr, weight_decay)
    scheduler = create_scheduler(optimizer, min_lr, max_lr, num_lr_decay_steps, num_warmup_steps)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    inputs, targets = get_batch('train')
    for itr in range(1, iters+1):
        loss = train_step(model, optimizer, scheduler, criterion, inputs, targets)
        inputs, targets = get_batch('train')

        if itr % 100 == 0:
            eval_loss = eval_step(model, criterion, inputs, targets)
            save_iter((itr, loss, eval_loss), "out/iter_data.csv")
            print(f"Iter {itr} | Training loss: {loss:.3f} | Evaluation loss: {eval_loss:.3f}")
            save_model(model, optimizer, scheduler, itr, f"out/model_checkpoint_{itr}.pth")
        else:
            save_iter((itr, loss, None), "out/iter_data.csv")
            print(f"Iter {itr} | Training loss: {loss:.3f}")
