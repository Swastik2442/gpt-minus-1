import os

import torch
import numpy as np

from model import GPTMinus1, create_optimizer, create_scheduler, save_model

batch_size = 8
vocab_size = 50257 # GPT-2 Vocab
n_ctx = 256
d_model = 256
n_heads = 4
n_layers = 2
lr = 1e-3
weight_decay = 0.01
epochs = 1000
num_warmup_steps = 4000
device_type = "cpu" # or cuda
device = "cuda" if torch.cuda.is_available() and device_type == "cuda" else "cpu"
dataset = "shakespeare" # or hiwiki
data_dir = os.path.join('data', dataset)

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

def save_epochs(epoch_data: tuple[float, float, float | None], file_name: str = "epoch_data.csv"):
    with open(file_name, 'a', encoding="utf-8") as f:
        f.write(','.join([f"{i}" for i in epoch_data if i is not None]) + '\n')

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

    optimizer = create_optimizer(model, lr, weight_decay)
    scheduler = create_scheduler(optimizer, d_model, num_warmup_steps)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    inputs, targets = get_batch('train')
    for epoch in range(1, epochs+1):
        loss = train_step(model, optimizer, scheduler, criterion, inputs, targets)
        inputs, targets = get_batch('train')

        if epoch % 100 == 0:
            eval_loss = eval_step(model, criterion, inputs, targets)
            save_epochs((epoch, loss, eval_loss), "out/epoch_data.csv")
            print(f"Epoch {epoch} | Training loss: {loss:.3f} | Evaluation loss: {eval_loss:.3f}")
            save_model(model, optimizer, scheduler, epoch, f"out/model_checkpoint_{epoch}.pth")
        else:
            save_epochs((epoch, loss, None), "out/epoch_data.csv")
            print(f"Epoch {epoch} | Training loss: {loss:.3f}")
