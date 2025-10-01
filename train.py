import os

import torch
import numpy as np

from config import Config
from model import GPTMinus1, create_optimizer, create_scheduler, save_metadata#, load_metadata

def get_batch(config: Config, split: str):
    "From https://github.com/karpathy/nanoGPT/blob/master/train.py"
    if split == 'train':
        data = np.memmap(os.path.join(config.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(config.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - config.n_ctx, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.n_ctx]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.n_ctx]).astype(np.int64)) for i in ix])
    if config.device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(config.device, non_blocking=True), y.pin_memory().to(config.device, non_blocking=True)
    else:
        x, y = x.to(config.device), y.to(config.device)
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
    cfg = Config()
    cfg.save()

    model = GPTMinus1(
        cfg.vocab_size,
        cfg.n_ctx,
        cfg.d_model,
        cfg.n_heads,
        cfg.n_layers,
        device=cfg.device
    ).to(cfg.device)

    optimizer = create_optimizer(model, cfg.max_lr, cfg.weight_decay)
    scheduler = create_scheduler(optimizer, cfg.min_lr, cfg.max_lr, cfg.num_lr_decay_steps, cfg.num_warmup_steps)
    criterion = torch.nn.CrossEntropyLoss().to(cfg.device)

    # model.load("out/model_checkpoint_1000.pth")
    # load_metadata(optimizer, scheduler, "out/meta_checkpoint_1000.pth")

    inputs, targets = get_batch(cfg, 'train')
    for itr in range(1, cfg.iters + 1):
        loss = train_step(model, optimizer, scheduler, criterion, inputs, targets)
        inputs, targets = get_batch(cfg, 'train')

        if itr % 100 == 0:
            eval_loss = eval_step(model, criterion, inputs, targets)
            save_iter((itr, loss, eval_loss), "out/iter_data.csv")
            print(f"Iter {itr} | Training loss: {loss:.3f} | Evaluation loss: {eval_loss:.3f}")
            model.save(f"out/model_checkpoint_{itr}.pth")
            save_metadata(optimizer, scheduler, itr, f"out/meta_checkpoint_{itr}.pth")
        else:
            save_iter((itr, loss, None), "out/iter_data.csv")
            print(f"Iter {itr} | Training loss: {loss:.3f}")
