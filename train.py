import os

import torch
import numpy as np

from config import Config
from model import GPTMinus1

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
    import glob
    import argparse
    from typing import Literal, get_args, get_origin

    from config import __FIELDS__
    from model import create_optimizer, create_scheduler, save_metadata, load_metadata

    # Set CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config file (if None, default config is used)')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint (in config directory)')
    for carg in __FIELDS__:
        if get_origin(carg.type) is Literal:
            choices = get_args(carg.type)
            parser.add_argument(
                f'--{carg.name}',
                type=type(choices[0]),
                choices=choices,
                default=None,
                help=f'Override {carg.name} config (default: {carg.default})'
            )
        else:
            parser.add_argument(
                f'--{carg.name}',
                type=carg.type,
                default=None,
                help=f'Override {carg.name} config (default: {carg.default})'
            )
    args = parser.parse_args()

    if args.resume and args.config is None:
        print("Error: --config must be set when using --resume")
        exit(1)

    print("Loading Config...")
    out_dir = "out"
    if args.config:
        cfg = Config.load(args.config)
        out_dir = os.path.dirname(args.config)
    else:
        cfg = Config(**{
            carg.name: getattr(args, carg.name)
            for carg in __FIELDS__
            if getattr(args, carg.name) is not None
        })
        out_dir = os.path.join("out", cfg.dataset)
        cfg.save(os.path.join(out_dir, "config.json"))

    # Setup Model, Optimizer, Scheduler, Criterion
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
    start_itr = 1

    # Resume from latest checkpoint if available
    if args.config and args.resume:
        print("Resuming from latest checkpoint...")
        ckpt_list = glob.glob(os.path.join(out_dir, "model_checkpoint_*.pth"))
        if len(ckpt_list) == 0:
            print("No checkpoints found. Starting from scratch.\n")
        else:
            latest_ckpt = max(ckpt_list, key=os.path.getctime)
            print(f"Loading checkpoint: {latest_ckpt}\n")
            model.load(latest_ckpt)
            start_itr = load_metadata(optimizer, scheduler, latest_ckpt.replace("model_checkpoint", "meta_checkpoint"))
    else:
        print("Starting Training...\n")

    iter_data_file = os.path.join(out_dir, "iter_data.csv")
    inputs, targets = get_batch(cfg, 'train')
    for itr in range(cfg.iters):
        loss = train_step(model, optimizer, scheduler, criterion, inputs, targets)
        inputs, targets = get_batch(cfg, 'train')

        itr += start_itr
        if itr % 100 == 0:
            eval_loss = eval_step(model, criterion, inputs, targets)
            save_iter((itr, loss, eval_loss), iter_data_file)
            print(f"Iter {itr} | Training loss: {loss:.3f} | Evaluation loss: {eval_loss:.3f}")
            model.save(os.path.join(out_dir, f"model_checkpoint_{itr}.pth"))
            save_metadata(optimizer, scheduler, itr, os.path.join(out_dir, f"meta_checkpoint_{itr}.pth"))
        else:
            save_iter((itr, loss, None), iter_data_file)
            print(f"Iter {itr} | Training loss: {loss:.3f}")
