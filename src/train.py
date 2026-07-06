"""Train the GPT on the tokenized shards.

Device-aware: uses CUDA if present (Colab), else Apple MPS (Mac), else CPU.

    # local smoke test (Mac) -- loss should tick DOWN from ~10.5 in a minute or two:
    .venv/bin/python3 src/train.py --max-iters 20 --eval-interval 10 --eval-iters 5 --batch-size 4

    # real run (Colab GPU):
    python src/train.py --max-iters 6000 --batch-size 24 --block-size 256
"""
import argparse
import math
import os
import time

import numpy as np
import torch

from data import get_batch
from model import GPT, GPTConfig


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, eval_iters, batch_size, block_size, device):
    """Average loss over a few random batches of train and val (no grad)."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, warmup, max_iters, lr, min_lr):
    """Linear warmup, then cosine decay from lr down to min_lr."""
    if it < warmup:
        return lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    ratio = (it - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))   # 1 -> 0
    return min_lr + coeff * (lr - min_lr)


def configure_optimizer(model, lr, weight_decay, betas):
    """Weight-decay the 2D matrices (matmuls, embeddings); leave biases and
    LayerNorm gains undecayed. Standard Transformer practice."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches to accumulate before a step (sim. bigger batch)")
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or pick_device()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device: {device}")

    # bfloat16 autocast on CUDA is a big speed/memory win; keep it simple elsewhere.
    use_amp = device == "cuda"
    amp_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else torch.autocast(device_type="cpu", enabled=False))

    config = GPTConfig(block_size=args.block_size)
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params")

    optimizer = configure_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    best_val = float("inf")
    t0 = time.time()

    for it in range(args.max_iters + 1):
        # set this step's learning rate
        lr = get_lr(it, args.warmup, args.max_iters, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # periodic eval + checkpoint
        if it % args.eval_interval == 0:
            losses = estimate_loss(model, args.eval_iters, args.batch_size,
                                   args.block_size, device)
            dt = time.time() - t0
            print(f"iter {it:5d} | train {losses['train']:.3f} | "
                  f"val {losses['val']:.3f} | lr {lr:.2e} | {dt:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "iter": it,
                    "best_val": best_val,
                }
                torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))
                print(f"  saved checkpoint (val {best_val:.3f})")

        if it == args.max_iters:
            break

        # one optimization step, with optional gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch("train", args.batch_size, args.block_size, device)
            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum   # average across micro-batches
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    print(f"done. best val loss: {best_val:.3f}")


if __name__ == "__main__":
    main()
