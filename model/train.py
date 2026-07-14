"""Pretraining loop per docs/PRETRAIN_BRIEF.md.

Muon on 2D block matrices + AdamW on embeddings/head/norms, WSD schedule
aligned with the stable->anneal data phase switch, bf16 autocast on CUDA,
grad accumulation to ~0.5M-token batches, fully resumable checkpoints
(model+optimizers+step; data sampling is stateless-deterministic in the
step number, so resume is exact), CSV + stdout logging with MFU.

M2 evidence (runs anywhere, no shards needed):
    python model/train.py --synthetic --overfit 300

Colab full run (downloads shards from the HF repo, then trains):
    python model/train.py --data-repo declan41/tinylm-shards \
        --out model/runs/full3b [--resume auto] [--compile]

Smoke on a real shard download:
    python model/train.py --data-repo ... --max-steps 20 --micro-batch 4
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import Net, NetConfig  # noqa: E402

# ------------------------------------------------------------------ config

@dataclass
class TrainConfig:
    total_tokens: int = 3_000_000_000
    batch_tokens: int = 524_288          # 256 x 2048 via grad accumulation
    seq_len: int = 2048
    stable_frac: float = 0.85            # anneal + LR decay start here
    warmup_frac: float = 0.005
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_wd: float = 0.01                # weight decay on matrices only
    adam_emb_lr: float = 3e-3
    adam_head_lr: float = 1e-3
    adam_other_lr: float = 1e-3
    adam_betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every_tokens: int = 250_000_000
    eval_rows: int = 64                  # rows per val group per eval
    ckpt_every_steps: int = 500
    seed: int = 0

    @property
    def total_steps(self):
        return self.total_tokens // self.batch_tokens

    @property
    def stable_end_step(self):
        return int(self.total_steps * self.stable_frac)


PEAK_BF16 = {"A100": 312e12, "H100": 989e12, "L4": 121e12,
             "4090": 165e12, "5090": 209e12, "T4": 65e12}

# ------------------------------------------------------------------ muon

def zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalize G via quintic Newton-Schulz (Keller Jordan's Muon)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.type_as(G)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                o = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(o, alpha=-group["lr"] * scale)


def build_optimizers(net, tc):
    matrices = [p for blk in net.blocks for p in blk.parameters() if p.ndim == 2]
    muon = Muon(matrices, lr=tc.muon_lr, momentum=tc.muon_momentum,
                nesterov=True, weight_decay=tc.muon_wd)
    norms = [p for n, p in net.named_parameters()
             if p.ndim == 1]
    adam = torch.optim.AdamW([
        {"params": [net.embed.weight], "lr": tc.adam_emb_lr},
        {"params": [net.head.weight], "lr": tc.adam_head_lr},
        {"params": norms, "lr": tc.adam_other_lr},
    ], betas=tc.adam_betas, weight_decay=0.0)
    n_m = sum(p.numel() for p in matrices)
    print(f"muon: {len(matrices)} matrices ({n_m/1e6:.1f}M params); "
          f"adamw: emb+head+{len(norms)} norms")
    return muon, adam


def lr_mult(step, tc):
    warm = max(1, int(tc.total_steps * tc.warmup_frac))
    if step < warm:
        return (step + 1) / warm
    if step < tc.stable_end_step:
        return 1.0
    left = tc.total_steps - step
    return max(0.0, left / (tc.total_steps - tc.stable_end_step))

# ------------------------------------------------------------------ data

class Group:
    def __init__(self, path, seq_len):
        meta = json.loads((path / "meta.json").read_text())
        maps = [np.memmap(path / s["file"], dtype=np.uint16, mode="r")
                for s in meta["shards"]]
        self.maps = [m for m in maps if len(m) > seq_len + 1]
        assert self.maps, f"group {path.name}: no shard is >= one window"
        self.sizes = np.array([len(m) for m in self.maps], dtype=np.int64)
        self.total = int(self.sizes.sum())

    def window(self, rng, seq_len):
        i = rng.choice(len(self.maps), p=self.sizes / self.sizes.sum())
        off = rng.integers(0, self.sizes[i] - seq_len - 1)
        return np.asarray(self.maps[i][off:off + seq_len + 1], dtype=np.int64)

    def fixed_window(self, k, seq_len):
        """k-th deterministic eval window, evenly spaced over the group."""
        stride = max(seq_len + 1, (self.total - seq_len - 1) // 1024)
        off = (k * stride) % (self.total - seq_len - 1)
        i = 0
        while off >= self.sizes[i] - seq_len - 1:
            off -= max(1, self.sizes[i] - seq_len - 1)
            i = (i + 1) % len(self.maps)
        return np.asarray(self.maps[i][off:off + seq_len + 1], dtype=np.int64)


class Data:
    """Stateless-deterministic sampler: row n of the run maps to a fixed
    (group, offset) via a seeded RNG, so resume-by-step is exact."""

    def __init__(self, shards_dir, tc, synthetic=False):
        self.tc = tc
        self.synthetic = synthetic
        if synthetic:
            rng = np.random.default_rng(0)
            self.fake = rng.integers(0, 32768, size=4_000_000).astype(np.int64)
            return
        man = json.loads((shards_dir / "manifest.json").read_text())
        self.groups = {g: Group(shards_dir / g, tc.seq_len)
                       for g in man["groups"]}
        self.phases = man["phases"]
        for ph in self.phases:
            names = sorted(ph["mix"])
            ph["_names"] = names
            ph["_probs"] = np.array([ph["mix"][n] for n in names])
            ph["_probs"] /= ph["_probs"].sum()

    def row(self, n, phase_idx):
        rng = np.random.default_rng([self.tc.seed, n])
        if self.synthetic:
            off = rng.integers(0, len(self.fake) - self.tc.seq_len - 1)
            return self.fake[off:off + self.tc.seq_len + 1]
        ph = self.phases[phase_idx]
        g = rng.choice(ph["_names"], p=ph["_probs"])
        return self.groups[g].window(rng, self.tc.seq_len)

    def batch(self, step, micro_idx, mb, device):
        phase = 0 if step < self.tc.stable_end_step else 1
        rows_per_step = self.tc.batch_tokens // self.tc.seq_len
        base = step * rows_per_step + micro_idx * mb
        rows = np.stack([self.row(base + r, phase) for r in range(mb)])
        t = torch.from_numpy(rows).to(device, non_blocking=True)
        return t[:, :-1], t[:, 1:]

    def val_batches(self, group, tc, device):
        g = self.groups[group]
        for k in range(0, tc.eval_rows, 8):
            rows = np.stack([g.fixed_window(k + j, tc.seq_len)
                             for j in range(min(8, tc.eval_rows - k))])
            t = torch.from_numpy(rows).to(device)
            yield t[:, :-1], t[:, 1:]

# ------------------------------------------------------------------ ckpt

def save_ckpt(out, name, net, muon, adam, step, tc, keep=3):
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "muon": muon.state_dict(),
                "adam": adam.state_dict(), "step": step, "cfg": asdict(tc)},
               out / f"{name}.pt")
    tenth = max(1, tc.total_steps // 10)
    numbered = sorted(out.glob("ckpt_[0-9]*.pt"))
    protect = {p for p in numbered
               if int(p.stem.split("_")[1]) % tenth == 0}
    for p in numbered[:-keep]:
        if p not in protect:
            p.unlink()


def load_ckpt(path, net, muon, adam, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(ck["model"])
    muon.load_state_dict(ck["muon"])
    adam.load_state_dict(ck["adam"])
    print(f"resumed from {path} at step {ck['step']}")
    return ck["step"]

# ------------------------------------------------------------------ misc

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def peak_flops(device):
    if device != "cuda":
        return None
    name = torch.cuda.get_device_name()
    for k, v in PEAK_BF16.items():
        if k in name:
            return v
    return None


def probe_micro_batch(net, data, tc, device, ctx):
    if device != "cuda":
        return 4
    for mb in (64, 48, 32, 24, 16, 12, 8, 4, 2, 1):
        if mb * tc.seq_len > tc.batch_tokens:
            continue
        try:
            x, y = data.batch(0, 0, mb, device)
            with ctx:
                _, loss = net(x, y)
            loss.backward()
            net.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"micro-batch auto-tuned to {mb}")
            return mb
        except torch.cuda.OutOfMemoryError:
            net.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
    raise RuntimeError("no micro-batch fits")


def maybe_download(repo):
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    path = Path(snapshot_download(repo, repo_type="dataset"))
    return path / "shards", path / "tokenizer.json"


def sample_texts(net, tok_path, device, eot_id):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))
    prompts = ["The most important thing about", "In 1969, the first",
               "Water is made of", "The capital of France is",
               "To solve the equation", "A neural network is",
               "The history of the Roman Empire", "Photosynthesis converts"]
    outs = []
    for p in prompts:
        ids = torch.tensor([tok.encode(p).ids], device=device)
        y = net.generate(ids, 48, eot_id=eot_id)
        outs.append(tok.decode(y[0].tolist(), skip_special_tokens=False))
    return outs

# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model/runs/dev")
    ap.add_argument("--data-dir", default="model/data/shards")
    ap.add_argument("--data-repo", help="HF dataset repo to download shards from")
    ap.add_argument("--resume", help="'auto' or a checkpoint path")
    ap.add_argument("--micro-batch", type=int, default=0, help="0 = auto")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--overfit", type=int, default=0,
                    help="train N steps on ONE fixed batch (M2 evidence)")
    ap.add_argument("--synthetic", action="store_true",
                    help="random tokens instead of shards (tests, overfit)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--total-tokens", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=0,
                    help="override effective batch (tests only)")
    args = ap.parse_args()

    tc = TrainConfig()
    if args.total_tokens:
        tc.total_tokens = args.total_tokens
    if args.batch_tokens:
        tc.batch_tokens = args.batch_tokens
    device = pick_device()
    torch.manual_seed(tc.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    net = Net(NetConfig(seq_len=tc.seq_len)).to(device)
    n = net.num_params()
    print(f"params: {n:,} ({n/1e6:.1f}M) on {device}; "
          f"{tc.total_steps} steps of {tc.batch_tokens} tokens")
    assert 190e6 <= n <= 230e6

    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if device == "cuda" else torch.autocast("cpu", enabled=False))
    if args.compile and device == "cuda":
        net = torch.compile(net)

    tok_path = Path("model/tokenizer/tokenizer.json")
    shards_dir = Path(args.data_dir)
    if args.data_repo:
        shards_dir, tok_path = maybe_download(args.data_repo)
    data = Data(shards_dir, tc, synthetic=args.synthetic)

    muon, adam = build_optimizers(net, tc)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(tc), indent=1))

    start = 0
    if args.resume:
        path = (max(out.glob("ckpt_[0-9]*.pt"), default=None)
                if args.resume == "auto" else Path(args.resume))
        if path and Path(path).exists():
            start = load_ckpt(path, net, muon, adam, device)
        else:
            print("no checkpoint found; starting fresh")

    mb = args.micro_batch or probe_micro_batch(net, data, tc, device, ctx)
    accum = max(1, tc.batch_tokens // (mb * tc.seq_len))
    if args.overfit:
        accum = 1                       # memorization test: one real batch
    print(f"micro-batch {mb} x accum {accum} x seq {tc.seq_len} "
          f"= {mb*accum*tc.seq_len:,} tokens/step")
    peak = peak_flops(device)
    flops_n = (net._orig_mod if hasattr(net, "_orig_mod") else net).flops_params()

    logf = open(out / "log.csv", "a", newline="")
    logger = csv.writer(logf)
    if start == 0:
        logger.writerow(["step", "tokens", "loss", "lr_mult", "grad_norm",
                         "tok_per_s", "mfu"])

    end = min(x for x in (tc.total_steps,
                          start + args.max_steps if args.max_steps else 1 << 60,
                          args.overfit if args.overfit else 1 << 60))
    fixed = data.batch(0, 0, mb, device) if args.overfit else None
    next_eval = ((start * tc.batch_tokens) // tc.eval_every_tokens + 1) \
        * tc.eval_every_tokens
    t_last, tokens_since = time.time(), 0

    for step in range(start, end):
        m = lr_mult(step, tc)
        for group in muon.param_groups:
            group["lr"] = tc.muon_lr * m
        for group, base in zip(adam.param_groups,
                               (tc.adam_emb_lr, tc.adam_head_lr,
                                tc.adam_other_lr)):
            group["lr"] = base * m

        loss_acc = 0.0
        for micro in range(accum):
            x, y = fixed if fixed is not None else data.batch(step, micro,
                                                              mb, device)
            with ctx:
                _, loss = net(x, y)
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        gnorm = nn.utils.clip_grad_norm_(net.parameters(), tc.grad_clip)
        muon.step()
        adam.step()
        net.zero_grad(set_to_none=True)

        tokens_since += mb * accum * tc.seq_len
        tokens_done = (step + 1) * tc.batch_tokens
        if step < start + 5 or (step + 1) % 10 == 0:
            dt = time.time() - t_last
            tps = tokens_since / max(dt, 1e-9)
            mfu = (tps * 6 * flops_n / peak) if peak else 0.0
            print(f"step {step+1:>6}/{tc.total_steps}  loss {loss_acc:.4f}  "
                  f"lr x{m:.3f}  gnorm {gnorm:.2f}  {tps/1e3:.1f}k tok/s"
                  + (f"  MFU {mfu*100:.1f}%" if peak else ""), flush=True)
            logger.writerow([step + 1, tokens_done, f"{loss_acc:.4f}",
                             f"{m:.4f}", f"{float(gnorm):.3f}",
                             f"{tps:.0f}", f"{mfu:.4f}"])
            logf.flush()
            t_last, tokens_since = time.time(), 0

        if not args.overfit and not args.synthetic \
                and tokens_done >= next_eval:
            next_eval += tc.eval_every_tokens
            net.eval()
            for g in ("val_fineweb", "val_qa"):
                with torch.no_grad(), ctx:
                    losses = [net(x, y)[1].item()
                              for x, y in data.val_batches(g, tc, device)]
                print(f"  [eval] {g}: {sum(losses)/len(losses):.4f}", flush=True)
            if tok_path.exists():
                eot = None
                for s in sample_texts(net, tok_path, device, eot):
                    print(f"  [sample] {s[:160]!r}", flush=True)
            net.train()

        if (step + 1) % tc.ckpt_every_steps == 0 or step + 1 == end:
            save_ckpt(out, f"ckpt_{step+1:07d}", net, muon, adam,
                      step + 1, tc)
        if step + 1 == tc.stable_end_step:
            save_ckpt(out, "ckpt_stable_end", net, muon, adam, step + 1, tc)
            print("  [ckpt] saved ckpt_stable_end.pt — KEEP THIS: the 8B "
                  "extension protocol resumes from here", flush=True)

    if args.overfit:
        print(f"overfit final loss: {loss_acc:.4f} "
              f"({'PASS' if loss_acc < 0.1 else 'not yet <0.1'})")
    logf.close()


if __name__ == "__main__":
    main()
