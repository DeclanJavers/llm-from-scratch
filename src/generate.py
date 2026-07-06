"""Generate text from a trained checkpoint (autoregressive sampling).

Feed a prompt, take the model's next-token distribution, sample one token,
append it, and repeat. Against a barely-trained checkpoint the output is
gibberish -- that's expected; it proves the sampling path works.

    .venv/bin/python3 src/generate.py --prompt "The city of" --max-new-tokens 100
"""
import argparse

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from model import GPT, GPTConfig   # importing GPTConfig also lets torch.load unpickle it


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # never feed more than block_size tokens of context
        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)          # (B, T, vocab)
        logits = logits[:, -1, :]            # only the last position predicts next
        logits = logits / temperature        # <1 sharpens, >1 flattens

        if top_k is not None:
            # keep only the top_k logits; zero out the rest's probability
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)   # sample, don't argmax
        idx = torch.cat((idx, next_id), dim=1)              # append and continue
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or pick_device()

    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")

    # rebuild the model from the saved config, then load the trained weights
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (trained to iter {ckpt['iter']}, val {ckpt['best_val']:.3f})")

    # encode the prompt; if empty, start from the document-boundary token
    ids = tok.encode(args.prompt).ids if args.prompt else [eot]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = generate(
        model, idx,
        max_new_tokens=args.max_new_tokens,
        block_size=config.block_size,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tok.decode(out[0].tolist())
    print("-" * 60)
    print(text)


if __name__ == "__main__":
    main()
