"""Build the FROZEN eval set from SQuAD 2.0's validation split.

Deterministic: same seed -> byte-identical file. This set is the eval gate.
Nothing is ever trained against it, resampled against it, or mined from it —
that includes synthetic-data filtering. Goodhart insurance.

Writes data/eval/squad2_frozen.jsonl, one example per line:
    {"id": ..., "question": ..., "document": ..., "answers": [...]}
answers == [] means the question is unanswerable (the model should abstain).
"""
import argparse
import json
import os
import random

from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-answerable", type=int, default=1000)
    ap.add_argument("--n-unanswerable", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/eval/squad2_frozen.jsonl")
    ap.add_argument("--exclude", help="jsonl whose ids to keep out (build the dev set "
                    "disjoint from the frozen set: --seed 1 --exclude data/eval/squad2_frozen.jsonl "
                    "--out data/eval/squad2_dev.jsonl)")
    args = ap.parse_args()

    excluded = set()
    if args.exclude:
        with open(args.exclude) as f:
            excluded = {json.loads(line)["id"] for line in f}

    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    answerable, unanswerable = [], []
    for ex in ds:
        if ex["id"] in excluded:
            continue
        row = {
            "id": ex["id"],
            "question": ex["question"],
            "document": ex["context"],
            # dedupe gold spans, keep order stable
            "answers": sorted(set(ex["answers"]["text"])),
        }
        (answerable if row["answers"] else unanswerable).append(row)

    rng = random.Random(args.seed)
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    rows = answerable[: args.n_answerable] + unanswerable[: args.n_unanswerable]
    rows.sort(key=lambda r: r["id"])   # order independent of sampling internals

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"wrote {len(rows)} examples -> {args.out}")
    print(f"  answerable:   {min(len(answerable), args.n_answerable)}")
    print(f"  unanswerable: {min(len(unanswerable), args.n_unanswerable)}")

if __name__ == "__main__":
    main()
