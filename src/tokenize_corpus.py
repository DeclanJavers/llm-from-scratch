"""Encode the streamed FineWeb-Edu sample into uint16 token-id shards.

Reuses the tokenizer trained by train_tokenizer.py. Streams documents,
encodes each one, separates them with <|endoftext|>, and writes the ids
to data/train.bin and data/val.bin as raw uint16 (2 bytes/token).

    # cheap plumbing test:
    python src/tokenize_corpus.py --sample-gb 0.02
    # real run:
    python src/tokenize_corpus.py --sample-gb 2.0
"""

import argparse
import json
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

FLUSH_EVERY = 1_000_000  # accumulate this many tokens before writing to disk


def text_iterator(sample_bytes: int):
    """Yield document text from FineWeb-Edu until ~sample_bytes have been seen."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    seen = 0
    for doc in ds:
        text = doc["text"]
        seen += len(text.encode("utf-8", errors="ignore"))
        yield text
        if seen >= sample_bytes:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32768)  # informational
    ap.add_argument("--sample-gb", type=float, default=2.0)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")  # the document-boundary id (0)
    assert eot is not None, "tokenizer has no <|endoftext|> token"

    buf = []
    total = 0        # total tokens produced
    val_written = 0  # tokens written to val.bin so far

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    with open(train_path, "wb") as f_train, open(val_path, "wb") as f_val:
        for text in text_iterator(int(args.sample_gb * 1e9)):
            ids = tok.encode(text).ids
            ids.append(eot)  # mark the end of this document
            buf.extend(ids)
            total += len(ids)

            if len(buf) >= FLUSH_EVERY:
                # Fill the validation shard first, then everything else trains.
                if val_written < args.val_tokens:
                    target, val_written = f_val, val_written + len(buf)
                else:
                    target = f_train
                np.array(buf, dtype=np.uint16).tofile(target)
                buf = []

        if buf:  # final leftover flush -> train
            np.array(buf, dtype=np.uint16).tofile(f_train)

    meta = {
        "total_tokens": total,
        "val_tokens": val_written,
        "train_tokens": total - val_written,
        "vocab_size": tok.get_vocab_size(),
        "dtype": "uint16",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"total tokens:  {total:,}")
    print(f"  train.bin:   {meta['train_tokens']:,} tokens")
    print(f"  val.bin:     {meta['val_tokens']:,} tokens")
    print(f"wrote {train_path}, {val_path}, and {args.out_dir}/meta.json")


if __name__ == "__main__":
    main()
