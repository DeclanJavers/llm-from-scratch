"""Train a 32k byte-level BPE tokenizer on a streamed FineWeb-Edu sample.

Streams documents (no local dataset cache) until ~2GB of text is collected,
trains BPE, and writes tokenizer/tokenizer.json.

    # cheap plumbing test:
    python src/train_tokenizer.py --sample-gb 0.05 --vocab-size 8192 --out /tmp/test_tok.json
    # real run:
    python src/train_tokenizer.py --sample-gb 2.0 --vocab-size 32768 --out tokenizer/tokenizer.json
"""

import argparse

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]


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


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--sample-gb", type=float, default=2.0)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default="tokenizer/tokenizer.json")
    args = ap.parse_args()

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=args.min_frequency,
        show_progress=True,
    )

    tok.train_from_iterator(
        text_iterator(int(args.sample_gb * 1e9)), trainer=trainer
    )
    tok.save(args.out)

    # --- sanity block: a tokenizer that "runs" can still be silently wrong ---
    print(f"\nsaved {args.out}")
    print(f"vocab size: {tok.get_vocab_size()}")
    for t in SPECIAL_TOKENS:
        print(f"  {t!r} -> id {tok.token_to_id(t)}")

    sample = "The year 2025 costs $42."
    enc = tok.encode(sample)
    print(f"\nencode {sample!r}")
    print(f"  tokens: {enc.tokens}")
    print(f"  ids:    {enc.ids}")
    roundtrip = tok.decode(enc.ids)
    print(f"  decode: {roundtrip!r}")
    print(f"  round-trip ok: {roundtrip == sample}")


if __name__ == "__main__":
    main()
