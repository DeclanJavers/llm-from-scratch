# tiny-lm

Pretraining a ~200M parameter LLM from scratch. 

## Plan

1. **Tokenizer** (Mac) — train a 32,768-vocab byte-level BPE on a FineWeb-Edu
   sample. Special tokens reserved up front: `<|endoftext|>`, `<|pad|>`,
   `<|im_start|>`, `<|im_end|>` (the last two are for the SFT stage later).
2. **Data** (Mac) — stream `HuggingFaceFW/fineweb-edu` `sample-10BT` (no local
   parquet cache — disk is tight), tokenize to uint16 `.bin` shards of 100M
   tokens each, upload shards to a HF Hub dataset repo as they finish.
3. **Model + train code** (Mac) — nanoGPT-style GPT in plain PyTorch. Runs on
   `mps` for smoke tests, `cuda` on Colab. Must resume cleanly from checkpoint
   (Colab sessions die).
4. **Pretrain** (Colab) — A100 preferred. Checkpoints to Google Drive every
   ~25 min. Budget: 8–10B tokens ≈ ~25 A100-hours.
5. **SFT** (Colab) — open-book QA + summarization mix (SQuAD 2.0, MS MARCO,
   CNN/DailyMail, SAMSum, DialogSum). Joint training, task prefixes.
6. **Eval** — held-out loss + ROUGE-L (summarization), EM/F1 (QA), and a
   "says 'not stated' when it should" check.

## Architecture target

GPT-2-style decoder-only, RoPE, ~200M class. Working numbers (finalized in
`src/model.py`): 16 layers, d_model 1024, 16 heads, ctx 2048, tied embeddings,
vocab 32,768 → ~235M params. Trim layers if we want to land closer to 200M.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
huggingface-cli login   # needed for shard upload
```

## Usage

```bash
# 1. Train tokenizer on ~2GB of streamed text (~20 min)
python src/train_tokenizer.py

# 2. Tokenize + shard + upload (streams, ~2-4 hrs, disk-light)
python src/tokenize_shards.py --max-tokens 10e9 \
    --hub-repo <you>/fineweb-edu-10b-tok32k --delete-after-upload
```

## Disk note

This Mac has ~20GB free. Everything streams; peak local disk usage with
`--delete-after-upload` is a couple of shards (~400MB). Do not `load_dataset`
without `streaming=True`.
