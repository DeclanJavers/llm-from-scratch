# tiny-lm

Pretraining a ~200M parameter LLM from scratch for **extractive open-note
question answering**, built validator-first. The hypothesis: a small model
paired with a robust validator (sample, check, resample, or abstain) can
match a much larger model — target Qwen 3.5 8B — at a narrow, verifiable
task, at a fraction of the inference cost.

All design decisions (task definition, output schema, tokenizer, architecture,
training plan, validator tiers) live in [docs/DESIGN.md](docs/DESIGN.md) and
are authoritative. This README only covers what to run.

## Gate (build first)

Nothing in training gets designed until the gate exists and baselines are
measured. The gate lives in `src/validator.py` (the checks) and
`src/run_gate.py` (the harness that grades predictions against it).

```bash
# run the validator's self-tests
python src/validator.py

# rebuild the frozen eval set (data/eval/squad2_frozen.jsonl) -- this is
# already committed and hash-pinned in docs/DESIGN.md, so don't regenerate
# unless you know what you're doing; it will change the hash
python src/build_eval_set.py

# smoke-test the harness itself against built-in floor baselines
python src/run_gate.py --baseline abstain
python src/run_gate.py --baseline first_sentence

# grade real predictions: one JSON line per example, {"id": ..., "output": ...}
python src/run_gate.py --preds preds/<model>.jsonl
```

## Baselines via LM Studio

Point `gen_preds.py` at an LM Studio server ("lms server start" on the other
Mac, default port 1234) to make any model take the exam:

```bash
python src/gen_preds.py --base-url http://<mac>.local:1234/v1 --list-models

# smoke test first (8 examples), then the full frozen run
python src/gen_preds.py --base-url http://<mac>.local:1234/v1 \
    --model qwen3-0.6b --limit 8 --out preds/smoke.jsonl
python src/gen_preds.py --base-url http://<mac>.local:1234/v1 \
    --model qwen3-0.6b --out preds/qwen3-0.6b.jsonl
python src/run_gate.py --preds preds/qwen3-0.6b.jsonl
```

Validator development happens on the dev set (`data/eval/squad2_dev.jsonl`,
id-disjoint from the frozen set): generate dev predictions with
`--eval-set data/eval/squad2_dev.jsonl --out preds/dev/<model>.jsonl`, build
the labeled bench with `src/build_validator_bench.py --preds preds/dev/*.jsonl`
(then `--review` to hand-label the ambiguous band), and score the V2 checks
with `src/v2_checks.py`. The frozen set is for reporting only.

## Training pipeline

Mac (M5 Max) does prep and smoke tests; the real training run happens on a
CUDA GPU.

- `src/train_tokenizer.py` — trains a 32,768-vocab byte-level BPE on a
  streamed FineWeb-Edu sample, writes `tokenizer/tokenizer.json`.
- `src/tokenize_corpus.py` — streams FineWeb-Edu again, encodes with that
  tokenizer, and writes `data/train.bin` / `data/val.bin` (uint16 token ids)
  plus `data/meta.json`.
- `src/data.py` — memory-maps the `.bin` shards and samples random windows
  for training.
- `src/model.py` — the GPT (decoder-only Transformer) definition.
- `src/train.py` — the training loop; device-aware (CUDA if present, MPS on
  Mac for smoke tests).
- `src/generate.py` — autoregressive sampling from a checkpoint.

**Architecture note:** the numbers currently in `src/model.py` (16 layers,
d_model 1024) are the old plan. `docs/DESIGN.md` specifies the current
target — deep-and-thin, ~24 layers x d_model 896, GQA (14 query / 2 KV
heads), QK-norm — and `model.py` has not been updated to match yet.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
huggingface-cli login   # needed for dataset streaming / shard upload
```

## Usage

```bash
# 1. Train tokenizer on ~2GB of streamed text (~20 min)
python src/train_tokenizer.py

# 2. Tokenize the corpus (streams, disk-light)
python src/tokenize_corpus.py --sample-gb 2.0

# 3. Local smoke test (Mac)
python src/train.py --max-iters 20 --eval-interval 10 --eval-iters 5 --batch-size 4

# 4. Real run (CUDA GPU)
python src/train.py --max-iters 6000 --batch-size 24 --block-size 256
```

## Disk note

Everything streams; `--sample-gb` controls how much text is pulled before
stopping. Do not `load_dataset` without `streaming=True`.
