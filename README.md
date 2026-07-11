# tiny-lm

Pretraining a ~200M parameter LLM from scratch for **extractive open-note
question answering**, built validator-first. The hypothesis: a small model
paired with a robust validator (sample, check, resample, or abstain) can
match a much larger model — target Qwen 3.5 8B — at a narrow, verifiable
task, at a fraction of the inference cost.

All design decisions (task definition, output schema, tokenizer, architecture,
training plan, validator tiers) live in [docs/DESIGN.md](docs/DESIGN.md) and
are authoritative. [docs/STATUS.md](docs/STATUS.md) is the living handoff —
read it before starting work.

## Layout

- `evals/` — the gate, the eval sets, baseline results, and everything that
  scores models or scores the validator itself. Done and characterized.
- `model/` — (not created yet) the model, tokenizer, and training code.
  The old GPT-2-style skeleton was deleted; the build spec is
  [docs/PRETRAIN_BRIEF.md](docs/PRETRAIN_BRIEF.md) (milestones M0–M5).
- `experiments/` — learning scratch code, not part of the pipeline.
- `docs/` — DESIGN.md (decisions), STATUS.md (state/handoff).

## The gate

The checks live in `evals/validator.py`; `evals/run_gate.py` grades
predictions against the hash-pinned frozen set (`evals/data/squad2_frozen.jsonl`
— never train or tune against it; validator development uses the id-disjoint
dev set `evals/data/squad2_dev.jsonl`). Run everything from the repo root:

```bash
# run the validator's self-tests
python evals/validator.py

# smoke-test the harness against built-in floor baselines
python evals/run_gate.py --baseline abstain
python evals/run_gate.py --baseline first_sentence

# grade real predictions: one JSON line per example, {"id": ..., "output": ...}
python evals/run_gate.py --preds evals/preds/<model>.jsonl
```

## Baselines via LM Studio

Point `gen_preds.py` at an LM Studio server ("lms server start" on the other
Mac, default port 1234) to make any model take the exam:

```bash
python evals/gen_preds.py --base-url http://<mac>.local:1234/v1 --list-models

# smoke test first (8 examples), then the full frozen run
python evals/gen_preds.py --base-url http://<mac>.local:1234/v1 \
    --model <model> --limit 8 --out evals/preds/smoke.jsonl
python evals/gen_preds.py --base-url http://<mac>.local:1234/v1 \
    --model <model> --out evals/preds/<model>.jsonl
python evals/run_gate.py --preds evals/preds/<model>.jsonl
```

Numbers so far: [evals/results/README.md](evals/results/README.md).

## Validating the validator

Generate dev predictions (`--eval-set evals/data/squad2_dev.jsonl --out
evals/preds/dev/<model>.jsonl`), build the labeled bench with
`evals/build_validator_bench.py` (then `--review` to hand-label the ambiguous
band), score the V2 checker probes with `evals/v2_checks.py`, and sweep
probe ensembles from the reply cache with `evals/v2_combos.py`.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
huggingface-cli login   # needed for dataset streaming
```
