# Design Decisions

Working record of decisions for the ~200M open-note QA model. Each entry is a
decision, not a suggestion — revisit deliberately, don't drift.

## Hypothesis

**A small LLM paired with a robust validator can match or beat a much larger
model (target: Qwen 3.5 8B) at a narrow, verifiable task.**

The unit under test is the *system* (small model + validator + resampling),
not the raw model. Success criterion, defined before training: at equal
coverage, the 200M system's verified accuracy ≥ Qwen 3.5 8B zero-shot, at a
fraction of the inference cost. Report both comparisons: system-vs-raw-8B
(the product claim) and system-vs-8B-with-validator (the honest science).

## Task

- **Extractive open-note question answering.** Not summarization (no
  mechanical gate exists for it), not free-form QA.
- Answers must be verbatim spans from the provided document.
- Unanswerable questions are first-class: the model must call
  `answerable: false`. Abstention quality (risk–coverage curve) is the
  headline metric, not a footnote.

## Output format

- Fixed JSON schema, byte-identical everywhere (fixed key order, no
  whitespace variance, short keys):

  ```json
  {"ok": true, "ans": "<verbatim span>", "ev": "<verbatim quote containing the answer>"}
  {"ok": false}
  ```

- Schema boilerplate fragments become single special tokens in the tokenizer
  (slots already reserved), so the scaffolding costs ~4 tokens and cannot be
  malformed.
- Evidence is a verbatim quote, not character offsets (models can copy;
  they can't count).
- Prompt order: **question → document → question repeated → answer.**
  Causal attention reads the document knowing what it's looking for.

## Tokenizer

- **Keep 32,768 vocab.** Vocab scaling laws put the optimum for ~200M at
  16–32k; dropping to 16k saves ~17M params (≈1 layer) but inflates every
  document 7–10% in tokens. For open-note QA, tokens-per-document is the
  binding constraint — context capacity beats parameters here.

## Architecture (~200M class)

- Decoder-only, **deep-and-thin: ~24 layers × d_model 896** (replaces
  16 × 1024; depth beats width sub-1B).
- **GQA**: 14 query heads, 2 KV heads (~7× smaller KV cache).
- RMSNorm, SwiGLU, no biases, RoPE, **QK-norm**, tied embeddings.
- Context 4096 if memory allows, else 2048 + RoPE-extension anneal later.
- **Not doing**: Mamba/SSM hybrids, MoE, ternary — research risk, not free
  wins, at this budget.

## Training plan

- ~8–10B FineWeb-Edu tokens (~2× Chinchilla for 235M; deliberately
  overtrained — deployment-bound models should be).
- **Three phases:**
  1. ~85%: FineWeb-Edu, as-is.
  2. ~15% (during LR decay): anneal on highest-quality slice + task-formatted
     data (QA pairs, schema examples). Tokens seen during decay punch above
     their weight.
  3. SFT: the schema exclusively.
- Cosine or WSD schedule, batch ~0.5M tokens. Muon optimizer if time permits
  (~20–30% fewer steps to same loss); AdamW fallback.
- Synthetic data: teacher model generates question/span pairs from
  FineWeb-Edu documents; **keep only pairs the validator verifies.**

## Validator (the gate) — build FIRST

Nothing in training phases 2–3 gets designed until the gate exists and
open-model baselines are measured.

Tiered, cheap-to-expensive:

- **V0 — mechanical (deterministic, ungameable, runs everywhere):**
  JSON parses; schema exact; `ev` is a verbatim substring of the document;
  `ans` is a substring of `ev`; `ok: false` ⇒ no other fields.
- **V1 — reference-based (train/eval only, gold answer available):**
  span EM / token-F1 vs gold; answerability call correct.
- **V2 — reference-free semantic (available at inference):**
  does `ev` actually answer the question (small entailment/NLI check or
  self-consistency round-trip); question-type vs answer-type agreement
  (a "when" question should yield a date).
- **V3 — LLM judge (offline only: data generation and audits, never in the
  inference loop).**

Inference loop: sample → V0 → V2 → accept / resample (best-of-N, adaptive N)
/ abstain.

**Validate the validator:** maintain a labeled set of model outputs
(correct/incorrect, from several models) and score the validator as a
classifier. False-accepts (hallucination passes) are the metric that matters
most. Keep a frozen held-out eval gate that is never trained against —
Goodhart insurance.

### Frozen gate artifact

`evals/data/squad2_frozen.jsonl` (2000 examples: 1000 answerable + 1000
unanswerable, SQuAD 2.0 validation, seed 0) is committed and byte-frozen:

    sha256 b28e813dce00985de005784fc476cf47f0a8090bba847da18ca6a94e0a068527

If that hash ever changes, every previously reported number is void.
Grade predictions with `evals/run_gate.py`; the checks live in `evals/validator.py`.

`evals/data/squad2_dev.jsonl` (2000 examples, seed 1, id-disjoint from the
frozen set) is the **dev set**: validator development, V2 tuning, and the
labeled validator bench all happen here. The frozen set is for reporting only.

### Gate tooling

- `evals/gen_preds.py` — runs any model on an eval set via an OpenAI-compatible
  server (LM Studio over the network). Extracts the first JSON object from
  replies (adapter for instruct models); resumable.
- `evals/build_validator_bench.py` — builds the labeled set that scores the
  validator itself: auto-labels clear cases from gold F1, `--review` gives an
  interactive loop for the ambiguous band.
- `evals/v2_checks.py` — type-agreement (rule-based) and round-trip (model-based)
  checks, scored as a classifier against the bench. The false-accept rate is
  the number that caps the whole system's verified accuracy.

## Baselines (before any training)

Benchmark on the gate: Qwen3-0.6B, Llama-3.2-1B, Gemma-3-1B, SmolLM3,
Qwen 3.5 8B (the target), one frontier API model (the ceiling).
If a frontier model scores badly, the gate is broken — fix the gate first.
Best small open model becomes the synthetic-data teacher.

## Evaluation

- Headline: **risk–coverage curve** (selective accuracy at 50/80/100%
  coverage) on SQuAD 2.0-style held-out data.
- Secondary: span EM/F1, JSON validity (should be ~100% by construction),
  cost per verified answer vs the 8B baseline.
