# Project status / handoff

Branch: `claude/thoughts-on-this-ua0f3y`. Read `docs/DESIGN.md` first (the
authoritative plan), then this file for current state. `results/README.md`
has the full numbers.

## The one-sentence thesis

A ~200–300M model paired with a robust validator (sample → check → resample
or abstain) can match a much larger model (target: Qwen3.5-8B) at extractive
open-note QA, at a fraction of the inference cost. We test the *system*, not
the raw model. Validator-first: the gate was built and characterized before
any training.

## Task / output contract (locked)

- Extractive open-note QA. Answers must be verbatim spans of the document.
- Unanswerable questions are first-class; the model must abstain.
- Fixed JSON schema, short keys:
  `{"ok": true, "ans": "<verbatim span>", "ev": "<verbatim quote>"}` or
  `{"ok": false}`.
- Headline metric = risk–coverage (selective accuracy + abstention), not raw F1.

## DONE (committed on the branch)

**Gate / validator — the front-loaded risky part, essentially finished.**
- `src/validator.py` — V0 mechanical checks (JSON parses, schema exact, `ev`
  is a verbatim substring of the doc, `ans` inside `ev`) + V1 EM/F1 grading.
  Canonical matching: forgives case/unicode-quote/whitespace/ellipsis
  cosmetics, fails on content differences. Self-tests pass.
- `data/eval/squad2_frozen.jsonl` — 2000 ex (1000 answerable + 1000
  unanswerable traps), SQuAD 2.0 val, seed 0, **hash-pinned in DESIGN.md**.
  Never train/resample/tune against it.
- `data/eval/squad2_dev.jsonl` — 2000 ex, seed 1, id-disjoint from frozen.
  All validator tuning happens here.
- `src/run_gate.py` — generation-agnostic grader. Reports V0 pass, coverage,
  selective F1, answerability acc, `answerable_answered` /
  `unanswerable_abstained` / `f1_when_committed` breakdown. `--show-fails N`,
  `--only-preds` (smoke), `--report-out`.
- `src/gen_preds.py` — runs any model on an eval set via an OpenAI-compatible
  server (LM Studio over LAN). Handles reasoning models (think-strip, last
  schema-shaped JSON object, reasoning-field fallback), `--no-think`,
  `--extra-body`, `--re-extract`, retries w/ backoff, resumable.
- `data/eval/validator_bench.jsonl` — 2,268 labeled model outputs
  (1,636 correct / 632 incorrect) for scoring the validator itself.
  Auto-labeled from gold F1; ambiguous band + 44 verbose-but-correct
  auto-mislabels fixed by LLM re-judging (`label_source` tracks provenance).
- `src/v2_checks.py` — semantic probes (type rule-based / roundtrip / verify),
  scored as classifiers. Reply cache (`cache/v2_replies.jsonl`) makes
  re-scoring free. `experiments/v2_combos.py` — cache-only ensemble sweep.

**Baselines on frozen gate** (`results/`):
- qwen3.5-2b (no-think): 0.698 overall F1, cautious/precise (79% attempt,
  0.879 F1 when committed, 70% trap abstention).
- gemma-4-e2b: 0.701 overall F1, eager/sloppy (93% attempt, 65% abstention).

**Validator result (the hypothesis-critical finding):**
- Self-check (Qwen→Qwen) combined FAR 0.234 / FRR 0.414.
- **Cross-model checking beats self-checking** — errors partly decorrelated.
  Frontier: FAR 0.242 @ FRR 0.337 → FAR 0.089 @ FRR 0.727.
- Measured FAR is an upper bound; a real chunk of "false accepts" is SQuAD
  gold-noise, not validator error.

## OPEN (not blocking training)

1. **8B target row** — hypothesis names Qwen3.5-8B; not yet run. Download in
   LM Studio, then: `gen_preds.py … --model <8b> --out preds/qwen3.5-8b.jsonl`
   then `run_gate.py --preds … --report-out results/qwen3.5-8b.frozen.json`.
   This is the actual "beat this" line.
2. **Full-document re-answer probe** — the hard validator survivors are
   evidence-local plausibility (e.g. Astra 2A: right within the quoted `ev`,
   wrong given the whole doc) and premise-mismatch traps. Bench rows carry
   the document; cache makes it cheap. Highest-value validator improvement.
3. Optional: audit surviving false-accepts with the LLM judge to quantify the
   gold-noise floor.

## NEXT: model building (decisions as of this handoff)

Architecture per DESIGN.md: decoder-only, **deep-and-thin ~24L × d896**, GQA
(14 Q / 2 KV), RMSNorm, SwiGLU, no biases, RoPE, QK-norm, tied embeddings,
**keep 32k vocab** (do NOT shrink — tokens-per-document is the binding
constraint for open-note). `src/model.py` is still the OLD GPT-2-style
skeleton (16×1024, LayerNorm, learned pos-emb, GELU) — **not yet updated**.

**User is now leaning ~300M instead of 235M** (round number, "sub-billion"
story). Fine — marginal.

**Precision decision (discussed, concluded):**
- **8-bit *training* = NO** at this scale. No FP8 tensor cores on A100/MPS
  (would emulate = slower), training cost isn't the bottleneck (~25 A100-hr,
  fits easily in bf16), and it adds instability for ~zero payoff. Same logic
  as DESIGN's "no ternary."
- **Cheaper training = bf16 mixed precision** (`torch.autocast`) — the
  automatic 2× win. Wire this into the training loop.
- **8-bit *inference* = YES, free** — post-training Q8 is nearly lossless and
  is on-thesis (cheap deployment). Do it at the end.
- **QAT** only if we later want a headline int4 claim; optional.

Immediate build steps when resuming:
1. Rewrite `src/model.py` to the DESIGN arch (RoPE, GQA, RMSNorm, SwiGLU,
   QK-norm), target ~300M, param-count + loss smoke test.
2. Add schema boilerplate as special tokens in `train_tokenizer.py`
   (slots reserved; makes JSON scaffolding ~4 tokens, unmalformable).
3. Wire bf16 autocast + resumable checkpointing into `src/train.py`.
4. Then: tokenize corpus → pretrain (3-phase: FineWeb-Edu 85% / task-anneal
   15% / SFT) → grade on frozen gate → wrap with validator + resampling.

## Environment notes

- LM Studio runs on another Mac over LAN: `http://192.168.1.181:1233/v1`.
  Models loaded: `qwen_qwen3.5-2b`, `google/gemma-4-e2b` (+ an embed model).
  Disable thinking via the LM Studio UI toggle (the `/no_think` soft switch
  is ignored); keep the server Mac awake (`caffeinate -dims`) for long runs.
- Mac prep + smoke tests; CUDA GPU for the real training run.
- zsh: `setopt interactive_comments` if pasting commented blocks.
