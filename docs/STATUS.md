# Project status / handoff

Branch: `claude/thoughts-on-this-ua0f3y`. Read `docs/DESIGN.md` first (the
authoritative plan), then this file for current state. `evals/results/README.md`
has the full numbers.

Repo layout (reorganized 2026-07-11): everything eval/validator-side lives in
`evals/` (code, eval sets, preds, cache, results). The old GPT-2-style training
pipeline (`src/`) was **deleted** — the new model gets built fresh in `model/`
(directory not created yet). Run all commands from the repo root.

## The one-sentence thesis

A ~200M model paired with a robust validator (sample → check → resample
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
- `evals/validator.py` — V0 mechanical checks (JSON parses, schema exact, `ev`
  is a verbatim substring of the doc, `ans` inside `ev`) + V1 EM/F1 grading.
  Canonical matching: forgives case/unicode-quote/whitespace/ellipsis
  cosmetics, fails on content differences. Self-tests pass.
- `evals/data/squad2_frozen.jsonl` — 2000 ex (1000 answerable + 1000
  unanswerable traps), SQuAD 2.0 val, seed 0, **hash-pinned in DESIGN.md**.
  Never train/resample/tune against it.
- `evals/data/squad2_dev.jsonl` — 2000 ex, seed 1, id-disjoint from frozen.
  All validator tuning happens here.
- `evals/run_gate.py` — generation-agnostic grader. Reports V0 pass, coverage,
  selective F1, answerability acc, `answerable_answered` /
  `unanswerable_abstained` / `f1_when_committed` breakdown. `--show-fails N`,
  `--only-preds` (smoke), `--report-out`.
- `evals/gen_preds.py` — runs any model on an eval set via an OpenAI-compatible
  server (LM Studio over LAN). Handles reasoning models (think-strip, last
  schema-shaped JSON object, reasoning-field fallback), `--no-think`,
  `--extra-body`, `--re-extract`, retries w/ backoff, resumable.
- `evals/data/validator_bench.jsonl` — 2,268 labeled model outputs
  (1,636 correct / 632 incorrect) for scoring the validator itself.
  Auto-labeled from gold F1; ambiguous band + 44 verbose-but-correct
  auto-mislabels fixed by LLM re-judging (`label_source` tracks provenance).
- `evals/v2_checks.py` — semantic probes (type rule-based / roundtrip / verify),
  scored as classifiers. Reply cache (`evals/cache/v2_replies.jsonl`) makes
  re-scoring free. `evals/v2_combos.py` — cache-only ensemble sweep.

**Baselines on frozen gate** (`evals/results/`):
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
   LM Studio, then: `gen_preds.py … --model <8b> --out evals/preds/qwen3.5-8b.jsonl`
   then `run_gate.py --preds … --report-out evals/results/qwen3.5-8b.frozen.json`.
   This is the actual "beat this" line.
2. **Full-document re-answer probe** — the hard validator survivors are
   evidence-local plausibility (e.g. Astra 2A: right within the quoted `ev`,
   wrong given the whole doc) and premise-mismatch traps. Bench rows carry
   the document; cache makes it cheap. Highest-value validator improvement.
3. Optional: audit surviving false-accepts with the LLM judge to quantify the
   gold-noise floor.

## NEXT: model building (decisions as of this handoff)

**The operative build spec is [PRETRAIN_BRIEF.md](PRETRAIN_BRIEF.md)** —
milestones M0–M5, built in order, evidence at each gate. Decisions settled
2026-07-11 (all reflected in the brief and in the amended DESIGN.md):

- **Architecture: 12L × d1024, MHA, untied embeddings, ~215M** (the
  modded-nanoGPT speedrun shape). This supersedes the earlier deep-and-thin
  24×896/GQA/tied plan and the ~300M leaning. 32k vocab stands.
- **QA data format = the locked JSON schema**, rendered via schema-fragment
  special tokens; every converted training row must pass the V0+V1 gate.
  Sources: SQuAD v1+v2, NQ, TriviaQA-rc, HotpotQA (DROP/RACE/CoQA cut —
  non-extractive).
- **Compute = Google Colab**: dev on T4/L4 (T4 has no bf16 — unit tests
  only), full 8B-token run on A100 (~22–27 hr ≈ 330–400 CU ≈ $35–40,
  spans ~2 sessions via checkpointing).
- **Precision:** bf16 autocast training (no 8-bit training — no FP8 cores
  on A100, instability for zero payoff). Int8 quantization at inference
  only, free, at the end. QAT only if an int4 headline is wanted later.
- The old GPT-2-style skeleton was **deleted 2026-07-11** (git history has
  it); everything gets built fresh under `model/`.

## Session coordination

Two sessions are active on this branch. Whichever session takes model
building owns `model/` (doesn't exist yet — it creates it); the other
should stick to the OPEN items above, which touch only `evals/`.
Pull before starting work; push small and often.

## Environment notes

- LM Studio runs on another Mac over LAN: `http://192.168.1.181:1233/v1`.
  Models loaded: `qwen_qwen3.5-2b`, `google/gemma-4-e2b` (+ an embed model).
  Disable thinking via the LM Studio UI toggle (the `/no_think` soft switch
  is ignored); keep the server Mac awake (`caffeinate -dims`) for long runs.
- Mac prep + smoke tests; CUDA GPU for the real training run.
- zsh: `setopt interactive_comments` if pasting commented blocks.
