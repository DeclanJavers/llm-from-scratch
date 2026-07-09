# Baseline results — frozen gate

All numbers from `src/run_gate.py` against the hash-pinned frozen set
(2000 examples: 1000 answerable, 1000 unanswerable traps). Full reports in
the `.json` files alongside this file.

| model | config | V0 pass | attempts real Qs | abstains on traps | F1 when committed | overall EM | overall F1 |
|---|---|---|---|---|---|---|---|
| qwen3.5-2b | no-think | 95.5% | 79.0% | 70.2% | 0.879 | 0.652 | 0.698 |
| gemma-4-e2b | default | 97.3% | 92.6% | 64.5% | 0.816 | 0.621 | 0.701 |

Floor baselines (harness sanity checks): always-abstain scores 0.50 overall
by construction; first-sentence scores ~0.05 F1.

## Reading

Two nearly identical overall scores reached by opposite personalities:

- **Qwen is cautious and precise.** It attempts only 79% of real questions,
  but when it commits it is very good (0.879 F1). Its losses are coverage.
- **Gemma is eager and sloppier.** It attempts nearly everything (92.6%),
  answers traps more (only 64.5% abstention), and is less precise when it
  commits (0.816).

Both models' biggest shared failure is the traps: ~300 (Qwen) and ~355
(Gemma) unanswerable questions got confidently answered with real quotes
that don't answer the question. That failure mode is invisible to V0 and is
exactly what the V2 round-trip check targets — the headroom for the
"system beats model" claim lives there.

Format failures run ~1% for both. A model trained on the schema with
constrained decoding gets this to ~0 by construction.

## Targets for the 200M system (model + validator + resampling)

Set before training; graded on the frozen set only:

1. **Overall F1 ≥ 0.70** — match both 10x-larger baselines.
2. **System abstention on traps > 0.70** — beat the best raw baseline,
   counting validator-forced abstention (no sample passed the gate).
3. **F1 when committed ≥ 0.88** — match Qwen's precision at equal or better
   coverage.

Still missing: the headline target row — a Qwen3.5-8B-class model (the
hypothesis names it). Download it in LM Studio and run the same two
commands when convenient.

## V2 validator results (checker probes on the labeled bench)

From `src/v2_checks.py` against the corrected validator bench (2,268 labeled
rows: 1,636 correct / 632 incorrect; 41 auto-labels flipped after semantic
re-judging). FAR = incorrect answers accepted (caps verified precision);
FRR = correct answers rejected (costs coverage/resamples). Combined = AND
of type+roundtrip+verify.

| checker | probe | FAR | FRR |
|---|---|---|---|
| — | type (rule-based) | 0.919 | 0.045 |
| qwen3.5-2b | roundtrip | 0.378 | 0.251 |
| qwen3.5-2b | verify | 0.476 | 0.254 |
| qwen3.5-2b | combined | 0.234 | 0.414 |
| gemma-4-e2b | roundtrip | 0.310 | 0.566 |
| gemma-4-e2b | verify | 0.468 | 0.151 |
| gemma-4-e2b | combined | 0.212 | 0.617 |

Cross-model ensembles (`experiments/v2_combos.py`, evaluated from the reply
cache), the useful frontier points:

| combo | FAR | FRR | note |
|---|---|---|---|
| type + qw_rt + gm_vf | 0.242 | 0.337 | same FAR as all-Qwen stack, −7.7 pts FRR |
| type + qw_rt + qw_vf + gm_vf | 0.171 | 0.450 | add Gemma verify to Qwen stack |
| qw_rt + qw_vf + gm_rt + gm_vf | 0.089 | 0.727 | sub-0.1 FAR costs ~73% rejection |

### Reading

- **Cross-model checking beats self-checking.** Adding one Gemma probe to
  the Qwen stack strictly improves the tradeoff — the checkers' errors are
  partly decorrelated, so the AND removes more bad accepts than good ones.
- **Measured FAR is an upper bound.** Auditing the surviving false-accepts
  shows a chunk are bench noise, not validator errors: ambiguous questions
  (Toyota said-vs-closed), incomplete gold lists (UK banned Sunday driving
  too), and "unanswerable" traps whose evidence answers them verbatim
  ("What was the triad?"). Gold noise puts a floor under measurable FAR.
- **The hard survivors are premise-mismatch traps and evidence-local
  plausibility.** Traps like "over 118 clubs" (text: 400) or "US's
  third-largest seaport" (text: Florida's third) look fully supported
  unless the checker notices the quantifier/scope mismatch; wrong spans
  like Astra 2A are correct within the quoted evidence and only wrong
  given the full document. Both motivate a full-document re-answer probe —
  the bench rows carry the document, and the cache makes it cheap to add.
- **Checker plumbing matters as much as checker judgment.** gemma-4-e2b
  reasons before answering via LM Studio; tight `max_tokens` truncated its
  CoT and produced degenerate rates (verify rejected everything, roundtrip
  accepted everything) until budgets were raised and verdicts parsed from
  the end of the reply. Fail-closed on empty replies kept the bug visible
  instead of silently inflating acceptance.

## Configuration notes

- qwen3.5-2b ran with thinking disabled in LM Studio (the `/no_think` soft
  switch is not honored by this model; use the UI toggle). Thinking-mode
  replies averaged ~10k chars and stalled the server; no-think replies
  average ~61 chars.
- Prompts and extraction adapter: `src/gen_preds.py` (temperature 0).
  The adapter takes the LAST schema-shaped JSON object in a reply and the
  validator forgives cosmetic quote differences (case, unicode punctuation,
  whitespace, ellipsis framing) — content differences still fail.
