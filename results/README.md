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

## Configuration notes

- qwen3.5-2b ran with thinking disabled in LM Studio (the `/no_think` soft
  switch is not honored by this model; use the UI toggle). Thinking-mode
  replies averaged ~10k chars and stalled the server; no-think replies
  average ~61 chars.
- Prompts and extraction adapter: `src/gen_preds.py` (temperature 0).
  The adapter takes the LAST schema-shaped JSON object in a reply and the
  validator forgives cosmetic quote differences (case, unicode punctuation,
  whitespace, ellipsis framing) — content differences still fail.
