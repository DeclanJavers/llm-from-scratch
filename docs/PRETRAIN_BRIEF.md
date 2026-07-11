# BUILD BRIEF: Pretraining pipeline for a ~200M-param open-book QA specialist LM

Amended 2026-07-11 from the original draft: (1) QA data uses the project's
locked JSON schema, not a free-text format; (2) architecture follows this
brief (DESIGN.md amended to match); (3) compute target is Google Colab;
(4) DROP/RACE/CoQA cut from sources; (5) M5 grades against the frozen gate.

You are building the PRETRAINING phase only. Post-training (distillation, SFT,
rejection sampling, QAT/quantization, long-context extension) happens in a
separate phase later — leave clean interfaces for it, don't build it.

All code lives under `model/` (this session owns that directory; `evals/` is
read-only from here — see docs/STATUS.md coordination note).

## Context (decisions already made — do not relitigate)
- Goal model: ~200M-param decoder-only transformer, trained from scratch,
  later specialized for open-note extractive QA (document + question →
  verbatim-span answer with verbatim evidence quote, or abstain). Deployment
  is quantized: int8 baseline (free, post-training), int4 only if QAT later.
- Compute: **Google Colab**. Real runs on A100 40GB (~13–15 CU/hr); dev and
  data prep on T4/L4. T4 has NO bf16 (Turing) — T4 is for unit tests and
  tiny fp32 smoke runs only, never training. Sessions die at ~24h max:
  everything must checkpoint and resume (expect the full run to span 2
  sessions). Detect the GPU and adapt micro-batch automatically.
- Budget: **3B pretraining tokens now** (~3.3e18 FLOPs ≈ 7–8 A100-hours or
  ~19–20 L4-hours at 35–45% MFU ≈ 100–115 compute units). This is mildly
  under Chinchilla for 215M (16 vs ~20 tokens/param — inside the flat part
  of the isoFLOP basin) and sized to the currently available units.
  **Extension protocol:** keep the end-of-stable-phase checkpoint (85% mark,
  before LR decay) permanently; a future budget resumes the stable phase
  from it, adds tokens toward the original 8B target, then runs a fresh
  15% anneal. Design nothing that assumes the 3B run is final.
- Recipe: modern speedrun-style stack (modded-nanoGPT is the reference
  implementation — crib hyperparameter defaults from it when unsure).

## Architecture (target ~200-220M total params)
- Decoder-only transformer: d_model=1024, n_layers=12, n_heads=16 (head_dim 64),
  SwiGLU MLP (hidden ≈ 2.67x d_model, rounded to multiple of 128 → 2688).
- RoPE position embeddings (theta=10000). No learned absolute positions.
- RMSNorm (pre-norm). QK-normalization on attention queries/keys.
- UNTIED input embedding and output head. Vocab 32k (see tokenizer).
- Zero-init on residual output projections. No biases anywhere.
- Attention via F.scaled_dot_product_attention with is_causal=True — do NOT
  hand-roll attention math or pass custom mask tensors (kills flash kernels).
- Print exact param count at startup; assert it's in [190M, 230M].

## Tokenizer
- Train a 32k byte-level BPE (HuggingFace tokenizers) on a ~2GB sample of the
  pretraining corpus. Special tokens, baked into the vocab NOW so
  post-training needs no surgery:
  - `<|endoftext|>` — document separator.
  - `<|q|>`, `<|doc|>` — prompt delimiters.
  - Schema fragments, one token each (detokenizing MUST reproduce the exact
    schema bytes the validator expects):
    - `<|ok_ans|>`  = `{"ok": true, "ans": "`
    - `<|ev|>`      = `", "ev": "`
    - `<|end|>`     = `"}`
    - `<|abstain|>` = `{"ok": false}`

## QA text format (LOCKED — this is the project's task contract)
Prompt order is question → document → question repeated (causal attention
reads the document knowing what it's looking for):

    <|q|>{question}<|doc|>{passage}<|q|>{question}<|ok_ans|>{span}<|ev|>{quote}<|end|><|endoftext|>

Unanswerable:

    <|q|>{question}<|doc|>{passage}<|q|>{question}<|abstain|><|endoftext|>

The JSON schema, verbatim-span requirement, and abstention semantics are
defined in docs/DESIGN.md and enforced by `evals/validator.py`. Never emit a
free-text answer format.

## Data pipeline
Token budget per bucket (3B plan + 50M held out; extension to 8B only adds
more FineWeb-Edu — no new sources):

| bucket                    | tokens  | source                          |
|---------------------------|---------|---------------------------------|
| stable: fluency           | ~2.04B  | FineWeb-Edu sample-10BT         |
| stable: facts (on-domain: | ~380M   | English Wikipedia               |
|   the gate IS wiki text)  |         |                                 |
| stable: QA format         | ~130M   | converted QA pool               |
| anneal: QA                | ~225M   | same QA pool, re-sampled        |
| anneal: top web           | ~225M   | FineWeb-Edu int_score=5 (≥4 if  |
|                           |         | the 5-slice is too thin)        |
| val (two sets)            | ~50M    | carved from FineWeb-Edu + QA    |

- Sources (all free, via HuggingFace datasets; NQ/TriviaQA/HotpotQA come
  via the MRQA 2019 distribution — pre-extractive, spans verified, a few
  GB instead of the ~140GB raw NQ). Implementation: `model/prepare_data.py`
  (Colab runbook: `model/COLAB.md`).
  1. FineWeb-Edu (sample-10BT config, ~10B tokens — covers the 8B extension).
  2. English Wikipedia (wikimedia/wikipedia, en) — need ~380M of ~4-5B.
  3. QA pool (~260-300M tokens rendered; ≤2 epochs total across both phases —
     well under the ~4-epoch degradation threshold). Convert to the locked
     format above:
     - SQuAD v2 train (~30M; v2 SUBSUMES v1 — do not also load v1, it
       double-counts the answerable questions).
     - Natural Questions short-answer (~35-40M).
     - HotpotQA (~90M; multi-paragraph + distractors = exactly the open-note
       skill; DROP its yes/no comparison rows — not extractive).
     - TriviaQA rc (~100-140M after truncating docs to fit context).
       **Distant-supervised = noisiest source**: the doc contains the answer
       string but may not answer the question. Before full inclusion, audit
       ~500 converted rows through the V2 cross-model checkers (LM Studio,
       reply cache) and downweight/cut based on the measured junk rate.
     (DROP, RACE, CoQA are CUT — non-extractive answers train against the
     gate.) Conversion rules:
     - `ev` = the sentence (or minimal window) of the passage containing the
       gold span; `ans` = the gold span itself.
     - SQuAD v2 unanswerables → the `<|abstain|>` form.
     - **Every converted row must pass `evals/validator.py` V0+V1 when
       rendered back to JSON** (ev verbatim in doc, ans inside ev). Drop rows
       that can't be rendered gate-passing; report per-source yield in the
       M1 data report.
     - **Decontamination:** never convert SQuAD's validation split, and
       n-gram-screen every QA row against the 2000 frozen-gate questions
       (gate passages appearing in Wikipedia text is fine — open-note QA
       provides the passage at test time; question/answer pairs must not
       leak).
  4. Directory `model/data/qa_synthetic/*.jsonl` (fields: context, question,
     answer) — populated later by post-training; loader picks up whatever is
     there at run start, converts and validates it like source 3.
- Shards are built once and pushed to a private HF dataset repo (3B tokens
  uint16 ≈ 6GB) — the canonical store; any GPU session streams them down.
- Pre-tokenize EVERYTHING offline into uint16 binary shards (~100M tokens per
  shard) with an index file; training reads via np.memmap. Never tokenize in
  the training loop. Concatenate docs with <|endoftext|>, slice into fixed
  2048-token rows (packing by concatenation; no padding anywhere).
- Two-phase data mix over the 3B-token budget (percentages hold under
  extension):
  - STABLE phase (first 85%, ~2.55B tokens): 80% FineWeb-Edu, 15% Wikipedia,
    5% QA-formatted.
  - ANNEAL phase (final 15%, ~450M tokens, coincides with LR decay): 50%
    QA-formatted (real + synthetic), 50% FineWeb-Edu (highest-quality slice,
    by the dataset's educational score field).
  - Implement as two separate shard-group manifests; mixing by sampling shards
    according to weights, deterministic under a seed.
- Hold out ~50M tokens: a FineWeb-Edu val set AND a separate schema-formatted
  QA val set. Report both losses separately at every eval (the QA val loss is
  the metric that matters most downstream).

## Optimization
- Muon optimizer for all 2D weight matrices; AdamW for embeddings, norms, and
  output head. Starting values (tune only via pilot runs, not vibes):
  Muon lr 0.02, momentum 0.95, nesterov; AdamW lr 3e-3 (embeddings) / 1e-3
  (head), betas (0.9, 0.95), weight decay 0.01 on matrices only.
  Take the Muon implementation from Keller Jordan's public repo.
- LR schedule: Warmup-Stable-Decay (trapezoidal). Warmup ~0.5% of steps, stable
  until 85%, then linear decay to 0 over the final 15% — aligned exactly with
  the ANNEAL data phase switch.
- bf16 autocast (fp32 master weights), grad clip 1.0.
- Effective batch ≈ 0.5M tokens (e.g., 256 x 2048) via gradient accumulation;
  auto-tune micro-batch size to the largest that fits the detected GPU.
- torch.compile the model. Keep ALL shapes static (fixed seq len, fixed micro
  batch). NO gradient checkpointing (not needed at this scale; wastes compute).

## Training loop requirements
- Fully resumable: checkpoint model+optimizer+dataloader-position+RNG every
  ~500 steps, keep last 3 + one every 10%. Must survive kill -9 and resume
  bit-exact-ish. Colab disks are ephemeral: save checkpoints to mounted
  Google Drive by default, with optional HF-hub upload behind a flag.
- Log every step: loss, LR, grad-norm, tokens/sec, and MFU (tokens/sec x 6N /
  GPU peak bf16 FLOPS; hardcode a small table of peak FLOPS per known GPU).
  CSV + stdout always; wandb optional behind a flag.
- Eval both val losses + a fixed set of 8 greedy text samples every ~250M tokens.
- Single config file (one Python dataclass or YAML) controls everything; every
  run writes its resolved config + git hash into the checkpoint dir.

## Milestones — build and verify IN THIS ORDER, show evidence at each gate
- M0 Scaffold: layout under `model/` (model/data/, model/configs/,
  model/scripts/, net + train + dataloader modules), config system, unit
  tests for shard writer/reader round-trip and schema-token detokenization
  (token ids → exact JSON bytes).
- M1 Data: tokenizer trained; all sources downloaded, converted, validated,
  tokenized to shards; print a data report (token counts per source, per
  phase manifest, per-source QA conversion yield).
- M2 Model correctness: param-count assert passes; a single batch overfits to
  loss < 0.1 in a few hundred steps (memorization sanity test); loss at init
  ≈ ln(32000) ≈ 10.4 (catches init/head bugs).
- M3 Throughput: on Colab A100, sustained MFU ≥ 35% (llm.c reaches ~50% on
  this model class — 35% is a floor, not a stretch) with stable memory.
  Profile and fix dataloader stalls before touching model code.
- M4 Pilot: 100M-token run (~20 min A100, ~5 CU). Acceptance: smooth monotone
  val-loss descent, no loss spikes > 0.3, resume-from-checkpoint verified
  mid-run, samples show babbling-but-English.
- M5 Full run: 3B tokens with phase switch. Track FineWeb val loss (expect
  roughly ~3.0-3.3 nats at this token count — sanity range, not target) and
  QA val loss (report its final value). Preserve the pre-decay checkpoint
  (85% mark) for the extension protocol. Then greedy-decode the frozen set
  and grade it: `python evals/run_gate.py --preds ...` — raw-pretrain gate
  numbers are the baseline SFT has to beat.

## Explicitly OUT of scope (do not build): synthetic data generation, teacher
## models/distillation, SFT, RL, quantization, context >2048, multi-GPU.

Work milestone by milestone. After each milestone, stop and show the evidence
(test output, data report, loss curves, MFU numbers) before proceeding.
