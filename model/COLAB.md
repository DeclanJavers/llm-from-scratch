# Colab runbook — M0/M1 data prep

Runs `model/prepare_data.py` end to end and pushes shards to a private HF
dataset repo. **Use a CPU high-RAM runtime** — this is download + tokenize
work; don't burn GPU units on it. Expect ~2–4 hours total and ~40GB of disk
(the runtime has plenty). Nothing needs to survive the session: the HF repo
is the canonical store.

Prerequisite: in Colab, open the key icon (Secrets), add `HF_TOKEN` with a
WRITE-scope HuggingFace token, and enable notebook access for it.

**Cell 1 — env + repo**

```python
!pip -q install -U tokenizers datasets huggingface_hub hf_transfer pyarrow
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"   # multi-threaded downloads
!git clone https://github.com/DeclanJavers/llm-from-scratch.git
%cd llm-from-scratch
!git checkout claude/thoughts-on-this-ua0f3y
```

**Cell 2 — M0 selftests (shard round-trip, schema rendering, escapes)**

```python
!python model/prepare_data.py --stage selftest
```

**Cell 3 — smoke the whole pipeline at 1/100 scale first (~10 min)**

```python
!python model/prepare_data.py --stage all --small --repo YOUR_HF_USER/tinylm-shards-smoke
```

If the smoke run's report looks sane (all groups filled, QA yields > 0,
schema round-trip OK), continue. The smoke shards go to a separate repo.

**Cell 4 — tokenizer (32k BPE on ~2GB of FineWeb-Edu, ~15 min)**

```python
!python model/prepare_data.py --stage tokenizer
```

**Cell 5 — QA pool (SQuAD v2 + MRQA; prints per-source yield table)**

```python
!python model/prepare_data.py --stage qa
```

This also writes `model/data/reports/triviaqa_audit.jsonl` (500 rows) —
run those through the V2 checkers (LM Studio, eval side) before deciding
TriviaQA's anneal weight.

**Cell 6 — the big one: FineWeb-Edu stable/anneal/val (~1–2h), then wiki**

```python
!python model/prepare_data.py --stage fineweb
!python model/prepare_data.py --stage wiki
```

**Cell 7 — manifests + M1 data report (the milestone evidence)**

```python
!python model/prepare_data.py --stage manifest
```

Check the printed table: every group at budget, no `** UNDERFILLED **`, no
`** >2 EPOCHS **` on any phase/group line.

**Cell 8 — push to the canonical HF repo (~6GB upload)**

```python
!python model/prepare_data.py --stage push --repo YOUR_HF_USER/tinylm-shards
```

Notes:
- Every stage writes a `.done` marker and skips itself on re-run; a crashed
  stage restarts cleanly with `--force` (stages are self-contained).
- If the session dies mid-`fineweb`, just re-run Cells 1 and 6 — completed
  stages skip themselves.
- Natural Questions / TriviaQA / HotpotQA come via the MRQA 2019 distribution
  (pre-extractive, answers verified, a few GB) instead of the ~140GB raw NQ.
