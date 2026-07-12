"""M0/M1: tokenizer + data shards for the 3B-token pretraining run.

Stages (run in order; each writes a .done marker and skips itself on re-run):

    python model/prepare_data.py --stage selftest    # M0 evidence, no downloads
    python model/prepare_data.py --stage tokenizer   # 32k BPE + special tokens
    python model/prepare_data.py --stage qa          # convert+validate QA pool
    python model/prepare_data.py --stage fineweb     # stable + anneal + val
    python model/prepare_data.py --stage wiki        # ~380M wikipedia tokens
    python model/prepare_data.py --stage manifest    # phase manifests + report
    python model/prepare_data.py --stage push --repo <user>/tinylm-shards
    python model/prepare_data.py --stage all --repo <user>/tinylm-shards

`--small` divides every token budget by 100 for an end-to-end smoke run.
Needs HF_TOKEN in the environment (Colab: export from Secrets first) and
HF_HUB_ENABLE_HF_TRANSFER=1 for fast downloads. See model/COLAB.md.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evals"))
from validator import canon, v0_check  # noqa: E402

# ---------------------------------------------------------------- constants

# Special tokens. The schema fragments MUST detokenize to these exact bytes —
# evals/validator.py parses the reconstructed JSON. Do not touch.
EOT = "<|endoftext|>"
Q, DOC = "<|q|>", "<|doc|>"
OK_ANS, EV, END = '{"ok": true, "ans": "', '", "ev": "', '"}'
ABSTAIN = '{"ok": false}'
SPECIALS = [EOT, Q, DOC, OK_ANS, EV, END, ABSTAIN]

VOCAB_SIZE = 32768
SEQ_LEN = 2048
MAX_ROW_TOKENS = SEQ_LEN - 8      # a QA example must fit one training window
CTX_CHAR_CAP = 6000               # ~1500 tokens; shrunk further if needed
SHARD_TOKENS = 100_000_000
TOKENIZER_SAMPLE_GB = 2.0

BUDGETS = {                        # tokens; --small divides by 100
    "fineweb_stable": 2_040_000_000,
    "fineweb_anneal":   225_000_000,
    "val_fineweb":       25_000_000,
    "wiki":             380_000_000,
    # qa group: take everything the filters pass (~260-300M expected);
    # 1-in-20 kept rows go to val_qa.
}

FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_PATTERN = "sample/10BT/*.parquet"
WIKI_REPO = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.en"
MRQA_IDS = ["mrqa-workshop/mrqa", "mrqa"]
MRQA_SUBSETS = {"NaturalQuestionsShort", "TriviaQA-web", "HotpotQA"}
QA_VAL_EVERY = 20
FINEWEB_VAL_EVERY = 100
TRIVIA_AUDIT_ROWS = 500

DATA = REPO / "model" / "data"
SHARDS = DATA / "shards"
REPORTS = DATA / "reports"
TOKENIZER_JSON = REPO / "model" / "tokenizer" / "tokenizer.json"


def esc(s):
    """JSON-escape string contents (no surrounding quotes)."""
    return json.dumps(s, ensure_ascii=False)[1:-1]


def render_answer_json(ans, ev):
    return OK_ANS + esc(ans) + EV + esc(ev) + END


def render_qa(question, context, ans=None, ev=None):
    """One training example in the locked format (question repeated)."""
    prompt = f"{Q}{question}{DOC}{context}{Q}{question}"
    if ans is None:
        return prompt + ABSTAIN + EOT
    return prompt + render_answer_json(ans, ev) + EOT


# ---------------------------------------------------------------- shard io

class ShardWriter:
    """Continuous uint16 token stream, split into ~SHARD_TOKENS files."""

    def __init__(self, group, shard_tokens=SHARD_TOKENS):
        self.dir = SHARDS / group
        self.dir.mkdir(parents=True, exist_ok=True)
        self.group, self.shard_tokens = group, shard_tokens
        self.files, self.total, self._fh, self._in_shard = [], 0, None, 0

    def _roll(self):
        if self._fh:
            self._fh.close()
            self.files[-1]["tokens"] = self._in_shard
        name = f"shard_{len(self.files):04d}.bin"
        self._fh = open(self.dir / name, "wb")
        self.files.append({"file": name, "tokens": 0})
        self._in_shard = 0

    def add(self, ids):
        arr = np.asarray(ids, dtype=np.uint16)
        while len(arr):
            if self._fh is None or self._in_shard >= self.shard_tokens:
                self._roll()
            take = min(len(arr), self.shard_tokens - self._in_shard)
            self._fh.write(arr[:take].tobytes())
            self._in_shard += take
            self.total += take
            arr = arr[take:]

    def close(self):
        if self._fh:
            self._fh.close()
            self.files[-1]["tokens"] = self._in_shard
        meta = {"group": self.group, "dtype": "uint16",
                "total_tokens": self.total, "shards": self.files}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=1))
        return meta


def read_group(group):
    meta = json.loads((SHARDS / group / "meta.json").read_text())
    parts = [np.memmap(SHARDS / group / s["file"], dtype=np.uint16, mode="r")
             for s in meta["shards"]]
    return np.concatenate(parts) if parts else np.empty(0, np.uint16)


# ---------------------------------------------------------------- tokenizer

def load_tokenizer():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    ids = {s: tok.token_to_id(s) for s in SPECIALS}
    assert None not in ids.values(), f"missing specials: {ids}"
    return tok, ids


def stage_tokenizer(args):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    files = fineweb_files()
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, special_tokens=SPECIALS, show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    budget = TOKENIZER_SAMPLE_GB * (0.01 if args.small else 1.0) * 1e9

    def sample():
        seen = 0
        for text in iter_parquet_texts(files, columns=["text"]):
            yield text
            seen += len(text)
            if seen >= budget:
                return

    t0 = time.time()
    tok.train_from_iterator(sample(), trainer=trainer)
    TOKENIZER_JSON.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOKENIZER_JSON))
    print(f"tokenizer: vocab {tok.get_vocab_size()} in {time.time()-t0:.0f}s "
          f"-> {TOKENIZER_JSON}")
    check_schema_roundtrip(verbose=True)


def check_schema_roundtrip(verbose=False):
    """Specials are single ids; encode->decode reproduces exact bytes; the
    reconstructed answer JSON passes V0. Tricky case: quotes needing escapes."""
    tok, ids = load_tokenizer()
    ctx = 'The unit said "no comment" — twice.\nIt was 1987.'
    q = 'What year was it?'
    for ans, ev in [("1987", "It was 1987."),
                    ('"no comment"', 'The unit said "no comment" — twice.')]:
        text = render_qa(q, ctx, ans, ev)
        enc = tok.encode(text)
        assert tok.decode(enc.ids, skip_special_tokens=False) == text, \
            "decode is not byte-exact"
        parsed, fail = v0_check(render_answer_json(ans, ev), ctx)
        assert fail is None, f"V0 failed on constructed row: {fail}"
        for frag in (OK_ANS, EV, END, ABSTAIN, Q, DOC, EOT):
            assert tok.token_to_id(frag) is not None
    abst = render_qa(q, ctx)
    assert tok.decode(tok.encode(abst).ids, skip_special_tokens=False) == abst
    if verbose:
        n = len(tok.encode(render_qa(q, ctx, "1987", "It was 1987.")).ids)
        print(f"schema round-trip OK (example row = {n} tokens; "
              f"scaffolding is {len(SPECIALS)} single tokens)")


# ---------------------------------------------------------------- downloads

def fineweb_files():
    from huggingface_hub import snapshot_download
    path = snapshot_download(FINEWEB_REPO, repo_type="dataset",
                             allow_patterns=FINEWEB_PATTERN)
    files = sorted(Path(path).glob("sample/10BT/*.parquet"))
    assert files, "fineweb-edu sample/10BT download came back empty"
    return files


def iter_parquet_texts(files, columns, batch_rows=1024):
    import pyarrow.parquet as pq
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=columns):
            cols = [batch.column(c).to_pylist() for c in columns]
            if len(columns) == 1:
                yield from cols[0]
            else:
                yield from zip(*cols)


# ---------------------------------------------------------------- fineweb

def stage_fineweb(args):
    tok, ids = load_tokenizer()
    eot = ids[EOT]
    budg = {k: scale(v, args) for k, v in BUDGETS.items()}
    w = {k: ShardWriter(k) for k in ("fineweb_stable", "fineweb_anneal",
                                     "val_fineweb")}
    stable_full = False
    n_doc = 0
    t0 = time.time()

    def full(k):
        return w[k].total >= budg[k]

    batch, routes = [], []

    def flush():
        if not batch:
            return
        for enc, route in zip(tok.encode_batch(batch), routes):
            w[route].add(enc.ids + [eot])
        batch.clear()
        routes.clear()

    for text, score in iter_parquet_texts(fineweb_files(),
                                          columns=["text", "int_score"]):
        n_doc += 1
        if all(full(k) for k in w):
            break
        if not full("val_fineweb") and n_doc % FINEWEB_VAL_EVERY == 0:
            route = "val_fineweb"
        elif score >= 5 and not full("fineweb_anneal"):
            route = "fineweb_anneal"
        elif not full("fineweb_stable"):
            route = "fineweb_stable"
        elif stable_full and score >= 4 and not full("fineweb_anneal"):
            route = "fineweb_anneal"   # top-up fallback, no overlap w/ stable
        else:
            stable_full = stable_full or full("fineweb_stable")
            continue
        stable_full = stable_full or full("fineweb_stable")
        batch.append(text)
        routes.append(route)
        if len(batch) >= 512:
            flush()
            if n_doc % 51200 < 512:
                done = sum(x.total for x in w.values())
                print(f"  {n_doc:,} docs, {done/1e9:.2f}B tokens, "
                      f"{done/max(time.time()-t0,1)/1e6:.1f}M tok/s")
    flush()
    metas = {k: x.close() for k, x in w.items()}
    for k, m in metas.items():
        short = "" if m["total_tokens"] >= budg[k] else "  ** UNDERFILLED **"
        print(f"{k}: {m['total_tokens']:,} / {budg[k]:,} tokens{short}")
    report(metas, "fineweb")


# ---------------------------------------------------------------- wikipedia

def stage_wiki(args):
    from huggingface_hub import HfApi, hf_hub_download
    tok, ids = load_tokenizer()
    eot = ids[EOT]
    budget = scale(BUDGETS["wiki"], args)
    api = HfApi()
    names = sorted(f for f in api.list_repo_files(WIKI_REPO, repo_type="dataset")
                   if f.startswith(WIKI_CONFIG + "/") and f.endswith(".parquet"))
    # spread across the dump rather than taking the head
    names = names[::max(1, len(names) // 12)]
    w = ShardWriter("wiki")
    for name in names:
        if w.total >= budget:
            break
        local = hf_hub_download(WIKI_REPO, name, repo_type="dataset")
        for title, text in iter_parquet_texts([Path(local)],
                                              columns=["title", "text"]):
            w.add(tok.encode(f"{title}\n\n{text}").ids + [eot])
            if w.total >= budget:
                break
        print(f"  {name}: cumulative {w.total/1e6:.0f}M tokens")
    meta = w.close()
    print(f"wiki: {meta['total_tokens']:,} / {budget:,} tokens")
    report({"wiki": meta}, "wiki")


# ---------------------------------------------------------------- QA pool

def load_gate_questions():
    grams = set()
    exact = set()
    for f in ("squad2_frozen.jsonl", "squad2_dev.jsonl"):
        for line in (REPO / "evals" / "data" / f).read_text().splitlines():
            qc = canon(json.loads(line)["question"]).split()
            exact.add(" ".join(qc))
            for i in range(len(qc) - 7):
                grams.add(tuple(qc[i:i + 8]))
    return exact, grams


def contaminated(question, exact, grams):
    qc = canon(question).split()
    if " ".join(qc) in exact:
        return True
    return any(tuple(qc[i:i + 8]) in grams for i in range(len(qc) - 7))


def evidence_window(ctx, start, end):
    lo, hi, bound = start, end, ".!?\n"
    while lo > 0 and ctx[lo - 1] not in bound:
        lo -= 1
    while hi < len(ctx) and ctx[hi] not in bound:
        hi += 1
    ev = ctx[lo:hi + 1].strip()
    if ctx[start:end] not in ev:                       # sentence split failed
        ev = ctx[max(0, start - 200):end + 200].strip()
    return ev


def clean_mrqa(ctx):
    for tag in ("[TLE]", "[DOC]", "[PAR]", "[SEP]"):
        ctx = ctx.replace(tag, "\n")
    return ctx.strip()


def truncate_ctx(ctx, ans_pos, cap):
    if len(ctx) <= cap:
        return ctx
    if ans_pos is None:                                # abstain row: keep head
        return ctx[:cap]
    lo = max(0, min(ans_pos - cap // 2, len(ctx) - cap))
    return ctx[lo:lo + cap]


def convert_row(tok, question, ctx, answer, stats):
    """-> rendered text or None; answer=None means an abstain row."""
    if answer is not None and answer.strip().lower() in ("yes", "no"):
        stats["drop_yesno"] += 1
        return None
    for cap in (CTX_CHAR_CAP, CTX_CHAR_CAP * 2 // 3, CTX_CHAR_CAP // 3):
        if answer is None:
            c = truncate_ctx(ctx, None, cap)
            text = render_qa(question, c)
        else:
            pos = ctx.find(answer)
            if pos < 0:
                stats["drop_ans_not_found"] += 1
                return None
            c = truncate_ctx(ctx, pos, cap)
            pos = c.find(answer)
            if pos < 0:
                stats["drop_ans_not_found"] += 1
                return None
            ev = evidence_window(c, pos, pos + len(answer))
            parsed, fail = v0_check(render_answer_json(answer, ev), c)
            if fail is not None:
                stats[f"drop_v0_{fail}"] += 1
                return None
            text = render_qa(question, c, answer, ev)
        if len(tok.encode(text).ids) <= MAX_ROW_TOKENS:
            return text
    stats["drop_too_long"] += 1
    return None


def iter_qa_rows():
    """Yields (source, question, context, answer-or-None)."""
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad_v2", split="train")
    for r in sq:
        ans = r["answers"]["text"][0] if r["answers"]["text"] else None
        yield "SQuADv2", r["question"], r["context"], ans
    ds = None
    for rid in MRQA_IDS:
        try:
            ds = load_dataset(rid, split="train", streaming=True)
            break
        except Exception as e:                         # noqa: BLE001
            print(f"  ({rid} failed: {e})")
    assert ds is not None, "could not load MRQA from any known id"
    print("  streaming MRQA (skips its unwanted subsets — quiet but alive)",
          flush=True)
    for r in ds:
        if r["subset"] not in MRQA_SUBSETS:
            continue
        ans = mrqa_surface_answer(r)
        if ans is None:
            continue
        yield r["subset"], r["question"], clean_mrqa(r["context"]), ans


def mrqa_surface_answer(r):
    """The answer as it literally appears in the context. detected_answers.text
    is a canonical alias whose casing often differs from the document (most of
    TriviaQA) — use the detected char span to pull the exact surface form."""
    det = r["detected_answers"]
    try:
        span = det["char_spans"][0]
        surface = r["context"][span["start"][0]:span["end"][0] + 1]
        if surface.strip():
            return surface
    except (KeyError, IndexError, TypeError):
        pass
    if det["text"]:
        return det["text"][0]
    return r["answers"][0] if r["answers"] else None


def stage_qa(args):
    from collections import defaultdict
    tok, ids = load_tokenizer()
    eot = ids[EOT]
    exact, grams = load_gate_questions()
    w_train, w_val = ShardWriter("qa"), ShardWriter("val_qa")
    stats = defaultdict(lambda: defaultdict(int))
    audit, kept = [], 0
    rng = random.Random(0)
    cap = 1500 if args.small else None    # per-source, so MRQA gets exercised
    wanted = {"SQuADv2"} | MRQA_SUBSETS
    n_in = 0
    for src, question, ctx, ans in iter_qa_rows():
        n_in += 1
        if n_in % 25000 == 0:
            prog = {k: v["rows_in"] for k, v in sorted(stats.items())}
            print(f"  scanned {n_in:,} rows; per-source in: {prog}", flush=True)
        s = stats[src]
        if cap and s["rows_in"] >= cap:
            if all(stats[w]["rows_in"] >= cap for w in wanted):
                break
            continue
        s["rows_in"] += 1
        if contaminated(question, exact, grams):
            s["drop_contaminated"] += 1
            continue
        text = convert_row(tok, question, ctx, ans, s)
        if text is None:
            continue
        toks = tok.encode(text).ids + [eot]
        kept += 1
        s["rows_out"] += 1
        s["tokens_out"] += len(toks)
        (w_val if kept % QA_VAL_EVERY == 0 else w_train).add(toks)
        if src == "TriviaQA-web" and len(audit) < TRIVIA_AUDIT_ROWS * 4:
            audit.append({"source": src, "question": question,
                          "context": ctx[:CTX_CHAR_CAP], "ans": ans})
        if kept % 50000 == 0:
            print(f"  kept {kept:,} rows, {w_train.total/1e6:.0f}M train tokens",
                  flush=True)
    metas = {"qa": w_train.close(), "val_qa": w_val.close()}
    REPORTS.mkdir(parents=True, exist_ok=True)
    rng.shuffle(audit)
    with open(REPORTS / "triviaqa_audit.jsonl", "w") as f:
        for row in audit[:TRIVIA_AUDIT_ROWS]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nQA yield per source:")
    for src, s in sorted(stats.items()):
        drops = {k: v for k, v in s.items() if k.startswith("drop_")}
        print(f"  {src}: {s['rows_in']:,} in -> {s['rows_out']:,} out "
              f"({s['tokens_out']/1e6:.1f}M tokens)  drops={dict(drops)}")
    report({**metas, "per_source": {k: dict(v) for k, v in stats.items()}}, "qa")
    print(f"TriviaQA audit sample -> {REPORTS/'triviaqa_audit.jsonl'} "
          f"(run through the V2 checkers before deciding its anneal weight)")


# ---------------------------------------------------------------- manifest

def stage_manifest(args):
    stable = {"fineweb_stable": 0.80, "wiki": 0.15, "qa": 0.05}
    anneal = {"qa": 0.50, "fineweb_anneal": 0.50}
    total = scale(3_000_000_000, args)
    out = {
        "seq_len": SEQ_LEN, "vocab_size": VOCAB_SIZE,
        "tokenizer": "model/tokenizer/tokenizer.json",
        "phases": [
            {"name": "stable", "tokens": int(total * 0.85), "mix": stable},
            {"name": "anneal", "tokens": int(total * 0.15), "mix": anneal},
        ],
        "val_groups": ["val_fineweb", "val_qa"],
        "groups": {},
    }
    print(f"{'group':16s} {'tokens':>15s}")
    for g in ("fineweb_stable", "fineweb_anneal", "wiki", "qa",
              "val_fineweb", "val_qa"):
        meta = json.loads((SHARDS / g / "meta.json").read_text())
        out["groups"][g] = meta
        print(f"{g:16s} {meta['total_tokens']:>15,}")
    for phase in out["phases"]:
        for g, frac in phase["mix"].items():
            need = phase["tokens"] * frac
            have = out["groups"][g]["total_tokens"]
            ep = need / max(have, 1)
            flag = "  ** >2 EPOCHS **" if ep > 2.05 else ""
            print(f"  {phase['name']}/{g}: needs {need/1e6:.0f}M, "
                  f"has {have/1e6:.0f}M -> {ep:.2f} epochs{flag}")
    (SHARDS / "manifest.json").write_text(json.dumps(out, indent=1))
    print(f"-> {SHARDS/'manifest.json'}")


# ---------------------------------------------------------------- push

def stage_push(args):
    from huggingface_hub import HfApi
    assert args.repo and "YOUR_HF_USER" not in args.repo, \
        "--repo needs your real HF username, e.g. --repo declan/tinylm-shards"
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(SHARDS), repo_id=args.repo,
                      repo_type="dataset", path_in_repo="shards")
    api.upload_file(path_or_fileobj=str(TOKENIZER_JSON), repo_id=args.repo,
                    repo_type="dataset", path_in_repo="tokenizer.json")
    print(f"pushed shards + tokenizer -> hf.co/datasets/{args.repo}")


# ---------------------------------------------------------------- selftest

def stage_selftest(args):
    import shutil
    import tempfile
    global SHARDS
    keep = SHARDS
    tmp = Path(tempfile.mkdtemp())
    SHARDS = tmp
    try:
        w = ShardWriter("t", shard_tokens=1000)
        rng = np.random.default_rng(0)
        chunks = [rng.integers(0, VOCAB_SIZE, size=n).astype(np.uint16)
                  for n in (700, 900, 1500, 3)]
        for c in chunks:
            w.add(c)
        w.close()
        back = read_group("t")
        want = np.concatenate(chunks)
        assert np.array_equal(back, want), "shard round-trip mismatch"
        assert len(list((tmp / "t").glob("*.bin"))) == 4  # split at 1000s
        print(f"shard round-trip OK ({len(want)} tokens across 4 shards)")
    finally:
        SHARDS = keep
        shutil.rmtree(tmp)

    parsed, fail = v0_check(render_answer_json("1987", "It was 1987."),
                            "It was 1987.")
    assert fail is None and parsed["ans"] == "1987"
    tricky = 'a "quoted\\thing"'
    parsed, fail = v0_check(render_answer_json(tricky, tricky), tricky)
    assert fail is None and parsed["ans"] == tricky, "escape handling broken"
    print("schema rendering + V0 escape handling OK")

    if TOKENIZER_JSON.exists():
        check_schema_roundtrip(verbose=True)
    else:
        print("(tokenizer not trained yet — encode/decode check will run "
              "at the end of the tokenizer stage)")


# ---------------------------------------------------------------- plumbing

def scale(n, args):
    return max(int(n / 100), 10_000) if args.small else n


def report(payload, name):
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{name}.json").write_text(json.dumps(payload, indent=1))


STAGES = {"selftest": stage_selftest, "tokenizer": stage_tokenizer,
          "qa": stage_qa, "fineweb": stage_fineweb, "wiki": stage_wiki,
          "manifest": stage_manifest, "push": stage_push}
ORDER = ["selftest", "tokenizer", "qa", "fineweb", "wiki", "manifest", "push"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    ap.add_argument("--repo", help="HF dataset repo for push, e.g. me/tinylm-shards")
    ap.add_argument("--small", action="store_true",
                    help="1/100 budgets for an end-to-end smoke run")
    ap.add_argument("--force", action="store_true", help="ignore .done markers")
    args = ap.parse_args()
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    DATA.mkdir(parents=True, exist_ok=True)
    stages = ORDER if args.stage == "all" else [args.stage]
    for s in stages:
        marker = DATA / f".done_{s}{'_small' if args.small else ''}"
        if marker.exists() and not args.force and s not in ("selftest", "push"):
            print(f"== {s}: already done (rm {marker} or --force to redo)")
            continue
        print(f"== stage: {s}")
        t0 = time.time()
        STAGES[s](args)
        marker.touch()
        print(f"== {s} done in {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
    # pyarrow/datasets background threads crash noisily during interpreter
    # finalization on Colab; all work is flushed to disk by now, exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
