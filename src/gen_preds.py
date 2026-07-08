"""Run models against an eval set via any OpenAI-compatible server.

Built for LM Studio running on another Mac ("lms server start", port 1234):
    python src/gen_preds.py --base-url http://<mac-hostname>.local:1234/v1 --list-models
    python src/gen_preds.py --base-url http://<mac-hostname>.local:1234/v1 \
        --model qwen3-0.6b --out preds/qwen3-0.6b.jsonl

Writes one line per example: {"id": ..., "output": <extracted json>, "raw": <full reply>}
then grade with: python src/run_gate.py --preds preds/qwen3-0.6b.jsonl

Resumable: reruns skip ids already present in the output file.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

SYSTEM_PROMPT = """You answer questions using ONLY the provided document. Reply with a single JSON object and nothing else.

If the document contains the answer:
{"ok": true, "ans": "<the answer, copied word-for-word from the document>", "ev": "<the full sentence or phrase from the document, copied word-for-word, that contains the answer>"}

If the document does NOT contain the answer:
{"ok": false}

Rules: "ans" and "ev" must be exact verbatim substrings of the document. "ans" must appear inside "ev". Never guess from outside knowledge."""

def api(base_url, path, payload=None, timeout=300):
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

def chat(base_url, model, question, document, temperature, max_tokens, no_think=False,
         timeout=600, retries=3):
    # question -> document -> question repeated (see docs/DESIGN.md)
    user = f"Question: {question}\n\nDocument:\n{document}\n\nQuestion: {question}"
    system = SYSTEM_PROMPT + (" /no_think" if no_think else "")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # LM Studio serves one request at a time; a slow generation ahead of us in
    # the queue looks like a timeout here, so wait long and retry with backoff
    for attempt in range(retries + 1):
        try:
            out = api(base_url, "/chat/completions", payload, timeout=timeout)
            break
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 ** (attempt + 1))
    msg = out["choices"][0]["message"]
    # reasoning models: LM Studio may split thinking into its own field and
    # leave content empty (e.g. truncated mid-think) — keep whatever exists
    return msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""

def extract_json(text):
    """Pull the model's answer object out of a reply (instruct models add prose
    and fences; the gate itself stays strict — this is the adapter for foreign
    models). Reasoning models restate the format spec while thinking, so the
    FIRST object is often the quoted template — the answer is the LAST
    schema-shaped object in the reply."""
    # drop <think>...</think> blocks; an unclosed <think> means truncated thinking
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    decoder = json.JSONDecoder()
    candidates = []
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            candidates.append(obj)
    schema_shaped = [c for c in candidates if isinstance(c, dict) and "ok" in c]
    if schema_shaped:
        return json.dumps(schema_shaped[-1], ensure_ascii=False)
    if candidates:
        return json.dumps(candidates[-1], ensure_ascii=False)
    return text  # nothing parseable; the gate will grade it not_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", help="e.g. http://mac.local:1234/v1")
    ap.add_argument("--model")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--eval-set", default="data/eval/squad2_frozen.jsonl")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, help="only run the first N examples (smoke test)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="generous by default: reasoning models spend most of it thinking")
    ap.add_argument("--no-think", action="store_true",
                    help="append the Qwen-style /no_think soft switch to the system prompt")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per request")
    ap.add_argument("--re-extract", metavar="PREDS",
                    help="re-run extraction over an existing preds file's raw replies "
                    "(after an extractor fix) instead of generating anything")
    args = ap.parse_args()

    if args.re_extract:
        with open(args.re_extract) as f:
            preds = [json.loads(line) for line in f]
        with open(args.re_extract, "w") as f:
            for p in preds:
                p["output"] = extract_json(p["raw"])
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"re-extracted {len(preds)} -> {args.re_extract}")
        return

    if not args.base_url:
        ap.error("--base-url required")
    if args.list_models:
        for m in api(args.base_url, "/models")["data"]:
            print(m["id"])
        return
    if not args.model or not args.out:
        ap.error("--model and --out required (or --list-models)")

    with open(args.eval_set) as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["id"] for line in f}
        print(f"resuming: {len(done)} already done")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        for i, ex in enumerate(rows):
            if ex["id"] in done:
                continue
            try:
                raw = chat(args.base_url, args.model, ex["question"], ex["document"],
                           args.temperature, args.max_tokens, no_think=args.no_think,
                           timeout=args.timeout)
            except Exception as e:   # keep going; rerun picks up the stragglers
                print(f"\n{ex['id']}: {e}", file=sys.stderr)
                continue
            f.write(json.dumps({"id": ex["id"], "output": extract_json(raw), "raw": raw},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"\r{i + 1}/{len(rows)}", end="", flush=True)
    print(f"\ndone -> {args.out}")

if __name__ == "__main__":
    main()
