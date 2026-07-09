"""V2 semantic checks: catch the failure V0 can't see — a real quote from the
document that doesn't actually answer the question.

Three probes, combined with AND (an output must pass every enabled probe):
  * type — rule-based, free: a "when" question should get a date-shaped
    answer, "how many" a number, "who" a name. Only votes when confident.
  * roundtrip — show a checker model ONLY the evidence quote and the
    question; if the evidence really contains the answer, the round-trip
    reproduces it.
  * verify — ask the checker model directly: does this text contain the
    answer to this question, YES or NO.

Score against the labeled bench (build_validator_bench.py):
    python src/v2_checks.py --checks type,roundtrip,verify \
        --base-url http://mac.local:1234/v1 --model qwen3-0.6b
Checker replies are cached (cache/v2_replies.jsonl) keyed by probe+checker+row,
so re-scoring with different combinations or thresholds is free after one pass.
Use a checker model DIFFERENT from the model that produced the bench rows where
possible — a checker shares blind spots with itself (correlated errors).

The number that matters is the false-accept rate: incorrect answers that pass.
It caps the verified precision of the whole system. False rejects only cost
coverage/resamples.
"""
import argparse
import json
import os
import re
import time

from validator import token_f1, canon
from gen_preds import api

MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"
NUMBER_WORDS = r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|hundred|thousand|million|billion"

def type_check(question, ans):
    """True = plausible, False = type mismatch. Only votes on question types it knows."""
    q = question.lower()
    a = ans.lower()
    if re.search(r"\bhow (many|much)\b|\bwhat (percent|percentage|year|number)\b", q):
        return bool(re.search(rf"\d|{NUMBER_WORDS}", a))
    if re.search(r"\bwhen\b|\bwhat (date|day|month|century|decade)\b", q):
        return bool(re.search(rf"\d|{MONTHS}", a))
    if re.search(r"^who\b|\bwho\b", q):
        return bool(re.search(r"[A-Z]", ans))   # names carry capitals
    return True

ROUNDTRIP_PROMPT = """Using ONLY the text below, answer the question in as few words as possible. If the text does not contain the answer, reply exactly: NO ANSWER

Text: {ev}

Question: {question}"""

VERIFY_PROMPT = """Question: {question}

Text: "{ev}"

Does the text explicitly state the answer to the question? Being on the same topic is not enough — the specific answer must be present. Reply with exactly one word: YES or NO."""

def checker_reply(base_url, model, prompt, cache, key, max_tokens=64, retries=3):
    if key in cache:
        return cache[key]["reply"]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries + 1):
        try:
            out = api(base_url, "/chat/completions", payload, timeout=600)
            break
        except Exception:
            if attempt == retries:
                raise   # cache holds everything done so far; rerun resumes
            time.sleep(2 ** (attempt + 1))
    reply = out["choices"][0]["message"]["content"].strip()
    entry = dict(zip(("probe", "checker", "row_model", "id"), key), reply=reply)
    cache[key] = entry
    with open(cache["__path__"], "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return reply

REFUSAL_MARKERS = ("no answer", "not contain", "not stated", "does not", "doesn't",
                   "cannot answer", "can't answer", "not mentioned", "not provided")

def roundtrip_accept(reply, ans):
    r = reply.lower()
    if any(m in r for m in REFUSAL_MARKERS):
        return False
    if token_f1(reply, ans) >= 0.5:
        return True
    # containment fallback for "the answer is X"-style verbosity — but a reply
    # that just restates the whole evidence contains everything; cap its length
    if len(reply.split()) <= max(20, 3 * len(ans.split())):
        return canon(ans) in canon(reply) or canon(reply) in canon(ans)
    return False

def verify_accept(reply):
    # first alphabetic word, markdown/punctuation stripped: "**Yes**, ..." -> yes
    words = re.findall(r"[a-zA-Z]+", reply)
    return bool(words) and words[0].lower() == "yes"

def load_cache(path):
    cache = {"__path__": path}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                cache[(e["probe"], e["checker"], e["row_model"], e["id"])] = e
    return cache

def confusion(rows, verdicts):
    """verdicts: list of bool accept aligned with rows. Returns FAR/FRR."""
    fa = sum(1 for r, v in zip(rows, verdicts) if v and r["label"] == "incorrect")
    fr = sum(1 for r, v in zip(rows, verdicts) if not v and r["label"] == "correct")
    n_inc = sum(1 for r in rows if r["label"] == "incorrect")
    n_cor = len(rows) - n_inc
    return {"far": fa / n_inc if n_inc else None, "frr": fr / n_cor if n_cor else None,
            "n_incorrect": n_inc, "n_correct": n_cor}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/eval/validator_bench.jsonl")
    ap.add_argument("--checks", default="type,roundtrip",
                    help="comma list from: type, roundtrip, verify")
    ap.add_argument("--base-url", help="needed for roundtrip/verify")
    ap.add_argument("--model", help="checker model id (prefer one that did NOT produce the bench rows)")
    ap.add_argument("--cache", default="cache/v2_replies.jsonl")
    ap.add_argument("--show-fa", type=int, default=0, metavar="N",
                    help="print N false-accepts that passed every enabled probe")
    args = ap.parse_args()
    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    needs_model = {"roundtrip", "verify"} & set(checks)
    if needs_model and not (args.base_url and args.model):
        ap.error(f"--base-url and --model required for {sorted(needs_model)}")

    with open(args.bench) as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r["label"]]

    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    cache = load_cache(args.cache)

    per_probe = {}   # probe -> aligned list of bools
    for i, r in enumerate(rows):
        out = json.loads(r["output"])
        if "type" in checks:
            per_probe.setdefault("type", []).append(type_check(r["question"], out["ans"]))
        if "roundtrip" in checks:
            key = ("roundtrip", args.model, r["model"], r["id"])
            reply = checker_reply(args.base_url, args.model,
                                  ROUNDTRIP_PROMPT.format(ev=out["ev"], question=r["question"]),
                                  cache, key)
            per_probe.setdefault("roundtrip", []).append(roundtrip_accept(reply, out["ans"]))
        if "verify" in checks:
            key = ("verify", args.model, r["model"], r["id"])
            reply = checker_reply(args.base_url, args.model,
                                  VERIFY_PROMPT.format(ev=out["ev"], question=r["question"]),
                                  cache, key, max_tokens=8)
            per_probe.setdefault("verify", []).append(verify_accept(reply))
        if needs_model:
            print(f"\r{i + 1}/{len(rows)}", end="", flush=True)
    if needs_model:
        print()

    combined = [all(per_probe[p][i] for p in per_probe) for i in range(len(rows))]

    # where do the surviving false-accepts come from?
    traps = [i for i, r in enumerate(rows) if r["label"] == "incorrect" and not r["gold"]]
    wrong_span = [i for i, r in enumerate(rows) if r["label"] == "incorrect" and r["gold"]]
    report = {
        "checks": checks,
        "checker": args.model,
        "n_labeled": len(rows),
        "combined": confusion(rows, combined),
        "per_probe": {p: confusion(rows, v) for p, v in per_probe.items()},
        "combined_far_on_answered_traps": (sum(combined[i] for i in traps) / len(traps)) if traps else None,
        "combined_far_on_wrong_spans": (sum(combined[i] for i in wrong_span) / len(wrong_span)) if wrong_span else None,
    }
    print(json.dumps(report, indent=2))

    if args.show_fa:
        shown = 0
        for i, r in enumerate(rows):
            if combined[i] and r["label"] == "incorrect" and shown < args.show_fa:
                shown += 1
                out = json.loads(r["output"])
                kind = "trap" if not r["gold"] else "wrong-span"
                print(f"--- false accept ({kind}, {r['model']}, {r['id']})")
                print(f"Q:    {r['question']}")
                print(f"gold: {r['gold']}")
                print(f"ans:  {out['ans']}")
                print(f"ev:   {out['ev'][:300]}")
                print()

if __name__ == "__main__":
    main()
