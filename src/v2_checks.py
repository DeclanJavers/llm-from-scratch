"""V2 semantic checks: catch the failure V0 can't see — a real quote from the
document that doesn't actually answer the question.

Two checks:
  * type agreement (rule-based, free): a "when" question should get a
    date-shaped answer, "how many" a number, "who" a name. Unknown question
    types pass by default — the check only votes when it's confident.
  * round-trip (needs a model): show a model ONLY the evidence quote and the
    question. If the evidence really contains the answer, the round-trip
    reproduces it; if the quote is a plausible dodge, it won't.

Score them as classifiers against the labeled bench (see build_validator_bench.py):
    python src/v2_checks.py --bench data/eval/validator_bench.jsonl \
        --base-url http://mac.local:1234/v1 --model qwen3-0.6b
The number that matters is the false-accept rate: incorrect answers that pass.
That rate caps the verified accuracy of the whole system.
"""
import argparse
import json
import re

from validator import token_f1
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

def roundtrip_check(base_url, model, question, ans, ev, max_tokens=64):
    """True = the evidence alone reproduces the answer."""
    out = api(base_url, "/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": ROUNDTRIP_PROMPT.format(ev=ev, question=question)}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    })
    reply = out["choices"][0]["message"]["content"].strip()
    if "NO ANSWER" in reply.upper():
        return False
    return token_f1(reply, ans) >= 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="data/eval/validator_bench.jsonl")
    ap.add_argument("--base-url", help="omit to score the type check alone")
    ap.add_argument("--model")
    args = ap.parse_args()

    with open(args.bench) as f:
        rows = [json.loads(line) for line in f if json.loads(line)["label"]]

    # confusion counts for the combined validator: accept = every check passes
    fa = fr = ta = tr = 0
    for i, r in enumerate(rows):
        out = json.loads(r["output"])
        accept = type_check(r["question"], out["ans"])
        if accept and args.base_url:
            accept = roundtrip_check(args.base_url, args.model, r["question"], out["ans"], out["ev"])
            print(f"\r{i + 1}/{len(rows)}", end="", flush=True)
        correct = r["label"] == "correct"
        if accept and correct: ta += 1
        elif accept and not correct: fa += 1
        elif not accept and correct: fr += 1
        else: tr += 1
    if args.base_url:
        print()

    n_correct, n_incorrect = ta + fr, fa + tr
    print(json.dumps({
        "checks": "type" + ("+roundtrip" if args.base_url else ""),
        "n_labeled": len(rows),
        # the load-bearing number: wrong answers that sneak through
        "false_accept_rate": fa / n_incorrect if n_incorrect else None,
        # the cost side: right answers we throw away (burns resamples/coverage)
        "false_reject_rate": fr / n_correct if n_correct else None,
    }, indent=2))

if __name__ == "__main__":
    main()
