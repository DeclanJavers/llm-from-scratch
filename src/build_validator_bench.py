"""Build the labeled benchmark that scores the VALIDATOR itself.

The V2 semantic checks are classifiers ("does this evidence really answer the
question?"), so they need labeled data: model outputs marked correct/incorrect.
This script builds that set from prediction files on the DEV set (never the
frozen set — V2 gets tuned on this data).

Auto-labeling uses the gold answers: clear scores label themselves, the
ambiguous middle band goes to a human review queue.
    answered on unanswerable question        -> incorrect
    F1 >= 0.75 vs gold                        -> correct
    F1 <= 0.25 vs gold                        -> incorrect
    in between                                -> review by hand

Build:   python src/build_validator_bench.py --preds preds/dev/*.jsonl
Review:  python src/build_validator_bench.py --review
Only V0-passing, answered outputs are included: V0 failures and abstentions
never reach V2, so they don't belong in its benchmark.
"""
import argparse
import json
import os

from validator import v0_check, v1_grade

BENCH = "data/eval/validator_bench.jsonl"

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def build(pred_paths, eval_path, out_path):
    eval_by_id = {ex["id"]: ex for ex in load_jsonl(eval_path)}
    existing = set()
    if os.path.exists(out_path):
        existing = {(r["model"], r["id"]) for r in load_jsonl(out_path)}

    rows, n_review = [], 0
    for path in pred_paths:
        model = os.path.splitext(os.path.basename(path))[0]
        for pred in load_jsonl(path):
            ex = eval_by_id.get(pred["id"])
            if ex is None or (model, pred["id"]) in existing:
                continue
            parsed, failure = v0_check(pred["output"], ex["document"])
            if failure is not None or parsed["ok"] is False:
                continue   # V0 already rejects these; V2 never sees them
            g = v1_grade(parsed, ex["answers"])
            if not ex["answers"]:
                label = "incorrect"          # answered an unanswerable question
            elif g["f1"] >= 0.75:
                label = "correct"
            elif g["f1"] <= 0.25:
                label = "incorrect"
            else:
                label, n_review = None, n_review + 1
            rows.append({"model": model, "id": pred["id"], "question": ex["question"],
                         "document": ex["document"], "output": pred["output"],
                         "gold": ex["answers"], "f1": round(g["f1"], 3),
                         "label": label, "label_source": "auto" if label else None})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_auto = sum(1 for r in rows if r["label"])
    print(f"added {len(rows)} outputs -> {out_path} ({n_auto} auto-labeled, {n_review} need review)")

def review(out_path):
    rows = load_jsonl(out_path)
    todo = [r for r in rows if r["label"] is None]
    print(f"{len(todo)} to review  (y = correct, n = incorrect, s = skip, q = quit)\n")
    for r in todo:
        out = json.loads(r["output"])
        print(f"Q: {r['question']}")
        print(f"gold: {r['gold']}  (auto F1 {r['f1']})")
        print(f"ans:  {out['ans']}")
        print(f"ev:   {out['ev']}")
        while True:
            key = input("[y/n/s/q] > ").strip().lower()
            if key in ("y", "n", "s", "q"):
                break
        if key == "q":
            break
        if key != "s":
            r["label"] = "correct" if key == "y" else "incorrect"
            r["label_source"] = "human"
        print()
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done = sum(1 for r in rows if r["label"])
    print(f"saved: {done}/{len(rows)} labeled")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", help="prediction jsonl files (from the dev set)")
    ap.add_argument("--eval-set", default="data/eval/squad2_dev.jsonl")
    ap.add_argument("--out", default=BENCH)
    ap.add_argument("--review", action="store_true", help="hand-label the ambiguous band")
    args = ap.parse_args()
    if args.review:
        review(args.out)
    elif args.preds:
        build(args.preds, args.eval_set, args.out)
    else:
        ap.error("either --preds or --review")

if __name__ == "__main__":
    main()
