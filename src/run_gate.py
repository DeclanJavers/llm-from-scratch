"""Grade a model's predictions against the frozen eval set.

The harness is generation-agnostic: run any model however you like (mps, cuda,
llama.cpp, an API) and write one JSON line per example:
    {"id": "<eval example id>", "output": "<raw model output string>"}
then:
    python src/run_gate.py --preds preds/qwen3-0.6b.jsonl

Two built-in baselines exist for smoke-testing the harness itself:
    python src/run_gate.py --baseline abstain          # floor: never answers
    python src/run_gate.py --baseline first_sentence   # floor: no reading comprehension

Headline numbers: coverage (how often it answers), selective F1 (how good the
answers it commits to are), answerability accuracy (does it know when to
abstain). The risk-coverage pair is the metric the whole project optimizes.
"""
import argparse
import json

from validator import grade

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

# ------------------------------------------------------------- baselines

def baseline_abstain(example):
    return '{"ok": false}'

def baseline_first_sentence(example):
    first = example["document"].split(". ")[0]
    return json.dumps({"ok": True, "ans": first, "ev": first}, ensure_ascii=False)

BASELINES = {"abstain": baseline_abstain, "first_sentence": baseline_first_sentence}

# ------------------------------------------------------------- reporting

def run(eval_rows, outputs, name):
    """outputs: dict id -> raw model output string. Missing id = graded as not_json."""
    results = []
    for ex in eval_rows:
        raw = outputs.get(ex["id"], "")
        r = grade(raw, ex["document"], ex["answers"])
        r["gold_answerable"] = len(ex["answers"]) > 0
        results.append(r)

    n = len(results)
    v0_pass = [r for r in results if r["v0_pass"]]
    answered = [r for r in v0_pass if r["answered"]]
    answered_and_answerable = [r for r in answered if r["gold_answerable"]]

    v0_failures = {}
    for r in results:
        if r["v0_failure"]:
            v0_failures[r["v0_failure"]] = v0_failures.get(r["v0_failure"], 0) + 1

    report = {
        "model": name,
        "n": n,
        "v0_pass_rate": len(v0_pass) / n,
        "v0_failures": v0_failures,
        # does it know when to abstain (graded over V0 passes; V0 fails count wrong)
        "answerability_acc": sum(r["answerable_correct"] for r in results) / n,
        # risk-coverage: how often it commits, and how good it is when it does
        "coverage": len(answered) / n,
        "selective_em": (sum(r["em"] for r in answered_and_answerable) / len(answered)) if answered else 0.0,
        "selective_f1": (sum(r["f1"] for r in answered_and_answerable) / len(answered)) if answered else 0.0,
        # SQuAD-comparable overall scores (abstentions on answerables score 0)
        "overall_em": sum(r["em"] if r["gold_answerable"] else float(r["answerable_correct"]) for r in results) / n,
        "overall_f1": sum(r["f1"] if r["gold_answerable"] else float(r["answerable_correct"]) for r in results) / n,
    }
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="data/eval/squad2_frozen.jsonl")
    ap.add_argument("--preds", help="predictions jsonl: {'id': ..., 'output': ...}")
    ap.add_argument("--baseline", choices=sorted(BASELINES))
    ap.add_argument("--report-out", help="also write the report as json")
    args = ap.parse_args()
    if bool(args.preds) == bool(args.baseline):
        ap.error("exactly one of --preds / --baseline required")

    eval_rows = load_jsonl(args.eval_set)
    if args.baseline:
        outputs = {ex["id"]: BASELINES[args.baseline](ex) for ex in eval_rows}
        name = f"baseline:{args.baseline}"
    else:
        outputs = {p["id"]: p["output"] for p in load_jsonl(args.preds)}
        name = args.preds

    report = run(eval_rows, outputs, name)
    print(json.dumps(report, indent=2))
    if args.report_out:
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
