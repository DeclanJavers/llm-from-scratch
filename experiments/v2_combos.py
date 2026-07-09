"""Sweep probe combinations across checker models, entirely from the reply
cache — no generations. For every AND-subset of the five available probes
(type, and roundtrip/verify per checker) plus 2-of-3-style majority rules,
report FAR/FRR, then print the Pareto frontier (no combo dominates another
on both rates).

    python experiments/v2_combos.py
"""
import itertools
import json
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from v2_checks import type_check, roundtrip_accept, verify_accept, load_cache, confusion

BENCH = "data/eval/validator_bench.jsonl"
CACHE = "cache/v2_replies.jsonl"
CHECKERS = {"qw": "qwen_qwen3.5-2b", "gm": "google/gemma-4-e2b"}

def main():
    with open(BENCH) as f:
        rows = [json.loads(line) for line in f if json.loads(line)["label"]]
    cache = load_cache(CACHE)

    probes = {"type": []}
    for name, checker in CHECKERS.items():
        probes[f"{name}_rt"] = []
        probes[f"{name}_vf"] = []
    missing = 0
    for r in rows:
        out = json.loads(r["output"])
        probes["type"].append(type_check(r["question"], out["ans"]))
        for name, checker in CHECKERS.items():
            for probe, accept in (("roundtrip", lambda rep: roundtrip_accept(rep, out["ans"])),
                                  ("verify", verify_accept)):
                e = cache.get((probe, checker, r["model"], r["id"]))
                if e is None:
                    missing += 1
                    verdict = False
                else:
                    verdict = accept(e["reply"])
                probes[f"{name}_{'rt' if probe == 'roundtrip' else 'vf'}"].append(verdict)
    if missing:
        print(f"warning: {missing} cache misses treated as reject", file=sys.stderr)

    results = []
    names = list(probes)
    # all AND-subsets
    for k in range(1, len(names) + 1):
        for combo in itertools.combinations(names, k):
            verdicts = [all(probes[p][i] for p in combo) for i in range(len(rows))]
            results.append(("AND(" + "+".join(combo) + ")", confusion(rows, verdicts)))
    # type AND (m-of-n over the four model probes)
    model_probes = [p for p in names if p != "type"]
    for n in (2, 3, 4):
        for combo in itertools.combinations(model_probes, n):
            for m in range(1, n):
                verdicts = [probes["type"][i] and sum(probes[p][i] for p in combo) >= m
                            for i in range(len(rows))]
                results.append((f"type+{m}of{n}(" + "+".join(combo) + ")", confusion(rows, verdicts)))

    results.sort(key=lambda x: (x[1]["far"], x[1]["frr"]))
    pareto = []
    best_frr = float("inf")
    for name, c in results:
        if c["frr"] < best_frr:
            pareto.append((name, c))
            best_frr = c["frr"]

    print(f"{len(results)} combos evaluated; Pareto frontier (FAR asc):\n")
    print(f"{'FAR':>7}  {'FRR':>7}  combo")
    for name, c in pareto:
        print(f"{c['far']:7.3f}  {c['frr']:7.3f}  {name}")

if __name__ == "__main__":
    main()
