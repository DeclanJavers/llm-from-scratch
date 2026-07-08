"""The gate, tier V0 + V1.

V0 (mechanical, no gold needed): output parses, schema is exact, evidence is a
verbatim substring of the document, answer is a substring of the evidence.
These checks are deterministic and ungameable — they run at train time,
inference time, and eval time.

V1 (reference-based, gold needed): answerability call correct, span exact
match, token F1. SQuAD-style answer normalization.

Expected model output, one JSON object and nothing else:
    {"ok": true, "ans": "<verbatim span>", "ev": "<verbatim quote>"}
    {"ok": false}
"""
import json
import re
import string
import unicodedata

# ---------------------------------------------------------------- V0 checks

def canon(s):
    """Canonical form for the containment checks: verbatim in *content*, but
    forgiving of cosmetics — curly vs straight quotes, dash variants, whitespace
    runs/newlines, and case (models capitalize sentence-initial quote words)."""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(str.maketrans({"‘": "'", "’": "'", "“": '"',
                                   "”": '"', "–": "-", "—": "-"}))
    return " ".join(s.lower().split())

# every way an output can fail V0, worst first; a result carries exactly one
V0_FAILURES = [
    "not_json",        # output is not a single JSON object
    "bad_schema",      # wrong keys or wrong types
    "empty_field",     # ans or ev present but empty/whitespace
    "ev_not_in_doc",   # evidence is not a verbatim substring of the document
    "ans_not_in_ev",   # answer is not a substring of the evidence
]

def v0_check(raw_output, document):
    """Run the mechanical checks. Returns (parsed_dict_or_None, failure_or_None)."""
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None, "not_json"
    if not isinstance(parsed, dict) or not isinstance(parsed.get("ok"), bool):
        return None, "bad_schema"

    if parsed["ok"] is False:
        # abstention carries no other fields
        if set(parsed.keys()) != {"ok"}:
            return None, "bad_schema"
        return parsed, None

    if set(parsed.keys()) != {"ok", "ans", "ev"}:
        return None, "bad_schema"
    ans, ev = parsed["ans"], parsed["ev"]
    if not isinstance(ans, str) or not isinstance(ev, str):
        return None, "bad_schema"
    # models frame quotes with literal "..." — trim the framing, keep the quote
    ans = re.sub(r"^\s*(?:\.{3}|…)\s*|\s*(?:\.{3}|…)\s*$", "", ans)
    ev = re.sub(r"^\s*(?:\.{3}|…)\s*|\s*(?:\.{3}|…)\s*$", "", ev)
    parsed["ans"], parsed["ev"] = ans, ev
    if not ans.strip() or not ev.strip():
        return None, "empty_field"
    if canon(ev) not in canon(document):
        return None, "ev_not_in_doc"
    if canon(ans) not in canon(ev):
        return None, "ans_not_in_ev"
    return parsed, None

# ---------------------------------------------------------------- V1 grading

def normalize_answer(s):
    """SQuAD-official normalization: lowercase, drop punctuation/articles, fix whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def token_f1(prediction, gold):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = 0
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t in pred_tokens:
        if gold_counts.get(t, 0) > 0:
            gold_counts[t] -= 1
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def v1_grade(parsed, gold_answers):
    """Grade a V0-passing output against gold. gold_answers is [] for unanswerable.

    Returns a dict:
        answerable_correct — did the model call answerable-vs-not right
        em, f1             — best score over gold answers (0.0 when it abstained
                             or the question was unanswerable)
    """
    gold_says_answerable = len(gold_answers) > 0
    model_says_answerable = parsed["ok"]
    result = {
        "answerable_correct": model_says_answerable == gold_says_answerable,
        "em": 0.0,
        "f1": 0.0,
    }
    if model_says_answerable and gold_says_answerable:
        pred = parsed["ans"]
        result["em"] = max(float(normalize_answer(pred) == normalize_answer(g)) for g in gold_answers)
        result["f1"] = max(token_f1(pred, g) for g in gold_answers)
    return result

def grade(raw_output, document, gold_answers):
    """V0 then V1 in one call. A V0 failure grades as wrong on everything."""
    parsed, failure = v0_check(raw_output, document)
    if failure is not None:
        return {"v0_pass": False, "v0_failure": failure, "answered": False,
                "answerable_correct": False, "em": 0.0, "f1": 0.0}
    graded = v1_grade(parsed, gold_answers)
    return {"v0_pass": True, "v0_failure": None, "answered": parsed["ok"], **graded}

if __name__ == "__main__":
    doc = "Marie Curie won the Nobel Prize in Physics in 1903. She later won a second Nobel Prize in Chemistry."
    cases = [
        # (raw output, gold answers, expected v0_failure, expected em)
        ('{"ok": true, "ans": "1903", "ev": "Nobel Prize in Physics in 1903"}', ["1903"], None, 1.0),
        ('{"ok": false}', [], None, 0.0),                                            # correct abstention
        ('{"ok": true, "ans": "1911", "ev": "Nobel Prize in Physics in 1903"}', ["1903"], "ans_not_in_ev", 0.0),
        ('{"ok": true, "ans": "1903", "ev": "won the prize in 1903"}', ["1903"], "ev_not_in_doc", 0.0),
        ('the answer is 1903', ["1903"], "not_json", 0.0),
        ('{"ok": false, "ans": "1903"}', [], "bad_schema", 0.0),                     # abstention smuggling an answer
        ('{"ok": true, "ans": " ", "ev": "in 1903"}', ["1903"], "empty_field", 0.0),
    ]
    for raw, gold, want_fail, want_em in cases:
        r = grade(raw, doc, gold)
        assert r["v0_failure"] == want_fail, (raw, r)
        assert r["em"] == want_em, (raw, r)
    assert token_f1("the Nobel Prize", "Nobel Prize") == 1.0     # articles normalize away
    assert 0.0 < token_f1("Prize in Physics", "Nobel Prize") < 1.0
    # canonical matching: cosmetic differences pass, content differences fail
    doc2 = "She said “no comment” —\n and left."
    ok, fail = v0_check('{"ok": true, "ans": "no comment", "ev": "said \\"no comment\\" - and left."}', doc2)
    assert fail is None, fail
    ok, fail = v0_check('{"ok": true, "ans": "no comment", "ev": "said no comment and stayed."}', doc2)
    assert fail == "ev_not_in_doc"
    doc3 = "went to the USC Trojans game."
    ok, fail = v0_check('{"ok": true, "ans": "Trojans", "ev": "The USC Trojans game"}', doc3)
    assert fail is None, fail   # sentence-initial capitalization is cosmetic
    print("all validator self-tests pass")
