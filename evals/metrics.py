"""Metric implementations.

We implement the metrics from scratch with no external NLP deps so the CI run
stays under 5 seconds. Each metric is a pure function from (candidate, case) to
either a 0/1 indicator or a float in [0, 1], so paired statistical tests work
without further transformation.
"""

from __future__ import annotations

from typing import Iterable

from .model import tokenize


# --------- ROUGE-L (longest common subsequence based F1) ---------


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    cand = tokenize(candidate)
    ref = tokenize(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


# --------- Faithfulness (mocked LLM judge) ---------
# In production this is a separate model call asking "is every claim in the
# candidate supported by the source note?". Here we approximate by checking
# whether every medication mentioned in the candidate also appears in the note.
# This is a defensible proxy because med hallucinations are the dominant
# faithfulness failure in clinical summarization.

_KNOWN_MEDS = {
    "aspirin", "atorvastatin", "metoprolol", "prednisone", "azithromycin",
    "albuterol", "methylprednisolone", "ceftriaxone", "levofloxacin",
    "insulin", "apixaban", "warfarin", "heparin", "clopidogrel",
    "rivaroxaban", "dabigatran",
}


def _meds_in(text: str) -> set[str]:
    toks = set(tokenize(text))
    return toks & _KNOWN_MEDS


def faithfulness(candidate: str, case: dict) -> float:
    cand_meds = _meds_in(candidate)
    src_meds = _meds_in(case["note"])
    if not cand_meds:
        # Nothing claimed = nothing to be unfaithful about, but penalize
        # mildly because an empty discharge summary is also unhelpful.
        return 0.9
    unsupported = cand_meds - src_meds
    if not unsupported:
        return 1.0
    return max(0.0, 1.0 - len(unsupported) / max(1, len(cand_meds)))


def is_hallucination(candidate: str, case: dict) -> int:
    """Binary: 1 if the candidate mentions a med not present in the source note."""
    return 0 if not (_meds_in(candidate) - _meds_in(case["note"])) else 1


# --------- Omission of critical findings ---------


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    low = text.lower()
    return any(kw.lower() in low for kw in keywords)


def is_omission(candidate: str, case: dict) -> int:
    """Binary: 1 if the candidate fails to mention any critical finding."""
    return 0 if _contains_any(candidate, case["critical_findings"]) else 1


def is_med_omission(candidate: str, case: dict) -> int:
    """Binary: 1 if any required discharge med is missing from the candidate."""
    cand_meds = _meds_in(candidate)
    missing = [m for m in case["required_meds"] if m.lower() not in cand_meds]
    return 1 if missing else 0


# --------- Completeness (fraction of required elements present) ---------


def completeness(candidate: str, case: dict) -> float:
    required = list(case["critical_findings"]) + list(case["required_meds"])
    if not required:
        return 1.0
    hits = 0
    for r in required:
        if r.lower() in candidate.lower():
            hits += 1
    return hits / len(required)


# --------- Top level convenience: score one candidate against one case ---------


def score_case(candidate: str, case: dict) -> dict:
    return {
        "rouge_l": rouge_l_f1(candidate, case["reference_summary"]),
        "faithfulness": faithfulness(candidate, case),
        "completeness": completeness(candidate, case),
        "is_hallucination": is_hallucination(candidate, case),
        "is_omission": is_omission(candidate, case),
        "is_med_omission": is_med_omission(candidate, case),
    }
