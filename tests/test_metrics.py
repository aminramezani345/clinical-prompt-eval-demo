"""Unit tests for the metric functions.

These are fast, deterministic, and DO NOT depend on the mock model. They prove
that ROUGE-L is symmetric in the right places, that hallucination detection
fires on out of context meds, and that omission detection fires when a critical
finding is missing.
"""

from evals import metrics


CASE = {
    "case_id": "test1",
    "note": "65 year old female with STEMI. Discharged on aspirin and metoprolol.",
    "critical_findings": ["STEMI"],
    "required_meds": ["aspirin", "metoprolol"],
    "reference_summary": "65 year old female admitted for STEMI, discharged on aspirin and metoprolol.",
}


def test_rouge_l_identical_is_one():
    assert metrics.rouge_l_f1("same text", "same text") == 1.0


def test_rouge_l_disjoint_is_zero():
    assert metrics.rouge_l_f1("aaa bbb ccc", "xxx yyy zzz") == 0.0


def test_faithfulness_no_meds_introduced():
    cand = "65 year old female with STEMI, discharged on aspirin and metoprolol."
    assert metrics.faithfulness(cand, CASE) == 1.0
    assert metrics.is_hallucination(cand, CASE) == 0


def test_faithfulness_hallucinated_med():
    # Introduces warfarin which is not in the note.
    cand = "65 year old female with STEMI discharged on warfarin."
    assert metrics.faithfulness(cand, CASE) < 1.0
    assert metrics.is_hallucination(cand, CASE) == 1


def test_omission_missing_critical_finding():
    # Omits STEMI.
    cand = "65 year old female discharged on aspirin and metoprolol."
    assert metrics.is_omission(cand, CASE) == 1


def test_omission_present_critical_finding():
    cand = "65 year old female with STEMI on aspirin and metoprolol."
    assert metrics.is_omission(cand, CASE) == 0


def test_med_omission_detected():
    # Drops metoprolol.
    cand = "65 year old female with STEMI discharged on aspirin."
    assert metrics.is_med_omission(cand, CASE) == 1


def test_completeness_fraction():
    cand = "65 year old female with STEMI on aspirin."
    # 2 of 3 required tokens present (STEMI, aspirin) -> 2/3
    assert abs(metrics.completeness(cand, CASE) - 2 / 3) < 1e-9
