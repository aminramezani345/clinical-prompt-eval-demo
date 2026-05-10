"""Deterministic mock LLM.

We do NOT call a real model in CI for this demo. Instead, model.generate is keyed
on (prompt_version, case_id) and returns canned outputs that show v2 outperforming
v1 on hallucination and omission. This makes the whole pipeline:
  - reproducible (no flaky network),
  - free (no API spend in CI),
  - safe (no PHI ever leaves the runner).

In a real production setup you would replace _MOCK_OUTPUTS with a call to your
in-VPC inference endpoint, and the rest of the pipeline stays identical.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


# Canned outputs for each prompt version and case. The v1 outputs are intentionally
# weaker: they drop discharge meds (omissions) and in two cases invent details
# not present in the source note (hallucinations).
_MOCK_OUTPUTS: dict[tuple[str, str], str] = {
    # ---- v1: minimal prompt, frequent omissions and one hallucination ----
    ("v1", "c001"): (
        "65 year old female admitted with chest pain found to have a STEMI. "
        "Stent was placed in the LAD and the patient was discharged home."
    ),
    ("v1", "c002"): (
        "72 year old male with COPD treated with nebulizers and steroids. "
        "Stable for discharge on home inhalers."
    ),
    ("v1", "c003"): (
        "45 year old female treated for pneumonia with IV antibiotics. "
        "Discharged on oral antibiotics."
    ),
    ("v1", "c004"): (
        "58 year old male admitted with elevated blood sugars and dehydration. "
        "Improved with IV fluids and discharged on insulin."
    ),
    ("v1", "c005"): (
        "30 year old female diagnosed with a pulmonary embolism. "
        "Started on warfarin and discharged home."
    ),
    # ---- v2: structured prompt, faithful and complete ----
    ("v2", "c001"): (
        "65 year old female admitted for anterior STEMI treated with a drug eluting "
        "stent placed in the proximal LAD. Discharged on aspirin, atorvastatin, "
        "and metoprolol."
    ),
    ("v2", "c002"): (
        "72 year old male admitted for an acute COPD exacerbation treated with "
        "nebulizers, prednisone, and azithromycin. Discharged on a prednisone "
        "taper and home inhalers."
    ),
    ("v2", "c003"): (
        "45 year old female admitted for community acquired pneumonia treated with "
        "IV ceftriaxone and azithromycin. Discharged on oral levofloxacin to "
        "complete a 7 day course."
    ),
    ("v2", "c004"): (
        "58 year old male admitted for diabetic ketoacidosis treated with IV "
        "insulin and aggressive fluid resuscitation. Discharged on a basal bolus "
        "insulin regimen."
    ),
    ("v2", "c005"): (
        "30 year old female admitted for segmental pulmonary embolism diagnosed "
        "on CT pulmonary angiogram, started on apixaban. Discharged on apixaban "
        "with no bleeding complications."
    ),
}


def _identify_prompt_version(prompt_template: str) -> str:
    """Map a prompt template to a stable version label.

    Real systems would key on a content hash of the prompt file. For the demo we
    detect the structured instruction unique to v2.
    """
    if "MUST include" in prompt_template:
        return "v2"
    return "v1"


def _simulated_latency_ms(prompt_version: str, case_id: str) -> float:
    """Deterministic fake latency: v2 is slightly slower because it is a longer prompt."""
    base = 1100.0 if prompt_version == "v1" else 1450.0
    # Per case jitter, deterministic via case_id hash.
    h = int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 400
    return base + h


def _token_counts(prompt_template: str, completion: str) -> tuple[int, int]:
    """A very crude token estimate (whitespace split) good enough for demo cost math."""
    return len(prompt_template.split()), len(completion.split())


def generate(prompt_template: str, case: dict) -> GenerationResult:
    """Run the mocked LLM. Replace this body with your real inference call later."""
    prompt_version = _identify_prompt_version(prompt_template)
    case_id = case["case_id"]
    key = (prompt_version, case_id)
    if key not in _MOCK_OUTPUTS:
        raise KeyError(
            f"No mock output configured for prompt version {prompt_version} "
            f"and case_id {case_id}. Add one to evals/model.py."
        )
    text = _MOCK_OUTPUTS[key]
    latency_ms = _simulated_latency_ms(prompt_version, case_id)
    p_tok, c_tok = _token_counts(prompt_template, text)
    # Tiny sleep so wall time in CI is non zero but still negligible.
    time.sleep(0.001)
    return GenerationResult(
        text=text,
        latency_ms=latency_ms,
        prompt_tokens=p_tok,
        completion_tokens=c_tok,
    )


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Toy price book. Update to match your real model's per token price."""
    in_price_per_1k = 0.0008
    out_price_per_1k = 0.0040
    return (prompt_tokens / 1000.0) * in_price_per_1k + (
        completion_tokens / 1000.0
    ) * out_price_per_1k


# Public so tests can introspect.
def list_mock_keys() -> list[tuple[str, str]]:
    return sorted(_MOCK_OUTPUTS.keys())


_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]
