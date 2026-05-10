"""Paired statistical comparison of two eval reports.

We use:
  - paired bootstrap (10k resamples) for continuous metrics like ROUGE-L,
    faithfulness, and completeness. Reported as mean delta with 95% CI.
  - McNemar's exact test for binary metrics (hallucination, omission).
    With N=5 this has almost no power, which is the whole point - the demo
    output will tell you the effect size is large but not statistically
    significant at the conventional alpha, exactly as it should.

Both tests are paired because the same cases pass through both prompts.

Usage:
    python -m evals.compare \
        --baseline reports/v1_results.json \
        --candidate reports/v2_results.json \
        --out reports/comparison.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable


CONTINUOUS_METRICS = ("rouge_l", "faithfulness", "completeness")
BINARY_METRICS = ("is_hallucination", "is_omission", "is_med_omission")


def _by_case(report: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in report["per_case"]}


def _paired_bootstrap_ci(
    pairs: list[tuple[float, float]],
    n_resamples: int = 10000,
    rng_seed: int = 1729,
) -> tuple[float, float, float, float]:
    """Returns (mean_delta, ci_low, ci_high, two_sided_p)."""
    if not pairs:
        return 0.0, 0.0, 0.0, 1.0
    deltas = [c - b for b, c in pairs]
    mean = sum(deltas) / len(deltas)
    rng = random.Random(rng_seed)
    boot_means: list[float] = []
    n = len(deltas)
    for _ in range(n_resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    ci_low = boot_means[int(0.025 * n_resamples)]
    ci_high = boot_means[int(0.975 * n_resamples)]
    # Two sided p value via the achieved significance level under H0: mean=0.
    # Approximated by reflecting the bootstrap distribution around the observed mean.
    centered = [b - mean for b in boot_means]
    extreme = sum(1 for c in centered if abs(c) >= abs(mean))
    p = (extreme + 1) / (n_resamples + 1)
    return mean, ci_low, ci_high, p


def _mcnemar_exact(b: list[int], c: list[int]) -> tuple[int, int, float, float]:
    """Exact two sided McNemar for paired binary outcomes.

    Returns (b_only_baseline, b_only_candidate, delta_rate, p_value).
    'b_only_baseline' is the count of cases where baseline failed but candidate
    succeeded (so a positive count is good for the candidate).
    """
    assert len(b) == len(c)
    only_baseline = sum(1 for bi, ci in zip(b, c) if bi == 1 and ci == 0)
    only_candidate = sum(1 for bi, ci in zip(b, c) if bi == 0 and ci == 1)
    n_disc = only_baseline + only_candidate
    if n_disc == 0:
        return only_baseline, only_candidate, 0.0, 1.0
    # Exact binomial two sided around p = 0.5.
    k = min(only_baseline, only_candidate)
    # Sum tail probabilities P(X<=k) and double.
    cdf = 0.0
    for i in range(0, k + 1):
        cdf += math.comb(n_disc, i) * (0.5 ** n_disc)
    p = min(1.0, 2 * cdf)
    delta_rate = (sum(c) - sum(b)) / len(b)
    return only_baseline, only_candidate, delta_rate, p


def compare(baseline_path: Path, candidate_path: Path, out_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    b_by = _by_case(baseline)
    c_by = _by_case(candidate)
    common = sorted(set(b_by) & set(c_by))
    if not common:
        raise SystemExit("No overlapping case_ids between baseline and candidate.")

    results = {"n": len(common), "metrics": {}}

    for m in CONTINUOUS_METRICS:
        pairs = [(b_by[cid][m], c_by[cid][m]) for cid in common]
        mean, lo, hi, p = _paired_bootstrap_ci(pairs)
        results["metrics"][m] = {
            "kind": "continuous",
            "baseline_mean": sum(b for b, _ in pairs) / len(pairs),
            "candidate_mean": sum(c for _, c in pairs) / len(pairs),
            "delta": mean,
            "ci_95": [lo, hi],
            "p_value": p,
        }

    for m in BINARY_METRICS:
        b_vals = [b_by[cid][m] for cid in common]
        c_vals = [c_by[cid][m] for cid in common]
        only_b, only_c, delta_rate, p = _mcnemar_exact(b_vals, c_vals)
        results["metrics"][m] = {
            "kind": "binary",
            "baseline_rate": sum(b_vals) / len(b_vals),
            "candidate_rate": sum(c_vals) / len(c_vals),
            "delta": delta_rate,
            "only_baseline_failures": only_b,
            "only_candidate_failures": only_c,
            "p_value": p,
        }

    # Operational metrics (no statistical test, just delta).
    for m in ("p95_latency_ms", "mean_cost_usd"):
        b_val = baseline["aggregate"][m]
        c_val = candidate["aggregate"][m]
        results["metrics"][m] = {
            "kind": "operational",
            "baseline": b_val,
            "candidate": c_val,
            "delta": c_val - b_val,
            "pct_change": (c_val - b_val) / b_val if b_val else 0.0,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    res = compare(args.baseline, args.candidate, args.out)
    print(f"Compared {res['n']} paired cases. Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
