"""Run a single prompt version against the full eval set.

Usage:
    python -m evals.run --prompt prompts/discharge_summary_v2.txt \
        --cases evals/data/eval_cases.jsonl \
        --out reports/v2_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from evals import metrics, model


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run(prompt_path: Path, cases_path: Path, out_path: Path) -> dict:
    prompt_template = prompt_path.read_text()
    cases = load_cases(cases_path)

    per_case = []
    total_cost = 0.0
    started = time.perf_counter()

    for case in cases:
        rendered = prompt_template.replace("{note}", case["note"])
        gen = model.generate(rendered, case)
        scored = metrics.score_case(gen.text, case)
        cost = model.cost_usd(gen.prompt_tokens, gen.completion_tokens)
        total_cost += cost
        per_case.append(
            {
                "case_id": case["case_id"],
                "service_line": case["service_line"],
                "candidate": gen.text,
                "latency_ms": gen.latency_ms,
                "cost_usd": cost,
                **scored,
            }
        )

    wall_s = time.perf_counter() - started

    # Aggregate. Means for continuous metrics, sums-as-rates for binary metrics.
    n = len(per_case)
    agg = {
        "n": n,
        "mean_rouge_l": _mean(per_case, "rouge_l"),
        "mean_faithfulness": _mean(per_case, "faithfulness"),
        "mean_completeness": _mean(per_case, "completeness"),
        "hallucination_rate": _mean(per_case, "is_hallucination"),
        "omission_rate": _mean(per_case, "is_omission"),
        "med_omission_rate": _mean(per_case, "is_med_omission"),
        "p50_latency_ms": _percentile([c["latency_ms"] for c in per_case], 50),
        "p95_latency_ms": _percentile([c["latency_ms"] for c in per_case], 95),
        "mean_cost_usd": total_cost / n,
        "total_cost_usd": total_cost,
        "wall_seconds": wall_s,
    }

    report = {
        "prompt_path": str(prompt_path),
        "cases_path": str(cases_path),
        "aggregate": agg,
        "per_case": per_case,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    return report


def _mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(r[key] for r in rows) / len(rows)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True, type=Path)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    report = run(args.prompt, args.cases, args.out)
    a = report["aggregate"]
    print(
        f"Ran {a['n']} cases. hallucination={a['hallucination_rate']:.1%} "
        f"omission={a['omission_rate']:.1%} "
        f"p95_latency={a['p95_latency_ms']:.0f}ms "
        f"cost=${a['total_cost_usd']:.5f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
