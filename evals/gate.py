"""Apply gates from gates.yaml to a comparison report.

Exit code 0 means all gates pass. Exit code 1 means at least one HARD gate
failed and the merge should be blocked. Exit code 2 means only SOFT gates
failed - the workflow can still allow merge if a labeled override is present.

Usage:
    python -m evals.gate \
        --comparison reports/comparison.json \
        --gates evals/gates.yaml \
        --report reports/gate_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Minimal YAML reader so we don't need PyYAML in CI. Supports just the structure
# we ship in gates.yaml. If you extend gates.yaml, swap this for PyYAML.
def _read_simple_yaml(text: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        key, _, value = raw.strip().partition(":")
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            new: dict = {}
            parent[key] = new
            stack.append((indent, new))
        else:
            parent[key] = _coerce(value)
    return root


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip().strip('"').strip("'")


def evaluate(comparison: dict, gates_cfg: dict) -> dict:
    findings = []
    hard_fail = False
    soft_fail = False
    for metric_name, rules in gates_cfg.get("gates", {}).items():
        m = comparison["metrics"].get(metric_name)
        if m is None:
            findings.append(
                {
                    "metric": metric_name,
                    "status": "missing",
                    "message": "metric not present in comparison report",
                }
            )
            continue
        verdict = _check_rules(metric_name, m, rules)
        findings.append(verdict)
        if verdict["status"] == "fail":
            if rules.get("hard", False):
                hard_fail = True
            else:
                soft_fail = True
    return {
        "hard_fail": hard_fail,
        "soft_fail": soft_fail,
        "findings": findings,
    }


def _check_rules(metric_name: str, metric: dict, rules: dict) -> dict:
    failures = []

    if "max_absolute" in rules:
        val = metric.get("candidate_rate", metric.get("candidate"))
        if val is not None and val > rules["max_absolute"]:
            failures.append(
                f"candidate value {val:.4f} exceeds max_absolute {rules['max_absolute']}"
            )

    if "min_absolute" in rules:
        val = metric.get("candidate_mean", metric.get("candidate"))
        if val is not None and val < rules["min_absolute"]:
            failures.append(
                f"candidate value {val:.4f} below min_absolute {rules['min_absolute']}"
            )

    if "max_delta" in rules:
        delta = metric.get("delta")
        if delta is not None and delta > rules["max_delta"]:
            failures.append(
                f"delta {delta:+.4f} exceeds max_delta {rules['max_delta']}"
            )

    if "max_delta_negative" in rules:
        delta = metric.get("delta")
        if delta is not None and delta < -abs(rules["max_delta_negative"]):
            failures.append(
                f"delta {delta:+.4f} is more negative than allowed "
                f"({-abs(rules['max_delta_negative'])})"
            )

    if "max_increase_pct" in rules:
        pct = metric.get("pct_change")
        if pct is not None and pct > rules["max_increase_pct"]:
            failures.append(
                f"pct_change {pct:+.1%} exceeds max_increase_pct "
                f"{rules['max_increase_pct']:.1%}"
            )

    return {
        "metric": metric_name,
        "status": "fail" if failures else "pass",
        "messages": failures,
        "hard": rules.get("hard", False),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--comparison", required=True, type=Path)
    p.add_argument("--gates", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    args = p.parse_args()

    comparison = json.loads(args.comparison.read_text())
    gates_cfg = _read_simple_yaml(args.gates.read_text())
    result = evaluate(comparison, gates_cfg)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2))

    for f in result["findings"]:
        prefix = "FAIL" if f["status"] == "fail" else f["status"].upper()
        hard_tag = " [HARD]" if f.get("hard") else ""
        print(f"{prefix}{hard_tag} {f['metric']}: {f.get('messages') or 'ok'}")

    if result["hard_fail"]:
        print("BLOCKING: at least one hard gate failed.")
        return 1
    if result["soft_fail"]:
        print("WARNING: soft gates failed. Waiver label required to merge.")
        return 2
    print("All gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
