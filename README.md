# clinical-prompt-eval-demo

A minimal, runnable example of a CI gate for prompt changes in a clinical LLM
product. Built as an interview-ready talking piece, not a production system.

The repo simulates one prompt change to a discharge-summary generator: from a
loose v1 prompt that drops medications and hallucinates a wrong anticoagulant,
to a structured v2 prompt that fixes both. CI runs both prompts on the same
five paired cases, computes metrics, runs paired statistical tests, and
applies hard / soft gates from a YAML policy. The PR gets a comment with the
full result table.

## What this demo proves you can talk about

- Designing the metric stack for a clinical LLM (lexical, judge-style,
  binary safety, operational).
- Paired statistical testing (paired bootstrap for continuous, exact
  McNemar for binary) and why pairing matters.
- Separating hard gates (safety) from soft gates (latency, cost) and how
  CODEOWNERS plus waiver labels reconcile speed with risk.
- Why you mock the model in CI even when you call a real one in production.
- The PHI boundary: this workflow runs on `ubuntu-latest` for the demo, but
  the comment in the workflow shows where you swap in a self-hosted runner
  inside your HIPAA VPC.

## Running it locally

```bash
pip install -r requirements.txt

# 1. Unit tests on the metric code.
python -m pytest tests/ -q

# 2. Run baseline (prompt v1) and candidate (prompt v2).
python -m evals.run --prompt prompts/discharge_summary_v1.txt \
    --cases evals/data/eval_cases.jsonl --out reports/v1_results.json
python -m evals.run --prompt prompts/discharge_summary_v2.txt \
    --cases evals/data/eval_cases.jsonl --out reports/v2_results.json

# 3. Paired comparison and gate.
python -m evals.compare --baseline reports/v1_results.json \
    --candidate reports/v2_results.json --out reports/comparison.json
python -m evals.gate --comparison reports/comparison.json \
    --gates evals/gates.yaml --report reports/gate_report.json
```

`evals/gate.py` exits with code 0 when all gates pass, 1 when a hard gate
fails (merge blocked), 2 when only soft gates fail (waiver path).

## Repo layout

```
prompts/                v1 (baseline) and v2 (candidate) prompts
evals/
  data/eval_cases.jsonl five paired discharge summary cases
  model.py              deterministic mock LLM, swap for real call in prod
  metrics.py            rouge_l, faithfulness, completeness, omission, hallucination
  run.py                runs one prompt over the corpus
  compare.py            paired bootstrap and exact McNemar
  gate.py               applies gates.yaml, exit code drives CI gate
  gates.yaml            hard and soft thresholds, the policy artifact
tests/                  unit tests on metric code
.github/
  workflows/prompt-eval.yml  pipeline that runs on every PR
  CODEOWNERS            forces clinical safety review on prompts and gates
```

## Trade-offs to be ready for in interviews

The eval set is only N=5 here. McNemar at N=5 has almost no power, which the
report will show as p ~ 0.06 to 0.13 even when the candidate is clearly
better. In production you would target N high enough for an 80% power test
at your minimum detectable effect (~5,000 paired cases for a 1pp absolute
change at a 5% baseline event rate). The structure of the code does not
change, only the size of the corpus and the statistical conclusions you can
draw.

The mock LLM is keyed on `(prompt_version, case_id)` for reproducibility.
Replace `evals/model.generate` with a call to your real in-VPC endpoint when
you wire this up for real. Keep the rest of the pipeline identical.

The faithfulness metric here is a deterministic medication-name check, not a
real LLM judge. A real judge would be a separate model call calibrated
against a human-labeled gold set with periodic agreement monitoring; if the
judge drifts off human agreement, the gate that depends on it is no longer
trustworthy and should be quarantined.
