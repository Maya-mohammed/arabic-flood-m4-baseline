"""Compare the baseline and the treatment on the same golden set and seeds.

Both conditions are scored with score_runs.score_run, so every metric uses the
same rules. The script stops if the two conditions do not cover the same seeds.

Outputs (in --out):
- comparison.csv: every metric as mean and SD per condition, and the
  difference in means (treatment minus baseline).
- per_class_f1.csv: per-class F1 for both conditions.
- paired_outcomes.csv: for each seed, how many cases both systems got right,
  only one got right, or both got wrong (same case, same seed).
- error_breakdown.csv: error types per condition, added over the seeds.
- comparison.md: the main table in Markdown, for the write-up.

Example:
    python compare_runs.py \
        --baseline m5/runs/baseline/seed_42.csv m5/runs/baseline/seed_123.csv m5/runs/baseline/seed_2026.csv \
        --treatment m6/runs/treatment/seed_42.csv m6/runs/treatment/seed_123.csv m6/runs/treatment/seed_2026.csv \
        --out m6/results/comparison --price-input 0.30 --price-output 2.50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from labels import INFO_TYPE_LABELS
from score_runs import load_golden, load_run, score_run, seed_from_path, valid_final

ROWS = [
    ("joint_exact_match", "Joint exact match", "pct"),
    ("valid_output_rate", "Valid output rate", "pct"),
    ("failed_requests", "Failed requests", "num"),
    ("relevance_macro_f1", "Relevance macro-F1", "pct"),
    ("info_type_macro_f1", "Information Type macro-F1", "pct"),
    ("first_output_violation_rate", "First-output consistency violations", "pct"),
    ("judge_override_rate", "Judge override rate", "pct"),
    ("helpful_override_rate", "Helpful override rate", "pct"),
    ("harmful_override_rate", "Harmful override rate", "pct"),
    ("mean_model_calls", "Model calls per case", "num"),
    ("mean_api_attempts", "API attempts per case", "num"),
    ("mean_repairs", "Repairs per case", "num"),
    ("mean_input_tokens", "Input tokens per case", "num"),
    ("mean_output_tokens", "Output tokens per case", "num"),
    ("mean_total_tokens", "Total tokens per case", "num"),
    ("mean_latency_seconds", "Latency per case (s)", "num"),
    ("mean_cost_usd", "Estimated cost per case (USD)", "usd"),
]


def score_condition(paths, golden, prices):
    runs, scores = {}, {}
    for path in sorted((Path(p) for p in paths), key=seed_from_path):
        seed = seed_from_path(path)
        if seed in runs:
            raise SystemExit(f"Seed {seed} given twice")
        runs[seed] = load_run(path, golden)
        scores[seed] = score_run(runs[seed], prices)
    return runs, scores


def mean_sd(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    if not values:
        return np.nan, np.nan
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else np.nan)


def correct(run):
    valid = valid_final(run)
    return valid & (run["pred_relevance"] == run["gold_relevance"]) & (
        run["pred_information_type"] == run["gold_information_type"])


def error_breakdown(runs):
    counts = {"cases": 0, "wrong": 0, "invalid_or_failed": 0,
              "not_related_labelled_related": 0, "related_labelled_not_related": 0,
              "relevance_right_info_type_wrong": 0}
    confusions: dict[str, int] = {}
    for run in runs.values():
        valid = valid_final(run)
        gr, gi = run["gold_relevance"], run["gold_information_type"]
        pr, pi = run["pred_relevance"], run["pred_information_type"]
        counts["cases"] += len(run)
        counts["wrong"] += int((~correct(run)).sum())
        counts["invalid_or_failed"] += int((~valid).sum())
        counts["not_related_labelled_related"] += int((valid & (gr == "Not related") & (pr == "Related")).sum())
        counts["related_labelled_not_related"] += int((valid & (gr == "Related") & (pr == "Not related")).sum())
        info_wrong = valid & (gr == pr) & (gi != pi)
        counts["relevance_right_info_type_wrong"] += int(info_wrong.sum())
        for g, p in zip(gi[valid & (gi != pi)], pi[valid & (gi != pi)]):
            key = f"{g} -> {p}"
            confusions[key] = confusions.get(key, 0) + 1
    top = sorted(confusions.items(), key=lambda kv: -kv[1])[:5]
    for rank, (key, n) in enumerate(top, start=1):
        counts[f"top_confusion_{rank}"] = f"{key} ({n})"
    return counts


def fmt(value, kind):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.2f}%"
    if kind == "usd":
        return f"{value:.6f}"
    return f"{value:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", default="m5/golden_set_ids.csv")
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--treatment", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--price-input", type=float)
    ap.add_argument("--price-output", type=float)
    args = ap.parse_args()
    prices = (None if args.price_input is None
              else (args.price_input, args.price_output))

    golden = load_golden(args.golden)
    base_runs, base = score_condition(args.baseline, golden, prices)
    treat_runs, treat = score_condition(args.treatment, golden, prices)
    if set(base) != set(treat):
        raise SystemExit(f"The conditions cover different seeds: {sorted(base)} vs {sorted(treat)}")
    seeds = sorted(base)

    table = []
    for key, label, kind in ROWS:
        b = [base[s]["metrics"].get(key) for s in seeds]
        t = [treat[s]["metrics"].get(key) for s in seeds]
        b_mean, b_sd = mean_sd(b)
        t_mean, t_sd = mean_sd(t)
        table.append({"metric": key, "label": label, "kind": kind,
                      "baseline_mean": b_mean, "baseline_sd": b_sd,
                      "treatment_mean": t_mean, "treatment_sd": t_sd,
                      "difference": t_mean - b_mean if not np.isnan(b_mean) and not np.isnan(t_mean) else np.nan,
                      **{f"baseline_seed_{s}": base[s]["metrics"].get(key) for s in seeds},
                      **{f"treatment_seed_{s}": treat[s]["metrics"].get(key) for s in seeds}})
    table = pd.DataFrame(table)

    per_class = []
    for task_label in base[seeds[0]]["per_class"]:
        b_mean, b_sd = mean_sd([base[s]["per_class"][task_label] for s in seeds])
        t_mean, t_sd = mean_sd([treat[s]["per_class"][task_label] for s in seeds])
        per_class.append({"task": task_label[0], "label": task_label[1],
                          "baseline_mean": b_mean, "baseline_sd": b_sd,
                          "treatment_mean": t_mean, "treatment_sd": t_sd,
                          "difference": t_mean - b_mean})
    per_class = pd.DataFrame(per_class)

    paired = []
    for s in seeds:
        b_ok, t_ok = correct(base_runs[s]), correct(treat_runs[s])
        paired.append({"seed": s, "both_correct": int((b_ok & t_ok).sum()),
                       "baseline_only_correct": int((b_ok & ~t_ok).sum()),
                       "treatment_only_correct": int((~b_ok & t_ok).sum()),
                       "both_wrong": int((~b_ok & ~t_ok).sum())})
    paired = pd.DataFrame(paired)

    errors = pd.DataFrame({"baseline": error_breakdown(base_runs),
                           "treatment": error_breakdown(treat_runs)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.drop(columns="kind").to_csv(out / "comparison.csv", index=False, float_format="%.6f")
    per_class.to_csv(out / "per_class_f1.csv", index=False, float_format="%.4f")
    paired.to_csv(out / "paired_outcomes.csv", index=False)
    errors.to_csv(out / "error_breakdown.csv", index_label="count")

    lines = [f"Same {len(golden)} cases and seeds {', '.join(map(str, seeds))} for both conditions. "
             "Mean ± sample SD across seeds.", "",
             "| Metric | Baseline | Treatment | Difference |", "|---|---|---|---|"]
    for row in table.itertuples():
        b = "n/a" if np.isnan(row.baseline_mean) else f"{fmt(row.baseline_mean, row.kind)} ± {fmt(row.baseline_sd, row.kind)}"
        t = "n/a" if np.isnan(row.treatment_mean) else f"{fmt(row.treatment_mean, row.kind)} ± {fmt(row.treatment_sd, row.kind)}"
        d = "n/a" if np.isnan(row.difference) else (f"{row.difference * 100:+.2f} points" if row.kind == "pct"
                                                    else f"{row.difference:+.6f}" if row.kind == "usd"
                                                    else f"{row.difference:+.2f}")
        if row.metric in ("mean_reasoning_tokens",):
            continue
        lines.append(f"| {row.label} | {b} | {t} | {d} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print("\nPaired outcomes per seed (same case, same seed):")
    print(paired.to_string(index=False))
    print("\nError breakdown, added over seeds:")
    print(errors.to_string())
    print(f"\nTables saved in {out}/")


if __name__ == "__main__":
    main()
