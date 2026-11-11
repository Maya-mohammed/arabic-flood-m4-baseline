"""Score saved evaluation runs against the golden set.

Reads one results CSV per seed (as written by evaluate.py), checks that every
run covers exactly the golden-set cases with the same gold labels, and reports
each metric per seed plus the mean and sample standard deviation across seeds.

Scoring rules:
- Exact match and F1 are recomputed from the labels; the saved exact_match
  column is only used as a cross-check.
- A case with an error, a label outside the closed sets, or a final pair that
  breaks the one-way rule counts as invalid and wrong. For F1 its prediction
  is the placeholder NO_VALID_OUTPUT, which lowers recall of the gold class.
- Macro-F1 averages over the closed label set only (2 Relevance labels,
  7 Information Type labels).
- Token, call and latency means are taken over cases that returned a result.
- USD cost is only computed when prices are given on the command line.

Example:
    python score_runs.py --condition baseline \
        --runs m5/runs/baseline/seed_42.csv m5/runs/baseline/seed_123.csv \
               m5/runs/baseline/seed_2026.csv \
        --out m5/results/baseline
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from labels import INFO_TYPE_LABELS, RELEVANCE_LABELS

NO_OUTPUT = "NO_VALID_OUTPUT"
USAGE_COLUMNS = ["model_calls", "api_attempts", "repairs", "input_tokens",
                 "output_tokens", "reasoning_tokens", "total_tokens",
                 "latency_seconds"]
REQUIRED = ["case_id", "gold_relevance", "gold_information_type",
            "initial_relevance", "initial_information_type",
            "pred_relevance", "pred_information_type", "exact_match",
            "initial_violation", "error", *USAGE_COLUMNS]


def is_true(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower() == "true"


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot read the seed from the file name: {path.name}")
    return int(match.group(1))


def load_golden(path: str) -> pd.DataFrame:
    golden = pd.read_csv(path, dtype=str)
    if golden["ID"].duplicated().any():
        raise ValueError("Golden set has duplicate IDs")
    return golden.set_index("ID")


def load_run(path: Path, golden: pd.DataFrame) -> pd.DataFrame:
    run = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in run.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    if run["case_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate case IDs")
    if set(run["case_id"]) != set(golden.index):
        raise ValueError(f"{path.name}: cases do not match the golden set")
    run = run.set_index("case_id").loc[golden.index]
    gold_ok = ((run["gold_relevance"] == golden["Relevance"])
               & (run["gold_information_type"] == golden["Information Type"]))
    if not gold_ok.all():
        raise ValueError(f"{path.name}: gold labels differ from the golden set "
                         f"for {int((~gold_ok).sum())} cases")
    return run


def valid_final(run: pd.DataFrame) -> pd.Series:
    rel, info = run["pred_relevance"], run["pred_information_type"]
    allowed = rel.isin(RELEVANCE_LABELS) & info.isin(INFO_TYPE_LABELS)
    consistent = (rel != "Not related") | (info == "Not applicable")
    no_error = run["error"].str.strip() == ""
    return allowed & consistent & no_error


def score_run(run: pd.DataFrame, prices: tuple[float, float] | None) -> dict:
    valid = valid_final(run)
    gold_rel, gold_info = run["gold_relevance"], run["gold_information_type"]
    pred_rel = run["pred_relevance"].where(valid, NO_OUTPUT)
    pred_info = run["pred_information_type"].where(valid, NO_OUTPUT)
    exact = (pred_rel == gold_rel) & (pred_info == gold_info)

    saved_exact = is_true(run["exact_match"])
    if not (saved_exact == exact).all():
        print(f"  Note: saved exact_match differs from the recomputed value "
              f"for {int((saved_exact != exact).sum())} cases; the recomputed value is used.")

    has_initial = (run["initial_relevance"].isin(RELEVANCE_LABELS)
                   & run["initial_information_type"].isin(INFO_TYPE_LABELS))
    returned = run["error"].str.strip() == ""

    metrics = {
        "cases": len(run),
        "joint_exact_match": exact.mean(),
        "valid_output_rate": valid.mean(),
        "failed_requests": int((~returned).sum()),
        "relevance_macro_f1": f1_score(gold_rel, pred_rel, labels=list(RELEVANCE_LABELS),
                                       average="macro", zero_division=0),
        "info_type_macro_f1": f1_score(gold_info, pred_info, labels=list(INFO_TYPE_LABELS),
                                       average="macro", zero_division=0),
        "first_output_violation_rate": (is_true(run["initial_violation"])[has_initial].mean()
                                        if has_initial.any() else np.nan),
        "cases_with_first_output": int(has_initial.sum()),
    }

    usage = run.loc[returned, USAGE_COLUMNS].replace("", np.nan).astype(float)
    for column in USAGE_COLUMNS:
        metrics[f"mean_{column}"] = usage[column].mean()  # NaN if never returned
    if prices is not None:
        input_price, output_price = prices
        output_like = usage["output_tokens"] + usage["reasoning_tokens"].fillna(0)
        cost = (usage["input_tokens"] * input_price + output_like * output_price) / 1e6
        metrics["mean_cost_usd"] = cost.mean()

    per_class = {
        ("Relevance", label): score
        for label, score in zip(RELEVANCE_LABELS, f1_score(
            gold_rel, pred_rel, labels=list(RELEVANCE_LABELS), average=None, zero_division=0))
    }
    per_class.update({
        ("Information Type", label): score
        for label, score in zip(INFO_TYPE_LABELS, f1_score(
            gold_info, pred_info, labels=list(INFO_TYPE_LABELS), average=None, zero_division=0))
    })

    pred_info_labels = list(INFO_TYPE_LABELS) + [NO_OUTPUT]
    pred_rel_labels = list(RELEVANCE_LABELS) + [NO_OUTPUT]
    confusions = {
        "information_type": pd.DataFrame(
            confusion_matrix(gold_info, pred_info, labels=pred_info_labels)[:len(INFO_TYPE_LABELS)],
            index=list(INFO_TYPE_LABELS), columns=pred_info_labels),
        "relevance": pd.DataFrame(
            confusion_matrix(gold_rel, pred_rel, labels=pred_rel_labels)[:len(RELEVANCE_LABELS)],
            index=list(RELEVANCE_LABELS), columns=pred_rel_labels),
    }
    return {"metrics": metrics, "per_class": per_class, "confusions": confusions}


def mean_sd(values: pd.DataFrame) -> pd.DataFrame:
    out = values.copy()
    out["mean"] = values.mean(axis=1)
    out["sd"] = values.std(axis=1, ddof=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", default="m5/golden_set_ids.csv")
    ap.add_argument("--runs", nargs="+", required=True, help="One results CSV per seed, named seed_<n>.csv")
    ap.add_argument("--condition", required=True, help="Name used in the output, e.g. baseline")
    ap.add_argument("--out", required=True, help="Folder for the result tables")
    ap.add_argument("--price-input", type=float, help="USD per 1M input tokens")
    ap.add_argument("--price-output", type=float, help="USD per 1M output tokens")
    args = ap.parse_args()

    if (args.price_input is None) != (args.price_output is None):
        raise SystemExit("Give both --price-input and --price-output, or neither.")
    prices = None if args.price_input is None else (args.price_input, args.price_output)

    golden = load_golden(args.golden)
    paths = sorted((Path(p) for p in args.runs), key=seed_from_path)
    seeds = [seed_from_path(p) for p in paths]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("Each seed must appear once")

    results = {}
    for seed, path in zip(seeds, paths):
        print(f"Scoring seed {seed}: {path}")
        results[seed] = score_run(load_run(path, golden), prices)

    columns = [f"seed_{s}" for s in seeds]
    per_seed = pd.DataFrame({f"seed_{s}": r["metrics"] for s, r in results.items()})[columns]
    summary = mean_sd(per_seed)
    per_class = mean_sd(pd.DataFrame({f"seed_{s}": r["per_class"] for s, r in results.items()})[columns])
    per_class.index.names = ["task", "label"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index_label="metric", float_format="%.6f")
    per_class.to_csv(out / "per_class_f1.csv", float_format="%.6f")
    for name in ("information_type", "relevance"):
        for seed, r in results.items():
            r["confusions"][name].to_csv(out / f"confusion_{name}_seed_{seed}.csv", index_label="gold")
        total = sum(r["confusions"][name] for r in results.values())
        total.to_csv(out / f"confusion_{name}_all_seeds.csv", index_label="gold")

    pct = ["joint_exact_match", "valid_output_rate", "relevance_macro_f1",
           "info_type_macro_f1", "first_output_violation_rate"]
    counts = ["cases", "failed_requests", "cases_with_first_output"]
    print(f"\n{args.condition}: {len(golden)} cases x {len(seeds)} seeds {seeds}")
    for metric, row in summary.iterrows():
        if metric in counts:
            print(f"  {metric:32s} " + ", ".join(f"{int(v)}" for v in row[columns]))
            continue
        if np.isnan(row["mean"]):
            print(f"  {metric:32s} not returned")
            continue
        scale = 100 if metric in pct else 1
        unit = "%" if metric in pct else ""
        places = 6 if "cost" in metric else 2
        sd = "" if np.isnan(row["sd"]) else f" ± {row['sd'] * scale:.{places}f}{unit}"
        print(f"  {metric:32s} {row['mean'] * scale:.{places}f}{unit}{sd}")
    print(f"\nTables saved in {out}/")


if __name__ == "__main__":
    main()
