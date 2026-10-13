"""Run the baseline on a labelled CSV of tweets and save one result row per case.

Works for the development files made by prepare_data.py (columns: ID, tweet_text,
Relevance, Information Type) and for the edge-case file (columns: case_id, tweet,
gold_relevance, gold_information_type).

Every case stays in the denominator: if a request fails (for example an API
error), the case is recorded as invalid and, when it has gold labels, as not an
exact match. Results are saved after every case, so a long run is not lost.

This is an M4 test runner. The reserved evaluation partition must not be used
until the settings are frozen (M5).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baseline import classify_tweet, is_consistent

FIELDS = [
    "case_id", "gold_relevance", "gold_information_type",
    "initial_relevance", "initial_information_type",
    "review_relevance", "review_information_type",
    "pred_relevance", "pred_information_type",
    "exact_match", "valid_final_pair", "initial_violation", "latency_seconds",
    "model_calls", "api_attempts", "repairs", "input_tokens", "output_tokens",
    "reasoning_tokens", "total_tokens", "error",
]


def read_cases(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as infile:
        return list(csv.DictReader(infile))


def run_case(idx: int, case: dict[str, str], seed: int | None) -> dict:
    row = dict.fromkeys(FIELDS, "")
    row["case_id"] = case.get("case_id") or case.get("ID") or str(idx)
    text = case.get("tweet_text") or case.get("tweet") or ""
    gold_rel = (case.get("Relevance") or case.get("gold_relevance") or "").strip()
    gold_info = (case.get("Information Type") or case.get("gold_information_type") or "").strip()
    row["gold_relevance"], row["gold_information_type"] = gold_rel, gold_info
    has_gold = bool(gold_rel and gold_info)
    try:
        result = classify_tweet(text, seed=seed)
        row.update({
            "initial_relevance": result["initial_prediction"]["relevance"],
            "initial_information_type": result["initial_prediction"]["information_type"],
            "review_relevance": result["self_review_prediction"]["relevance"],
            "review_information_type": result["self_review_prediction"]["information_type"],
            "pred_relevance": result["relevance"],
            "pred_information_type": result["information_type"],
            "valid_final_pair": is_consistent(result),
            "initial_violation": result["initial_consistency_violation"],
            "latency_seconds": result["latency_seconds"],
            "model_calls": result["successful_model_calls"],
            "api_attempts": result["api_attempts"],
            "repairs": result["repairs"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "reasoning_tokens": result["reasoning_tokens"],
            "total_tokens": result["total_tokens"],
        })
        if has_gold:
            row["exact_match"] = (result["relevance"] == gold_rel
                                  and result["information_type"] == gold_info)
    except Exception as exc:  # Keep the failed case; do not drop it from the results.
        row["valid_final_pair"] = False
        if has_gold:
            row["exact_match"] = False
        row["error"] = f"{type(exc).__name__}: {str(exc)[:250]}"
    return row


def save(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summary(rows: list[dict]) -> str:
    n = len(rows)
    valid = sum(r["valid_final_pair"] is True for r in rows)
    lines = [f"Valid final pairs: {valid}/{n} ({100 * valid / n:.1f}%)"]
    labelled = [r for r in rows if r["exact_match"] in (True, False)]
    if labelled:
        match = sum(r["exact_match"] is True for r in labelled)
        lines.append(f"Joint exact match (all labelled cases, failures count as wrong): "
                     f"{match}/{len(labelled)} ({100 * match / len(labelled):.1f}%)")
    failed = sum(bool(r["error"]) for r in rows)
    lines.append(f"Failed requests: {failed}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True, help="CSV of labelled cases")
    ap.add_argument("--out", required=True, help="Where to save the results CSV")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N cases (0 = all)")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed sent to the model")
    args = ap.parse_args()

    cases = read_cases(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No cases found")

    out = Path(args.out)
    rows: list[dict] = []
    for idx, case in enumerate(cases, start=1):
        row = run_case(idx, case, args.seed)
        rows.append(row)
        save(rows, out)  # Save after every case.
        status = (f"FAILED ({row['error'].split(':')[0]})" if row["error"] else
                  f"{row['pred_relevance']} / {row['pred_information_type']}; "
                  f"match={row['exact_match']}; {row['latency_seconds']}s")
        print(f"[{idx}/{len(cases)}] {row['case_id']}: {status}")
    print(summary(rows))
    print("Results saved to", out)


if __name__ == "__main__":
    main()
