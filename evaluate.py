"""Run the baseline or the treatment on a labelled CSV of tweets and save one result row per case.

Works for the development files made by prepare_data.py and make_golden_set.py
(columns: ID, tweet_text, Relevance, Information Type) and for the edge-case
file (columns: case_id, tweet, gold_relevance, gold_information_type).

Every case stays in the denominator: if a request fails (for example a server
error or an output that is still invalid after the bounded repair), the case is
recorded as invalid and, when it has gold labels, as not an exact match.
Results are saved after every case, so a long run is not lost.

Quota stop: if the API still answers 429 (rate limit) after all retries, the
run stops before recording that case, because this is a quota limit and not a
failure of the system being tested. Run the same command again with --resume
(for example after the daily quota resets) to continue from the next case.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import treatment
from baseline import MODEL_NAME, classify_tweet, is_consistent

SYSTEMS = ("baseline", "treatment")

FIELDS = [
    "case_id", "gold_relevance", "gold_information_type",
    "initial_relevance", "initial_information_type",
    "review_relevance", "review_information_type",
    "pred_relevance", "pred_information_type",
    "exact_match", "valid_final_pair", "initial_violation", "latency_seconds",
    "model_calls", "api_attempts", "repairs", "input_tokens", "output_tokens",
    "reasoning_tokens", "total_tokens", "error",
    "system", "model", "seed", "judge_changed_pair",
]


class QuotaStop(Exception):
    """The API kept answering 429 after all retries."""


def read_cases(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as infile:
        return list(csv.DictReader(infile))


def case_id_of(idx: int, case: dict[str, str]) -> str:
    return case.get("case_id") or case.get("ID") or str(idx)


def classifier_for(system: str):
    # Looked up at call time, so tests can replace classify_tweet.
    return classify_tweet if system == "baseline" else treatment.classify_tweet


def run_case(idx: int, case: dict[str, str], seed: int | None,
             system: str = "baseline") -> dict:
    row = dict.fromkeys(FIELDS, "")
    row["case_id"] = case_id_of(idx, case)
    row.update({"system": system, "model": MODEL_NAME, "seed": "" if seed is None else seed})
    text = case.get("tweet_text") or case.get("tweet") or ""
    gold_rel = (case.get("Relevance") or case.get("gold_relevance") or "").strip()
    gold_info = (case.get("Information Type") or case.get("gold_information_type") or "").strip()
    row["gold_relevance"], row["gold_information_type"] = gold_rel, gold_info
    has_gold = bool(gold_rel and gold_info)
    try:
        result = classifier_for(system)(text, seed=seed)
        review = result.get("self_review_prediction") or {}
        row.update({
            "initial_relevance": result["initial_prediction"]["relevance"],
            "initial_information_type": result["initial_prediction"]["information_type"],
            "review_relevance": review.get("relevance", ""),
            "review_information_type": review.get("information_type", ""),
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
            "judge_changed_pair": result.get("judge_changed_pair", ""),
        })
        if has_gold:
            row["exact_match"] = (result["relevance"] == gold_rel
                                  and result["information_type"] == gold_info)
    except Exception as exc:  # Keep the failed case; do not drop it from the results.
        if getattr(exc, "code", None) == 429:
            raise QuotaStop(str(exc)[:250]) from exc
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


def load_saved(path: Path) -> list[dict]:
    """Rows from an earlier, interrupted run of the same command."""
    rows = []
    for row in read_cases(str(path)):
        for key in ("exact_match", "valid_final_pair"):
            if row.get(key) in ("True", "False"):
                row[key] = row[key] == "True"
        rows.append({k: row.get(k, "") for k in FIELDS})
    return rows


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
    ap.add_argument("--system", choices=SYSTEMS, default="baseline", help="Which system to run")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N cases (0 = all)")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed sent to the model")
    ap.add_argument("--pause", type=float, default=0, help="Seconds to wait between tweets (helps with free-tier rate limits)")
    ap.add_argument("--resume", action="store_true",
                    help="Keep the cases already saved in --out and run only the rest")
    args = ap.parse_args()

    cases = read_cases(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No cases found")

    out = Path(args.out)
    rows: list[dict] = load_saved(out) if args.resume and out.exists() else []
    done = {r["case_id"] for r in rows}
    if done:
        print(f"Resuming: {len(done)} cases already saved in {out}")
    first = True
    for idx, case in enumerate(cases, start=1):
        if case_id_of(idx, case) in done:
            continue
        if not first and args.pause:
            time.sleep(args.pause)
        first = False
        try:
            row = run_case(idx, case, args.seed, args.system)
        except QuotaStop as exc:
            print(f"\nStopped at case {idx}/{len(cases)}: the API quota is used up ({exc}).")
            print(f"{len(rows)} cases are saved in {out}. Run the same command with --resume to continue.")
            sys.exit(2)
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
