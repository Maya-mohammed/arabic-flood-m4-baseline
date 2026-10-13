"""Run M4 labelled edge cases and save a CSV after every case.

Labelled requests with API errors are counted as incorrect for end-to-end
exact-match; no gold labels are guessed or modified. This is an M4 test
runner, not the formal frozen M5 evaluation harness.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from baseline import classify_tweet, RELEVANCE_LABELS, INFO_TYPE_LABELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="private_data/dev_5.csv")
    ap.add_argument("--out", default="private_data/dev_5_results.csv")
    ap.add_argument("--limit", type=int, default=0, help="0 means all provided cases")
    args = ap.parse_args()
    with open(args.cases, newline="", encoding="utf-8-sig") as infile:
        cases = list(csv.DictReader(infile))
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("No cases found")
    fields = ["case_id", "gold_relevance", "gold_information_type", "pred_relevance", "pred_information_type",
              "exact_match", "valid_final_pair", "initial_violation", "latency_seconds",
              "model_calls", "api_attempts", "repairs", "input_tokens", "output_tokens",
              "reasoning_tokens", "total_tokens", "error"]
    results = []
    for idx, case in enumerate(cases, start=1):
        case_id = case.get("case_id") or case.get("ID") or str(idx)
        text = case.get("tweet_text") or case.get("tweet") or ""
        gold_rel = (case.get("Relevance") or case.get("gold_relevance") or "").strip()
        gold_info = (case.get("Information Type") or case.get("gold_information_type") or "").strip()
        row = dict.fromkeys(fields, "")
        row.update({"case_id": case_id, "gold_relevance": gold_rel,
                    "gold_information_type": gold_info})
        try:
            result = classify_tweet(text)
            row.update({
                "pred_relevance": result["relevance"],
                "pred_information_type": result["information_type"],
                "valid_final_pair": (result["relevance"] in RELEVANCE_LABELS
                                     and result["information_type"] in INFO_TYPE_LABELS
                                     and (result["relevance"] != "Not related"
                                          or result["information_type"] == "Not applicable")),
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
            if gold_rel and gold_info:
                row["exact_match"] = (result["relevance"] == gold_rel
                                      and result["information_type"] == gold_info)
            print(f"[{idx}/{len(cases)}] {case_id}: {result['relevance']} / "
                  f"{result['information_type']}; match={row['exact_match']}; "
                  f"latency={result['latency_seconds']}s")
        except Exception as exc:
            row["valid_final_pair"] = False
            if gold_rel and gold_info:
                row["exact_match"] = False
            row["error"] = f"{type(exc).__name__}: {str(exc)[:250]}"
            print(f"[{idx}/{len(cases)}] {case_id} FAILED: {type(exc).__name__}")
        results.append(row)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", newline="", encoding="utf-8-sig") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8-sig") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    labelled = [r for r in results if r["exact_match"] in (True, False)]
    valid = sum(bool(r["valid_final_pair"]) for r in results)
    print(f"Valid pair rate: {valid}/{len(results)} ({100*valid/len(results):.1f}%)")
    if labelled:
        match = sum(bool(r["exact_match"]) for r in labelled)
        print(f"Joint exact match (labelled only): {match}/{len(labelled)} ({100*match/len(labelled):.1f}%)")
    print("Local results saved to", output)


if __name__ == "__main__":
    main()
