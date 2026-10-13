"""Prepare private, non-overlapping development/evaluation splits from the source CSV.

Never commit the input corpus or private_data/*.csv to a public repository.
Original 4,037 source rows are not modified on disk.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from labels import RELEVANCE_LABELS, INFO_TYPE_LABELS

REQUIRED = ["ID", "Tweet Text", "Relevance", "Information Type", "Event"]


def unwrap_tweet(raw):
    """Unwrap dataset's one-element list notation without dropping malformed records.

    Most values parse as ['tweet']. Some original values use mismatched quotes;
    for those remove only the verified surrounding markers. No NLP normalization.
    """
    s = str(raw)
    try:
        candidate = ast.literal_eval(s)
        if isinstance(candidate, list) and len(candidate) == 1 and isinstance(candidate[0], str):
            return candidate[0]
    except (ValueError, SyntaxError, TypeError, MemoryError):
        pass
    if s.startswith("['") and s.endswith("']\""):
        return s[2:-3]
    if s.startswith("['") and s.endswith("']"):
        return s[2:-2]
    return s


def build_splits(source_csv):
    df = pd.read_csv(source_csv, dtype=str)
    if any(col not in df.columns for col in REQUIRED):
        raise ValueError("CSV must contain: " + ", ".join(REQUIRED))
    if df[REQUIRED].isna().any().any():
        raise ValueError("Source contains missing values in required columns")
    invalid_labels = df.loc[
        ~df["Relevance"].isin(RELEVANCE_LABELS)
        | ~df["Information Type"].isin(INFO_TYPE_LABELS)
    ]
    if not invalid_labels.empty:
        raise ValueError(f"Invalid source labels ({len(invalid_labels)} rows)")
    conflicting = df.groupby("ID")[["Relevance", "Information Type"]].nunique().gt(1).any(axis=1)
    bad_ids = set(conflicting[conflicting].index)
    clean = df.loc[~df["ID"].isin(bad_ids)].drop_duplicates("ID").reset_index(drop=True).copy()
    clean["tweet_text"] = clean["Tweet Text"].map(unwrap_tweet)
    clean["joint_label"] = clean["Relevance"] + " | " + clean["Information Type"]

    dev, evaluation = train_test_split(
        clean, test_size=.80, random_state=42, stratify=clean["joint_label"]
    )
    dev, evaluation = dev.copy(), evaluation.copy()
    # The original team split has an equal raw tweet text with two distinct IDs
    # on both sides. Swap an unrelated evaluation example with the same gold pair
    # and event into dev, then move the duplicate dev example to evaluation.
    # This preserves row counts, joint-label proportions and event counts.
    overlaps = set(dev["tweet_text"]) & set(evaluation["tweet_text"])
    swaps = []
    for duplicated_text in sorted(overlaps):
        conflict_rows = dev.loc[dev["tweet_text"] == duplicated_text]
        for dev_index, row in conflict_rows.iterrows():
            occupied = set(dev["tweet_text"]) | set(evaluation["tweet_text"])
            elig = evaluation.loc[
                (evaluation["joint_label"] == row["joint_label"])
                & (evaluation["Event"] == row["Event"])
                & (~evaluation["tweet_text"].isin(overlaps))
                & (~evaluation["tweet_text"].duplicated(keep=False))
            ]
            if len(elig) == 0:
                raise ValueError("Could not repair overlap while matching joint label and event")
            replacement_index = elig.sort_values("ID").index[0]
            replacement = evaluation.loc[[replacement_index]]
            dev_row = dev.loc[[dev_index]]
            dev = pd.concat([dev.drop(index=dev_index), replacement], axis=0)
            evaluation = pd.concat([evaluation.drop(index=replacement_index), dev_row], axis=0)
            swaps.append((row["ID"], replacement.iloc[0]["ID"]))
    if set(dev["ID"]) & set(evaluation["ID"]):
        raise AssertionError("ID leakage detected")
    if set(dev["tweet_text"]) & set(evaluation["tweet_text"]):
        raise AssertionError("Text leakage detected")
    if not ((dev["Relevance"] == "Not related") <= (dev["Information Type"] == "Not applicable")).all():
        raise AssertionError("Source contains invalid Relevance implication")
    if not ((evaluation["Relevance"] == "Not related") <= (evaluation["Information Type"] == "Not applicable")).all():
        raise AssertionError("Source contains invalid Relevance implication")
    return df, clean, dev, evaluation, bad_ids, swaps


def choose_normal_cases(dev):
    # One per valid joint-label pair (8 total), then ensure all 4 flood events.
    normal = (
        dev.groupby("joint_label", group_keys=False)
        .sample(n=1, random_state=42)
        .sort_values(["joint_label", "ID"])
    )
    for event in sorted(set(dev["Event"]) - set(normal["Event"])):
        other = dev.loc[(dev["Event"] == event) & (~dev["ID"].isin(normal["ID"]))]
        normal = pd.concat([normal, other.sort_values("ID").head(1)])
    assert len(set(normal["joint_label"])) == len(set(dev["joint_label"])) == 8
    assert len(set(normal["Event"])) == len(set(dev["Event"])) == 4
    return normal


def choose_dev_five(normal_cases):
    """Five different joint pairs spanning all four flood events."""
    chosen = normal_cases.sort_values(["Event", "joint_label"]).drop_duplicates("Event")
    candidates = normal_cases.loc[~normal_cases["ID"].isin(chosen["ID"])].sort_values("joint_label")
    chosen = pd.concat([chosen, candidates.head(max(0, 5-len(chosen)))])
    assert len(chosen) == 5 and chosen["Event"].nunique() == 4
    assert chosen["joint_label"].nunique() == 5
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Source ArabicFloods.csv path")
    parser.add_argument("--out", default="private_data")
    args = parser.parse_args()
    original, clean, dev, evaluation, bad_ids, swaps = build_splits(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export_cols = ["ID", "tweet_text", "Relevance", "Information Type", "Event", "joint_label"]
    normal_cases = choose_normal_cases(dev)
    for name, frame in (
        ("development.csv", dev),
        ("evaluation.csv", evaluation),
        ("dev_5.csv", choose_dev_five(normal_cases)),
        ("normal_cases.csv", normal_cases),
    ):
        frame[export_cols].to_csv(out / name, index=False)
    print(f"Original rows: {len(original)}; conflicting IDs removed: {len(bad_ids)}")
    print(f"Clean: {len(clean)}; development: {len(dev)}; evaluation: {len(evaluation)}")
    print(f"Cross-split duplicate tweet text repaired via swaps: {len(swaps)}")
    print(f"All output files stored LOCALLY in {out}/ (excluded from Git)")


if __name__ == "__main__":
    main()
