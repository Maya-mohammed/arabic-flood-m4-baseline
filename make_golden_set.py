"""Build the M5 golden set from the reserved evaluation split.

The golden set is 20 tweets per Information Type (7 x 20 = 140), drawn from
private_data/evaluation.csv with a fixed random_state. Two files are written:

- m5/golden_set_ids.csv: IDs, gold labels and flood event, no tweet text.
  This file is committed so the exact cases are on record.
- private_data/golden_set.csv: the same rows with tweet text, needed by
  evaluate.py. It stays local because we do not redistribute the corpus.

If m5/golden_set_ids.csv already exists, the new sample is compared with it
and the script stops if they differ, so the frozen set is never replaced.

Run prepare_data.py first.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PER_CLASS = 20
RANDOM_STATE = 2026
ID_COLUMNS = ["ID", "Relevance", "Information Type", "Event", "joint_label"]


def sample_golden_set(evaluation: pd.DataFrame) -> pd.DataFrame:
    return (
        evaluation.groupby("Information Type", group_keys=False)
        .sample(n=PER_CLASS, random_state=RANDOM_STATE)
        .sort_values("ID")
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluation", default="private_data/evaluation.csv")
    parser.add_argument("--ids-out", default="m5/golden_set_ids.csv")
    parser.add_argument("--text-out", default="private_data/golden_set.csv")
    args = parser.parse_args()

    evaluation = pd.read_csv(args.evaluation, dtype=str)
    golden = sample_golden_set(evaluation)

    ids_path = Path(args.ids_out)
    if ids_path.exists():
        frozen = pd.read_csv(ids_path, dtype=str)
        if not frozen[ID_COLUMNS].equals(golden[ID_COLUMNS]):
            raise SystemExit(
                f"The new sample does not match the frozen set in {ids_path}. "
                "Nothing was written."
            )
        print(f"Sample matches the frozen golden set in {ids_path}.")
    else:
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        golden[ID_COLUMNS].to_csv(ids_path, index=False)
        print(f"Wrote {ids_path}")

    text_path = Path(args.text_out)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(text_path, index=False)
    print(f"Wrote {text_path} ({len(golden)} cases, kept local)")


if __name__ == "__main__":
    main()
