#!/usr/bin/env python
"""Check the shipped splits against the fold assignment of the upstream checkpoints.

This matters because each fold initialises from the upstream weights of the *same* fold. That
only keeps a fold's validation cases out of its starting point if both use the same assignment
-- otherwise the initialisation has already trained on cases we then validate on, and the
validation score is optimistic.

The upstream release records its assignment implicitly: each fold_<n>/validation/summary.json
lists the cases that fold validated. This compares those sets against splits/fold_<n>_val.csv.

Usage:
    python scripts/verify_fold_alignment.py <nnunet-model-dir> [--splits-dir splits]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib

N_FOLDS = 5


def upstream_val_cases(model_dir: pathlib.Path, fold: int) -> set[str]:
    """Case identifiers the upstream run validated on for this fold."""
    summary = model_dir / f"fold_{fold}" / "validation" / "summary.json"
    if not summary.exists():
        raise SystemExit(f"missing validation summary: {summary}")
    payload = json.loads(summary.read_text())
    return {os.path.basename(case["reference_file"]).replace(".nii.gz", "") for case in payload["metric_per_case"]}


def our_val_cases(splits_dir: pathlib.Path, fold: int) -> set[str]:
    """Case identifiers in this repository's validation split for this fold."""
    path = splits_dir / f"fold_{fold}_val.csv"
    with path.open(newline="") as handle:
        return {os.path.basename(row["mask"]).replace(".npy", "") for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", type=pathlib.Path, help="directory of fold_<n> subdirectories")
    parser.add_argument("--splits-dir", type=pathlib.Path, default=pathlib.Path("splits"))
    args = parser.parse_args()

    print(f"{'fold':<6}{'upstream':>10}{'ours':>7}{'identical':>12}")
    aligned = True
    for fold in range(N_FOLDS):
        theirs = upstream_val_cases(args.model_dir, fold)
        ours = our_val_cases(args.splits_dir, fold)
        same = theirs == ours
        aligned &= same
        print(f"{fold:<6}{len(theirs):>10}{len(ours):>7}{str(same):>12}")
        if not same:
            for label, extra in (("upstream only", theirs - ours), ("ours only", ours - theirs)):
                for case in sorted(extra)[:3]:
                    print(f"       {label}: {case}")

    print()
    if aligned:
        print("Fold assignments match: same-fold initialisation never saw that fold's val cases.")
    else:
        raise SystemExit(
            "Fold assignments DIFFER. Same-fold initialisation would train on cases that are "
            "later validated, making validation scores optimistic."
        )


if __name__ == "__main__":
    main()
