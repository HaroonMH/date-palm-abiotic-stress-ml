#!/usr/bin/env python3
"""
Step 3: one global, stratified 7:3 train/test split.

The single-task pipeline split each stress independently, so a protein could
appear in cold's training set and salt's test set simultaneously. Four
separate models made that harmless. A shared backbone makes it leakage. This
script produces ONE split that every head is evaluated on.

Stratification is on the full label combination (e.g. '0101' = heat+salt), so
rare co-response patterns are represented proportionally in both partitions.
Combinations with fewer than 2 members cannot be stratified and are assigned
to the training set.

Output:
  data/splits/split_assignments.tsv   ProteinID, labels, combo, source, split
  outputs/tables/split_summary.tsv    per-stress and per-combo train/test counts
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    LABELS_DIR, NR_DIR, SPLIT_DIR, TABLES_DIR, STRESSES, RANDOM_STATE, parse_fasta,
)

TRAIN_FRACTION = 0.7


def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    nr_path = os.path.join(NR_DIR, "dataset_nr.fasta")
    if not os.path.exists(nr_path):
        sys.exit(f"ERROR: {nr_path} not found -- run 21_run_cdhit_global.sh first")

    df = pd.read_csv(os.path.join(LABELS_DIR, "multilabel_dataset.tsv"), sep="\t")
    kept = set(parse_fasta(nr_path).keys())
    before = len(df)
    df = df[df["ProteinID"].isin(kept)].reset_index(drop=True)
    print(f"After global CD-HIT: {len(df)} of {before} proteins retained")

    rng = np.random.RandomState(RANDOM_STATE)
    split = pd.Series(index=df.index, dtype=object)

    # Stratify by label combination so rare co-response patterns are
    # proportionally represented in both partitions.
    for combo, group in df.groupby("combo"):
        idx = np.array(group.index, dtype=int)  # copy: .to_numpy() is read-only
        rng.shuffle(idx)
        if len(idx) < 2:
            split.loc[idx] = "train"      # cannot stratify a singleton
            continue
        n_train = max(1, int(round(len(idx) * TRAIN_FRACTION)))
        n_train = min(n_train, len(idx) - 1)   # guarantee >=1 test sample
        split.loc[idx[:n_train]] = "train"
        split.loc[idx[n_train:]] = "test"

    df["split"] = split
    out = os.path.join(SPLIT_DIR, "split_assignments.tsv")
    df.to_csv(out, sep="\t", index=False)

    rows = []
    for s in STRESSES:
        rows.append({
            "stress": s,
            "train_positive": int(df[(df.split == "train") & (df[s] == 1)].shape[0]),
            "train_negative": int(df[(df.split == "train") & (df[s] == 0)].shape[0]),
            "test_positive": int(df[(df.split == "test") & (df[s] == 1)].shape[0]),
            "test_negative": int(df[(df.split == "test") & (df[s] == 0)].shape[0]),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(TABLES_DIR, "split_summary.tsv"), sep="\t", index=False)

    combo_tab = (df.groupby(["combo", "split"]).size().unstack(fill_value=0)
                   .sort_values("train", ascending=False))
    combo_tab.to_csv(os.path.join(TABLES_DIR, "split_by_combo.tsv"), sep="\t")

    print(f"\ntrain: {int((df.split == 'train').sum())}   "
          f"test: {int((df.split == 'test').sum())}")
    print("\nPer-stress positives:")
    print(summary.to_string(index=False))
    print(f"\nWrote {out}")
    print("Next: python3 23_extract_esm_embeddings.py")


if __name__ == "__main__":
    main()
