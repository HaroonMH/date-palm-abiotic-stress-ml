#!/usr/bin/env python3
"""
Step 3: assemble positive/negative sets per stress and do the
7:3 train/test split -- the final "curated non-redundant FASTA dataset"

For stress X:
  - positive = its own CD-HIT'd (non-redundant) single-stress set
  - negative = union of the OTHER THREE stresses' CD-HIT'd single-stress sets
               (straight concatenation, no extra dedup -- this reproduces the
               reference paper's Table 1 arithmetic exactly, e.g. cabbage
               cold negative(1163) = heat(515)+drought(409)+salt(239))

Then split 7:3 train/test, stratified by label, per stress, fixed seed.

Outputs (under stress_gene_prediction/data/final/):
  - {stress}_train.fasta, {stress}_test.fasta
  - {stress}_labels.tsv        (ProteinID, Label[1=positive/0=negative], Split[train/test])
  - dataset_summary.tsv         (Table-1-style counts report)
"""
import csv
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from fasta_utils import parse_fasta, write_fasta  # noqa: E402

DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
NR_DIR = os.path.join(DATA_DIR, "nr")
FINAL_DIR = os.path.join(DATA_DIR, "final")

STRESSES = ["cold", "heat", "drought", "salt"]
TRAIN_FRACTION = 0.7
RANDOM_SEED = 42


def stratified_split(ids, fraction, rng):
    ids = list(ids)
    rng.shuffle(ids)
    n_train = round(len(ids) * fraction)
    return ids[:n_train], ids[n_train:]


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    nr_records = {}
    for stress in STRESSES:
        path = os.path.join(NR_DIR, f"{stress}_nr.fasta")
        if not os.path.exists(path):
            sys.exit(f"ERROR: {path} not found -- run 02_run_cdhit.sh first")
        nr_records[stress] = parse_fasta(path)

    summary_rows = []

    for stress in STRESSES:
        positive_records = nr_records[stress]
        positive_ids = list(positive_records.keys())

        negative_records = {}
        for other in STRESSES:
            if other == stress:
                continue
            negative_records.update(nr_records[other])
        negative_ids = list(negative_records.keys())

        all_records = {}
        all_records.update(positive_records)
        all_records.update(negative_records)

        pos_train, pos_test = stratified_split(positive_ids, TRAIN_FRACTION, rng)
        neg_train, neg_test = stratified_split(negative_ids, TRAIN_FRACTION, rng)

        train_ids = pos_train + neg_train
        test_ids = pos_test + neg_test
        rng.shuffle(train_ids)
        rng.shuffle(test_ids)

        write_fasta(all_records, os.path.join(FINAL_DIR, f"{stress}_train.fasta"), ids=train_ids)
        write_fasta(all_records, os.path.join(FINAL_DIR, f"{stress}_test.fasta"), ids=test_ids)

        labels_path = os.path.join(FINAL_DIR, f"{stress}_labels.tsv")
        with open(labels_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["ProteinID", "Label", "Split"])
            for pid in pos_train:
                writer.writerow([pid, 1, "train"])
            for pid in pos_test:
                writer.writerow([pid, 1, "test"])
            for pid in neg_train:
                writer.writerow([pid, 0, "train"])
            for pid in neg_test:
                writer.writerow([pid, 0, "test"])

        print(f"=== {stress} ===")
        print(f"  positive set (post CD-HIT): {len(positive_ids)}")
        print(f"  negative set (post CD-HIT): {len(negative_ids)}")
        print(f"  train: {len(train_ids)} (pos={len(pos_train)}, neg={len(neg_train)})")
        print(f"  test:  {len(test_ids)} (pos={len(pos_test)}, neg={len(neg_test)})")
        print(f"  -> {labels_path}")
        print()

        summary_rows.append({
            "stress": stress,
            "positive_set": len(positive_ids),
            "negative_set": len(negative_ids),
            "train_total": len(train_ids),
            "train_positive": len(pos_train),
            "train_negative": len(neg_train),
            "test_total": len(test_ids),
            "test_positive": len(pos_test),
            "test_negative": len(neg_test),
        })

    summary_path = os.path.join(DATA_DIR, "dataset_summary.tsv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "stress", "positive_set", "negative_set",
            "train_total", "train_positive", "train_negative",
            "test_total", "test_positive", "test_negative",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Dataset summary (Table-1 equivalent) written to: {summary_path}")


if __name__ == "__main__":
    main()
