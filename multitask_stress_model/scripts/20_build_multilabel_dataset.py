#!/usr/bin/env python3
"""
Step 1: build the multi-label dataset.

Changes versus the single-task pipeline, and why:

1. NO multi-stress exclusion. The single-task pipeline discarded every protein
   whose GO terms spanned more than one stress -- 735 proteins, 24.1% of all
   labelled data, and biologically the most interesting subset (239 are
   drought+salt, 50 respond to all four). Here each protein simply carries a
   4-bit label vector, which is what a multi-label model consumes.

2. A REAL negative class. The single-task pipeline used "genes responsive to
   the other three stresses" as negatives, so the models were never taught to
   reject an ordinary non-stress protein -- the cause of the 15-26% of the
   proteome being called stress-responsive. Here negatives are proteins that
   ARE GO-annotated (hence functionally characterised) but carry no stress
   term at all. Proteins with no GO annotation whatsoever are EXCLUDED from
   training, because absence of annotation is absence of evidence, not
   evidence of absence -- they would be unreliable negatives. They are still
   scored at genome-wide prediction time.

Output:
  data/labels/multilabel_dataset.tsv   ProteinID, cold, heat, drought, salt,
                                        combo, source ('stress' | 'background')
  data/labels/dataset_all.fasta        sequences for every retained protein
  outputs/tables/label_composition.tsv label-combination counts
"""
import csv
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    GO_FULL_TSV, GO_ANNOTATIONS, PROTEOME_FASTA, LABELS_DIR, TABLES_DIR,
    STRESSES, STRESS_GO, parse_fasta, write_fasta, sanitize, label_combo,
)

MIN_LENGTH = 30  # below this, k-mer/PLM signal is unreliable


def main():
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    for p in (GO_FULL_TSV, GO_ANNOTATIONS, PROTEOME_FASTA):
        if not os.path.exists(p):
            sys.exit(f"ERROR: required input not found: {p}")

    # ---- stress-labelled proteins -> multi-label vector -------------------
    protein_to_stresses = {}
    with open(GO_FULL_TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            go_id = row["GO_ID"].strip()
            for stress, go_set in STRESS_GO.items():
                if go_id in go_set:
                    for pid in (x.strip() for x in row["Proteins"].split(";") if x.strip()):
                        protein_to_stresses.setdefault(pid, set()).add(stress)
    print(f"Stress-labelled proteins: {len(protein_to_stresses)}")
    n_multi = sum(1 for v in protein_to_stresses.values() if len(v) > 1)
    print(f"  of which multi-stress:  {n_multi} "
          f"({100 * n_multi / len(protein_to_stresses):.1f}%) -- retained, not discarded")

    # ---- proteins carrying any GO annotation at all ----------------------
    print("\nScanning GO annotations to identify functionally characterised proteins...")
    annotated = set()
    for chunk in pd.read_csv(GO_ANNOTATIONS, sep="\t", chunksize=200_000):
        annotated.update(chunk["DatePalm_Protein"].unique())
    print(f"  proteins with >=1 GO term: {len(annotated)}")

    # ---- sequences --------------------------------------------------------
    proteome = parse_fasta(PROTEOME_FASTA)
    print(f"  proteome sequences: {len(proteome)}")

    stress_ids = set(protein_to_stresses)
    background_ids = (annotated & set(proteome)) - stress_ids
    unannotated = set(proteome) - annotated - stress_ids
    print(f"\nConfident negatives (annotated, no stress term): {len(background_ids)}")
    print(f"Excluded as ambiguous (no GO annotation at all) : {len(unannotated)}")

    # ---- assemble ---------------------------------------------------------
    rows, keep_records, too_short = [], {}, 0
    for pid in sorted(stress_ids | background_ids):
        if pid not in proteome:
            continue
        header, seq = proteome[pid]
        seq = sanitize(seq)
        if len(seq) < MIN_LENGTH:
            too_short += 1
            continue
        labels = protein_to_stresses.get(pid, set())
        row = {"ProteinID": pid}
        row.update({s: int(s in labels) for s in STRESSES})
        row["source"] = "stress" if pid in stress_ids else "background"
        row["combo"] = label_combo(row)
        row["length"] = len(seq)
        rows.append(row)
        keep_records[pid] = (header, seq)

    df = pd.DataFrame(rows)[["ProteinID"] + STRESSES + ["combo", "source", "length"]]
    out_tsv = os.path.join(LABELS_DIR, "multilabel_dataset.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)
    write_fasta(keep_records, os.path.join(LABELS_DIR, "dataset_all.fasta"),
                ids=df["ProteinID"].tolist())

    comp = (df.groupby(["combo", "source"]).size()
              .reset_index(name="count").sort_values("count", ascending=False))
    comp.to_csv(os.path.join(TABLES_DIR, "label_composition.tsv"), sep="\t", index=False)

    print(f"\nDropped {too_short} sequences shorter than {MIN_LENGTH} residues")
    print(f"Dataset: {len(df)} proteins "
          f"({int((df.source == 'stress').sum())} stress-labelled, "
          f"{int((df.source == 'background').sum())} background negatives)")
    print("\nPositives per stress (multi-label):")
    for s in STRESSES:
        print(f"  {s:8s} {int(df[s].sum()):5d}")
    print(f"\nWrote {out_tsv}")
    print("Next: bash 21_run_cdhit_global.sh")


if __name__ == "__main__":
    main()
