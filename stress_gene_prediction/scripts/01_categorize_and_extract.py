#!/usr/bin/env python3
"""
Step 1 of Milestone 1 (post-GO-annotation): categorize date palm proteins into
single-stress groups (cold / heat / drought / salt) using the GO analysis
already completed (abiotic_stress_GO_full.tsv), then pull their sequences
from the full proteome FASTA (protein.faa).

Mirrors the reference paper's methodology (Section 2.1 / Figure 3):
  - Each stress is defined by a small set of *specific* GO terms (not every
    stress-adjacent term), same idea as the paper's "five GO terms closely
    associated with cold stress" approach.
  - A protein counts for a stress only if it carries one of that stress's
    core GO IDs.
  - A protein hit by core terms of >1 stress is EXCLUDED entirely
    ("Exclude overlapped gene ID" in Figure 3) -- it goes into neither
    positive nor negative pools for any stress.

Inputs (paths relative to project root, adjust GO_FULL_TSV / PROTEOME_FASTA
below if yours differ):
  - Date_Palm_Project Data/DatePalm_Annotation/Results files/abiotic_stress_GO_full.tsv
  - protein.faa   (master proteome, 48,801 sequences)

Outputs (under stress_gene_prediction/data/):
  - raw/cold_raw.fasta, raw/heat_raw.fasta, raw/drought_raw.fasta, raw/salt_raw.fasta
  - summary_step1.tsv   (counts: candidates matched / single-stress / overlap-excluded)
"""
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
from fasta_utils import parse_fasta, write_fasta  # noqa: E402

GO_FULL_TSV = os.path.join(
    PROJECT_ROOT,
    "Date_Palm_Project Data", "DatePalm_Annotation", "Results files",
    "abiotic_stress_GO_full.tsv",
)
PROTEOME_FASTA = os.path.join(PROJECT_ROOT, "protein.faa")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# Core GO IDs that DEFINE membership in each stress category.
# Chosen to be specific to that single stress (mirrors the reference paper
# picking a short, specific GO-term list per stress rather than every
# stress-adjacent term). Generic/shared terms (response to osmotic stress,
# response to temperature stimulus, response to abscisic acid, oxidative
# stress, defense response, etc.) are deliberately left out because they
# don't discriminate between the four target stresses.
STRESS_GO = {
    "cold": {
        "GO:0009409",  # response to cold
        "GO:0009631",  # cold acclimation
        "GO:0070417",  # cellular response to cold
        "GO:0050826",  # response to freezing
        "GO:0071497",  # cellular response to freezing
    },
    "heat": {
        "GO:0009408",  # response to heat
        "GO:0010286",  # heat acclimation
        "GO:0034605",  # cellular response to heat
        "GO:0070370",  # cellular heat acclimation
    },
    "drought": {
        "GO:0009414",  # response to water deprivation
        "GO:0042631",  # cellular response to water deprivation
        "GO:0009819",  # drought recovery
        "GO:2000070",  # regulation of response to water deprivation
        "GO:0080148",  # negative regulation of response to water deprivation
        "GO:1902584",  # positive regulation of response to water deprivation
    },
    "salt": {
        "GO:0009651",  # response to salt stress
        "GO:1902074",  # response to salt
        "GO:0042538",  # hyperosmotic salinity response
        "GO:0071472",  # cellular response to salt stress
        "GO:1901000",  # regulation of response to salt stress
        "GO:1901001",  # negative regulation of response to salt stress
        "GO:1901002",  # positive regulation of response to salt stress
    },
}

STRESSES = ["cold", "heat", "drought", "salt"]


def main():
    if not os.path.exists(GO_FULL_TSV):
        sys.exit(f"ERROR: GO file not found: {GO_FULL_TSV}")
    if not os.path.exists(PROTEOME_FASTA):
        sys.exit(f"ERROR: proteome FASTA not found: {PROTEOME_FASTA}")

    os.makedirs(RAW_DIR, exist_ok=True)

    # protein_id -> set of stresses whose core GO terms it matched
    protein_to_stresses = {}

    with open(GO_FULL_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            go_id = row["GO_ID"].strip()
            matched_stress = None
            for stress, go_set in STRESS_GO.items():
                if go_id in go_set:
                    matched_stress = stress
                    break
            if matched_stress is None:
                continue
            proteins = [p.strip() for p in row["Proteins"].split(";") if p.strip()]
            for p in proteins:
                protein_to_stresses.setdefault(p, set()).add(matched_stress)

    total_candidates = len(protein_to_stresses)

    single_stress_ids = {s: [] for s in STRESSES}
    overlap_excluded = 0
    for protein, stresses in protein_to_stresses.items():
        if len(stresses) == 1:
            (only_stress,) = stresses
            single_stress_ids[only_stress].append(protein)
        else:
            overlap_excluded += 1

    print("Loading proteome FASTA (this can take a few seconds)...")
    proteome = parse_fasta(PROTEOME_FASTA)
    print(f"  {len(proteome)} sequences loaded from {PROTEOME_FASTA}")

    summary_rows = []
    for stress in STRESSES:
        ids = sorted(single_stress_ids[stress])
        missing = [i for i in ids if i not in proteome]
        if missing:
            print(f"WARNING: {len(missing)} {stress} IDs have no sequence in "
                  f"protein.faa (first few: {missing[:5]})")
        found_ids = [i for i in ids if i in proteome]

        out_path = os.path.join(RAW_DIR, f"{stress}_raw.fasta")
        write_fasta(proteome, out_path, ids=found_ids)
        print(f"  wrote {len(found_ids)} sequences -> {out_path}")

        summary_rows.append({
            "stress": stress,
            "single_stress_candidates": len(ids),
            "sequences_written": len(found_ids),
            "missing_sequence": len(missing),
        })

    summary_path = os.path.join(DATA_DIR, "summary_step1.tsv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["stress", "single_stress_candidates", "sequences_written", "missing_sequence"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"Total GO-matched candidate proteins (any stress): {total_candidates}")
    print(f"Excluded for matching >1 stress (overlap): {overlap_excluded}")
    print(f"Summary written to: {summary_path}")
    print()
    print("Next step: run cd-hit on each raw FASTA -- see 02_run_cdhit.sh")


if __name__ == "__main__":
    main()
