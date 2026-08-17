#!/usr/bin/env python3
"""
Phase 2, step 1: generate protein sequence features via iFeatureOmega for
each stress's train/test FASTA (produced by Milestone 1's
03_build_pos_neg_split.py), using the six descriptors named in the pipeline
doc: CKSAAP, DDE, DPC, TPC, PAAC, APAAC.

Uses "type 2" (raw count / unnormalized) for CKSAAP/DPC/TPC, matching the
reference paper's description of those encodings as raw counts. Default
iFeatureOmegaCLI parameters are used (CKSAAP kspace=3, PAAC/APAAC lambda=3);
these are safe for our data since the shortest sequence per stress (checked
via CD-HIT's banner output) is well above lambda+1.

Output: stress_gene_prediction/data/features/{stress}_{method}_{split}.csv
        (one row per protein, iFeatureOmega's native columns, index=ProteinID)
"""
import os
import sys

import iFeatureOmegaCLI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from ml_utils import STRESSES, FEATURE_METHODS, FINAL_DIR, FEATURES_DIR  # noqa: E402


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)

    for stress in STRESSES:
        for split in ["train", "test"]:
            fasta_path = os.path.join(FINAL_DIR, f"{stress}_{split}.fasta")
            if not os.path.exists(fasta_path):
                sys.exit(f"ERROR: {fasta_path} not found -- run Milestone 1 scripts first")

            print(f"=== {stress} / {split} ===")
            for label, descriptor in FEATURE_METHODS.items():
                seq_obj = iFeatureOmegaCLI.iProtein(fasta_path)
                seq_obj.get_descriptor(descriptor)
                if seq_obj.encodings is None:
                    print(f"  WARNING: {descriptor} produced no encodings, skipping")
                    continue
                out_path = os.path.join(FEATURES_DIR, f"{stress}_{label}_{split}.csv")
                seq_obj.encodings.to_csv(out_path)
                print(f"  {label:8s} -> {seq_obj.encodings.shape[1]:5d} features"
                      f"  ({seq_obj.encodings.shape[0]} sequences) -> {out_path}")
            print()

    print("Feature extraction complete.")
    print("Next step: python3 05_screen_feature_methods.py")


if __name__ == "__main__":
    main()
