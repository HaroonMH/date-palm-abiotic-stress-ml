#!/usr/bin/env python3
"""
Milestone 3, step 2: rank the genome-wide predictions and build the candidate
gene tables (pipeline doc Phase 3; reference paper Section 3.4 / Table 4).

Following the reference paper's "unlabeled set" definition, each stress is
ranked over every protein EXCEPT that stress's own known single-stress genes
(the pre-CD-HIT positives from Milestone 1) -- those are the training signal,
not new discoveries. Genes known for a DIFFERENT stress stay in the pool, so a
gene can legitimately surface as a novel candidate for another stress.

Functional annotation is NOT recomputed here -- the project's existing
BLAST-vs-Arabidopsis and GO annotation results are joined onto the candidates.

Output (under data/genomewide/):
  {stress}_ranked_predictions.tsv   every unlabeled protein, ranked by score
  {stress}_top10.tsv                 top-10 candidates + annotation
  table4_top10_candidates.tsv        all four stresses side by side (Table 4)
  srg_counts.tsv                     predicted SRG counts at several thresholds
"""
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from fasta_utils import parse_fasta  # noqa: E402
from ml_utils import STRESSES, DATA_DIR  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ANNOT_DIR = os.path.join(PROJECT_ROOT, "Date_Palm_Project Data", "DatePalm_Annotation")
BLAST_BEST_HITS = os.path.join(ANNOT_DIR, "blast_best_hits.txt")
GO_ANNOTATIONS = os.path.join(ANNOT_DIR, "datepalm_GO_annotations.tsv")
PROTEOME_FASTA = os.path.join(PROJECT_ROOT, "protein.faa")

GW_DIR = os.path.join(DATA_DIR, "genomewide")
RAW_DIR = os.path.join(DATA_DIR, "raw")

THRESHOLDS = [0.5, 0.7, 0.9]
TOP_N = 10


def load_descriptions():
    """ProteinID -> free-text product description from the FASTA header."""
    descriptions = {}
    with open(PROTEOME_FASTA) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            parts = header.split(None, 1)
            seq_id = parts[0]
            desc = parts[1] if len(parts) > 1 else ""
            desc = desc.replace("[Phoenix dactylifera]", "").strip()
            descriptions[seq_id] = desc
    return descriptions


def load_blast_hits():
    """ProteinID -> (arabidopsis_hit, percent_identity, evalue)."""
    if not os.path.exists(BLAST_BEST_HITS):
        print(f"  WARNING: {BLAST_BEST_HITS} not found -- skipping homolog column")
        return {}
    hits = {}
    with open(BLAST_BEST_HITS) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 12:
                continue
            hits.setdefault(cols[0], (cols[1], cols[2], cols[10]))
    return hits


def load_go_terms(wanted_ids):
    """ProteinID -> '; '-joined GO names, restricted to wanted_ids."""
    if not os.path.exists(GO_ANNOTATIONS):
        print(f"  WARNING: {GO_ANNOTATIONS} not found -- skipping GO column")
        return {}
    go_map = {}
    for chunk in pd.read_csv(GO_ANNOTATIONS, sep="\t", chunksize=100_000):
        sub = chunk[chunk["DatePalm_Protein"].isin(wanted_ids)]
        for protein, group in sub.groupby("DatePalm_Protein"):
            names = group["GO_Name"].dropna().unique().tolist()
            go_map.setdefault(protein, []).extend(names)
    return {p: "; ".join(sorted(set(names))) for p, names in go_map.items()}


def main():
    predictions_path = os.path.join(GW_DIR, "all_predictions.tsv")
    if not os.path.exists(predictions_path):
        sys.exit(f"ERROR: {predictions_path} not found -- run 10_genomewide_predict.py first")

    preds = pd.read_csv(predictions_path, sep="\t", index_col=0)
    print(f"Loaded predictions for {len(preds)} proteins\n")

    print("Loading existing annotation (BLAST homologs, GO terms, descriptions)...")
    descriptions = load_descriptions()
    blast_hits = load_blast_hits()

    ranked_per_stress = {}
    count_rows = []

    for stress in STRESSES:
        raw_fasta = os.path.join(RAW_DIR, f"{stress}_raw.fasta")
        known = set(parse_fasta(raw_fasta).keys()) if os.path.exists(raw_fasta) else set()

        col = f"{stress}_prob"
        unlabeled = preds[~preds.index.isin(known)][[col]].copy()
        unlabeled = unlabeled.sort_values(col, ascending=False)
        unlabeled.insert(0, "Rank", range(1, len(unlabeled) + 1))
        unlabeled = unlabeled.rename(columns={col: "Likeliness"})
        ranked_per_stress[stress] = unlabeled

        row = {"stress": stress, "known_excluded": len(known), "unlabeled_scored": len(unlabeled)}
        for thr in THRESHOLDS:
            row[f"predicted_SRGs_p>{thr}"] = int((unlabeled["Likeliness"] > thr).sum())
        count_rows.append(row)

        out_path = os.path.join(GW_DIR, f"{stress}_ranked_predictions.tsv")
        unlabeled.to_csv(out_path, sep="\t")
        print(f"  {stress:8s} {len(unlabeled)} unlabeled proteins ranked "
              f"({row[f'predicted_SRGs_p>0.5']} above p=0.5) -> {out_path}")

    top_ids = set()
    for stress in STRESSES:
        top_ids.update(ranked_per_stress[stress].head(TOP_N).index)
    print(f"\nFetching GO annotation for {len(top_ids)} top candidates...")
    go_terms = load_go_terms(top_ids)

    table4_frames = []
    for stress in STRESSES:
        top = ranked_per_stress[stress].head(TOP_N).copy()
        top["Description"] = [descriptions.get(i, "") for i in top.index]
        top["Arabidopsis_Hit"] = [blast_hits.get(i, ("", "", ""))[0] for i in top.index]
        top["Percent_Identity"] = [blast_hits.get(i, ("", "", ""))[1] for i in top.index]
        top["BLAST_evalue"] = [blast_hits.get(i, ("", "", ""))[2] for i in top.index]
        top["GO_Terms"] = [go_terms.get(i, "") for i in top.index]

        out_path = os.path.join(GW_DIR, f"{stress}_top10.tsv")
        top.to_csv(out_path, sep="\t")
        print(f"  {stress:8s} top-{TOP_N} -> {out_path}")

        compact = top.reset_index()[["Rank", "ProteinID", "Likeliness"]].copy()
        compact.columns = ["Rank", f"{stress.capitalize()}_Protein", f"{stress.capitalize()}_Likeliness"]
        table4_frames.append(compact.set_index("Rank"))

    table4 = pd.concat(table4_frames, axis=1)
    table4_path = os.path.join(GW_DIR, "table4_top10_candidates.tsv")
    table4.to_csv(table4_path, sep="\t")

    counts_path = os.path.join(GW_DIR, "srg_counts.tsv")
    counts_df = pd.DataFrame(count_rows)
    counts_df.to_csv(counts_path, sep="\t", index=False)

    print(f"\nTable 4 equivalent -> {table4_path}")
    print(table4.to_string())
    print(f"\nPredicted SRG counts -> {counts_path}")
    print(counts_df.to_string(index=False))
    print("\nNext steps: bash 12_meme_motif_analysis.sh  (motif discovery)")
    print("            python3 13_phase3_figures.py   (figures)")


if __name__ == "__main__":
    main()
