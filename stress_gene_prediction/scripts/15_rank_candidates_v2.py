#!/usr/bin/env python3
"""
Milestone 3, revised step 2: rank candidates by decision margin and collapse
splice isoforms so the candidate table holds distinct GENES.

Two fixes over 11_rank_candidates.py
------------------------------------
1. Ranking uses the SVM decision margin, not `predict_proba`. The probability
   saturates at ~1.0 and produced large tie groups (60 proteins tied at the
   top for cold, and salt's entire top-10 was one tie block), so the previous
   ordering inside those blocks was arbitrary. The margin is monotonically
   related to the probability, so the ordering is the same one the model
   intended -- just at full resolution.

2. Splice isoforms are collapsed to one representative per gene. Previously
   heat's "top 10" was only 4 distinct genes: seven entries were isoforms
   X1-X10 of a single ubiquitin-conjugating enzyme. CD-HIT deduplicated the
   training set but never the prediction set.

   Grouping uses NCBI GeneID from data/genomewide_v2/datepalm_gene_map.tsv
   (built by 14b_build_gene_map.sh) when that file exists -- that is the
   authoritative answer.

   If the map is absent it falls back to grouping by product description with
   the trailing "isoform X<n>" stripped, which is how RefSeq names isoforms.
   That fallback is fine for picking a top-N but over-merges paralogs
   genome-wide: 105 distinct proteins share the description "putative disease
   resistance protein RGA3" and are separate genes, not isoforms. Gene-level
   counts from the fallback are therefore an undercount, and the script says
   so in its output.

Reads data/genomewide_v2/all_predictions.tsv (from 14_genomewide_rescore.py)
and writes into the same folder, leaving data/genomewide/ untouched.

Output (under data/genomewide_v2/):
  {stress}_ranked_proteins.tsv        every unlabeled protein, ranked by margin
  {stress}_ranked_genes.tsv           isoform-collapsed, ranked by margin
  {stress}_top10.tsv                  top-10 distinct genes + annotation
  {stress}_top10_isoform_members.tsv  which proteins each top-10 entry absorbed
  table4_top10_candidates.tsv         Table 4 equivalent
  srg_counts.tsv                      protein-level and gene-level counts
"""
import os
import re
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

GW_DIR = os.path.join(DATA_DIR, "genomewide_v2")
RAW_DIR = os.path.join(DATA_DIR, "raw")

PROB_THRESHOLDS = [0.5, 0.7, 0.9]
TOP_N = 10

ISOFORM_RE = re.compile(r"\s+isoform\s+X?\d+\s*$", re.IGNORECASE)
LOWQUAL_RE = re.compile(r"^LOW QUALITY PROTEIN:\s*", re.IGNORECASE)


def gene_key(protein_id, description):
    """Fallback group key: description with the isoform suffix stripped."""
    desc = description or ""
    desc = LOWQUAL_RE.sub("", desc)
    desc = ISOFORM_RE.sub("", desc)
    desc = re.sub(r"\s+", " ", desc).strip().lower()
    return desc if desc else f"__nodesc__{protein_id}"


def load_gene_map():
    """ProteinID -> GeneID from 14b_build_gene_map.sh, or {} if not built."""
    path = os.path.join(GW_DIR, "datepalm_gene_map.tsv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str)
    return dict(zip(df["ProteinID"], df["GeneID"]))


def build_keys(protein_ids, descriptions, gene_map):
    """GeneID where known, description-based key otherwise."""
    keys, n_by_geneid = [], 0
    for pid in protein_ids:
        gid = gene_map.get(pid)
        if gid:
            keys.append(f"GeneID:{gid}")
            n_by_geneid += 1
        else:
            keys.append(gene_key(pid, descriptions.get(pid, "")))
    return keys, n_by_geneid


def load_descriptions():
    descriptions = {}
    with open(PROTEOME_FASTA) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            parts = line[1:].strip().split(None, 1)
            desc = parts[1] if len(parts) > 1 else ""
            descriptions[parts[0]] = desc.replace("[Phoenix dactylifera]", "").strip()
    return descriptions


def load_blast_hits():
    if not os.path.exists(BLAST_BEST_HITS):
        print(f"  WARNING: {BLAST_BEST_HITS} not found -- skipping homolog columns")
        return {}
    hits = {}
    with open(BLAST_BEST_HITS) as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 12:
                hits.setdefault(cols[0], (cols[1], cols[2], cols[10]))
    return hits


def load_go_terms(wanted_ids):
    if not os.path.exists(GO_ANNOTATIONS):
        print(f"  WARNING: {GO_ANNOTATIONS} not found -- skipping GO column")
        return {}
    go_map = {}
    for chunk in pd.read_csv(GO_ANNOTATIONS, sep="\t", chunksize=100_000):
        sub = chunk[chunk["DatePalm_Protein"].isin(wanted_ids)]
        for protein, group in sub.groupby("DatePalm_Protein"):
            go_map.setdefault(protein, []).extend(group["GO_Name"].dropna().tolist())
    return {p: "; ".join(sorted(set(v))) for p, v in go_map.items()}


def main():
    predictions_path = os.path.join(GW_DIR, "all_predictions.tsv")
    if not os.path.exists(predictions_path):
        sys.exit(f"ERROR: {predictions_path} not found -- run 14_genomewide_rescore.py first")

    preds = pd.read_csv(predictions_path, sep="\t", index_col=0)
    print(f"Loaded predictions for {len(preds)} proteins\n")

    print("Loading existing annotation (descriptions, BLAST homologs)...")
    descriptions = load_descriptions()
    blast_hits = load_blast_hits()

    gene_map = load_gene_map()
    if gene_map:
        print(f"Using authoritative GeneID map for {len(gene_map)} proteins "
              f"({len(set(gene_map.values()))} distinct genes)\n")
    else:
        print("NOTE: datepalm_gene_map.tsv not found -- falling back to")
        print("      description-based isoform grouping. The top-N tables are")
        print("      still valid, but genome-wide gene counts will UNDERCOUNT")
        print("      because paralogs sharing a description get merged.")
        print("      Run 14b_build_gene_map.sh for exact counts.\n")

    ranked_genes = {}
    members_by_stress = {}
    count_rows = []

    for stress in STRESSES:
        raw_fasta = os.path.join(RAW_DIR, f"{stress}_raw.fasta")
        known = set(parse_fasta(raw_fasta).keys()) if os.path.exists(raw_fasta) else set()

        df = preds[~preds.index.isin(known)][[f"{stress}_prob", f"{stress}_margin"]].copy()
        df.columns = ["Likeliness", "Margin"]
        df["Description"] = [descriptions.get(i, "") for i in df.index]
        keys, n_by_geneid = build_keys(list(df.index), descriptions, gene_map)
        df["GeneKey"] = keys
        df["GeneID"] = [gene_map.get(i, "") for i in df.index]

        df = df.sort_values("Margin", ascending=False)
        df.insert(0, "Rank", range(1, len(df) + 1))
        df.to_csv(os.path.join(GW_DIR, f"{stress}_ranked_proteins.tsv"), sep="\t")

        # Collapse: the highest-margin protein represents its gene.
        collapsed = df.drop_duplicates(subset="GeneKey", keep="first").copy()
        group_sizes = df.groupby("GeneKey").size()
        collapsed["N_Isoforms"] = collapsed["GeneKey"].map(group_sizes).astype(int)
        collapsed["Rank"] = range(1, len(collapsed) + 1)
        collapsed.to_csv(os.path.join(GW_DIR, f"{stress}_ranked_genes.tsv"), sep="\t")
        ranked_genes[stress] = collapsed
        members_by_stress[stress] = df

        row = {
            "stress": stress,
            "known_excluded": len(known),
            "unlabeled_proteins": len(df),
            "distinct_genes": len(collapsed),
            "grouping": "GeneID" if gene_map else "description(approx)",
        }
        for thr in PROB_THRESHOLDS:
            row[f"proteins_p>{thr}"] = int((df["Likeliness"] > thr).sum())
            row[f"genes_p>{thr}"] = int((collapsed["Likeliness"] > thr).sum())
        count_rows.append(row)

        n_tied_prob = int((df["Likeliness"] >= df["Likeliness"].max() - 1e-12).sum())
        n_tied_margin = int((df["Margin"] >= df["Margin"].max() - 1e-12).sum())
        print(f"  {stress:8s} {len(df):6d} proteins -> {len(collapsed):6d} distinct genes "
              f"| top-score ties: prob={n_tied_prob}, margin={n_tied_margin}")

    top_ids = set()
    for stress in STRESSES:
        top_ids.update(ranked_genes[stress].head(TOP_N).index)
    print(f"\nFetching GO annotation for {len(top_ids)} top candidates...")
    go_terms = load_go_terms(top_ids)

    table4_frames = []
    for stress in STRESSES:
        top = ranked_genes[stress].head(TOP_N).copy()
        top["Arabidopsis_Hit"] = [blast_hits.get(i, ("", "", ""))[0] for i in top.index]
        top["Percent_Identity"] = [blast_hits.get(i, ("", "", ""))[1] for i in top.index]
        top["BLAST_evalue"] = [blast_hits.get(i, ("", "", ""))[2] for i in top.index]
        top["GO_Terms"] = [go_terms.get(i, "") for i in top.index]
        top.drop(columns=["GeneKey"]).to_csv(
            os.path.join(GW_DIR, f"{stress}_top10.tsv"), sep="\t"
        )

        full = members_by_stress[stress]
        member_rows = []
        for rep_id in ranked_genes[stress].head(TOP_N).index:
            key = ranked_genes[stress].loc[rep_id, "GeneKey"]
            for member_id, m in full[full["GeneKey"] == key].iterrows():
                member_rows.append({
                    "Representative": rep_id,
                    "MemberProtein": member_id,
                    "Likeliness": m["Likeliness"],
                    "Margin": m["Margin"],
                    "Description": m["Description"],
                })
        pd.DataFrame(member_rows).to_csv(
            os.path.join(GW_DIR, f"{stress}_top10_isoform_members.tsv"), sep="\t", index=False
        )

        compact = top.reset_index()[["Rank", "ProteinID", "Likeliness", "Margin"]].copy()
        compact.columns = ["Rank", f"{stress.capitalize()}_Protein",
                           f"{stress.capitalize()}_Likeliness", f"{stress.capitalize()}_Margin"]
        table4_frames.append(compact.set_index("Rank"))

        n_groups = top["GeneKey"].nunique()
        print(f"  {stress:8s} top-{TOP_N}: {n_groups} distinct genes "
              f"(must equal {TOP_N})")

    table4 = pd.concat(table4_frames, axis=1)
    table4.to_csv(os.path.join(GW_DIR, "table4_top10_candidates.tsv"), sep="\t")

    counts_df = pd.DataFrame(count_rows)
    counts_df.to_csv(os.path.join(GW_DIR, "srg_counts.tsv"), sep="\t", index=False)

    print("\n=== Table 4 (isoform-collapsed, margin-ranked) ===")
    print(table4.round(4).to_string())
    print("\n=== Predicted SRG counts ===")
    print(counts_df.to_string(index=False))
    print("\nNext step: python3 16_phase3_figures_v2.py")


if __name__ == "__main__":
    main()
