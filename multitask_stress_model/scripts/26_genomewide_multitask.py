#!/usr/bin/env python3
"""
Step 7: genome-wide prediction with the multi-task model.

Beyond ranking each stress separately, this produces the output the reference
paper's architecture structurally cannot generate: per-gene CO-RESPONSE
predictions. Because the four heads are independent sigmoids, a gene can be
predicted responsive to drought AND salt simultaneously, and the joint pattern
is itself a prediction.

Requires proteome embeddings:
  python3 23_extract_esm_embeddings.py --proteome

Output (outputs/tables/):
  genomewide_multitask_predictions.tsv  every protein x 4 probabilities
  {stress}_ranked_genes_multitask.tsv   isoform-collapsed ranking per stress
  {stress}_top10_multitask.tsv          top-10 distinct genes
  coresponse_predictions.tsv            genes predicted for >=2 stresses
  coresponse_matrix.tsv                 pairwise co-response counts
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    EMB_DIR, MODELS_DIR, TABLES_DIR, LABELS_DIR, GENE_MAP, PROTEOME_FASTA,
    STRESSES, parse_fasta,
)
from importlib import import_module

MultiTaskNet = import_module("24_train_multitask").MultiTaskNet

THRESHOLD = 0.5
TOP_N = 10


def load_descriptions():
    d = {}
    with open(PROTEOME_FASTA) as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split(None, 1)
                d[parts[0]] = (parts[1] if len(parts) > 1 else "").replace(
                    "[Phoenix dactylifera]", "").strip()
    return d


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)

    ckpt_path = os.path.join(MODELS_DIR, "esm_multitask.pt")
    if not os.path.exists(ckpt_path):
        sys.exit("ERROR: model not found -- run 24_train_multitask.py first")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cands = [f for f in os.listdir(EMB_DIR)
             if f.startswith("proteome_") and f.endswith(".npz")]
    if not cands:
        sys.exit("ERROR: proteome embeddings missing -- run:\n"
                 "  python3 23_extract_esm_embeddings.py --proteome")
    z = np.load(os.path.join(EMB_DIR, sorted(cands)[-1]), allow_pickle=True)
    ids, X = z["ids"].tolist(), z["emb"]
    print(f"Scoring {len(ids)} proteins")

    X = (X - ckpt["mu"]) / ckpt["sd"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskNet(ckpt["in_dim"], ckpt["hidden"], ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    probs = []
    with torch.no_grad():
        for b in range(0, len(X), 4096):
            xb = torch.tensor(X[b:b + 4096], dtype=torch.float32, device=device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    probs = np.concatenate(probs)

    pred = pd.DataFrame(probs, columns=[f"{s}_prob" for s in STRESSES])
    pred.insert(0, "ProteinID", ids)

    # annotate: gene id, description, and which proteins were training labels
    desc = load_descriptions()
    pred["Description"] = [desc.get(p, "") for p in pred["ProteinID"]]
    if os.path.exists(GENE_MAP):
        gm = pd.read_csv(GENE_MAP, sep="\t", dtype=str)
        g = dict(zip(gm["ProteinID"], gm["GeneID"]))
        pred["GeneID"] = [g.get(p, "") for p in pred["ProteinID"]]
        print(f"GeneID map applied ({len(set(g.values()))} distinct genes)")
    else:
        pred["GeneID"] = ""
        print("NOTE: GeneID map absent -- isoform collapsing falls back to description")

    labels_path = os.path.join(LABELS_DIR, "multilabel_dataset.tsv")
    known = {}
    if os.path.exists(labels_path):
        lab = pd.read_csv(labels_path, sep="\t")
        for s in STRESSES:
            known[s] = set(lab.loc[lab[s] == 1, "ProteinID"])
    else:
        known = {s: set() for s in STRESSES}

    pred.to_csv(os.path.join(TABLES_DIR, "genomewide_multitask_predictions.tsv"),
                sep="\t", index=False)

    def gene_key(row):
        if row["GeneID"]:
            return f"GeneID:{row['GeneID']}"
        d = str(row["Description"]).lower()
        import re
        d = re.sub(r"\s+isoform\s+x?\d+\s*$", "", d).strip()
        return d if d else f"__{row['ProteinID']}"

    pred["GeneKey"] = pred.apply(gene_key, axis=1)

    print("\nPredicted stress-responsive genes (p > %.2f):" % THRESHOLD)
    for s in STRESSES:
        sub = pred[~pred["ProteinID"].isin(known[s])].copy()
        sub = sub.sort_values(f"{s}_prob", ascending=False)
        collapsed = sub.drop_duplicates("GeneKey", keep="first").copy()
        collapsed["Rank"] = range(1, len(collapsed) + 1)
        collapsed.to_csv(
            os.path.join(TABLES_DIR, f"{s}_ranked_genes_multitask.tsv"),
            sep="\t", index=False)
        collapsed.head(TOP_N).to_csv(
            os.path.join(TABLES_DIR, f"{s}_top10_multitask.tsv"), sep="\t", index=False)
        n = int((collapsed[f"{s}_prob"] > THRESHOLD).sum())
        print(f"  {s:8s} {n:6d} / {len(collapsed):6d} genes "
              f"({100*n/len(collapsed):.1f}%)")

    # ---- co-response: the output the single-task design cannot produce ----
    calls = (pred[[f"{s}_prob" for s in STRESSES]].to_numpy() > THRESHOLD)
    pred["n_stresses"] = calls.sum(1)
    pred["stress_pattern"] = ["+".join([s for s, c in zip(STRESSES, row) if c]) or "none"
                              for row in calls]

    multi = pred[pred["n_stresses"] >= 2].copy()
    multi = multi.drop_duplicates("GeneKey", keep="first")
    multi = multi.sort_values("n_stresses", ascending=False)
    multi.to_csv(os.path.join(TABLES_DIR, "coresponse_predictions.tsv"),
                 sep="\t", index=False)

    mat = pd.DataFrame(0, index=STRESSES, columns=STRESSES)
    uniq = pred.drop_duplicates("GeneKey", keep="first")
    ucalls = (uniq[[f"{s}_prob" for s in STRESSES]].to_numpy() > THRESHOLD)
    for i, a in enumerate(STRESSES):
        for j, b in enumerate(STRESSES):
            mat.loc[a, b] = int((ucalls[:, i] & ucalls[:, j]).sum())
    mat.to_csv(os.path.join(TABLES_DIR, "coresponse_matrix.tsv"), sep="\t")

    print(f"\nGenes predicted responsive to >=2 stresses: {len(multi)}")
    print("\nMost frequent predicted co-response patterns:")
    print(multi["stress_pattern"].value_counts().head(8).to_string())
    print("\nPairwise co-response counts (distinct genes):")
    print(mat.to_string())
    print("\nNext: python3 27_figures_multitask.py")


if __name__ == "__main__":
    main()
