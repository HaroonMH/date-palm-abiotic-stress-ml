#!/usr/bin/env python3
"""
Step 9: genome-wide candidate lists from the per-stress SVM on ESM-2 features
(model "A" in the ablation), so its candidate biology can be compared directly
against the multi-task model's.

Model A won the ablation on auROC for all four stresses, and it already
benefits from three of the four dataset improvements: recovered multi-label
positives, confident background negatives, and ESM-2 features. Only the shared
backbone is absent.

Ranking uses the SVM decision margin, never predict_proba. Platt scaling
saturates near 1.0 and produced 60-protein tie blocks in the earlier pipeline,
which made top-N ordering arbitrary. Counts are reported at the natural
decision boundary (margin > 0), which needs no calibration assumption.

Output (outputs/tables/):
  svm_esm_genomewide_predictions.tsv    all proteins x 4 margins
  svm_{stress}_ranked_genes.tsv         isoform-collapsed ranking
  svm_{stress}_top10.tsv                top-10 distinct genes + annotation
  svm_coresponse_matrix.tsv             pairwise co-response counts
  candidate_biology_comparison.tsv      SVM vs multi-task top-10, side by side
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    EMB_DIR, SPLIT_DIR, LABELS_DIR, TABLES_DIR, MODELS_DIR, GENE_MAP,
    PROTEOME_FASTA, ANNOT_DIR, STRESSES, RANDOM_STATE,
)

TOP_N = 10
ISOFORM_RE = re.compile(r"\s+isoform\s+x?\d+\s*$", re.IGNORECASE)


def load_descriptions():
    d = {}
    with open(PROTEOME_FASTA) as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].strip().split(None, 1)
                d[parts[0]] = (parts[1] if len(parts) > 1 else "").replace(
                    "[Phoenix dactylifera]", "").strip()
    return d


def load_blast():
    p = os.path.join(ANNOT_DIR, "blast_best_hits.txt")
    hits = {}
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) >= 12:
                    hits.setdefault(c[0], (c[1], c[2], c[10]))
    return hits


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)

    ds = [f for f in os.listdir(EMB_DIR) if f.startswith("dataset_") and f.endswith(".npz")]
    pr = [f for f in os.listdir(EMB_DIR) if f.startswith("proteome_") and f.endswith(".npz")]
    if not ds or not pr:
        sys.exit("ERROR: need both dataset and proteome embeddings (script 23)")

    z = np.load(os.path.join(EMB_DIR, sorted(ds)[-1]), allow_pickle=True)
    id2vec = {i: v for i, v in zip(z["ids"].tolist(), z["emb"])}
    zp = np.load(os.path.join(EMB_DIR, sorted(pr)[-1]), allow_pickle=True)
    prot_ids, Xp = zp["ids"].tolist(), zp["emb"]

    split = pd.read_csv(os.path.join(SPLIT_DIR, "split_assignments.tsv"), sep="\t")
    split = split[split["ProteinID"].isin(id2vec)].reset_index(drop=True)
    train = split[split.split == "train"].reset_index(drop=True)

    Xtr = np.stack([id2vec[p] for p in train["ProteinID"]])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xp_s = (Xp - mu) / sd

    lab = pd.read_csv(os.path.join(LABELS_DIR, "multilabel_dataset.tsv"), sep="\t")
    known = {s: set(lab.loc[lab[s] == 1, "ProteinID"]) for s in STRESSES}

    pred = pd.DataFrame({"ProteinID": prot_ids})
    for s in STRESSES:
        y = train[s].to_numpy()
        print(f"Training SVM for {s} ({int(y.sum())} positives)...")
        clf = SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(Xtr, y)
        margins = []
        for b in range(0, len(Xp_s), 4096):
            margins.append(clf.decision_function(Xp_s[b:b + 4096]))
        pred[f"{s}_margin"] = np.concatenate(margins)
        print(f"  scored proteome; margin>0: {int((pred[f'{s}_margin'] > 0).sum())}")

    desc = load_descriptions()
    blast = load_blast()
    pred["Description"] = [desc.get(p, "") for p in pred["ProteinID"]]
    if os.path.exists(GENE_MAP):
        gm = pd.read_csv(GENE_MAP, sep="\t", dtype=str)
        g = dict(zip(gm["ProteinID"], gm["GeneID"]))
        pred["GeneID"] = [g.get(p, "") for p in pred["ProteinID"]]
    else:
        pred["GeneID"] = ""

    def key(r):
        if r["GeneID"]:
            return f"GeneID:{r['GeneID']}"
        d = ISOFORM_RE.sub("", str(r["Description"]).lower()).strip()
        return d if d else f"__{r['ProteinID']}"

    pred["GeneKey"] = pred.apply(key, axis=1)
    pred.to_csv(os.path.join(TABLES_DIR, "svm_esm_genomewide_predictions.tsv"),
                sep="\t", index=False)

    print("\nPredicted stress-responsive genes (margin > 0, isoform-collapsed):")
    comparison = []
    for s in STRESSES:
        sub = pred[~pred["ProteinID"].isin(known[s])].copy()
        sub = sub.sort_values(f"{s}_margin", ascending=False)
        col = sub.drop_duplicates("GeneKey", keep="first").copy()
        col["Rank"] = range(1, len(col) + 1)
        col.to_csv(os.path.join(TABLES_DIR, f"svm_{s}_ranked_genes.tsv"),
                   sep="\t", index=False)

        top = col.head(TOP_N).copy()
        top["Arabidopsis_Hit"] = [blast.get(i, ("", "", ""))[0] for i in top["ProteinID"]]
        top["Percent_Identity"] = [blast.get(i, ("", "", ""))[1] for i in top["ProteinID"]]
        top.to_csv(os.path.join(TABLES_DIR, f"svm_{s}_top10.tsv"), sep="\t", index=False)

        n = int((col[f"{s}_margin"] > 0).sum())
        print(f"  {s:8s} {n:6d} / {len(col):6d} genes ({100*n/len(col):.1f}%)")

        mt_path = os.path.join(TABLES_DIR, f"{s}_top10_multitask.tsv")
        mt_desc = []
        if os.path.exists(mt_path):
            mt_desc = pd.read_csv(mt_path, sep="\t")["Description"].astype(str).tolist()
        for i in range(TOP_N):
            comparison.append({
                "stress": s, "rank": i + 1,
                "SVM_candidate": str(top["Description"].iloc[i])[:70] if i < len(top) else "",
                "MultiTask_candidate": mt_desc[i][:70] if i < len(mt_desc) else "",
            })

    calls = np.column_stack([pred[f"{s}_margin"] > 0 for s in STRESSES])
    uniq = pred.drop_duplicates("GeneKey", keep="first")
    ucalls = np.column_stack([uniq[f"{s}_margin"] > 0 for s in STRESSES])
    mat = pd.DataFrame(0, index=STRESSES, columns=STRESSES)
    for i, a in enumerate(STRESSES):
        for j, b in enumerate(STRESSES):
            mat.loc[a, b] = int((ucalls[:, i] & ucalls[:, j]).sum())
    mat.to_csv(os.path.join(TABLES_DIR, "svm_coresponse_matrix.tsv"), sep="\t")

    cmp_df = pd.DataFrame(comparison)
    cmp_df.to_csv(os.path.join(TABLES_DIR, "candidate_biology_comparison.tsv"),
                  sep="\t", index=False)

    print(f"\nGenes predicted for >=2 stresses (margin>0): "
          f"{int((ucalls.sum(1) >= 2).sum())}")
    print("\n=== Top-10 candidates: SVM vs multi-task ===")
    for s in STRESSES:
        print(f"\n--- {s} ---")
        sub = cmp_df[cmp_df.stress == s]
        for _, r in sub.iterrows():
            print(f"  {r['rank']:2d}  SVM: {r['SVM_candidate'][:52]:54s} "
                  f"MT: {r['MultiTask_candidate'][:44]}")
    print(f"\nWrote {os.path.join(TABLES_DIR, 'candidate_biology_comparison.tsv')}")


if __name__ == "__main__":
    main()
