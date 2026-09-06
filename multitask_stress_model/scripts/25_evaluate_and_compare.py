#!/usr/bin/env python3
"""
Step 6: ablation study isolating what actually causes any improvement.

A direct comparison against the original single-task SVM pipeline would be
confounded: that pipeline used a different split, different negatives ("genes
responsive to the other three stresses") and different features, so any
difference could come from the data redesign rather than the architecture.

This script therefore trains every baseline on the IDENTICAL dataset, split
and ESM-2 features as the multi-task model, so the only thing that varies is
the architecture:

  A. per-stress SVM              -- classical, four independent models
  B. per-stress independent MLP  -- same capacity as the multi-task net, but
                                    four separate backbones (no sharing)
  C. multi-task shared backbone  -- trained by 24_train_multitask.py

A vs C measures the total gain. B vs C isolates the contribution of the SHARED
backbone specifically, since B and C differ only in whether the representation
is shared across stresses.

The original pipeline's numbers are reported alongside for context, clearly
marked as not directly comparable.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    EMB_DIR, SPLIT_DIR, MODELS_DIR, TABLES_DIR, PROJECT_ROOT, STRESSES, RANDOM_STATE,
)
from importlib import import_module

mt = import_module("24_train_multitask")
MultiTaskNet = mt.MultiTaskNet


def load_data():
    cands = [f for f in os.listdir(EMB_DIR) if f.startswith("dataset_") and f.endswith(".npz")]
    if not cands:
        sys.exit("ERROR: no dataset embeddings found -- run 23_extract_esm_embeddings.py")
    z = np.load(os.path.join(EMB_DIR, sorted(cands)[-1]), allow_pickle=True)
    id_to_vec = {i: v for i, v in zip(z["ids"].tolist(), z["emb"])}
    split = pd.read_csv(os.path.join(SPLIT_DIR, "split_assignments.tsv"), sep="\t")
    split = split[split["ProteinID"].isin(id_to_vec)].reset_index(drop=True)
    tr = split[split.split == "train"].reset_index(drop=True)
    te = split[split.split == "test"].reset_index(drop=True)
    Xtr = np.stack([id_to_vec[p] for p in tr["ProteinID"]])
    Xte = np.stack([id_to_vec[p] for p in te["ProteinID"]])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return ((Xtr - mu) / sd, tr[STRESSES].to_numpy().astype(float),
            (Xte - mu) / sd, te[STRESSES].to_numpy().astype(float))


def score(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)


def run_svm(Xtr, Ytr, Xte, Yte):
    rows = []
    for i, s in enumerate(STRESSES):
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced",
                  random_state=RANDOM_STATE)
        clf.fit(Xtr, Ytr[:, i])
        p = clf.predict_proba(Xte)[:, 1]
        a, pr = score(Yte[:, i], p)
        prev = float(Yte[:, i].mean())
        rows.append({"model": "A: per-stress SVM", "stress": s,
                     "auROC": round(a * 100, 2), "auPRC": round(pr * 100, 2),
                     "positive_rate": round(prev * 100, 2),
                     "auPRC_lift_over_random": round(pr / prev, 2) if prev else float("nan")})
        print(f"  SVM  {s:8s} auROC {a*100:6.2f}  auPRC {pr*100:6.2f}  "
              f"lift {pr/prev if prev else float('nan'):.1f}x")
    return rows


def run_single_mlps(Xtr, Ytr, Xte, Yte, epochs=200, patience=25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for i, s in enumerate(STRESSES):
        torch.manual_seed(RANDOM_STATE)
        net = MultiTaskNet(Xtr.shape[1], (512, 256), 0.3, n_tasks=1).to(device)
        pos = Ytr[:, i].sum()
        pw = torch.tensor([(len(Ytr) - pos) / max(pos, 1)], dtype=torch.float32, device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
        Yt = torch.tensor(Ytr[:, i:i + 1], dtype=torch.float32, device=device)

        n_val = int(0.15 * len(Xt))
        g = torch.Generator(device="cpu").manual_seed(RANDOM_STATE)
        perm = torch.randperm(len(Xt), generator=g).to(device)
        vi, ti = perm[:n_val], perm[n_val:]
        best, bad, best_state = -np.inf, 0, None
        for ep in range(epochs):
            net.train()
            p2 = ti[torch.randperm(len(ti), device=device)]
            for b in range(0, len(p2), 256):
                sel = p2[b:b + 256]
                opt.zero_grad()
                crit(net(Xt[sel]), Yt[sel]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vp = torch.sigmoid(net(Xt[vi])).cpu().numpy().ravel()
            _, vpr = score(Yt[vi].cpu().numpy().ravel(), vp)
            if not np.isnan(vpr) and vpr > best:
                best, bad = vpr, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                break
        if best_state:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32,
                                               device=device))).cpu().numpy().ravel()
        a, pr = score(Yte[:, i], p)
        prev = float(Yte[:, i].mean())
        rows.append({"model": "B: independent MLP", "stress": s,
                     "auROC": round(a * 100, 2), "auPRC": round(pr * 100, 2),
                     "positive_rate": round(prev * 100, 2),
                     "auPRC_lift_over_random": round(pr / prev, 2) if prev else float("nan")})
        print(f"  MLP  {s:8s} auROC {a*100:6.2f}  auPRC {pr*100:6.2f}  "
              f"lift {pr/prev if prev else float('nan'):.1f}x")
    return rows


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    Xtr, Ytr, Xte, Yte = load_data()
    print(f"train {len(Xtr)}  test {len(Xte)}  dim {Xtr.shape[1]}\n")

    print("A: per-stress SVM on ESM-2 features")
    rows = run_svm(Xtr, Ytr, Xte, Yte)

    print("\nB: per-stress independent MLPs (no shared backbone)")
    rows += run_single_mlps(Xtr, Ytr, Xte, Yte)

    print("\nC: multi-task shared backbone")
    mt_path = os.path.join(TABLES_DIR, "esm_multitask_test_metrics.tsv")
    if os.path.exists(mt_path):
        for r in pd.read_csv(mt_path, sep="\t").to_dict("records"):
            rows.append({"model": "C: multi-task (shared)", "stress": r["stress"],
                         "auROC": r["auROC"], "auPRC": r["auPRC"],
                         "positive_rate": r.get("positive_rate", float("nan")),
                         "auPRC_lift_over_random": r.get("auPRC_lift_over_random",
                                                         float("nan"))})
            print(f"  MT   {r['stress']:8s} auROC {r['auROC']:6.2f}  "
                  f"auPRC {r['auPRC']:6.2f}  lift {r.get('auPRC_lift_over_random', float('nan'))}x")
    else:
        print("  (not found -- run 24_train_multitask.py first)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "ablation_comparison.tsv"), sep="\t", index=False)

    print("\n=== auROC by model (identical data, split and features) ===")
    print(df.pivot(index="stress", columns="model", values="auROC")
            .reindex(STRESSES).to_string())
    print("\n=== auPRC ===")
    print(df.pivot(index="stress", columns="model", values="auPRC")
            .reindex(STRESSES).to_string())

    old = os.path.join(PROJECT_ROOT, "stress_gene_prediction", "data",
                       "final_model_metrics.tsv")
    if os.path.exists(old):
        o = pd.read_csv(old, sep="\t")
        print("\n--- original single-task pipeline, for context only ---")
        print("NOT directly comparable: different split, different negative class")
        print("(other-stress genes rather than a genome background), k-mer features.")
        print("Its test sets were ~26% positive; these are ~2%. auPRC scales with")
        print("the positive rate, so compare auPRC_lift_over_random, never raw auPRC.")
        print(o[["stress", "classifier", "test_auROC", "test_auPRC"]].to_string(index=False))

    print(f"\nWrote {os.path.join(TABLES_DIR, 'ablation_comparison.tsv')}")
    print("Next: python3 26_genomewide_multitask.py")


if __name__ == "__main__":
    main()
