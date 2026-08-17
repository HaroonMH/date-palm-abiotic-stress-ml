#!/usr/bin/env python3
"""
Phase 2, step 2: initial screening of all six feature construction methods
across all six classifiers via 5-fold CV on the TRAINING set only (mirrors
the reference paper's Figure 5 heatmap / Section 3.2). No feature selection
yet -- this just identifies which encodings are worth carrying forward into
SVM-RFE (the paper found CKSAAP and DDE consistently best, and dropped
PAAC/APAAC/DPC/TPC at this stage).

Output:
  data/screening_results.tsv     long format: stress, feature_method, classifier,
                                  cv_auROC_mean, cv_auROC_std, cv_auPRC_mean
  data/top2_feature_methods.tsv  the two best-performing methods overall
                                  (by mean auROC across stresses & classifiers),
                                  used as input to the SVM-RFE step
"""
import os
import sys
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from ml_utils import STRESSES, FEATURE_METHODS, DATA_DIR, get_classifiers, get_xy  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for stress in STRESSES:
        for method in FEATURE_METHODS:
            X, y = get_xy(stress, method, "train")
            for clf_name, clf in get_classifiers().items():
                pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
                scores = cross_validate(
                    pipe, X, y, cv=cv,
                    scoring=["roc_auc", "average_precision"],
                    n_jobs=-1,
                )
                row = {
                    "stress": stress,
                    "feature_method": method,
                    "classifier": clf_name,
                    "cv_auROC_mean": np.mean(scores["test_roc_auc"]),
                    "cv_auROC_std": np.std(scores["test_roc_auc"]),
                    "cv_auPRC_mean": np.mean(scores["test_average_precision"]),
                }
                rows.append(row)
                print(f"{stress:8s} {method:8s} {clf_name:5s} "
                      f"auROC={row['cv_auROC_mean']:.4f} auPRC={row['cv_auPRC_mean']:.4f}")

    import csv
    out_path = os.path.join(DATA_DIR, "screening_results.tsv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")

    # Rank feature methods by mean auROC averaged across stresses & classifiers
    method_scores = {m: [] for m in FEATURE_METHODS}
    for r in rows:
        method_scores[r["feature_method"]].append(r["cv_auROC_mean"])
    method_avg = {m: np.mean(v) for m, v in method_scores.items()}
    ranked = sorted(method_avg.items(), key=lambda kv: -kv[1])

    print("\nFeature method ranking (mean auROC across all stresses & classifiers):")
    for m, score in ranked:
        print(f"  {m:8s} {score:.4f}")

    top2 = [m for m, _ in ranked[:2]]
    top2_path = os.path.join(DATA_DIR, "top2_feature_methods.tsv")
    with open(top2_path, "w") as f:
        f.write("rank\tfeature_method\tmean_auROC\n")
        for i, (m, score) in enumerate(ranked[:2], start=1):
            f.write(f"{i}\t{m}\t{score:.4f}\n")
    print(f"\nTop 2 methods selected for SVM-RFE: {top2}")
    print(f"Wrote {top2_path}")
    print("\nNext step: python3 06_svm_rfe_selection.py")


if __name__ == "__main__":
    main()
