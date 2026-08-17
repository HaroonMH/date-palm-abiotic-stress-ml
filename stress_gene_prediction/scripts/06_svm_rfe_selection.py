#!/usr/bin/env python3
"""
Phase 2, step 3: SVM-RFE feature selection (mirrors reference paper Section
3.2 / Figure 6). Reads the top-2 feature methods from 05_screen_feature_methods.py
and, for each stress, runs recursive feature elimination with cross-validation
(RFECV, linear-kernel SVM, 5-fold, auROC scoring) on THREE feature sets:
  - method 1 alone
  - method 2 alone
  - method 1 + method 2 concatenated

Features are standardized once before RFECV (StandardScaler fit on the full
training set) -- this is the common, lightweight approach for feature
selection; the actual model evaluation later still uses proper CV.

Output per (stress, feature_set):
  data/rfe/{stress}_{feature_set}_selected_features.txt   selected column names
  data/rfe/{stress}_{feature_set}_rfe_curve.tsv           approx. n_features vs mean CV auROC
  data/rfe_summary.tsv  (stress, feature_set, n_total, n_selected, best_cv_auROC)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from ml_utils import STRESSES, DATA_DIR, get_xy, get_xy_combined  # noqa: E402

warnings.filterwarnings("ignore")

RFE_DIR = os.path.join(DATA_DIR, "rfe")


def run_rfe(X, y, min_features_to_select=1):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_total = X.shape[1]
    step = max(1, n_total // 150)  # ~150 points on the curve regardless of dimensionality

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    estimator = SVC(kernel="linear", random_state=42)
    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring="roc_auc",
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )
    selector.fit(X_scaled, y)

    scores = selector.cv_results_["mean_test_score"]
    feature_counts = [
        min(n_total, min_features_to_select + i * step) for i in range(len(scores))
    ]
    selected_columns = X.columns[selector.support_].tolist()
    return selected_columns, selector.n_features_, feature_counts, scores


def main():
    os.makedirs(RFE_DIR, exist_ok=True)

    top2_path = os.path.join(DATA_DIR, "top2_feature_methods.tsv")
    if not os.path.exists(top2_path):
        sys.exit(f"ERROR: {top2_path} not found -- run 05_screen_feature_methods.py first")
    top2_df = pd.read_csv(top2_path, sep="\t")
    method1, method2 = top2_df["feature_method"].tolist()
    print(f"Using top-2 methods from screening: {method1}, {method2}\n")

    summary_rows = []

    for stress in STRESSES:
        feature_sets = {
            method1: lambda s=stress: get_xy(s, method1, "train"),
            method2: lambda s=stress: get_xy(s, method2, "train"),
            f"{method1}+{method2}": lambda s=stress: get_xy_combined(s, [method1, method2], "train"),
        }
        for fs_name, loader in feature_sets.items():
            X, y = loader()
            print(f"=== {stress} / {fs_name} ({X.shape[1]} features, {X.shape[0]} samples) ===")
            selected_cols, n_selected, feature_counts, scores = run_rfe(X, y)
            best_auroc = max(scores)

            safe_name = fs_name.replace("+", "_plus_")
            feats_path = os.path.join(RFE_DIR, f"{stress}_{safe_name}_selected_features.txt")
            with open(feats_path, "w") as f:
                f.write("\n".join(selected_cols) + "\n")

            curve_path = os.path.join(RFE_DIR, f"{stress}_{safe_name}_rfe_curve.tsv")
            pd.DataFrame({"n_features": feature_counts, "mean_cv_auROC": scores}).to_csv(
                curve_path, sep="\t", index=False
            )

            print(f"  optimal features: {n_selected} / {X.shape[1]}  best CV auROC: {best_auroc:.4f}")
            print(f"  -> {feats_path}\n  -> {curve_path}\n")

            summary_rows.append({
                "stress": stress,
                "feature_set": fs_name,
                "n_total": X.shape[1],
                "n_selected": n_selected,
                "best_cv_auROC": round(best_auroc, 4),
            })

    summary_path = os.path.join(DATA_DIR, "rfe_summary.tsv")
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    print(f"RFE summary written to: {summary_path}")
    print("\nNext step: python3 07_train_evaluate_classifiers.py")


if __name__ == "__main__":
    main()
