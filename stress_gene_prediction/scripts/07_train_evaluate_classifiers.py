#!/usr/bin/env python3
"""
Phase 2, step 4: train & evaluate the six classifiers (SVM, XGB, RF, GBDT,
ADB, BAG) on each stress's three RFE-selected feature sets (mirrors reference
paper Section 3.3 / Figure 7 / Table 2).

For each (stress, feature_set, classifier):
  - 5-fold CV on the training set (accuracy, auROC, auPRC)
  - fit on full training set, evaluate once on the held-out test set
    (accuracy, auROC, auPRC) -- this is the number reported in Table 2

Output:
  data/table2_results.tsv       stress, feature_set, classifier, cv_auROC,
                                 test_accuracy, test_auROC, test_auPRC
  data/table3_best_models.tsv   best (feature_set, classifier) per stress
                                 by test auROC -- mirrors reference Table 3
  data/roc_data/{stress}_{feature_set}_{classifier}.tsv
                                 per-test-sample ProteinID, y_true, y_proba --
                                 kept so 09_generate_figures.py can redraw
                                 ROC curves (paper Figure 7/8) without retraining
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from ml_utils import (  # noqa: E402
    STRESSES, DATA_DIR, get_classifiers, get_xy, get_xy_combined,
)

warnings.filterwarnings("ignore")

RFE_DIR = os.path.join(DATA_DIR, "rfe")
ROC_DATA_DIR = os.path.join(DATA_DIR, "roc_data")


def load_selected(stress, safe_name):
    path = os.path.join(RFE_DIR, f"{stress}_{safe_name}_selected_features.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    os.makedirs(ROC_DATA_DIR, exist_ok=True)

    top2_path = os.path.join(DATA_DIR, "top2_feature_methods.tsv")
    top2_df = pd.read_csv(top2_path, sep="\t")
    method1, method2 = top2_df["feature_method"].tolist()
    combined_name = f"{method1}+{method2}"

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for stress in STRESSES:
        feature_sets = {
            method1: (lambda s=stress: get_xy(s, method1, "train"),
                      lambda s=stress: get_xy(s, method1, "test")),
            method2: (lambda s=stress: get_xy(s, method2, "train"),
                      lambda s=stress: get_xy(s, method2, "test")),
            combined_name: (lambda s=stress: get_xy_combined(s, [method1, method2], "train"),
                            lambda s=stress: get_xy_combined(s, [method1, method2], "test")),
        }

        for fs_name, (train_loader, test_loader) in feature_sets.items():
            safe_name = fs_name.replace("+", "_plus_")
            selected_cols = load_selected(stress, safe_name)

            X_train, y_train = train_loader()
            X_test, y_test = test_loader()
            X_train = X_train[selected_cols]
            X_test = X_test[selected_cols]

            print(f"=== {stress} / {fs_name} ({len(selected_cols)} selected features) ===")
            for clf_name, clf in get_classifiers().items():
                pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])

                cv_auroc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

                pipe.fit(X_train, y_train)
                y_proba = pipe.predict_proba(X_test)[:, 1]
                y_pred = pipe.predict(X_test)

                test_auroc = roc_auc_score(y_test, y_proba)
                test_auprc = average_precision_score(y_test, y_proba)
                test_acc = accuracy_score(y_test, y_pred)

                safe_fs = fs_name.replace("+", "_plus_")
                roc_path = os.path.join(ROC_DATA_DIR, f"{stress}_{safe_fs}_{clf_name}.tsv")
                pd.DataFrame({
                    "ProteinID": X_test.index,
                    "y_true": y_test.values,
                    "y_proba": y_proba,
                }).to_csv(roc_path, sep="\t", index=False)

                print(f"  {clf_name:5s}  CV-auROC={np.mean(cv_auroc):.4f}  "
                      f"test-auROC={test_auroc:.4f}  test-auPRC={test_auprc:.4f}  "
                      f"test-acc={test_acc:.4f}")

                rows.append({
                    "stress": stress,
                    "feature_set": fs_name,
                    "classifier": clf_name,
                    "cv_auROC": round(np.mean(cv_auroc), 4),
                    "test_accuracy": round(test_acc * 100, 2),
                    "test_auROC": round(test_auroc * 100, 2),
                    "test_auPRC": round(test_auprc * 100, 2),
                })
            print()

    results_df = pd.DataFrame(rows)
    table2_path = os.path.join(DATA_DIR, "table2_results.tsv")
    results_df.to_csv(table2_path, sep="\t", index=False)
    print(f"Wrote {table2_path}")

    best_rows = []
    for stress in STRESSES:
        sub = results_df[results_df["stress"] == stress]
        best = sub.loc[sub["test_auROC"].idxmax()]
        best_rows.append(best.to_dict())
    best_df = pd.DataFrame(best_rows)
    table3_path = os.path.join(DATA_DIR, "table3_best_models.tsv")
    best_df.to_csv(table3_path, sep="\t", index=False)

    print(f"\nBest (feature_set, classifier) per stress -> {table3_path}")
    print(best_df.to_string(index=False))
    print("\nNext step: python3 08_finalize_best_models.py")


if __name__ == "__main__":
    main()
