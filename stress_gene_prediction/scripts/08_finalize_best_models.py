#!/usr/bin/env python3
"""
Phase 2, step 5 (final): hyperparameter-tune the winning (feature_set,
classifier) combination per stress from table3_best_models.tsv, refit on the
full training set, evaluate once more on the test set, and save the trained
model -- mirrors reference paper Section 3.3's final tuning step that produced
Table 3.

Output:
  data/final_models/{stress}_model.joblib   fitted Pipeline(StandardScaler, classifier)
  data/final_models/{stress}_meta.json      feature_set, selected feature columns,
                                             best hyperparameters
  data/final_model_metrics.tsv              stress, feature_set, classifier,
                                             best_params, test_accuracy, test_auROC, test_auPRC
"""
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from ml_utils import DATA_DIR, get_classifiers, get_xy, get_xy_combined  # noqa: E402

warnings.filterwarnings("ignore")

RFE_DIR = os.path.join(DATA_DIR, "rfe")
FINAL_MODELS_DIR = os.path.join(DATA_DIR, "final_models")

PARAM_GRIDS = {
    "SVM": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01, 0.1]},
    "XGB": {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.8, 1.0],
    },
    "RF": {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "GBDT": {
        "clf__n_estimators": [100, 300],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__max_depth": [2, 3, 5],
    },
    "ADB": {
        "clf__n_estimators": [50, 100, 200, 300],
        "clf__learning_rate": [0.01, 0.1, 0.5, 1.0],
    },
    "BAG": {
        "clf__n_estimators": [10, 50, 100],
        "clf__max_samples": [0.5, 0.7, 1.0],
    },
}


def load_selected(stress, safe_name):
    path = os.path.join(RFE_DIR, f"{stress}_{safe_name}_selected_features.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_xy(stress, feature_set_name, split):
    if "+" in feature_set_name:
        m1, m2 = feature_set_name.split("+")
        return get_xy_combined(stress, [m1, m2], split)
    return get_xy(stress, feature_set_name, split)


def main():
    os.makedirs(FINAL_MODELS_DIR, exist_ok=True)

    table3_path = os.path.join(DATA_DIR, "table3_best_models.tsv")
    if not os.path.exists(table3_path):
        sys.exit(f"ERROR: {table3_path} not found -- run 07_train_evaluate_classifiers.py first")
    best_df = pd.read_csv(table3_path, sep="\t")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_rows = []

    for _, row in best_df.iterrows():
        stress = row["stress"]
        feature_set = row["feature_set"]
        clf_name = row["classifier"]
        safe_name = feature_set.replace("+", "_plus_")

        print(f"=== {stress}: tuning {clf_name} on {feature_set} ===")
        selected_cols = load_selected(stress, safe_name)
        X_train, y_train = load_xy(stress, feature_set, "train")
        X_test, y_test = load_xy(stress, feature_set, "test")
        X_train = X_train[selected_cols]
        X_test = X_test[selected_cols]

        base_clf = get_classifiers()[clf_name]
        pipe = Pipeline([("scale", StandardScaler()), ("clf", base_clf)])
        param_grid = PARAM_GRIDS[clf_name]

        search = RandomizedSearchCV(
            pipe, param_distributions=param_grid, n_iter=20, cv=cv,
            scoring="roc_auc", random_state=42, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_

        y_proba = best_pipe.predict_proba(X_test)[:, 1]
        y_pred = best_pipe.predict(X_test)
        test_auroc = roc_auc_score(y_test, y_proba)
        test_auprc = average_precision_score(y_test, y_proba)
        test_acc = accuracy_score(y_test, y_pred)

        print(f"  best params: {search.best_params_}")
        print(f"  test auROC={test_auroc:.4f}  auPRC={test_auprc:.4f}  accuracy={test_acc:.4f}\n")

        model_path = os.path.join(FINAL_MODELS_DIR, f"{stress}_model.joblib")
        joblib.dump(best_pipe, model_path)

        meta = {
            "stress": stress,
            "feature_set": feature_set,
            "classifier": clf_name,
            "selected_features": selected_cols,
            "best_params": search.best_params_,
        }
        meta_path = os.path.join(FINAL_MODELS_DIR, f"{stress}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        metrics_rows.append({
            "stress": stress,
            "feature_set": feature_set,
            "classifier": clf_name,
            "n_features": len(selected_cols),
            "best_params": json.dumps(search.best_params_),
            "test_accuracy": round(test_acc * 100, 2),
            "test_auROC": round(test_auroc * 100, 2),
            "test_auPRC": round(test_auprc * 100, 2),
        })

    metrics_path = os.path.join(DATA_DIR, "final_model_metrics.tsv")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, sep="\t", index=False)
    print(f"Final tuned model metrics written to: {metrics_path}")
    print(f"Trained models saved under: {FINAL_MODELS_DIR}")


if __name__ == "__main__":
    main()
