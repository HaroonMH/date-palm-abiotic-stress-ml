"""Shared helpers for Phase 2 (feature screening / RFE / classifier training)."""
import os

import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from xgboost import XGBClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
FINAL_DIR = os.path.join(DATA_DIR, "final")

STRESSES = ["cold", "heat", "drought", "salt"]

# The six iFeatureOmega descriptors named in the pipeline doc / reference paper.
# type 2 = raw count (matches paper's "raw count of ... pairs/tripeptides" wording).
FEATURE_METHODS = {
    "CKSAAP": "CKSAAP type 2",
    "DDE": "DDE",
    "DPC": "DPC type 2",
    "TPC": "TPC type 2",
    "PAAC": "PAAC",
    "APAAC": "APAAC",
}

RANDOM_STATE = 42


def get_classifiers():
    """The six classifiers named in the pipeline doc: SVM, XGB, RF, GBDT, ADB, BAG."""
    return {
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "XGB": XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "RF": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "GBDT": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "ADB": AdaBoostClassifier(random_state=RANDOM_STATE),
        "BAG": BaggingClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    }


def load_labels(stress):
    """Return dict: ProteinID -> {'Label': int, 'Split': 'train'/'test'}."""
    path = os.path.join(FINAL_DIR, f"{stress}_labels.tsv")
    df = pd.read_csv(path, sep="\t")
    return df.set_index("ProteinID")


def load_feature_csv(stress, method, split):
    path = os.path.join(FEATURES_DIR, f"{stress}_{method}_{split}.csv")
    return pd.read_csv(path, index_col=0)


def get_xy(stress, method, split, labels=None):
    """Load a feature CSV and align with labels -> (X DataFrame, y Series)."""
    if labels is None:
        labels = load_labels(stress)
    feats = load_feature_csv(stress, method, split)
    split_ids = labels.index[labels["Split"] == split]
    common = [i for i in split_ids if i in feats.index]
    X = feats.loc[common]
    y = labels.loc[common, "Label"]
    return X, y


def get_xy_combined(stress, methods, split, labels=None):
    """Concatenate feature columns from multiple methods (e.g. CKSAAP + DDE)."""
    if labels is None:
        labels = load_labels(stress)
    parts = [load_feature_csv(stress, m, split) for m in methods]
    common_ids = set(parts[0].index)
    for p in parts[1:]:
        common_ids &= set(p.index)
    split_ids = [i for i in labels.index if labels.loc[i, "Split"] == split and i in common_ids]
    X = pd.concat([p.loc[split_ids] for p in parts], axis=1)
    y = labels.loc[split_ids, "Label"]
    return X, y
