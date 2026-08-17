#!/usr/bin/env python3
"""
Phase 2, step 6: publication-style figures from the artifacts produced by
scripts 01-08, mirroring the reference paper's figures:

  fig1_dataset_composition   Table 1 / Figure 1 equivalent (positive/negative,
                              train/test counts per stress)
  fig2_feature_screening     Figure 5 equivalent (auROC heatmap: classifier x
                              feature method, one panel per stress)
  fig3_rfe_curves            Figure 6 equivalent (auROC vs number of features
                              during SVM-RFE, one panel per stress)
  fig4_roc_curves            Figure 7/8 equivalent (ROC curves for all six
                              classifiers on the winning feature set, one
                              panel per stress)
  fig5_final_model_metrics   Table 3 (tuned) as a bar chart

Uses a fixed, colorblind-safe categorical palette (Okabe-Ito) so classifier
and stress identity are consistent across every figure, and a single-hue
sequential colormap (viridis) for the auROC heatmap. Output: PNG (300dpi) +
PDF under data/figures/.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from ml_utils import STRESSES, FEATURE_METHODS, DATA_DIR  # noqa: E402

FIGURES_DIR = os.path.join(DATA_DIR, "figures")
ROC_DATA_DIR = os.path.join(DATA_DIR, "roc_data")
RFE_DIR = os.path.join(DATA_DIR, "rfe")

# Okabe-Ito palette -- colorblind-safe, fixed assignment (never re-cycled).
OKABE_ITO = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#F0E442"]

STRESS_ORDER = ["cold", "heat", "drought", "salt"]
STRESS_COLORS = dict(zip(STRESS_ORDER, OKABE_ITO[:4]))

CLASSIFIER_ORDER = ["SVM", "XGB", "RF", "GBDT", "ADB", "BAG"]
CLASSIFIER_COLORS = dict(zip(CLASSIFIER_ORDER, OKABE_ITO[:6]))

plt.rcParams.update({
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / {name}.pdf")


def fig1_dataset_composition():
    path = os.path.join(DATA_DIR, "dataset_summary.tsv")
    if not os.path.exists(path):
        print(f"  skip (missing {path})")
        return
    df = pd.read_csv(path, sep="\t").set_index("stress").loc[STRESS_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    x = np.arange(len(STRESS_ORDER))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width / 2, df["positive_set"], width, label="Positive", color=OKABE_ITO[3])
    ax.bar(x + width / 2, df["negative_set"], width, label="Negative", color=OKABE_ITO[1])
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in STRESS_ORDER])
    ax.set_ylabel("Number of proteins")
    ax.set_title("Positive vs. negative set (post CD-HIT)")
    ax.legend(frameon=False)
    for xi, (p, n) in enumerate(zip(df["positive_set"], df["negative_set"])):
        ax.text(xi - width / 2, p, str(p), ha="center", va="bottom", fontsize=8)
        ax.text(xi + width / 2, n, str(n), ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bottom = np.zeros(len(STRESS_ORDER))
    parts = [
        ("train_positive", "Train positive", OKABE_ITO[3]),
        ("train_negative", "Train negative", OKABE_ITO[1]),
        ("test_positive", "Test positive", OKABE_ITO[5]),
        ("test_negative", "Test negative", OKABE_ITO[2]),
    ]
    for col, label, color in parts:
        ax.bar(x, df[col], bottom=bottom, label=label, color=color)
        bottom += df[col].values
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in STRESS_ORDER])
    ax.set_ylabel("Number of proteins")
    ax.set_title("Train / test split (7:3)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    fig.suptitle("Curated dataset composition per stress type", y=1.02)
    save(fig, "fig1_dataset_composition")


def fig2_feature_screening():
    path = os.path.join(DATA_DIR, "screening_results.tsv")
    if not os.path.exists(path):
        print(f"  skip (missing {path})")
        return
    df = pd.read_csv(path, sep="\t")
    methods = list(FEATURE_METHODS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    im = None
    for ax, stress in zip(axes.flat, STRESS_ORDER):
        sub = df[df["stress"] == stress]
        grid = sub.pivot(index="classifier", columns="feature_method", values="cv_auROC_mean")
        grid = grid.reindex(index=CLASSIFIER_ORDER, columns=methods)
        im = ax.imshow(grid.values, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASSIFIER_ORDER)))
        ax.set_yticklabels(CLASSIFIER_ORDER)
        ax.set_title(stress.capitalize())
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid.values[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.75 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Mean 5-fold CV auROC")
    fig.suptitle("Feature method x classifier screening (training set, 5-fold CV)", y=0.98)
    save(fig, "fig2_feature_screening")


def fig3_rfe_curves():
    summary_path = os.path.join(DATA_DIR, "rfe_summary.tsv")
    if not os.path.exists(summary_path):
        print(f"  skip (missing {summary_path})")
        return
    summary = pd.read_csv(summary_path, sep="\t")
    feature_sets = summary["feature_set"].unique().tolist()
    fs_colors = dict(zip(feature_sets, OKABE_ITO[:len(feature_sets)]))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, stress in zip(axes.flat, STRESS_ORDER):
        for fs_name in feature_sets:
            safe_name = fs_name.replace("+", "_plus_")
            curve_path = os.path.join(RFE_DIR, f"{stress}_{safe_name}_rfe_curve.tsv")
            if not os.path.exists(curve_path):
                continue
            curve = pd.read_csv(curve_path, sep="\t")
            ax.plot(curve["n_features"], curve["mean_cv_auROC"],
                     color=fs_colors[fs_name], label=fs_name, linewidth=1.5)

            row = summary[(summary["stress"] == stress) & (summary["feature_set"] == fs_name)]
            if not row.empty:
                n_sel = int(row["n_selected"].iloc[0])
                best_auroc = float(row["best_cv_auROC"].iloc[0])
                ax.axvline(n_sel, color=fs_colors[fs_name], linestyle="--", linewidth=0.8, alpha=0.6)
                ax.scatter([n_sel], [best_auroc], color=fs_colors[fs_name], s=25, zorder=5)

        ax.set_title(stress.capitalize())
        ax.set_xlabel("Number of features selected")
        ax.set_ylabel("Mean 5-fold CV auROC")

    handles = [plt.Line2D([0], [0], color=fs_colors[fs], lw=2) for fs in feature_sets]
    fig.legend(handles, feature_sets, loc="upper center", ncol=len(feature_sets),
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("SVM-RFE: auROC vs. number of features", y=1.06)
    save(fig, "fig3_rfe_curves")


def fig4_roc_curves():
    table3_path = os.path.join(DATA_DIR, "table3_best_models.tsv")
    if not os.path.exists(table3_path):
        print(f"  skip (missing {table3_path})")
        return
    best_df = pd.read_csv(table3_path, sep="\t").set_index("stress")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, stress in zip(axes.flat, STRESS_ORDER):
        winning_fs = best_df.loc[stress, "feature_set"]
        safe_fs = winning_fs.replace("+", "_plus_")
        for clf_name in CLASSIFIER_ORDER:
            pred_path = os.path.join(ROC_DATA_DIR, f"{stress}_{safe_fs}_{clf_name}.tsv")
            if not os.path.exists(pred_path):
                continue
            preds = pd.read_csv(pred_path, sep="\t")
            fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_proba"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=CLASSIFIER_COLORS[clf_name],
                     label=f"{clf_name} (AUC={roc_auc:.2f})", linewidth=1.5)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{stress.capitalize()} (feature set: {winning_fs})")
        ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("ROC curves on held-out test set", y=1.02)
    save(fig, "fig4_roc_curves")


def fig5_final_model_metrics():
    path = os.path.join(DATA_DIR, "final_model_metrics.tsv")
    if not os.path.exists(path):
        print(f"  skip (missing {path})")
        return
    df = pd.read_csv(path, sep="\t").set_index("stress").loc[STRESS_ORDER]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(STRESS_ORDER))
    width = 0.25
    metrics = [("test_accuracy", "Accuracy", OKABE_ITO[5]),
               ("test_auROC", "auROC", OKABE_ITO[0]),
               ("test_auPRC", "auPRC", OKABE_ITO[2])]
    for i, (col, label, color) in enumerate(metrics):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, df[col], width, label=label, color=color)
        for xi, v in zip(x + offset, df[col]):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.capitalize()}\n({df.loc[s, 'classifier']})" for s in STRESS_ORDER])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Final tuned model performance (test set)")
    ax.legend(frameon=False)
    save(fig, "fig5_final_model_metrics")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating figures...")
    fig1_dataset_composition()
    fig2_feature_screening()
    fig3_rfe_curves()
    fig4_roc_curves()
    fig5_final_model_metrics()
    print(f"\nAll available figures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
