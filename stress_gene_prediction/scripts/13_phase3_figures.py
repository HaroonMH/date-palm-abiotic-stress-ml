#!/usr/bin/env python3
"""
Milestone 3, step 4: figures for the genome-wide prediction stage, using the
same fixed colorblind-safe palette as 09_generate_figures.py so stress colors
stay consistent across the whole manuscript.

  fig6_prediction_distributions  score distribution over the unlabeled proteome
                                  per stress, with the p=0.5 call threshold
  fig7_predicted_srg_counts      number of predicted SRGs per stress at
                                  several probability thresholds
  fig8_top10_candidates          top-10 candidate genes and their likeliness
                                  scores per stress (reference paper Table 4)

Output: PNG (300dpi) + PDF under data/figures/.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from ml_utils import DATA_DIR  # noqa: E402

GW_DIR = os.path.join(DATA_DIR, "genomewide")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")

OKABE_ITO = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#F0E442"]
STRESS_ORDER = ["cold", "heat", "drought", "salt"]
STRESS_COLORS = dict(zip(STRESS_ORDER, OKABE_ITO[:4]))

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


def load_ranked(stress):
    path = os.path.join(GW_DIR, f"{stress}_ranked_predictions.tsv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t", index_col=0)


def fig6_prediction_distributions():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    any_data = False
    for ax, stress in zip(axes.flat, STRESS_ORDER):
        ranked = load_ranked(stress)
        if ranked is None:
            continue
        any_data = True
        ax.hist(ranked["Likeliness"], bins=60, color=STRESS_COLORS[stress], alpha=0.85)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.2)
        n_above = int((ranked["Likeliness"] > 0.5).sum())
        ax.set_yscale("log")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Number of proteins (log)")
        ax.set_title(f"{stress.capitalize()}  ({n_above:,} above p=0.5)")

    if not any_data:
        plt.close(fig)
        print("  skip fig6 (no ranked predictions found)")
        return
    fig.suptitle("Genome-wide prediction score distribution (unlabeled proteome)", y=1.00)
    fig.tight_layout()
    save(fig, "fig6_prediction_distributions")


def fig7_predicted_srg_counts():
    counts_path = os.path.join(GW_DIR, "srg_counts.tsv")
    if not os.path.exists(counts_path):
        print(f"  skip fig7 (missing {counts_path})")
        return
    df = pd.read_csv(counts_path, sep="\t").set_index("stress").loc[STRESS_ORDER]
    thr_cols = [c for c in df.columns if c.startswith("predicted_SRGs_p>")]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(STRESS_ORDER))
    width = 0.8 / len(thr_cols)
    for i, col in enumerate(thr_cols):
        offset = (i - (len(thr_cols) - 1) / 2) * width
        label = col.replace("predicted_SRGs_", "")
        bars = ax.bar(x + offset, df[col], width, label=label, color=OKABE_ITO[i])
        for xi, v in zip(x + offset, df[col]):
            ax.text(xi, v, f"{int(v):,}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in STRESS_ORDER])
    ax.set_ylabel("Predicted stress-responsive proteins")
    ax.set_title("Genome-wide predicted SRGs per stress")
    ax.legend(frameon=False, title="Threshold")
    save(fig, "fig7_predicted_srg_counts")


def fig8_top10_candidates():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    any_data = False
    for ax, stress in zip(axes.flat, STRESS_ORDER):
        path = os.path.join(GW_DIR, f"{stress}_top10.tsv")
        if not os.path.exists(path):
            continue
        any_data = True
        top = pd.read_csv(path, sep="\t", index_col=0).sort_values("Likeliness")
        ypos = np.arange(len(top))
        ax.barh(ypos, top["Likeliness"], color=STRESS_COLORS[stress], height=0.7)
        ax.set_yticks(ypos)
        ax.set_yticklabels(top.index, fontsize=7)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Likeliness score")
        ax.set_title(stress.capitalize())
        for yi, v in zip(ypos, top["Likeliness"]):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=7)

    if not any_data:
        plt.close(fig)
        print("  skip fig8 (no top10 tables found)")
        return
    fig.suptitle("Top-10 predicted candidate genes per stress", y=1.00)
    fig.tight_layout()
    save(fig, "fig8_top10_candidates")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating Milestone 3 figures...")
    fig6_prediction_distributions()
    fig7_predicted_srg_counts()
    fig8_top10_candidates()
    print(f"\nFigures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
