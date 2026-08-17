#!/usr/bin/env python3
"""
Milestone 3, revised step 3: figures from the margin-ranked, isoform-collapsed
results. Reads data/genomewide_v2/ and writes data/figures_v2/, leaving the
originals in place.

  fig6_prediction_distributions  probability distribution over the unlabeled
                                  proteome per stress
  fig7_predicted_srg_counts      predicted SRGs per stress -- protein-level vs
                                  gene-level (isoform-collapsed), by threshold
  fig8_top10_candidates          top-10 distinct genes per stress, by decision
                                  margin (the quantity actually ranked on, and
                                  unlike probability it does not saturate)

Same fixed colorblind-safe palette as the earlier figure scripts.
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

GW_DIR = os.path.join(DATA_DIR, "genomewide_v2")
FIGURES_DIR = os.path.join(DATA_DIR, "figures_v2")

OKABE_ITO = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
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


def fig6_prediction_distributions():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    found = False
    for ax, stress in zip(axes.flat, STRESS_ORDER):
        path = os.path.join(GW_DIR, f"{stress}_ranked_genes.tsv")
        if not os.path.exists(path):
            continue
        found = True
        genes = pd.read_csv(path, sep="\t", index_col=0)
        ax.hist(genes["Likeliness"], bins=60, color=STRESS_COLORS[stress], alpha=0.85)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Number of genes (log)")
        ax.set_title(f"{stress.capitalize()}  ({int((genes['Likeliness'] > 0.5).sum()):,} above p=0.5)")
    if not found:
        plt.close(fig)
        print("  skip fig6 (no ranked_genes files)")
        return
    fig.suptitle("Genome-wide prediction scores (isoform-collapsed genes)", y=1.00)
    fig.tight_layout()
    save(fig, "fig6_prediction_distributions")


def fig7_predicted_srg_counts():
    path = os.path.join(GW_DIR, "srg_counts.tsv")
    if not os.path.exists(path):
        print(f"  skip fig7 (missing {path})")
        return
    df = pd.read_csv(path, sep="\t").set_index("stress").loc[STRESS_ORDER]
    thresholds = [c.replace("proteins_p>", "") for c in df.columns if c.startswith("proteins_p>")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.arange(len(STRESS_ORDER))
    width = 0.8 / len(thresholds)

    for ax, level, prefix in zip(axes, ["Protein-level", "Gene-level (collapsed)"],
                                 ["proteins_p>", "genes_p>"]):
        for i, thr in enumerate(thresholds):
            offset = (i - (len(thresholds) - 1) / 2) * width
            vals = df[f"{prefix}{thr}"]
            ax.bar(x + offset, vals, width, label=f"p>{thr}", color=OKABE_ITO[i])
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v, f"{int(v):,}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in STRESS_ORDER])
        ax.set_title(level)
    axes[0].set_ylabel("Predicted stress-responsive genes")
    axes[1].legend(frameon=False, title="Threshold")
    fig.suptitle("Predicted SRGs per stress, before and after isoform collapsing", y=1.02)
    save(fig, "fig7_predicted_srg_counts")


def fig8_top10_candidates():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    found = False
    for ax, stress in zip(axes.flat, STRESS_ORDER):
        path = os.path.join(GW_DIR, f"{stress}_top10.tsv")
        if not os.path.exists(path):
            continue
        found = True
        top = pd.read_csv(path, sep="\t", index_col=0).sort_values("Margin")
        ypos = np.arange(len(top))
        ax.barh(ypos, top["Margin"], color=STRESS_COLORS[stress], height=0.7)
        labels = [
            f"{pid}  {str(d)[:42]}" for pid, d in zip(top.index, top["Description"])
        ]
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("SVM decision margin")
        ax.set_title(stress.capitalize())
        for yi, v in zip(ypos, top["Margin"]):
            ax.text(v, yi, f" {v:.2f}", va="center", fontsize=6)
    if not found:
        plt.close(fig)
        print("  skip fig8 (no top10 files)")
        return
    fig.suptitle("Top-10 predicted candidate genes per stress (isoform-collapsed)", y=1.01)
    fig.tight_layout()
    save(fig, "fig8_top10_candidates")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating revised Milestone 3 figures...")
    fig6_prediction_distributions()
    fig7_predicted_srg_counts()
    fig8_top10_candidates()
    print(f"\nFigures written to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
