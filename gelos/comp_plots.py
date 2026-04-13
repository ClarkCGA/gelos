from __future__ import annotations

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pca_ablation_table(
    metric_result: dict,
    output_path: str | Path = None,
    format: str = "heatmap",
    **kwargs,
) -> None:
    """Render a PCA ablation comparison as a heatmap.

    Rows are experiments, columns are variance thresholds, cell values are
    the number of PCA components needed.

    Args:
        metric_result: Output from ``pca_ablation_comparison`` metric.
        output_path: Path to save the figure. Shows interactively if None.
        format: Plot format (currently only "heatmap" is supported).
    """
    df = metric_result.get("comparison_df", pd.DataFrame())
    if df.empty:
        logger.warning("no data for PCA ablation table, skipping")
        return

    pivot = df.pivot_table(
        index="experiment",
        columns="threshold",
        values="proportion_of_total_components",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(10, max(3, len(pivot) * 0.8)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0%}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Variance Threshold")
    ax.set_ylabel("Experiment")
    ax.set_title("PCA Components Needed by Variance Threshold as Percent of Total Dimensions")

    # Build a matching pivot of raw component counts for annotation
    n_comp_pivot = df.pivot_table(
        index="experiment",
        columns="threshold",
        values="n_components",
        aggfunc="first",
    ).reindex(index=pivot.index, columns=pivot.columns)

    # Annotate cells with proportion and total count
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            pct = pivot.values[i, j]
            n = n_comp_pivot.values[i, j]
            if not np.isnan(pct):
                ax.text(
                    j,
                    i,
                    f"{int(pct * 100)}%\n({int(n)})",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    fig.colorbar(im, ax=ax, label="Percentage of Components")
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def distance_matrix(
    metric_result: dict,
    output_path: str | Path = None,
    **kwargs,
) -> None:
    """Render a pairwise distance/similarity matrix as a heatmap.

    Args:
        metric_result: Output from ``cosine_distance`` or ``wasserstein_distance``.
        output_path: Path to save the figure. Shows interactively if None.
    """
    labels = metric_result["labels"]
    matrix = metric_result["matrix"]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    im = ax.imshow(matrix, cmap="coolwarm", aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_title("Pairwise Distance Matrix")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def knn_purity_plot(
    metric_result: dict,
    output_path: str | Path = None,
    **kwargs,
) -> None:
    """Render KNN class purity comparison as line plots.

    Creates a figure with two subplots:
    - Top: overall purity vs k, one line per experiment
    - Bottom: per-class purity vs k, faceted by experiment

    Args:
        metric_result: Output from ``knn_purity_comparison`` metric.
        output_path: Path to save the figure. Shows interactively if None.
    """
    df = metric_result.get("comparison_df", pd.DataFrame())
    if df.empty:
        logger.warning("no data for KNN purity plot, skipping")
        return

    overall = df[df["class"] == "overall"]
    per_class = df[df["class"] != "overall"]
    experiments = overall["experiment"].unique()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8))

    # --- Top: overall purity ---
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, exp in enumerate(experiments):
        exp_data = overall[overall["experiment"] == exp].sort_values("k")
        marker = markers[i % len(markers)]
        ax_top.plot(exp_data["k"], exp_data["purity"], marker=marker, label=exp)

    ax_top.set_xlabel("k")
    ax_top.set_ylabel("Purity")
    ax_top.set_ylim(0, 1.05)
    ax_top.set_title("Overall KNN Class Purity by Experiment")
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)

    # --- Bottom: per-class purity ---
    classes = sorted(per_class["class"].unique())
    n_classes = len(classes)
    n_experiments = len(experiments)
    k_values = sorted(per_class["k"].unique())
    x = np.arange(len(k_values))
    total_bars = n_classes * n_experiments
    width = 0.8 / max(total_bars, 1)

    for i, exp in enumerate(experiments):
        for j, cls in enumerate(classes):
            subset = per_class[
                (per_class["experiment"] == exp) & (per_class["class"] == cls)
            ].sort_values("k")
            offset = (i * n_classes + j - total_bars / 2) * width + width / 2
            bar_vals = [
                subset[subset["k"] == k]["purity"].values[0]
                if len(subset[subset["k"] == k]) > 0
                else 0
                for k in k_values
            ]
            ax_bot.bar(x + offset, bar_vals, width, label=f"{exp} — {cls}")

    ax_bot.set_xlabel("k")
    ax_bot.set_ylabel("Purity")
    ax_bot.set_ylim(0, 1.05)
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([str(k) for k in k_values])
    ax_bot.set_title("Per-Class KNN Purity by Experiment")
    ax_bot.legend(fontsize=7, ncol=2)
    ax_bot.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


COMP_PLOTS: dict[str, callable] = {
    "pca_ablation_table": pca_ablation_table,
    "distance_matrix": distance_matrix,
    "knn_purity_plot": knn_purity_plot,
}
