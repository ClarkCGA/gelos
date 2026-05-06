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
    class_labels: dict[str, str] | None = None,
    **kwargs,
) -> None:
    """Render KNN class purity comparison as a per-class facet grid.

    One subplot per class, each showing purity vs k with one line per
    experiment — makes cross-model comparison direct.

    Args:
        metric_result: Output from ``knn_purity_comparison`` metric.
        output_path: Path to save the figure. Shows interactively if None.
        class_labels: Optional mapping from class id (as string) to display
            name. Used for facet subplot titles. Falls back to raw class id.
    """
    df = metric_result.get("comparison_df", pd.DataFrame())
    if df.empty:
        logger.warning("no data for KNN purity plot, skipping")
        return

    class_labels = class_labels or {}

    per_class = df[df["class"] != "overall"]
    experiments = list(per_class["experiment"].unique())
    classes = sorted(per_class["class"].unique())
    n_classes = len(classes)

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    n_cols = min(6, n_classes) if n_classes else 1
    n_rows = (n_classes + n_cols - 1) // n_cols if n_classes else 1
    fig_height = max(3, 2.5 * n_rows)
    fig = plt.figure(figsize=(3.5 * n_cols, fig_height))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.5, wspace=0.3)

    facet_axes = []
    for idx, cls in enumerate(classes):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        facet_axes.append(ax)

        for i, exp in enumerate(experiments):
            subset = per_class[
                (per_class["experiment"] == exp) & (per_class["class"] == cls)
            ].sort_values("k")
            if subset.empty:
                continue
            ax.plot(
                subset["k"],
                subset["purity"],
                marker=markers[i % len(markers)],
                label=exp,
            )

        display = class_labels.get(str(cls), str(cls))
        ax.set_title(display)
        ax.set_xlabel("k")
        ax.set_ylabel("Purity")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # Single shared legend for facets, placed below the grid.
    if facet_axes:
        handles, labels = facet_axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(len(labels), 4),
                bbox_to_anchor=(0.5, -0.02),
            )

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
