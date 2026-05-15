from __future__ import annotations

from itertools import combinations
from pathlib import Path

from loguru import logger
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


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
                loc="upper center",
                ncol=min(len(labels), 4),
                bbox_to_anchor=(0.5, -0.02),
            )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def knn_purity_distribution_plot(
    metric_result: dict,
    output_path: str | Path = None,
    class_labels: dict[str, str] | None = None,
    **kwargs,
) -> None:
    """Render KNN per-query purity distribution as a per-class facet grid.

    One subplot per class. Within each subplot, the x-axis enumerates k
    values and at each k position one boxplot per experiment is drawn
    side-by-side — exposes the spread of per-query purity that the
    aggregated line plot collapses to a mean.

    Args:
        metric_result: Output from ``knn_purity_per_query_comparison`` metric.
        output_path: Path to save the figure. Shows interactively if None.
        class_labels: Optional mapping from class id (as string) to display
            name. Used for facet subplot titles. Falls back to raw class id.
    """
    df = metric_result.get("comparison_df", pd.DataFrame())
    if df.empty:
        logger.warning("no data for KNN purity distribution plot, skipping")
        return

    class_labels = class_labels or {}

    experiments = sorted(df["experiment"].unique())
    classes = sorted(df["class"].unique())
    k_values = sorted(df["k"].unique())
    n_classes = len(classes)
    n_experiments = len(experiments)
    n_k = len(k_values)

    colors = [plt.colormaps["tab10"](i % 10) for i in range(n_experiments)]
    box_width = 0.8 / max(n_experiments, 1)

    n_cols = min(6, n_classes) if n_classes else 1
    n_rows = (n_classes + n_cols - 1) // n_cols if n_classes else 1
    fig_height = max(3, 2.5 * n_rows)
    fig = plt.figure(figsize=(3.5 * n_cols, fig_height))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.5, wspace=0.3)

    for idx, cls in enumerate(classes):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        for i, exp in enumerate(experiments):
            data_per_k = []
            positions = []
            offset = (i - (n_experiments - 1) / 2) * box_width
            for k_idx, k in enumerate(k_values):
                values = df[(df["experiment"] == exp) & (df["class"] == cls) & (df["k"] == k)][
                    "purity"
                ].to_numpy()
                if values.size == 0:
                    continue
                data_per_k.append(values)
                positions.append(k_idx + offset)
            if not data_per_k:
                continue
            bp = ax.boxplot(
                data_per_k,
                positions=positions,
                widths=box_width * 0.85,
                patch_artist=True,
                showfliers=False,
            )
            for box in bp["boxes"]:
                box.set_facecolor(colors[i])
                box.set_edgecolor("black")
            for median in bp["medians"]:
                median.set_color("black")

        display = class_labels.get(str(cls), str(cls))
        ax.set_title(display)
        ax.set_xlabel("k")
        ax.set_ylabel("Purity")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(range(n_k))
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_xlim(-0.5, n_k - 0.5)
        ax.grid(True, alpha=0.3)

    if n_experiments:
        handles = [
            Patch(facecolor=colors[i], edgecolor="black", label=experiments[i])
            for i in range(n_experiments)
        ]
        fig.legend(
            handles,
            experiments,
            loc="upper center",
            ncol=min(n_experiments, 4),
            bbox_to_anchor=(0.5, -0.02),
        )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def per_class_similarity_distribution_plot(
    metric_result: dict,
    output_path: str | Path = None,
    class_labels: dict[str, str] | None = None,
    *,
    control_label: str,
    n_cols: int = 3,
    bins: int = 40,
    x_range: tuple[float, float] | None = None,
    max_ks_annotations: int = 6,
    include_control_series: bool = False,
    **kwargs,
) -> None:
    """Per-class overlaid histograms of per-chip cosine similarity to the control.

    One subplot per class; one histogram series per ablation experiment.
    Dashed vertical lines mark each series' median. KS-test stats for the
    experiment pairs present in that class are annotated upper-left.

    Args:
        metric_result: Output from ``per_chip_similarity_to_control``.
        output_path: Path to save the figure. Shows interactively if None.
        class_labels: Optional mapping from class id (as string) to display name.
        control_label: Label of the control experiment. Used to filter the
            control series from the overlay (since all values are 1.0) unless
            ``include_control_series`` is True.
        n_cols: Number of subplot columns in the facet grid.
        bins: Number of histogram bins.
        x_range: Optional fixed (min, max) x-axis range. Autoscales if None.
        max_ks_annotations: When all-pairs KS would exceed this count, fall
            back to ``<ablation> vs control`` pairs only.
        include_control_series: If True, render the control's self-similarity
            series (all values = 1.0). Off by default — it's uninformative.
    """
    df = metric_result.get("comparison_df", pd.DataFrame())
    if df.empty:
        logger.warning("no data for per-class similarity distribution plot, skipping")
        return

    class_labels = class_labels or {}

    plot_df = df if include_control_series else df[df["experiment"] != control_label]
    if plot_df.empty:
        logger.warning("no ablation series to plot (only control present), skipping")
        return

    experiments = sorted(plot_df["experiment"].unique().tolist())
    classes = sorted(plot_df["class"].unique().tolist())
    n_classes = len(classes)
    n_experiments = len(experiments)

    colors = {exp: plt.colormaps["tab10"](i % 10) for i, exp in enumerate(experiments)}

    n_cols_effective = min(n_cols, n_classes) if n_classes else 1
    n_rows = (n_classes + n_cols_effective - 1) // n_cols_effective if n_classes else 1
    fig_height = max(3, 2.5 * n_rows)
    fig = plt.figure(figsize=(3.5 * n_cols_effective, fig_height))
    gs = fig.add_gridspec(n_rows, n_cols_effective, hspace=0.5, wspace=0.3)

    for idx, cls in enumerate(classes):
        row = idx // n_cols_effective
        col = idx % n_cols_effective
        ax = fig.add_subplot(gs[row, col])
        is_bottom_row = row == n_rows - 1
        is_left_col = col == 0

        # Per-class data: KS pairs are computed against the *full* df so the
        # comparison can include the control series even when it isn't drawn.
        ks_pool = df[df["class"] == cls]
        ks_experiments = sorted(ks_pool["experiment"].unique().tolist())
        all_pairs = list(combinations(ks_experiments, 2))
        if len(all_pairs) > max_ks_annotations and control_label in ks_experiments:
            ks_pairs = [(a, b) for a, b in all_pairs if control_label in (a, b)]
            logger.info(
                f"class '{cls}': truncating {len(all_pairs)} KS pairs to "
                f"{len(ks_pairs)} control-only pairs"
            )
        else:
            ks_pairs = all_pairs

        ks_lines: list[str] = []
        for a, b in ks_pairs:
            va = ks_pool[ks_pool["experiment"] == a]["cosine_similarity"].to_numpy()
            vb = ks_pool[ks_pool["experiment"] == b]["cosine_similarity"].to_numpy()
            if va.size == 0 or vb.size == 0:
                continue
            res = ks_2samp(va, vb)
            ks_lines.append(f"{a} vs {b}: KS={res.statistic:.3f}, p={res.pvalue:.2e}")

        any_drawn = False
        for exp in experiments:
            values = plot_df[(plot_df["experiment"] == exp) & (plot_df["class"] == cls)][
                "cosine_similarity"
            ].to_numpy()
            if values.size == 0:
                continue
            any_drawn = True
            weights = np.full_like(values, 100.0 / values.size, dtype=float)
            hist_kwargs = {
                "bins": bins,
                "alpha": 0.5,
                "color": colors[exp],
                "weights": weights,
                "label": exp,
            }
            if x_range is not None:
                hist_kwargs["range"] = x_range
            ax.hist(values, **hist_kwargs)
            ax.axvline(np.median(values), linestyle="--", color=colors[exp], linewidth=1.2)

        if ks_lines:
            ax.text(
                0.02,
                0.98,
                "\n".join(ks_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
            )

        display = class_labels.get(str(cls), str(cls))
        ax.set_title(display)
        if is_bottom_row:
            ax.set_xlabel("Cosine Similarity to Control Embeddings")
        if is_left_col:
            ax.set_ylabel("Percentage of Samples (%)")
        if x_range is not None:
            ax.set_xlim(*x_range)
        ax.grid(True, alpha=0.3)
        if not any_drawn:
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )

    if n_experiments:
        handles = [
            Patch(facecolor=colors[exp], edgecolor="none", alpha=0.5, label=exp)
            for exp in experiments
        ]
        fig.legend(
            handles,
            experiments,
            loc="upper center",
            ncol=min(n_experiments, 4),
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
    "knn_purity_distribution_plot": knn_purity_distribution_plot,
    "per_class_similarity_distribution_plot": per_class_similarity_distribution_plot,
}
