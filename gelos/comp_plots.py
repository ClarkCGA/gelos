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


def _resolve_experiment_colors(
    experiments: list[str],
    experiment_colors: dict[str, str] | None,
) -> dict[str, str]:
    """Map experiment labels to colors, falling back to tab10 when unset."""
    experiment_colors = experiment_colors or {}
    return {
        exp: experiment_colors.get(exp, plt.colormaps["tab10"](i % 10))
        for i, exp in enumerate(experiments)
    }


def knn_purity_plot(
    metric_result: dict,
    output_path: str | Path = None,
    class_labels: dict[str, str] | None = None,
    experiment_colors: dict[str, str] | None = None,
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

    colors = _resolve_experiment_colors(experiments, experiment_colors)
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
                color=colors[exp],
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
    experiment_colors: dict[str, str] | None = None,
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

    color_map = _resolve_experiment_colors(experiments, experiment_colors)
    colors = [color_map[exp] for exp in experiments]
    box_width = 0.8 / max(n_experiments, 1)

    n_cols = min(2, n_classes) if n_classes else 1
    n_rows = (n_classes + n_cols - 1) // n_cols if n_classes else 1
    fig_height = max(3.5, 3.5 * n_rows)
    fig = plt.figure(figsize=(12, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols)

    for idx, cls in enumerate(classes):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])

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
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": colors[i],
                    "markeredgecolor": "black",
                    "markersize": 4,
                },
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


def knn_purity_violin_distribution_plot(
    metric_result: dict,
    output_path: str | Path = None,
    class_labels: dict[str, str] | None = None,
    experiment_colors: dict[str, str] | None = None,
    *,
    split: bool | None = None,
    split_pairs: list[list[str]] | None = None,
    **kwargs,
) -> None:
    """Render KNN per-query purity distribution as violin plots.

    A violin alternative to :func:`knn_purity_distribution_plot`. Consumes the
    identical metric output (``knn_purity_per_query_comparison``). One subplot
    per class; the x-axis enumerates k values. When exactly two experiments are
    compared the violins at each k are drawn as a *split* violin (experiment 0 =
    left half, experiment 1 = right half) which is especially effective for
    contrasting two groups (e.g. multitemporal vs single time step). Otherwise
    (1 or 3+ experiments) side-by-side single violins are drawn. The mean is
    marked with a diamond and the median with a short horizontal line, matching
    the box-plot variant.

    Args:
        metric_result: Output from ``knn_purity_per_query_comparison`` metric.
        output_path: Path to save the figure. Shows interactively if None.
        class_labels: Optional mapping from class id (as string) to display
            name. Used for facet subplot titles. Falls back to raw class id.
        experiment_colors: Optional mapping from experiment label to color.
        split: Force split (True) or side-by-side single (False) violins. When
            None (default) split is used automatically iff there are exactly two
            experiments. Ignored when ``split_pairs`` is provided.
        split_pairs: Optional list of 2-element experiment-name lists. Each pair
            is rendered as one split violin (``exp[0]`` = left half, ``exp[1]`` =
            right half) per k tick, with the pairs laid out side-by-side. Any
            experiment not named in a valid, retained pair is appended as a
            single violin (in sorted ``experiments`` order). When provided,
            ``split_pairs`` takes precedence over ``split`` (and the automatic
            two-experiment split rule). A malformed pair (not exactly two names)
            or a pair naming an experiment absent from the data logs a warning
            and is skipped.
    """
    df = metric_result.get("comparison_df", pd.DataFrame())
    if df.empty:
        logger.warning("no data for KNN purity violin distribution plot, skipping")
        return

    class_labels = class_labels or {}

    experiments = sorted(df["experiment"].unique())
    classes = sorted(df["class"].unique())
    k_values = sorted(df["k"].unique())
    n_classes = len(classes)
    n_experiments = len(experiments)
    n_k = len(k_values)

    color_map = _resolve_experiment_colors(experiments, experiment_colors)
    colors = [color_map[exp] for exp in experiments]
    violin_width = 0.8 / max(n_experiments, 1)

    do_split = split if split is not None else (n_experiments == 2)

    # Build a validated slot layout when split_pairs is provided. This takes
    # precedence over `split`/`do_split` (see docstring).
    slots: list[tuple] = []
    slot_width = 0.8
    if split_pairs is not None:
        experiment_set = set(experiments)
        valid_pairs: list[tuple[str, str]] = []
        for pair in split_pairs:
            if len(pair) != 2:
                logger.warning(
                    f"split_pairs entry {pair!r} is not a pair of two experiments, skipping"
                )
                continue
            e0, e1 = pair
            missing = [e for e in (e0, e1) if e not in experiment_set]
            if missing:
                logger.warning(
                    f"split_pairs entry {pair!r} names experiment(s) {missing!r} "
                    "not present in the data, skipping"
                )
                continue
            valid_pairs.append((e0, e1))
        paired = {e for pair in valid_pairs for e in pair}
        unpaired = [e for e in experiments if e not in paired]
        slots = [("pair", e0, e1) for (e0, e1) in valid_pairs]
        slots += [("single", e) for e in unpaired]
        n_slots = len(slots)
        slot_width = 0.8 / max(n_slots, 1)

    def _style_body(body, color):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.7)

    def _mean_median(ax, x, values, color, half=None):
        """Draw the mean diamond and a short median line for one violin/half."""
        if half is None:
            half = violin_width * 0.2
        ax.plot(
            x,
            float(np.mean(values)),
            marker="D",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=4,
            linestyle="none",
            zorder=5,
        )
        med = float(np.median(values))
        ax.hlines(med, x - half, x + half, color="black", linewidth=1)

    n_cols = min(2, n_classes) if n_classes else 1
    n_rows = (n_classes + n_cols - 1) // n_cols if n_classes else 1
    fig_height = max(3.5, 3.5 * n_rows)
    fig = plt.figure(figsize=(12, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols)

    for idx, cls in enumerate(classes):
        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])

        if split_pairs is not None:
            for k_idx, k in enumerate(k_values):
                center = float(k_idx)
                for s_idx, slot in enumerate(slots):
                    slot_center = center + (s_idx - (n_slots - 1) / 2) * slot_width
                    if slot[0] == "pair":
                        _, e0, e1 = slot
                        drew_center_line = False
                        for half_idx, exp in enumerate((e0, e1)):
                            values = df[
                                (df["experiment"] == exp) & (df["class"] == cls) & (df["k"] == k)
                            ]["purity"].to_numpy()
                            if values.size == 0:
                                continue
                            left = half_idx == 0
                            marker_x = slot_center + (
                                -slot_width * 0.25 if left else slot_width * 0.25
                            )
                            if values.size >= 2 and np.ptp(values) > 0:
                                parts = ax.violinplot(
                                    [values],
                                    positions=[slot_center],
                                    # Clamp the split-body width to the slot so
                                    # adjacent slots do not overlap.
                                    widths=min(slot_width * 1.7, slot_width),
                                    showextrema=False,
                                    showmeans=False,
                                    showmedians=False,
                                )
                                for body in parts["bodies"]:
                                    verts = body.get_paths()[0].vertices
                                    if left:
                                        verts[:, 0] = np.minimum(verts[:, 0], slot_center)
                                    else:
                                        verts[:, 0] = np.maximum(verts[:, 0], slot_center)
                                    _style_body(body, color_map[exp])
                                if not drew_center_line:
                                    ax.axvline(
                                        slot_center,
                                        color="black",
                                        linewidth=0.6,
                                        alpha=0.4,
                                        zorder=1,
                                    )
                                    drew_center_line = True
                            _mean_median(
                                ax, marker_x, values, color_map[exp], half=slot_width * 0.2
                            )
                    else:
                        _, exp = slot
                        values = df[
                            (df["experiment"] == exp) & (df["class"] == cls) & (df["k"] == k)
                        ]["purity"].to_numpy()
                        if values.size == 0:
                            continue
                        if values.size >= 2 and np.ptp(values) > 0:
                            parts = ax.violinplot(
                                [values],
                                positions=[slot_center],
                                widths=slot_width * 0.85,
                                showextrema=False,
                                showmeans=False,
                                showmedians=False,
                            )
                            for body in parts["bodies"]:
                                _style_body(body, color_map[exp])
                        _mean_median(
                            ax, slot_center, values, color_map[exp], half=slot_width * 0.2
                        )
        elif do_split and n_experiments == 2:
            for k_idx, k in enumerate(k_values):
                center = float(k_idx)
                drew_center_line = False
                for i, exp in enumerate(experiments):
                    values = df[(df["experiment"] == exp) & (df["class"] == cls) & (df["k"] == k)][
                        "purity"
                    ].to_numpy()
                    if values.size == 0:
                        continue
                    left = i == 0
                    # Nudge the mean/median marker toward the populated half.
                    marker_x = center + (-violin_width * 0.25 if left else violin_width * 0.25)
                    if values.size >= 2 and np.ptp(values) > 0:
                        parts = ax.violinplot(
                            [values],
                            positions=[center],
                            widths=violin_width * 1.7,
                            showextrema=False,
                            showmeans=False,
                            showmedians=False,
                        )
                        for body in parts["bodies"]:
                            verts = body.get_paths()[0].vertices
                            if left:
                                verts[:, 0] = np.minimum(verts[:, 0], center)
                            else:
                                verts[:, 0] = np.maximum(verts[:, 0], center)
                            _style_body(body, colors[i])
                        if not drew_center_line:
                            ax.axvline(center, color="black", linewidth=0.6, alpha=0.4, zorder=1)
                            drew_center_line = True
                    _mean_median(ax, marker_x, values, colors[i])
        else:
            for i, exp in enumerate(experiments):
                offset = (i - (n_experiments - 1) / 2) * violin_width
                for k_idx, k in enumerate(k_values):
                    values = df[(df["experiment"] == exp) & (df["class"] == cls) & (df["k"] == k)][
                        "purity"
                    ].to_numpy()
                    if values.size == 0:
                        continue
                    x = k_idx + offset
                    if values.size >= 2 and np.ptp(values) > 0:
                        parts = ax.violinplot(
                            [values],
                            positions=[x],
                            widths=violin_width * 0.85,
                            showextrema=False,
                            showmeans=False,
                            showmedians=False,
                        )
                        for body in parts["bodies"]:
                            _style_body(body, colors[i])
                    _mean_median(ax, x, values, colors[i])

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
    experiment_colors: dict[str, str] | None = None,
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

    colors = _resolve_experiment_colors(experiments, experiment_colors)

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
    "knn_purity_violin_distribution_plot": knn_purity_violin_distribution_plot,
    "per_class_similarity_distribution_plot": per_class_similarity_distribution_plot,
}
