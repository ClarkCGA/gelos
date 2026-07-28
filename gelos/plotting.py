from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from gelos.transforms import TRANSFORM_TITLES


def format_lat_lon(lat: float, lon: float) -> str:
    """Return formatted coords like 12.34°N, 56.78°E."""
    lat_hemisphere = "N" if lat >= 0 else "S"
    lon_hemisphere = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{lat_hemisphere}, {abs(lon):.2f}°{lon_hemisphere}"


def scatter_2d(
    embeddings: np.ndarray,
    chip_gdf: gpd.GeoDataFrame,
    chip_indices: list[int],
    style_cfg: dict,
    experiment_name: str,
    strategy_title: str,
    t_type: str,
    embedding_layer: str,
    output_path: str | Path = None,
    axis_lim: int = 120,
    legend_loc: str = "upper left",
) -> None:
    """
    plot a 2d transform of embeddings colored according to chip category
    """
    category_column, color_dict, legend_patches = build_style_from_config(style_cfg)
    colors = chip_gdf[category_column].loc[chip_indices].astype(str).map(color_dict)
    transform_title = TRANSFORM_TITLES[t_type]

    fig = plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 1], -embeddings[:, 0], c=colors, s=2)
    plt.suptitle(
        f"{transform_title} Visualization of Embeddings for {experiment_name}",
        fontsize=14,
    )
    plt.title(strategy_title)
    plt.figtext(
        0.5,
        0.01,
        "Embeddings from the model's final transformer layer.",
        ha="center",
        fontsize=8,
        color="gray",
    )
    plt.xlabel(f"{transform_title} Dimension 1", fontsize=12)
    plt.ylabel(f"{transform_title} Dimension 2", fontsize=12)
    if axis_lim:
        plt.xlim([-axis_lim, axis_lim])
        plt.ylim([-axis_lim, axis_lim])
    plt.legend(handles=legend_patches, loc=legend_loc, fontsize=10, framealpha=0.9)

    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def build_style_from_config(style_cfg: dict) -> tuple[str, dict, list[Patch]]:
    """Extract category_column, color_dict, and legend_patches from the style config section."""
    category_column = style_cfg["category_column"]
    color_dict = {str(k): v for k, v in style_cfg["colors"].items()}
    label_dict = {str(k): v for k, v in style_cfg["labels"].items()}
    legend_patches = [Patch(color=color, label=label_dict[k]) for k, color in color_dict.items()]
    return category_column, color_dict, legend_patches


def temporal_cosine_similarity(
    embeddings: np.ndarray,
    chip_gdf: gpd.GeoDataFrame,
    chip_indices: list[int],
    style_cfg: dict,
    experiment_name: str,
    strategy_title: str,
    t_type: str,
    embedding_layer: str,
    output_path: str | Path = None,
    n_timesteps: int = 4,
    timestep_labels: list[str] | None = None,
    n_cols: int = 6,
    ylim: tuple[float, float] = (0.5, 1),
) -> None:
    """Plot cosine similarity between consecutive timesteps per land-cover category.

    Reshapes flat embeddings into (N, n_timesteps, D), computes cosine similarity
    of each timestep to its predecessor, then plots median + IQR + min/max per category.
    """
    if not isinstance(n_timesteps, int) or n_timesteps <= 1:
        raise ValueError(f"n_timesteps must be an int > 1, got {n_timesteps!r}")
    if timestep_labels is not None and len(timestep_labels) != n_timesteps:
        raise ValueError(
            f"timestep_labels length ({len(timestep_labels)}) must equal "
            f"n_timesteps ({n_timesteps})"
        )

    n_samples, flat_dim = embeddings.shape
    embed_depth = flat_dim // n_timesteps
    reshaped = embeddings.reshape(n_samples, n_timesteps, embed_depth)
    previous = np.roll(reshaped, shift=1, axis=1)

    cos_sim = np.array(
        [cosine_similarity(reshaped[i], previous[i]).diagonal() for i in range(n_samples)]
    )  # shape (N, n_timesteps)

    category_column, color_dict, _ = build_style_from_config(style_cfg)
    categories = chip_gdf[category_column].loc[chip_indices]
    label_map = style_cfg.get("labels", {})

    unique_cats = sorted(categories.unique())
    n_categories = len(unique_cats)
    n_rows = math.ceil(n_categories / n_cols)
    timesteps = np.arange(n_timesteps)
    x_labels = timestep_labels if timestep_labels else [str(t) for t in timesteps]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4 * n_rows),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes).flatten()

    for i, cat in enumerate(unique_cats):
        ax = axes[i]
        mask = categories.values == cat
        cat_sim = cos_sim[mask]
        cat_color = color_dict.get(str(cat), color_dict.get(cat, "gray"))

        median = np.median(cat_sim, axis=0)
        q1 = np.percentile(cat_sim, 25, axis=0)
        q3 = np.percentile(cat_sim, 75, axis=0)
        sim_min = cat_sim.min(axis=0)
        sim_max = cat_sim.max(axis=0)

        ax.plot(timesteps, median, label="Median", color=cat_color)
        ax.plot(timesteps, sim_min, label="Minimum", color=cat_color, linestyle="--")
        ax.plot(timesteps, sim_max, label="Maximum", color=cat_color, linestyle=":")
        ax.fill_between(timesteps, q1, q3, alpha=0.8, color=cat_color, label="IQR")

        cat_label = label_map.get(str(cat), str(cat))
        ax.set_title(cat_label)
        if i == 0:
            ax.set_ylabel("Cosine Similarity to Previous Time Step")
            ax.legend()
        ax.set_xticks(timesteps)
        ax.set_xticklabels(x_labels)
        ax.set_ylim(ylim[0], ylim[1])
        ax.grid(True, linestyle="--", alpha=0.6)

    # Hide unused axes
    for j in range(n_categories, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{experiment_name} Embedding Trajectories by Land Cover Category ({strategy_title})",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.005,
        "Embeddings from the model's final transformer layer.",
        ha="center",
        fontsize=8,
        color="gray",
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    chip_indices: list[int],
    style_cfg: dict,
    experiment_name: str,
    strategy_title: str,
    model_type: str,
    embedding_layer: str,
    output_path: str | Path = None,
) -> None:
    """Render a row-normalized confusion matrix as a heatmap.

    Cells are annotated with the row-normalized proportion and the raw count.
    Tick labels resolve through ``style_cfg["labels"]``, falling back to the
    raw class ID. Auto-dispatched from ``analysis.run_analysis`` after each
    model — not registered in ``PLOTS``, since it is a fixed artifact rather
    than a YAML-configurable step.
    """
    label_map = {str(k): v for k, v in style_cfg.get("labels", {}).items()}

    labels_str = np.asarray(labels).astype(str)
    predictions_str = np.asarray(predictions).astype(str)
    classes = sorted(set(labels_str.tolist()) | set(predictions_str.tolist()))

    cm = sklearn_confusion_matrix(labels_str, predictions_str, labels=classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    tick_labels = [label_map.get(c, c) for c in classes]
    n = len(classes)

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    fig.suptitle(
        f"{experiment_name} — {model_type.upper()} Confusion Matrix",
        fontsize=14,
    )
    ax.set_title(strategy_title, fontsize=11)
    fig.text(
        0.5,
        0.01,
        "Embeddings from the model's final transformer layer.",
        ha="center",
        fontsize=8,
        color="gray",
    )

    for i in range(n):
        for j in range(n):
            value = cm_norm[i, j]
            count = int(cm[i, j])
            text_color = "white" if value > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}\n({count})",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    fig.colorbar(im, ax=ax, label="Proportion (row-normalized)")
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


PLOTS: dict[str, callable] = {
    "scatter_2d": scatter_2d,
    "temporal_cosine_similarity": temporal_cosine_similarity,
}
