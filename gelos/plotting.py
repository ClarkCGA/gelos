from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
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
    colors = chip_gdf[category_column].loc[chip_indices].map(color_dict)
    transform_title = TRANSFORM_TITLES[t_type]

    fig = plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 1], -embeddings[:, 0], c=colors, s=2)
    plt.suptitle(
        f"{transform_title} Visualization of Embeddings for {experiment_name} Layer {embedding_layer}",
        fontsize=14,
    )
    plt.title(strategy_title)
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
    color_dict = style_cfg["colors"]
    legend_patches = [
        Patch(color=color, label=style_cfg["labels"][k]) for k, color in color_dict.items()
    ]
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
    ylim: tuple[float, float] = (0,1)
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
        f"{experiment_name} Embedding Trajectories by Land Cover Category "
        f"(Layer {embedding_layer}, {strategy_title})",
        fontsize=16,
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


PLOTS: dict[str, callable] = {
    "scatter_2d": scatter_2d,
    "temporal_cosine_similarity": temporal_cosine_similarity,
}
