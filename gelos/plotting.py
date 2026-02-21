from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np


def format_lat_lon(lat: float, lon: float) -> str:
    """Return formatted coords like 12.34°N, 56.78°E."""
    lat_hemisphere = "N" if lat >= 0 else "S"
    lon_hemisphere = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}°{lat_hemisphere}, {abs(lon):.2f}°{lon_hemisphere}"


def plot_from_tsne(
    embeddings_tsne: np.array,
    chip_gdf: gpd.GeoDataFrame,
    experiment_name: str,
    strategy_title: str,
    legend_patches: list[Patch],
    category_column: Any,
    color_dict: dict[Any, str],
    chip_indices: list[int],
    axis_lim: int = 90,
    output_path: str | Path = None,
    legend_loc: str = "upper left",
) -> None:
    """
    plot a tSNE transform of embeddings colored according to land cover
    """
    chip_gdf["color"] = chip_gdf[category_column].map(color_dict)
    colors = chip_gdf.loc[chip_indices]["color"]

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_tsne[:, 1], -embeddings_tsne[:, 0], c=colors, s=2)
    plt.suptitle(f"t-SNE Visualization of Embeddings for {experiment_name}", fontsize=14)
    plt.title(strategy_title)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    if axis_lim:
        plt.xlim([-axis_lim, axis_lim])
        plt.ylim([-axis_lim, axis_lim])
    plt.legend(handles=legend_patches, loc=legend_loc, fontsize=10, framealpha=0.9)

    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
    else:
        plt.show()


def tsne_scatter(
    transformed: np.ndarray,
    chip_gdf: gpd.GeoDataFrame,
    chip_indices: list[int],
    style_cfg: dict,
    output_path: Path,
    experiment_name: str,
    strategy_title: str,
    embedding_layer: str,
    **params: Any,
) -> None:
    """Registry-compatible wrapper around ``plot_from_tsne``.

    Args:
        transformed: t-SNE coordinates of shape (N, 2).
        chip_gdf: GeoDataFrame with chip metadata.
        chip_indices: Indices into chip_gdf for this embedding set.
        style_cfg: Style section from the YAML config.
        output_path: Full path for the output figure.
        experiment_name: Human-readable experiment name for plot title.
        strategy_title: Display title for the extraction strategy.
        embedding_layer: Name of the embedding layer.
        **params: Additional keyword arguments forwarded to ``plot_from_tsne``.
    """
    from gelos.analysis import _build_style_from_config

    category_column, color_dict, legend_patches = _build_style_from_config(style_cfg)
    plot_from_tsne(
        transformed,
        chip_gdf,
        experiment_name,
        strategy_title,
        legend_patches,
        category_column,
        color_dict,
        chip_indices,
        output_path=output_path,
        **params,
    )


PLOTS: dict[str, callable] = {
    "tsne_scatter": tsne_scatter,
}
