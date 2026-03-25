from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
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
    plt.suptitle(f"{transform_title} Visualization of Embeddings for {experiment_name} Layer {embedding_layer}", fontsize=14)
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


PLOTS: dict[str, callable] = {
    "scatter_2d": scatter_2d,
}
