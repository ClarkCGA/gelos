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
    model_title: str,
    extraction_strategy: str,
    embedding_layer: str,
    legend_patches: list[Patch],
    category_column: Any,
    color_dict: dict[Any, str],
    chip_indices: list[int],
    axis_lim: int = 90,
    output_dir: str | Path = None,
    legend_loc: str = "upper left",
) -> None:
    """
    plot a tSNE transform of embeddings colored according to land cover
    """
    chip_gdf["color"] = chip_gdf[category_column].map(color_dict)
    colors = chip_gdf.loc[chip_indices]["color"]

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_tsne[:, 1], -embeddings_tsne[:, 0], c=colors, s=2)
    plt.suptitle(f"t-SNE Visualization of Embeddings for {model_title}", fontsize=14)
    plt.title(extraction_strategy)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    if axis_lim:
        plt.xlim([-axis_lim, axis_lim])
        plt.ylim([-axis_lim, axis_lim])
    plt.legend(handles=legend_patches, loc=legend_loc, fontsize=10, framealpha=0.9)

    if output_dir:
        model_title_lower = model_title.replace(" ", "").lower()
        extraction_strategy_lower = extraction_strategy.replace(" ", "").lower()
        embedding_layer_lower = embedding_layer.replace("_", "").lower()
        figure_path = (
            output_dir
            / f"{model_title_lower}_{extraction_strategy_lower}_{embedding_layer_lower}_tsneplot.png"
        )
        plt.savefig(figure_path, dpi=600, bbox_inches="tight")
    else:
        plt.show()
