from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np


def mean_pool(footprint_raster: np.ndarray) -> np.ndarray:
    """Reduce a ``[H, W, C]`` footprint raster to a ``[C]`` vector via NaN-aware mean."""
    if footprint_raster.ndim != 3:
        raise ValueError(
            f"footprint_raster must be 3D [H, W, C], got shape {footprint_raster.shape}"
        )
    return np.nanmean(footprint_raster, axis=(0, 1))


def flatten_pool(footprint_raster: np.ndarray) -> np.ndarray:
    """Reduce ``[H, W, C]`` to a 1-D ``[H*W*C]`` vector in row-major order."""
    if footprint_raster.ndim != 3:
        raise ValueError(
            f"footprint_raster must be 3D [H, W, C], got shape {footprint_raster.shape}"
        )
    return footprint_raster.reshape(-1)


POOLS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "mean": mean_pool,
    "flatten": flatten_pool,
}


def write_chip_parquet(
    chip_filename: str,
    file_id: Any,
    patch_vectors: np.ndarray,
    out_dir: Path,
) -> Path:
    """Write a chip's stacked patch vectors as ``{stem}_embedding.parquet``.

    Schema mirrors :func:`gelos.raw_pixels.RawPixelEmbeddingTask.write_parquet`:
    ``{"embedding": list[list[float]], "file_id": int}``.

    Args:
        chip_filename: Chip identifier; ``Path(chip_filename).stem`` is used as the basename.
        file_id: Stored as a metadata column; pass ``None`` to omit.
        patch_vectors: ``(n_patches, C)`` array.
        out_dir: Destination directory (created if missing).

    Returns:
        The output Parquet path.
    """
    if patch_vectors.ndim != 2:
        raise ValueError(
            f"patch_vectors must be 2D (n_patches, C), got shape {patch_vectors.shape}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(chip_filename).stem
    out_path = out_dir / f"{stem}_embedding.parquet"

    row: dict[str, Any] = {"embedding": patch_vectors.tolist()}
    if file_id is not None:
        row["file_id"] = file_id

    df = gpd.GeoDataFrame([row])
    df.to_parquet(out_path, index=False)
    return out_path
