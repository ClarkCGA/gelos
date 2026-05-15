from typing import Any

import geopandas as gpd
from loguru import logger
import rioxarray
from shapely.geometry import box
from tqdm import tqdm

from gelos.cloud_embeddings.patches import resolve_patch_indices
from gelos.gelosdataset import GELOSDataSet


def compute_footprint_manifest(
    dataset: GELOSDataSet,
    reference_sensor: str,
    patch_size: int,
    strategy_specs: dict[str, dict[str, Any]],
) -> gpd.GeoDataFrame:
    """Build a per-chip per-patch footprint manifest from the reference-sensor raster.

    Args:
        dataset: Instantiated GELOS dataset.
        reference_sensor: Sensor whose first-timestep raster defines the patch grid
            (CRS, transform, spatial dimensions).
        patch_size: Side length in pixels for each patch (must divide H and W).
        strategy_specs: Mapping ``strategy_name -> {"patches": <spec>}``; the same
            ``patches`` syntax as :func:`gelos.cloud_embeddings.patches.resolve_patch_indices`.

    Returns:
        A GeoDataFrame with one row per (chip, strategy, patch). Columns:
        ``file_id, chip_index, filename, strategy, patch_row, patch_col,
        bbox_minx, bbox_miny, bbox_maxx, bbox_maxy, crs, geometry``.
        The frame's CRS is set to the chips' common CRS, or ``None`` when chips
        span multiple CRSes (per-row ``crs`` column is always authoritative).
    """
    rows: list[dict] = []
    for index in tqdm(range(len(dataset)), desc="footprint manifest", unit="chip"):
        path = dataset._get_file_paths(index, reference_sensor)[0]
        raster = rioxarray.open_rasterio(path)
        try:
            transform = raster.rio.transform()
            crs = raster.rio.crs
            H = raster.rio.height
            W = raster.rio.width
        finally:
            raster.close()

        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"chip {index} ({path.name}): H ({H}) and W ({W}) must be "
                f"divisible by patch_size ({patch_size})"
            )
        if transform.b != 0 or transform.d != 0 or transform.e >= 0:
            raise ValueError(
                f"chip {index} ({path.name}): non-north-up affine transform "
                f"{transform!r}; require b == 0, d == 0, e < 0"
            )

        px = transform.a
        py = -transform.e
        x0 = transform.c
        y0 = transform.f

        n_rows = H // patch_size
        n_cols = W // patch_size

        filename, file_id = dataset._get_sample_id(index)

        for strategy_name, spec in strategy_specs.items():
            patch_indices = resolve_patch_indices(spec, n_rows, n_cols)
            for r, c in patch_indices:
                bbox_minx = x0 + c * patch_size * px
                bbox_maxx = x0 + (c + 1) * patch_size * px
                bbox_maxy = y0 - r * patch_size * py
                bbox_miny = y0 - (r + 1) * patch_size * py

                rows.append(
                    {
                        "file_id": file_id,
                        "chip_index": index,
                        "filename": str(filename),
                        "strategy": strategy_name,
                        "patch_row": r,
                        "patch_col": c,
                        "bbox_minx": bbox_minx,
                        "bbox_miny": bbox_miny,
                        "bbox_maxx": bbox_maxx,
                        "bbox_maxy": bbox_maxy,
                        "crs": str(crs) if crs is not None else None,
                        "geometry": box(bbox_minx, bbox_miny, bbox_maxx, bbox_maxy),
                    }
                )

    if not rows:
        return gpd.GeoDataFrame(
            columns=[
                "file_id",
                "chip_index",
                "filename",
                "strategy",
                "patch_row",
                "patch_col",
                "bbox_minx",
                "bbox_miny",
                "bbox_maxx",
                "bbox_maxy",
                "crs",
                "geometry",
            ],
            geometry="geometry",
        )

    crs_set = {r["crs"] for r in rows if r["crs"] is not None}
    if len(crs_set) == 1:
        frame_crs = next(iter(crs_set))
    else:
        logger.warning(
            f"footprint manifest contains {len(crs_set)} distinct chip CRSes; "
            "GeoDataFrame.crs set to None — use the per-row 'crs' column"
        )
        frame_crs = None

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=frame_crs)
