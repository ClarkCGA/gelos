from pathlib import Path

from gelos.cloud_embeddings.backends.zarr_mosaic import ZarrMosaicBackend

DEFAULT_MOSAIC_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/"


class AlphaEarthBackend(ZarrMosaicBackend):
    """Fetches AlphaEarth (AEF) 64-band annual embeddings from the GeoZarr mosaic.

    Reads from the TGE Labs single-mosaic GeoZarr v3 store at
    ``s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/``
    (shape ``(time=9 [2017–2025], band=64, y, x)``, EPSG:4326, int8 quantized,
    nodata ``-128``). All fetch machinery — cache lookup, bbox reprojection to
    WGS84, one-pixel edge padding, nearest-neighbor reprojection pinned to the
    request bbox, int8 cache files, chunk-streaming :meth:`fetch_batch` — is
    inherited from :class:`~gelos.cloud_embeddings.backends.zarr_mosaic.ZarrMosaicBackend`;
    see its docstring for the per-request pipeline. Values are returned as
    stored (float32 with ``-128`` mapped to NaN; no dequantization).

    Cache key format is ``"mosaic_{crs}_{year}_{minx:.3f}_{miny:.3f}_{maxx:.3f}_"``
    ``"{maxy:.3f}_{8-char-sha256}.tif"``. The ``mosaic_`` prefix namespaces away
    from any pre-existing COG-era cache. The hash includes the full-precision
    bbox tuple plus ``out_shape`` and a ``pad1`` version token (bumped when the
    edge-padding fix landed, so pre-fix caches with false edge nodata are
    orphaned rather than silently reused) — two distinct footprints, output
    shapes, or extraction versions can never collide. Invalidate the cache by
    deleting the directory.

    Years map onto the ``time`` axis arithmetically from ``year_origin``
    (``idx = year - year_origin``), matching the mosaic's contiguous annual
    cadence. Peak resident chunks under :meth:`fetch_batch` is
    ``prefetch_workers + extract_workers`` (≈ ``N × 64 MB`` for the production
    mosaic).
    """

    LABEL = "AlphaEarth"
    CACHE_PREFIX = "mosaic"
    CACHE_VERSION = "pad1"
    MISSING_ZARR_HINT = "Install with: pip install 'gelos[alphaearth]'"

    def __init__(
        self,
        url: str = DEFAULT_MOSAIC_URL,
        cache_dir: str | Path | None = None,
        chunk_cache_dir: str | Path | None = None,
        anon: bool = True,
        variable: str | None = None,
        nodata: int = -128,
        year_origin: int = 2017,
        prefetch_workers: int = 16,
        extract_workers: int = 8,
    ) -> None:
        super().__init__(
            url,
            cache_dir=cache_dir,
            chunk_cache_dir=chunk_cache_dir,
            anon=anon,
            variable=variable,
            nodata=nodata,
            prefetch_workers=prefetch_workers,
            extract_workers=extract_workers,
            source_crs="EPSG:4326",
        )
        self.year_origin = year_origin

    def _year_to_index(self, year: int) -> int:
        idx = year - self.year_origin
        n_times = int(self.data_array.sizes["time"])
        if not 0 <= idx < n_times:
            available = [self.year_origin + i for i in range(n_times)]
            raise ValueError(
                f"AlphaEarth mosaic: year={year} out of range; available years: {available}"
            )
        return idx
