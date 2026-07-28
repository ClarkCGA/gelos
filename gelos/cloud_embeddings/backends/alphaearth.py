from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
from pathlib import Path
import threading

from loguru import logger
import numpy as np
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
import rioxarray  # noqa: F401 — registers .rio accessor on xarray objects
import xarray as xr

from gelos.cloud_embeddings.backends.base import FetchRequest

DEFAULT_MOSAIC_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/"


def _require_zarr() -> None:
    """Fail fast at __init__ with an actionable error when 'zarr' is missing.

    Without this, ``xr.open_zarr`` raises a generic backend-resolution error at
    first fetch — well past the point where the user can tell what's wrong.
    """
    if importlib.util.find_spec("zarr") is None:
        raise ImportError(
            "AlphaEarthBackend requires 'zarr'. Install with: pip install 'gelos[alphaearth]'"
        )


def _require_s3fs() -> None:
    """Fail at dataset-open time when reading s3:// without 's3fs' installed."""
    if importlib.util.find_spec("s3fs") is None:
        raise ImportError(
            "Reading AlphaEarth from s3:// requires 's3fs'. Install with: "
            "pip install 'gelos[alphaearth]'"
        )


class AlphaEarthBackend:
    """Fetches AlphaEarth (AEF) 64-band annual embeddings from the GeoZarr mosaic.

    Reads from the TGE Labs single-mosaic GeoZarr v3 store at
    ``s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic/``
    (shape ``(time=9 [2017–2025], band=64, y, x)``, EPSG:4326, int8 quantized,
    nodata ``-128``). Per ``fetch`` call:

    1. cache lookup keyed on ``(crs, year, bbox, out_shape)`` — short-circuits S3 if present,
    2. reproject the requested bbox to WGS84,
    3. lazy-isel the year slab and ``.sel`` the bbox along x/y — padded by one
       source-mosaic pixel per side when the request will be reprojected, so
       nearest-neighbor resampling near the footprint edge always finds a real
       source pixel instead of falling off the clipped slab into nodata,
    4. write the source CRS and nodata onto the slab via rioxarray,
    5. reproject to the requested ``crs`` (and optionally to ``out_shape``) with
       nearest-neighbor to preserve raw embedding values; the output grid is
       pinned to the request's original bbox (``from_bounds`` when ``out_shape``
       is given, ``clip_box`` otherwise) so the padding never widens the output
       footprint,
    6. persist the clipped + reprojected raster into the cache so re-runs and
       aggregation experiments skip the network entirely,
    7. dequantize int8 → float32 with ``-128`` mapped to NaN.

    Cache key format is ``"mosaic_{crs}_{year}_{minx:.3f}_{miny:.3f}_{maxx:.3f}_"``
    ``"{maxy:.3f}_{8-char-sha256}.tif"``. The ``mosaic_`` prefix namespaces away
    from any pre-existing COG-era cache. The hash includes the full-precision
    bbox tuple plus ``out_shape`` and a ``pad1`` version token (bumped when the
    edge-padding fix landed, so pre-fix caches with false edge nodata are
    orphaned rather than silently reused) — two distinct footprints, output
    shapes, or extraction versions can never collide. Invalidate the cache by
    deleting the directory.

    :meth:`fetch_batch` inverts the conventional chip-first loop: it iterates
    over the unique ``(year, cy, cx)`` zarr chunks touched by any uncached
    request, fetches each chunk exactly once through a bounded thread pool,
    and streams per-request extracts onto a second pool as soon as all chunks
    they depend on are resident. The single-request :meth:`fetch` is a thin
    wrapper. ``prefetch_workers`` caps in-flight S3 fetches; ``extract_workers``
    caps concurrent GDAL reproject/cache writes. Peak resident chunks is
    bounded by ``prefetch_workers + extract_workers`` (≈ ``(N) × 64 MB`` for
    the production mosaic). When ``chunk_cache_dir`` is set (and the URL is
    an fsspec protocol like ``s3://``), the dataset is opened with the fsspec
    ``simplecache::`` wrapper so every Zarr chunk file is cached to local
    disk on first read — invalidate by deleting that dir.
    """

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
        _require_zarr()
        self.url = url
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.chunk_cache_dir = Path(chunk_cache_dir) if chunk_cache_dir is not None else None
        self.anon = anon
        self.variable = variable
        self.nodata = nodata
        self.year_origin = year_origin
        self.prefetch_workers = prefetch_workers
        self.extract_workers = extract_workers
        self._dataset: xr.Dataset | None = None
        self._da: xr.DataArray | None = None
        self._y_descending: bool | None = None
        self._chunk_layout: tuple[int, int, float, float, float, float] | None = None

    @property
    def dataset(self) -> xr.Dataset:
        if self._dataset is None:
            logger.info(f"opening AlphaEarth Zarr mosaic at {self.url}")
            url = str(self.url)
            is_fsspec = "://" in url

            open_kwargs: dict = {"decode_coords": "all"}
            if is_fsspec:
                inner_protocol = "s3" if url.startswith("s3://") else url.split("://", 1)[0]
                if inner_protocol == "s3":
                    _require_s3fs()
                inner_options: dict = {}
                if inner_protocol == "s3":
                    inner_options["anon"] = self.anon

                if self.chunk_cache_dir is not None:
                    # fsspec protocol chaining: simplecache::s3://... caches each
                    # Zarr chunk file to local disk on first read.
                    self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
                    effective_url = f"simplecache::{url}"
                    open_kwargs["storage_options"] = {
                        "simplecache": {"cache_storage": str(self.chunk_cache_dir)},
                        inner_protocol: inner_options,
                    }
                else:
                    effective_url = url
                    open_kwargs["storage_options"] = inner_options
            else:
                effective_url = url

            try:
                self._dataset = xr.open_zarr(effective_url, consolidated=True, **open_kwargs)
            except (KeyError, ValueError, OSError) as exc:
                logger.info(
                    f"consolidated metadata unavailable "
                    f"({type(exc).__name__}: {exc}); opening with consolidated=False"
                )
                self._dataset = xr.open_zarr(effective_url, consolidated=False, **open_kwargs)
            dim_summary = ", ".join(f"{k}={v}" for k, v in self._dataset.sizes.items())
            logger.info(
                f"opened mosaic — vars={list(self._dataset.data_vars)}, dims=({dim_summary})"
            )
        return self._dataset

    @property
    def data_array(self) -> xr.DataArray:
        if self._da is None:
            ds = self.dataset
            if self.variable is not None:
                if self.variable not in ds.data_vars:
                    raise ValueError(
                        f"AlphaEarth mosaic at {self.url} has no data variable "
                        f"{self.variable!r}; available: {sorted(ds.data_vars)}"
                    )
                da = ds[self.variable]
            else:
                names = list(ds.data_vars)
                if len(names) != 1:
                    raise ValueError(
                        f"AlphaEarth mosaic at {self.url} has {len(names)} data "
                        f"variables {names}; pass variable=<name> to disambiguate."
                    )
                da = ds[names[0]]
            self._da = da
            self._y_descending = float(da.y[0]) > float(da.y[-1])
        return self._da

    def _ensure_chunk_layout(self) -> None:
        if self._chunk_layout is not None:
            return
        da = self.data_array
        chunks = da.chunks
        if chunks is None:
            encoding_chunks = da.encoding.get("chunks")
            if encoding_chunks is None:
                raise RuntimeError(
                    "Zarr chunk shape unavailable: da.chunks is None and "
                    "da.encoding has no 'chunks' key — cannot batch by chunk"
                )
            y_chunk_px = int(encoding_chunks[-2])
            x_chunk_px = int(encoding_chunks[-1])
        else:
            y_chunk_px = int(chunks[-2][0])
            x_chunk_px = int(chunks[-1][0])
        y_step = float(da.y.values[1]) - float(da.y.values[0])
        x_step = float(da.x.values[1]) - float(da.x.values[0])
        y_origin = float(da.y.values[0])
        x_origin = float(da.x.values[0])
        self._chunk_layout = (y_chunk_px, x_chunk_px, y_origin, x_origin, y_step, x_step)

    def _chunks_intersecting_bbox(
        self, bbox_wgs84: tuple[float, float, float, float]
    ) -> set[tuple[int, int]]:
        """Return the set of ``(cy, cx)`` zarr-chunk indices the bbox touches.

        Clamps to the valid chunk-grid range so out-of-range bboxes return ∅
        — :meth:`fetch_batch` uses that empty set to raise the
        ``does not intersect`` ValueError without ever calling
        :meth:`_load_chunk` on invalid coordinates.
        """
        self._ensure_chunk_layout()
        y_chunk_px, x_chunk_px, y_origin, x_origin, y_step, x_step = self._chunk_layout
        da = self.data_array
        n_y = int(da.sizes["y"])
        n_x = int(da.sizes["x"])
        n_cy = (n_y + y_chunk_px - 1) // y_chunk_px
        n_cx = (n_x + x_chunk_px - 1) // x_chunk_px

        minx, miny, maxx, maxy = bbox_wgs84
        px_x_lo = int(np.floor((minx - x_origin) / x_step))
        px_x_hi = int(np.floor((maxx - x_origin) / x_step))
        px_y_lo = int(np.floor((miny - y_origin) / y_step))
        px_y_hi = int(np.floor((maxy - y_origin) / y_step))

        cx_lo, cx_hi = sorted((px_x_lo // x_chunk_px, px_x_hi // x_chunk_px))
        cy_lo, cy_hi = sorted((px_y_lo // y_chunk_px, px_y_hi // y_chunk_px))
        cx_lo = max(cx_lo, 0)
        cy_lo = max(cy_lo, 0)
        cx_hi = min(cx_hi, n_cx - 1)
        cy_hi = min(cy_hi, n_cy - 1)
        if cx_lo > cx_hi or cy_lo > cy_hi:
            return set()
        return {(cy, cx) for cy in range(cy_lo, cy_hi + 1) for cx in range(cx_lo, cx_hi + 1)}

    def _load_chunk(self, year_idx: int, cy: int, cx: int) -> xr.DataArray:
        """Load one zarr chunk at fixed pixel bounds and materialize into numpy.

        Returns a numpy-backed DataArray carrying its own y/x coords so
        :func:`xr.combine_by_coords` can stitch straddling neighbors downstream.
        Uses positional ``isel`` (not coordinate ``sel``) so y-axis direction
        doesn't matter at this layer.
        """
        self._ensure_chunk_layout()
        y_chunk_px, x_chunk_px, _y0, _x0, _ys, _xs = self._chunk_layout
        da = self.data_array
        n_y = int(da.sizes["y"])
        n_x = int(da.sizes["x"])
        y_lo = cy * y_chunk_px
        y_hi = min(y_lo + y_chunk_px, n_y)
        x_lo = cx * x_chunk_px
        x_hi = min(x_lo + x_chunk_px, n_x)
        chunk = da.isel(time=year_idx, y=slice(y_lo, y_hi), x=slice(x_lo, x_hi))
        chunk = chunk.rio.write_crs("EPSG:4326").rio.write_nodata(self.nodata)
        return chunk.load()

    def _cache_key(
        self,
        crs: str,
        year: int,
        bbox: tuple[float, float, float, float],
        out_shape: tuple[int, int] | None = None,
    ) -> str:
        minx, miny, maxx, maxy = bbox
        # "pad1" versions the extraction algorithm (1-pixel edge padding +
        # bbox-pinned output grid): bumping it orphans pre-fix cache entries
        # whose edges hold false nodata, instead of silently reusing them.
        raw = f"{crs}|{year}|{minx!r}|{miny!r}|{maxx!r}|{maxy!r}|{out_shape!r}|pad1"
        short_hash = hashlib.sha256(raw.encode()).hexdigest()[:8]
        safe_crs = crs.replace(":", "").replace("/", "_")
        return (
            f"mosaic_{safe_crs}_{year}_{minx:.3f}_{miny:.3f}_{maxx:.3f}_{maxy:.3f}_"
            f"{short_hash}.tif"
        )

    def _year_to_index(self, year: int) -> int:
        idx = year - self.year_origin
        n_times = int(self.data_array.sizes["time"])
        if not 0 <= idx < n_times:
            available = [self.year_origin + i for i in range(n_times)]
            raise ValueError(
                f"AlphaEarth mosaic: year={year} out of range; available years: {available}"
            )
        return idx

    def _read_cache(self, cache_path: Path) -> np.ndarray:
        cached = rioxarray.open_rasterio(cache_path, masked=True)
        try:
            arr = cached.to_numpy()
        finally:
            cached.close()
        return np.transpose(arr, (1, 2, 0))

    def _select_bbox(
        self, slab: xr.DataArray, bbox_wgs84: tuple[float, float, float, float]
    ) -> xr.DataArray:
        minx, miny, maxx, maxy = bbox_wgs84
        slab = slab.sel(x=slice(minx, maxx))
        if self._y_descending:
            slab = slab.sel(y=slice(maxy, miny))
        else:
            slab = slab.sel(y=slice(miny, maxy))
        return slab

    def _pad_bbox_wgs84(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Expand a WGS84 bbox by one source-mosaic pixel step per side.

        :meth:`_select_bbox` keeps only source pixels whose *centers* fall
        inside the bbox, so an unpadded selection starves nearest-neighbor
        reprojection at the footprint edge: a target pixel whose center lies
        inside the request bbox can have its nearest source-pixel center just
        *outside* the clipped slab, and rasterio then fills it with nodata.
        One pixel of padding suffices for ``Resampling.nearest`` — any target
        pixel center inside the bbox has its nearest source-pixel center
        within half a source pixel of the bbox edge. Over-padding past the
        mosaic edge is harmless: ``.sel`` clips to the coordinate range,
        :meth:`_chunks_intersecting_bbox` clamps chunk indices, and genuinely
        missing mosaic data stays nodata.
        """
        self._ensure_chunk_layout()
        _y_chunk_px, _x_chunk_px, _y0, _x0, y_step, x_step = self._chunk_layout
        minx, miny, maxx, maxy = bbox
        return (
            minx - abs(x_step),
            miny - abs(y_step),
            maxx + abs(x_step),
            maxy + abs(y_step),
        )

    def fetch(
        self,
        bbox: tuple[float, float, float, float],
        crs: str,
        year: int,
        out_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        return self.fetch_batch([FetchRequest(bbox, crs, year, out_shape)])[0]

    def _extract_request(
        self,
        i: int,
        req: FetchRequest,
        bbox_wgs84: tuple[float, float, float, float],
        needed: dict[tuple[int, int, int], xr.DataArray],
        cache_path: Path | None,
    ) -> tuple[int, np.ndarray]:
        """Extract one request from its already-loaded chunk DataArray(s).

        **Refcount-snapshot safety contract.** ``needed`` is a dict of chunk
        DataArrays the orchestrator constructs *before* it decrements
        refcounts on those chunks. Its values are local Python references to
        the underlying numpy buffers; this is what keeps those buffers alive
        even after the orchestrator's outer ``loaded`` dict evicts the
        entries (because their refcount hit zero). Callers MUST construct
        ``needed`` before decrementing — otherwise this is a use-after-evict.
        The contract is documented here, not buried in a one-liner, so a
        future refactor of the orchestrator cannot trivially break it.

        ``needed`` keys are ``(year_idx, cy, cx)`` triples; only one ``year_idx``
        appears per call because chunks_per_req for any single request is
        built against one year.
        """
        if len(needed) == 1:
            slab = next(iter(needed.values()))
        else:
            combined = xr.combine_by_coords(list(needed.values()))
            # combine_by_coords returns a Dataset (not DataArray) on current
            # xarray when inputs carry a `.name`; coerce back to the single
            # named DataArray so rioxarray's nodata accessor is available.
            if isinstance(combined, xr.Dataset):
                var_name = next(iter(combined.data_vars))
                combined = combined[var_name]
            slab = combined.rio.write_crs("EPSG:4326").rio.write_nodata(self.nodata)

        sub = self._select_bbox(slab, bbox_wgs84)
        if sub.size == 0:
            raise ValueError(
                f"AlphaEarth mosaic: bbox {bbox_wgs84} does not intersect data for year={req.year}"
            )

        if req.crs != "EPSG:4326" or req.out_shape is not None:
            # ``bbox_wgs84`` arrives padded by one source pixel (see
            # _pad_bbox_wgs84), so pin the output grid to the request's
            # ORIGINAL bbox — the pad feeds the resampler without widening
            # the output footprint.
            if req.out_shape is not None:
                sub = sub.rio.reproject(
                    req.crs,
                    resampling=Resampling.nearest,
                    nodata=self.nodata,
                    shape=req.out_shape,
                    transform=from_bounds(
                        *req.bbox, width=req.out_shape[1], height=req.out_shape[0]
                    ),
                )
            else:
                sub = sub.rio.reproject(
                    req.crs,
                    resampling=Resampling.nearest,
                    nodata=self.nodata,
                )
                # Mean-pool path: trim the pad ring so it can't leak into
                # pooled statistics or the cache.
                sub = sub.rio.clip_box(*req.bbox)

        if cache_path is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            sub.rio.to_raster(cache_path)

        arr = sub.to_numpy().astype("float32")
        arr[arr == self.nodata] = np.nan
        return (i, np.transpose(arr, (1, 2, 0)))

    def fetch_batch(self, requests: list[FetchRequest]) -> list[np.ndarray]:
        """Fetch many bboxes via chunk-streaming inversion to maximize parallelism.

        Cache hits short-circuit the Zarr open entirely, matching the
        single-request :meth:`fetch` contract. For uncached requests, the
        loop is inverted to iterate over unique ``(year, cy, cx)`` zarr
        chunks: each chunk is fetched exactly once through a bounded thread
        pool (``prefetch_workers``), and each request is dispatched onto a
        second pool (``extract_workers``) as soon as every chunk it depends
        on is resident. Refcount eviction frees a chunk once no remaining
        request needs it. Output order matches input order.

        Single-uncached batches (e.g. notebook callers of :meth:`fetch`) take
        a short-circuit fast path that skips the inversion machinery.
        """
        if not requests:
            return []

        n = len(requests)
        results: list[np.ndarray | None] = [None] * n
        cache_paths: list[Path | None] = [None] * n
        uncached: list[int] = []

        for i, req in enumerate(requests):
            if self.cache_dir is not None:
                cache_paths[i] = self.cache_dir / self._cache_key(
                    req.crs, req.year, req.bbox, req.out_shape
                )
                if cache_paths[i].exists():
                    logger.debug(f"AlphaEarth cache hit: {cache_paths[i].name}")
                    continue
            uncached.append(i)

        if uncached:
            logger.info(
                f"AlphaEarth fetch_batch: {n} requests "
                f"({len(uncached)} uncached, {n - len(uncached)} cached)"
            )

            bboxes_wgs84: dict[int, tuple[float, float, float, float]] = {}
            sel_bboxes: dict[int, tuple[float, float, float, float]] = {}
            for i in uncached:
                req = requests[i]
                bboxes_wgs84[i] = (
                    req.bbox
                    if req.crs == "EPSG:4326"
                    else tuple(transform_bounds(req.crs, "EPSG:4326", *req.bbox))
                )
                # Pad the source selection when the request will be
                # reprojected (see _pad_bbox_wgs84). The padded bbox drives
                # BOTH chunk intersection and slab slicing — otherwise a bbox
                # flush against a zarr-chunk boundary would pad into a chunk
                # that was never loaded and _select_bbox would silently clip
                # the pad away. Pure-WGS84 pass-through stays unpadded so its
                # output shape is unchanged.
                needs_reproject = req.crs != "EPSG:4326" or req.out_shape is not None
                sel_bboxes[i] = (
                    self._pad_bbox_wgs84(bboxes_wgs84[i])
                    if needs_reproject
                    else bboxes_wgs84[i]
                )

            if len(uncached) == 1:
                # Notebook fast path: skip pools + inversion bookkeeping. fetch()
                # routes through fetch_batch([req]), so this overhead is paid
                # on every single-bbox call when the inversion can't amortize.
                i = uncached[0]
                req = requests[i]
                year_idx = self._year_to_index(req.year)
                chunk_pairs = self._chunks_intersecting_bbox(sel_bboxes[i])
                if not chunk_pairs:
                    raise ValueError(
                        f"AlphaEarth mosaic: bbox {bboxes_wgs84[i]} "
                        f"does not intersect data for year={req.year}"
                    )
                needed = {
                    (year_idx, cy, cx): self._load_chunk(year_idx, cy, cx)
                    for (cy, cx) in sorted(chunk_pairs)
                }
                _, arr = self._extract_request(i, req, sel_bboxes[i], needed, cache_paths[i])
                results[i] = arr
            else:
                chunks_per_req: dict[int, set[tuple[int, int, int]]] = {}
                chunk_to_reqs: dict[tuple[int, int, int], list[int]] = {}
                for i in uncached:
                    year_idx = self._year_to_index(requests[i].year)
                    chunk_pairs = self._chunks_intersecting_bbox(sel_bboxes[i])
                    if not chunk_pairs:
                        raise ValueError(
                            f"AlphaEarth mosaic: bbox {bboxes_wgs84[i]} "
                            f"does not intersect data for year={requests[i].year}"
                        )
                    chunk_keys = {(year_idx, cy, cx) for (cy, cx) in chunk_pairs}
                    chunks_per_req[i] = chunk_keys
                    for ck in chunk_keys:
                        chunk_to_reqs.setdefault(ck, []).append(i)

                refcount: dict[tuple[int, int, int], int] = {
                    ck: len(reqs) for ck, reqs in chunk_to_reqs.items()
                }
                chunk_order = sorted(chunk_to_reqs.keys())

                loaded: dict[tuple[int, int, int], xr.DataArray] = {}
                completed: set[int] = set()
                lock = threading.Lock()
                pending_extract: list = []

                with (
                    ThreadPoolExecutor(max_workers=self.prefetch_workers) as fetch_pool,
                    ThreadPoolExecutor(max_workers=self.extract_workers) as extract_pool,
                ):
                    future_to_ck = {
                        fetch_pool.submit(self._load_chunk, *ck): ck for ck in chunk_order
                    }
                    for fut in as_completed(future_to_ck):
                        ck = future_to_ck[fut]
                        chunk_arr = fut.result()
                        with lock:
                            loaded[ck] = chunk_arr
                            newly_ready = [
                                j
                                for j in chunk_to_reqs[ck]
                                if j not in completed and chunks_per_req[j] <= loaded.keys()
                            ]
                            # Snapshot chunk refs BEFORE decrementing refcounts:
                            # see _extract_request docstring on the safety contract.
                            for j in newly_ready:
                                completed.add(j)
                                needed = {k: loaded[k] for k in chunks_per_req[j]}
                                pending_extract.append(
                                    extract_pool.submit(
                                        self._extract_request,
                                        j,
                                        requests[j],
                                        sel_bboxes[j],
                                        needed,
                                        cache_paths[j],
                                    )
                                )
                            for j in newly_ready:
                                for k in chunks_per_req[j]:
                                    refcount[k] -= 1
                                    if refcount[k] == 0:
                                        loaded.pop(k, None)
                    for fut in pending_extract:
                        j, arr = fut.result()
                        results[j] = arr

        for i in range(n):
            if results[i] is None:
                results[i] = self._read_cache(cache_paths[i])

        return results
