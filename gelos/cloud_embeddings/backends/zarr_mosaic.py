from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
from pathlib import Path
import random
import threading
import time

from loguru import logger
import numpy as np
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
import rioxarray  # also registers the .rio accessor on xarray objects
import xarray as xr

from gelos.cloud_embeddings.backends.base import FetchRequest


def _require_zarr(label: str, hint: str) -> None:
    """Fail fast at __init__ with an actionable error when 'zarr' is missing.

    Without this, ``xr.open_zarr`` raises a generic backend-resolution error at
    first fetch — well past the point where the user can tell what's wrong.
    """
    if importlib.util.find_spec("zarr") is None:
        raise ImportError(f"{label} backend requires 'zarr'. {hint}")


def _require_s3fs(label: str, hint: str) -> None:
    """Fail at dataset-open time when reading s3:// without 's3fs' installed."""
    if importlib.util.find_spec("s3fs") is None:
        raise ImportError(f"Reading {label} from s3:// requires 's3fs'. {hint}")


def _require_aiohttp(label: str, hint: str) -> None:
    """Fail at dataset-open time when reading http(s):// without 'aiohttp' installed.

    fsspec's HTTP filesystem is implemented on top of aiohttp, which is only a
    transitive dependency of the project.
    """
    if importlib.util.find_spec("aiohttp") is None:
        raise ImportError(f"Reading {label} over http(s):// requires 'aiohttp'. {hint}")


# Indirection so tests can stub the backoff sleep without touching ``time``.
_sleep = time.sleep


def _is_transient_error(exc: BaseException) -> bool:
    """Classify an exception as a transient network failure worth retrying.

    Transient: ``ConnectionError`` (``ConnectionResetError`` & co),
    ``TimeoutError`` (also ``asyncio.TimeoutError`` and fsspec's
    ``FSTimeoutError``), aiohttp's ``ClientError`` family — in practice
    ``ClientPayloadError`` ("Response payload is not completed ...
    ConnectionResetError(104)"), which is what data.source.coop produces every
    few minutes under 16 concurrent readers — and an
    ``aiohttp.ClientResponseError`` whose HTTP status is 429 or 5xx (502/503/
    504 from the Source Cooperative proxy; fsspec raises these unwrapped from
    both store opens and chunk reads). Any other ``ClientResponseError``
    (403/404 and the rest of 4xx), ``FileNotFoundError`` (fsspec's 404) and
    ``PermissionError`` (403), plus anything that is not a connection/timeout
    error (``KeyError``, ``ValueError``, ...), are configuration or
    programming errors and are not retried. The ``__cause__``/``__context__``
    chain is walked so a transient error wrapped by zarr/fsspec is still
    recognised.
    """
    try:
        import aiohttp
    except ImportError:  # pragma: no cover - aiohttp is optional
        aiohttp = None

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (FileNotFoundError, PermissionError)):
            return False
        if aiohttp is not None and isinstance(cur, aiohttp.ClientResponseError):
            status = int(getattr(cur, "status", 0) or 0)
            return status == 429 or status >= 500
        if isinstance(cur, (ConnectionError, TimeoutError)):
            return True
        if aiohttp is not None and isinstance(cur, aiohttp.ClientError):
            return True
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    return False


class ZarrMosaicBackend:
    """Fetches footprint rasters from a single-grid ``(time, band, y, x)`` Zarr mosaic.

    This is the shared chunk-streaming machinery behind the concrete cloud
    embedding backends (AlphaEarth, one OlmoEarth UTM zone, ...). A mosaic is
    one Zarr array in one CRS (``source_crs``), with ``y``/``x`` coordinates
    at pixel centers, a ``time`` coordinate of annual slabs, int8 (or other)
    quantized values and a ``nodata`` sentinel. Per ``fetch`` call:

    1. cache lookup keyed on ``(crs, year, bbox, out_shape)`` — short-circuits the
       store if present,
    2. reproject the requested bbox to ``source_crs``,
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
    6. persist the clipped + reprojected raster into the cache (raw stored
       dtype, lossless) so re-runs and aggregation experiments skip the
       network entirely,
    7. cast to float32 with ``nodata`` mapped to NaN and run :meth:`_dequantize`
       (identity here; subclasses undo their store's quantization).

    Cache key format is ``"{CACHE_PREFIX}_{crs}_{year}_{minx:.3f}_{miny:.3f}_"``
    ``"{maxx:.3f}_{maxy:.3f}_{8-char-sha256}.tif"``. The hash includes the
    full-precision bbox tuple plus ``out_shape`` and the ``CACHE_VERSION``
    token (bump it when the extraction algorithm changes so stale entries are
    orphaned rather than silently reused) — two distinct footprints, output
    shapes, or extraction versions can never collide. Invalidate the cache by
    deleting the directory.

    :meth:`fetch_batch` inverts the conventional chip-first loop: it iterates
    over the unique ``(year, cy, cx)`` zarr chunks touched by any uncached
    request, fetches each chunk exactly once through a bounded thread pool,
    and streams per-request extracts onto a second pool as soon as all chunks
    they depend on are resident. The single-request :meth:`fetch` is a thin
    wrapper. ``prefetch_workers`` caps in-flight store fetches; ``extract_workers``
    caps concurrent GDAL reproject/cache writes. Peak resident chunks is
    bounded by ``prefetch_workers + extract_workers``. When ``chunk_cache_dir``
    is set (and the URL is an fsspec protocol like ``s3://``), the dataset is
    opened with the fsspec ``simplecache::`` wrapper so every Zarr chunk file
    is cached to local disk on first read — invalidate by deleting that dir.

    ``use_dask`` (default ``True``) opens the store with xarray's default dask
    chunking. Pass ``False`` to open with ``chunks=None`` — a lazily-indexed
    numpy-backed array — for huge sharded stores where dask's per-read graph
    construction/optimisation (one task per inner chunk across the whole
    mosaic) costs far more than the windowed I/O itself. Only ever touch such
    an array through windowed ``.isel`` + ``.load()``; anything that
    materialises the whole variable (``.data``, ``.values``, ``.compute()``)
    would try to allocate the entire mosaic.

    ``max_retries`` / ``retry_backoff`` bound the retry (:meth:`_with_retries`)
    around every store access: the ``xr.open_zarr`` call in :attr:`dataset`
    (which eagerly reads the coordinate arrays, so a 503 from the proxy can
    surface there before the first chunk read) and each chunk read in
    :meth:`_load_chunk`. A call that fails with a transient network error
    (connection reset, aiohttp client errors, timeouts, HTTP 429/5xx — see
    :func:`_is_transient_error`) is retried up to ``max_retries`` more times,
    sleeping ``retry_backoff * 2**attempt`` seconds (plus up to 10 % jitter)
    between attempts; chunk retries run in the worker thread without holding
    any orchestrator lock. Only the failing chunk is retried; chunks already
    resident and the batch bookkeeping are untouched. Chunk loads never write
    the footprint cache (that happens in :meth:`_extract_request`, after every
    chunk a request needs is loaded), so a failed load cannot leave a partial
    cache entry, and a failed open leaves ``dataset`` unset so the next call
    opens afresh. Non-transient errors (``KeyError``, ``ValueError``, 404/403,
    ...) are raised immediately; after the last attempt the original
    transient error is re-raised unchanged.

    ``pin_native_grid`` (default ``False``) changes what a request *without*
    ``out_shape`` returns. Off, source pixels whose centres fall in the bbox
    are passed through (same-CRS request) or reprojected and ``clip_box``-ed
    (other CRS) — the window is whatever the source grid happens to cover,
    e.g. 17x17 for a 160 m bbox whose edges sit half a source pixel off the
    source grid. On, the output grid is derived from the bbox at the source's
    native resolution (``round(extent / pixel_size)`` per axis) and pinned to
    the bbox with ``from_bounds`` + nearest resampling, exactly like the
    ``out_shape`` path — so a 160 m footprint over a 10 m store is always a
    16x16 window aligned to the footprint, in both the same-CRS and the
    reproject branch. Cache keys still carry the request's own ``out_shape``
    (``None``); bump ``CACHE_VERSION`` when toggling this on an existing cache.

    Subclass hooks: ``LABEL`` (log/error prefix), ``CACHE_PREFIX`` /
    ``CACHE_VERSION`` (cache-key namespace), ``MISSING_ZARR_HINT`` (install
    hint), :meth:`_year_to_index` (year -> ``time`` index) and
    :meth:`_dequantize` (float32-with-NaN -> float32-with-NaN).
    """

    LABEL = "Zarr mosaic"
    CACHE_PREFIX = "mosaic"
    CACHE_VERSION = "pad1"
    MISSING_ZARR_HINT = "Install with: pip install 'gelos[alphaearth]'"

    def __init__(
        self,
        url: str,
        cache_dir: str | Path | None = None,
        chunk_cache_dir: str | Path | None = None,
        anon: bool = True,
        variable: str | None = None,
        nodata: int = -128,
        prefetch_workers: int = 16,
        extract_workers: int = 8,
        *,
        group: str | None = None,
        source_crs: str = "EPSG:4326",
        storage_options: dict | None = None,
        endpoint_url: str | None = None,
        band_slice: tuple[int, int] | None = None,
        prefer_inner_chunks: bool = False,
        use_dask: bool = True,
        max_retries: int = 4,
        retry_backoff: float = 1.0,
        pin_native_grid: bool = False,
    ) -> None:
        _require_zarr(self.LABEL, self.MISSING_ZARR_HINT)
        if int(max_retries) < 0:
            raise ValueError(f"{self.LABEL}: max_retries must be >= 0, got {max_retries!r}")
        if float(retry_backoff) < 0:
            raise ValueError(f"{self.LABEL}: retry_backoff must be >= 0, got {retry_backoff!r}")
        self.url = url
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.chunk_cache_dir = Path(chunk_cache_dir) if chunk_cache_dir is not None else None
        self.anon = anon
        self.variable = variable
        self.nodata = nodata
        self.prefetch_workers = prefetch_workers
        self.extract_workers = extract_workers
        self.group = group
        self.source_crs = source_crs
        self.storage_options = dict(storage_options) if storage_options else None
        self.endpoint_url = endpoint_url
        self.band_slice = tuple(band_slice) if band_slice is not None else None
        self.prefer_inner_chunks = prefer_inner_chunks
        self.use_dask = use_dask
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self.pin_native_grid = bool(pin_native_grid)
        self._dataset: xr.Dataset | None = None
        self._da: xr.DataArray | None = None
        self._y_descending: bool | None = None
        self._chunk_layout: tuple[int, int, float, float, float, float] | None = None

    @property
    def dataset(self) -> xr.Dataset:
        if self._dataset is None:
            group_note = f" (group={self.group!r})" if self.group else ""
            logger.info(f"opening {self.LABEL} Zarr mosaic at {self.url}{group_note}")
            url = str(self.url)
            is_fsspec = "://" in url

            open_kwargs: dict = {"decode_coords": "all"}
            if self.group is not None:
                open_kwargs["group"] = self.group
            if not self.use_dask:
                # No dask: xarray hands back a lazily-indexed backend array, so
                # `_load_chunk`'s windowed isel+load reads straight from the
                # store. With dask, a (9, 128, ~1.9M, ~98k) array at
                # (1, 64, 256, 256) inner chunks has ~50M graph keys, and every
                # window read pays tens of seconds of GIL-bound graph
                # culling/optimisation for ~1 s of actual I/O.
                open_kwargs["chunks"] = None
            if is_fsspec:
                inner_protocol = "s3" if url.startswith("s3://") else url.split("://", 1)[0]
                if inner_protocol == "s3":
                    _require_s3fs(self.LABEL, self.MISSING_ZARR_HINT)
                elif inner_protocol in ("http", "https"):
                    _require_aiohttp(self.LABEL, self.MISSING_ZARR_HINT)
                inner_options: dict = {}
                if inner_protocol == "s3":
                    inner_options["anon"] = self.anon
                    if self.endpoint_url is not None:
                        # Non-AWS S3-compatible hosts (e.g. Source Cooperative).
                        inner_options["client_kwargs"] = {"endpoint_url": self.endpoint_url}
                if self.storage_options:
                    inner_options.update(self.storage_options)

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

            def _open() -> xr.Dataset:
                # Opening reads the metadata plus every dimension coordinate
                # (``time``/``x``/``y`` become in-memory indexes), so this is
                # the first network round trip and the one a transient 5xx
                # from the proxy hits before any chunk read.
                try:
                    return xr.open_zarr(effective_url, consolidated=True, **open_kwargs)
                except (KeyError, ValueError, OSError) as exc:
                    logger.info(
                        f"consolidated metadata unavailable "
                        f"({type(exc).__name__}: {exc}); opening with consolidated=False"
                    )
                    return xr.open_zarr(effective_url, consolidated=False, **open_kwargs)

            # Assign only on success: a failed open must not cache a
            # half-initialised dataset, so the next access opens afresh.
            ds = self._with_retries(_open, f"store open at {url}{group_note}")
            dim_summary = ", ".join(f"{k}={v}" for k, v in ds.sizes.items())
            logger.info(f"opened mosaic — vars={list(ds.data_vars)}, dims=({dim_summary})")
            self._dataset = ds
        return self._dataset

    def _with_retries(self, fn, what: str):
        """Call ``fn()`` and retry it on transient network errors.

        Up to ``max_retries`` extra attempts with exponential backoff from
        ``retry_backoff`` seconds (plus up to 10 % jitter) between them, logged
        as warnings; non-transient errors (see :func:`_is_transient_error`)
        and the last transient error are re-raised unchanged. ``what``
        describes the operation in the log lines (``"... load"``, ``"store
        open at ..."``). No lock is held here, so a sleeping retry never
        blocks other workers.
        """
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:
                if not _is_transient_error(exc):
                    raise
                if attempt >= self.max_retries:
                    logger.error(
                        f"{self.LABEL} {what} failed with transient error "
                        f"{type(exc).__name__}: {exc} — giving up after "
                        f"{attempt + 1} attempt(s) (max_retries={self.max_retries}). "
                        "Raise init_args.max_retries / retry_backoff or lower "
                        "prefetch_workers if this recurs."
                    )
                    raise
                delay = self.retry_backoff * (2**attempt)
                delay += random.uniform(0.0, 0.1 * delay)
                attempt += 1
                logger.warning(
                    f"{self.LABEL} {what} failed with transient error "
                    f"{type(exc).__name__}: {exc}; retry {attempt}/{self.max_retries} "
                    f"in {delay:.1f}s"
                )
                _sleep(delay)

    @property
    def data_array(self) -> xr.DataArray:
        if self._da is None:
            ds = self.dataset
            if self.variable is not None:
                if self.variable not in ds.data_vars:
                    raise ValueError(
                        f"{self.LABEL} mosaic at {self.url} has no data variable "
                        f"{self.variable!r}; available: {sorted(ds.data_vars)}"
                    )
                da = ds[self.variable]
            else:
                names = list(ds.data_vars)
                if len(names) != 1:
                    raise ValueError(
                        f"{self.LABEL} mosaic at {self.url} has {len(names)} data "
                        f"variables {names}; pass variable=<name> to disambiguate."
                    )
                da = ds[names[0]]
            if self.band_slice is not None:
                # Positional: the store may carry a `band` dimension without a
                # coordinate, so coordinate-based `.sel` is not an option.
                da = da.isel(band=slice(*self.band_slice))
            self._da = da
            self._y_descending = float(da.y[0]) > float(da.y[-1])
        return self._da

    def _ensure_chunk_layout(self) -> None:
        if self._chunk_layout is not None:
            return
        da = self.data_array
        encoding_chunks = da.encoding.get("chunks")
        chunks = da.chunks
        if self.prefer_inner_chunks and encoding_chunks is not None:
            # Sharded stores: `da.chunks` (with dask) reports the shard shape,
            # while `encoding["chunks"]` is the inner chunk — the real unit of
            # I/O. Batch on the inner chunk so each load stays small.
            y_chunk_px = int(encoding_chunks[-2])
            x_chunk_px = int(encoding_chunks[-1])
        elif chunks is None:
            # No dask (use_dask=False, or dask not installed): the on-disk
            # chunk shape is only available through the encoding.
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
        self, bbox_src: tuple[float, float, float, float]
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

        minx, miny, maxx, maxy = bbox_src
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
        doesn't matter at this layer. Windowed ``isel`` + ``load`` is the only
        access pattern that is safe for both dask-backed and (``use_dask=False``)
        lazily-indexed arrays — never materialise ``da`` as a whole here.

        The store read is retried on transient network errors through
        :meth:`_with_retries` (see the class docstring and
        :func:`_is_transient_error`). Runs on a prefetch-pool worker thread,
        so the backoff sleep never blocks the orchestrator or other workers.
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

        def _read() -> xr.DataArray:
            chunk = da.isel(time=year_idx, y=slice(y_lo, y_hi), x=slice(x_lo, x_hi))
            chunk = chunk.rio.write_crs(self.source_crs).rio.write_nodata(self.nodata)
            return chunk.load()

        return self._with_retries(_read, f"chunk (time_idx={year_idx}, cy={cy}, cx={cx}) load")

    def _cache_key(
        self,
        crs: str,
        year: int,
        bbox: tuple[float, float, float, float],
        out_shape: tuple[int, int] | None = None,
    ) -> str:
        minx, miny, maxx, maxy = bbox
        # CACHE_VERSION versions the extraction algorithm (e.g. "pad1": 1-pixel
        # edge padding + bbox-pinned output grid): bumping it orphans stale
        # cache entries instead of silently reusing them.
        raw = (
            f"{crs}|{year}|{minx!r}|{miny!r}|{maxx!r}|{maxy!r}|{out_shape!r}|{self.CACHE_VERSION}"
        )
        short_hash = hashlib.sha256(raw.encode()).hexdigest()[:8]
        safe_crs = crs.replace(":", "").replace("/", "_")
        return (
            f"{self.CACHE_PREFIX}_{safe_crs}_{year}_{minx:.3f}_{miny:.3f}_{maxx:.3f}_{maxy:.3f}_"
            f"{short_hash}.tif"
        )

    def _year_to_index(self, year: int) -> int:
        """Map a calendar year onto the ``time`` axis by coordinate lookup."""
        times = self.data_array["time"].values
        matches = np.flatnonzero(times == int(year))
        if matches.size == 0:
            available = sorted(int(t) for t in times)
            raise ValueError(
                f"{self.LABEL}: year={year} out of range; available years: {available}"
            )
        return int(matches[0])

    def _dequantize(self, arr: np.ndarray) -> np.ndarray:
        """Undo the store's quantization on a float32 array that already has NaN at nodata.

        The base mosaic is returned as stored; subclasses override this to map
        quantized codes back to embedding values. Must preserve NaN.
        """
        return arr

    def _read_cache(self, cache_path: Path) -> np.ndarray:
        cached = rioxarray.open_rasterio(cache_path, masked=True)
        try:
            arr = cached.to_numpy()
        finally:
            cached.close()
        arr = np.transpose(arr, (1, 2, 0)).astype("float32", copy=False)
        return self._dequantize(arr)

    def _select_bbox(
        self, slab: xr.DataArray, bbox_src: tuple[float, float, float, float]
    ) -> xr.DataArray:
        minx, miny, maxx, maxy = bbox_src
        slab = slab.sel(x=slice(minx, maxx))
        if self._y_descending:
            slab = slab.sel(y=slice(maxy, miny))
        else:
            slab = slab.sel(y=slice(miny, maxy))
        return slab

    def _pad_bbox_src(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Expand a source-CRS bbox by one source-mosaic pixel step per side.

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

    def _resolve_out_shape(self, req: FetchRequest) -> tuple[int, int] | None:
        """Output grid ``(rows, cols)`` for a request, or ``None`` for source pass-through.

        An explicit ``req.out_shape`` always wins. Without one, a
        ``pin_native_grid`` backend derives the grid from the request bbox
        at the source's native resolution — the bbox is measured in the
        source CRS (exact for a same-zone UTM request, an approximation for
        geographic requests) and divided by the pixel step, rounded to the
        nearest integer and floored at 1 — so the request is resampled onto a
        footprint-aligned grid like the ``out_shape`` path. Otherwise ``None``.
        """
        if req.out_shape is not None:
            return (int(req.out_shape[0]), int(req.out_shape[1]))
        if not self.pin_native_grid:
            return None
        self._ensure_chunk_layout()
        _y_chunk_px, _x_chunk_px, _y0, _x0, y_step, x_step = self._chunk_layout
        if req.crs == self.source_crs:
            minx, miny, maxx, maxy = req.bbox
        else:
            minx, miny, maxx, maxy = transform_bounds(req.crs, self.source_crs, *req.bbox)
        n_cols = max(1, round((maxx - minx) / abs(x_step)))
        n_rows = max(1, round((maxy - miny) / abs(y_step)))
        return (n_rows, n_cols)

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
        bbox_src: tuple[float, float, float, float],
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
            slab = combined.rio.write_crs(self.source_crs).rio.write_nodata(self.nodata)

        sub = self._select_bbox(slab, bbox_src)
        if sub.size == 0:
            raise ValueError(
                f"{self.LABEL}: bbox {bbox_src} does not intersect data for year={req.year}"
            )

        out_shape = self._resolve_out_shape(req)
        if req.crs != self.source_crs or out_shape is not None:
            # ``bbox_src`` arrives padded by one source pixel (see
            # _pad_bbox_src), so pin the output grid to the request's
            # ORIGINAL bbox — the pad feeds the resampler without widening
            # the output footprint.
            if out_shape is not None:
                sub = sub.rio.reproject(
                    req.crs,
                    resampling=Resampling.nearest,
                    nodata=self.nodata,
                    shape=out_shape,
                    transform=from_bounds(*req.bbox, width=out_shape[1], height=out_shape[0]),
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
        arr = self._dequantize(arr)
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
                    logger.debug(f"{self.LABEL} cache hit: {cache_paths[i].name}")
                    continue
            uncached.append(i)

        if uncached:
            logger.info(
                f"{self.LABEL} fetch_batch: {n} requests "
                f"({len(uncached)} uncached, {n - len(uncached)} cached)"
            )

            bboxes_src: dict[int, tuple[float, float, float, float]] = {}
            sel_bboxes: dict[int, tuple[float, float, float, float]] = {}
            for i in uncached:
                req = requests[i]
                bboxes_src[i] = (
                    req.bbox
                    if req.crs == self.source_crs
                    else tuple(transform_bounds(req.crs, self.source_crs, *req.bbox))
                )
                # Pad the source selection when the request will be
                # reprojected (see _pad_bbox_src). The padded bbox drives
                # BOTH chunk intersection and slab slicing — otherwise a bbox
                # flush against a zarr-chunk boundary would pad into a chunk
                # that was never loaded and _select_bbox would silently clip
                # the pad away. Pure source-CRS pass-through stays unpadded so
                # its output shape is unchanged.
                needs_reproject = (
                    req.crs != self.source_crs or self._resolve_out_shape(req) is not None
                )
                sel_bboxes[i] = (
                    self._pad_bbox_src(bboxes_src[i]) if needs_reproject else bboxes_src[i]
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
                        f"{self.LABEL}: bbox {bboxes_src[i]} "
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
                            f"{self.LABEL}: bbox {bboxes_src[i]} "
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
