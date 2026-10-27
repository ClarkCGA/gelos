import importlib.util
from pathlib import Path
import re
import threading

from loguru import logger
import numpy as np
from rasterio.warp import transform
import xarray as xr

from gelos.cloud_embeddings.backends.base import FetchRequest
from gelos.cloud_embeddings.backends.zarr_mosaic import ZarrMosaicBackend

DEFAULT_STORE_URL = "https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
S3_STORE_URL = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
DEFAULT_ENDPOINT_URL = "https://data.source.coop"
NODATA = -128
QUANT_SCALE = 127.5
QUANT_POWER = 2.0
VALID_DIMS = (64, 128)

MISSING_ZARR_HINT = "Install with: pip install 'gelos[olmoearth-embeddings]'"

_UTM_CRS_RE = re.compile(r"^EPSG:32[67](\d{2})$", re.IGNORECASE)


def _require_zarr() -> None:
    """Fail fast at __init__ with an actionable error when 'zarr' is missing."""
    if importlib.util.find_spec("zarr") is None:
        raise ImportError(f"OlmoEarthBackend requires 'zarr'. {MISSING_ZARR_HINT}")


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable, no I/O)
# ---------------------------------------------------------------------------


def dequantize(q: np.ndarray) -> np.ndarray:
    """Undo OlmoEarth's signed-square-root int8 quantization.

    The store holds ``q = round(sign(v) * sqrt(|v|) * 127.5)`` clipped to
    ``[-127, 127]`` with ``-128`` as nodata; the inverse is
    ``v = (|q| / 127.5) ** 2 * sign(q)``, so values land in ``[-1, 1]``.
    Accepts raw int8 codes (``-128`` -> NaN) or float input that already has
    NaN at nodata (what :class:`ZarrMosaicBackend` hands to the hook); NaN is
    preserved either way. Returns float32.
    """
    v = np.asarray(q)
    v = np.where(v == NODATA, np.nan, v).astype(np.float32, copy=False)
    return ((np.abs(v) / QUANT_SCALE) ** QUANT_POWER * np.sign(v)).astype(np.float32, copy=False)


def zone_from_lon(lon: float) -> int:
    """Nominal UTM zone (1..60) for a longitude in degrees."""
    zone = int((lon + 180.0) // 6.0) + 1
    return min(max(zone, 1), 60)


def zone_for_crs(crs: str) -> int | None:
    """Return the UTM zone encoded by an ``EPSG:326NN`` / ``EPSG:327NN`` code, else ``None``."""
    m = _UTM_CRS_RE.match(str(crs).strip())
    if m is None:
        return None
    zone = int(m.group(1))
    return zone if 1 <= zone <= 60 else None


def zone_group(zone: int) -> str:
    """Zarr group name for a UTM zone, e.g. ``utm18``."""
    return f"utm{zone:02d}"


def zone_crs(zone: int) -> str:
    """CRS of the store's group for a UTM zone — northern (``EPSG:326NN``) for every zone."""
    return f"EPSG:326{zone:02d}"


def resolve_zone(bbox: tuple[float, float, float, float], crs: str) -> int:
    """Pick the store group a footprint is read from.

    Fast path: a chip authored in a UTM CRS (``EPSG:326NN``/``EPSG:327NN``)
    is read from group ``utmNN`` — each group extends >=160 km past the
    nominal zone edge, so a chip native to zone NN always lies inside it.
    Otherwise the bbox centre is reprojected to WGS84 and the nominal zone of
    its longitude is used. A footprint is never split across zones.
    """
    zone = zone_for_crs(crs)
    if zone is not None:
        return zone
    minx, miny, maxx, maxy = bbox
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    if str(crs).upper() == "EPSG:4326":
        lon = cx
    else:
        lons, _lats = transform(crs, "EPSG:4326", [cx], [cy])
        lon = float(lons[0])
    return zone_from_lon(lon)


# ---------------------------------------------------------------------------
# Per-zone reader
# ---------------------------------------------------------------------------


class _OlmoEarthZoneReader(ZarrMosaicBackend):
    """One UTM-zone group of the OlmoEarth store, read through the shared mosaic machinery.

    Configured with the group's northern-hemisphere CRS, a positional band
    slice ``0:dims`` (the store has no ``band`` coordinate), inner-chunk
    batching (the shards are ~512 MB; the inner chunks ~4 MB) and the
    signed-sqrt dequantization hook. Cache files stay int8 on disk.

    ``pin_native_grid=True``: GELOS chips are authored on a grid offset half
    a pixel (5 m) from the store's 10 m zone grid, so a 160 m footprint
    requested *without* ``out_shape`` (``pool: mean``) would otherwise pass
    through the 17x17 source pixels whose centres fall inside it — a 170 m
    window hanging half a pixel outside the footprint. Pinning resamples
    every such request onto a ``round(extent / 10 m)`` grid aligned to the
    footprint (16x16 for a 160 m chip footprint) with nearest-neighbour, the
    same way the ``flatten`` path is pinned via ``out_shape``, in both the
    same-zone and the reprojected (``EPSG:327NN``) branch.

    The group is opened **without dask** (``use_dask=False`` -> ``chunks=None``):
    a zone array is ``(9, 128, 1867776, 98304)`` at ``(1, 64, 256, 256)`` inner
    chunks, so a dask-wrapped open has ~50M graph keys and every windowed read
    spends ~a minute of GIL-bound graph optimisation for ~1-2 s of I/O, which
    serialises the prefetch pool. The lazily-indexed numpy-backed array reads
    a window in a couple of seconds and is only ever accessed via windowed
    ``isel`` + ``load`` in :meth:`ZarrMosaicBackend._load_chunk`.
    """

    LABEL = "OlmoEarth"
    MISSING_ZARR_HINT = MISSING_ZARR_HINT

    def __init__(
        self,
        url: str,
        *,
        zone: int,
        dims: int,
        cache_dir: str | Path | None = None,
        anon: bool = True,
        nodata: int = NODATA,
        prefetch_workers: int = 16,
        extract_workers: int = 8,
        endpoint_url: str | None = None,
        storage_options: dict | None = None,
        max_retries: int = 4,
        retry_backoff: float = 1.0,
    ) -> None:
        group = zone_group(zone)
        # Instance attributes shadow the class-level defaults the base reads
        # through ``self`` — namespace cache keys per zone and width.
        self.LABEL = f"OlmoEarth {group}"
        self.CACHE_PREFIX = f"olmoearth_{group}_d{dims}"
        # olmo2: mean-path (no out_shape) footprints are pinned to the request
        # bbox at native resolution instead of passing 17x17 source pixels
        # through; olmo1 caches hold the misaligned window and must not be reused.
        self.CACHE_VERSION = f"olmo2|d{dims}"
        self.zone = zone
        self.dims = dims
        super().__init__(
            url,
            cache_dir=cache_dir,
            chunk_cache_dir=None,
            anon=anon,
            nodata=nodata,
            prefetch_workers=prefetch_workers,
            extract_workers=extract_workers,
            group=group,
            source_crs=zone_crs(zone),
            storage_options=storage_options,
            endpoint_url=endpoint_url,
            band_slice=(0, dims),
            prefer_inner_chunks=True,
            use_dask=False,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            pin_native_grid=True,
        )

    @property
    def dataset(self) -> xr.Dataset:
        first_open = self._dataset is None
        ds = super().dataset
        if first_open:
            proj_code = ds.attrs.get("proj:code")
            if proj_code is None or str(proj_code).upper() != self.source_crs.upper():
                raise ValueError(
                    f"{self.LABEL}: group {self.group!r} at {self.url} declares "
                    f"proj:code={proj_code!r} but the zone CRS is {self.source_crs!r}; "
                    "the store layout has changed — refusing to read it."
                )
            if "time" not in ds.coords:
                raise ValueError(
                    f"{self.LABEL}: group {self.group!r} at {self.url} has no 'time' "
                    f"coordinate; coords: {sorted(ds.coords)}"
                )
        return ds

    def _dequantize(self, arr: np.ndarray) -> np.ndarray:
        return dequantize(arr)


# ---------------------------------------------------------------------------
# Public backend: router over per-zone readers
# ---------------------------------------------------------------------------


class OlmoEarthBackend:
    """Fetches Ai2's precomputed OlmoEarth v1.3 embeddings from Source Cooperative.

    **Store** (public, anonymous, Zarr v3):

    ================  =============================================================
    S3 URL            ``s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr``
                      with custom endpoint ``https://data.source.coop`` (needs s3fs)
    HTTPS URL         ``https://data.source.coop/olmoearth/olmoearth-v1-3-embeddings/
                      embeddings.zarr`` (default; fsspec HTTP + aiohttp)
    layout            root group, one child group per UTM zone ``utm01``..``utm60``
    per-zone array    ``embeddings`` dims ``(time, band, y, x)``, int8, nodata ``-128``;
                      coords ``time`` (int years), ``x``/``y`` (metres, pixel centres);
                      no ``band`` coordinate
    time              ``2017..2025``; 2023-2025 global, 2017-2022 selected countries
                      only (nodata elsewhere)
    grid              ``proj:code`` ``EPSG:326NN`` for every zone (southern
                      hemisphere has negative ``y``), 10 m pixels, ``y`` descending
    chunking          shard ``(1, 128, 2048, 2048)`` (~512 MB), inner chunk
                      ``(1, 64, 256, 256)`` (~4 MB)
    quantization      ``v = (|q| / 127.5) ** 2 * sign(q)``, valid ``[-127, 127]``
    matryoshka        bands ``0:64`` are a standalone 64-d embedding
    license           OlmoEarth Artifact License
    ================  =============================================================

    **Zone rule.** Each request is routed to exactly one zone group: the zone
    of its CRS when that is ``EPSG:326NN``/``EPSG:327NN``, else the nominal
    zone of its bbox-centre longitude. The whole footprint is read from that
    group and reprojected to the request CRS as one raster — footprints are
    never split or stitched across zones (groups overlap >=160 km past the
    nominal edge). Each zone gets its own lazily-built
    :class:`_OlmoEarthZoneReader` (a :class:`ZarrMosaicBackend`), so all the
    cache / padding / chunk-streaming behaviour is shared with AlphaEarth.

    **Values.** Cache GeoTIFFs stay lossless int8; callers always receive
    float32 in ``[-1, 1]`` with NaN at nodata (dequantized on both the fresh
    and the cache path). ``mean`` pooling is NaN-aware, ``flatten`` keeps NaN.
    Footprints are always resampled onto a grid pinned to the request bbox at
    the store's 10 m resolution (a 160 m footprint -> 16x16 samples), whether
    or not ``out_shape`` is given — see :class:`_OlmoEarthZoneReader`.
    A year that exists in the store but is not computed for the region
    (2017-2022 outside the backfilled countries) comes back as all-NaN
    vectors; :meth:`fetch_batch` logs one warning per batch with the count.

    **dims.** ``128`` (default) returns the full embedding; ``64`` reads only
    bands ``0:64`` — the matryoshka 64-d embedding, stored in their own inner
    chunks so downloads halve. 64-d and 128-d vectors live in different
    spaces: never compare them with each other.

    **Cache keys** are ``olmoearth_{utmNN}_d{dims}_{crs}_{year}_{bbox:.3f...}_{8-hash}.tif``;
    the hash carries an ``olmo2|d{dims}`` version token so a layout or
    extraction change orphans old entries (``olmo1`` held the unpinned 17x17
    mean-path windows).

    ``chunk_cache_dir`` is rejected: the store is sharded, and fsspec's
    ``simplecache`` rejects the suffix (negative-start) byte requests zarr v3
    sharding issues for shard-index reads.

    **Transient-error retry.** Reading ``data.source.coop`` over HTTPS with the
    default 16 prefetch workers drops a connection every few minutes
    (``aiohttp.ClientPayloadError ... ConnectionResetError(104)``) and the
    proxy occasionally answers ``503 Service Unavailable`` — also while a
    zone group is being opened (the coordinate arrays are read eagerly).
    Each zone-group open and each chunk read is therefore retried up to
    ``max_retries`` times with exponential backoff starting at
    ``retry_backoff`` seconds (see :class:`ZarrMosaicBackend`); only
    connection/timeout errors and HTTP 429/5xx responses are retried, never
    404/403 or programming errors.
    """

    def __init__(
        self,
        url: str = DEFAULT_STORE_URL,
        cache_dir: str | Path | None = None,
        chunk_cache_dir: str | Path | None = None,
        anon: bool = True,
        endpoint_url: str | None = DEFAULT_ENDPOINT_URL,
        dims: int = 128,
        nodata: int = NODATA,
        prefetch_workers: int = 16,
        extract_workers: int = 8,
        storage_options: dict | None = None,
        max_retries: int = 4,
        retry_backoff: float = 1.0,
    ) -> None:
        _require_zarr()
        if dims not in VALID_DIMS:
            raise ValueError(
                f"OlmoEarthBackend: dims={dims!r} is not supported; choose one of "
                f"{list(VALID_DIMS)} (the matryoshka widths published with the store)."
            )
        if chunk_cache_dir is not None:
            raise ValueError(
                "OlmoEarthBackend does not support chunk_cache_dir: the OlmoEarth store is "
                "sharded (zarr v3 sharding codec) and fsspec's simplecache wrapper rejects "
                "the negative-start suffix reads the shard index needs. Use cache_dir "
                "(footprint GeoTIFF cache) instead."
            )
        if int(max_retries) < 0:
            raise ValueError(f"OlmoEarthBackend: max_retries must be >= 0, got {max_retries!r}")
        if float(retry_backoff) < 0:
            raise ValueError(
                f"OlmoEarthBackend: retry_backoff must be >= 0, got {retry_backoff!r}"
            )
        self.url = str(url)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.chunk_cache_dir = None
        self.anon = anon
        # endpoint_url only means something for s3://; the HTTPS store carries
        # the host in its URL.
        self.endpoint_url = endpoint_url if self.url.startswith("s3://") else None
        self.dims = int(dims)
        self.nodata = nodata
        self.prefetch_workers = prefetch_workers
        self.extract_workers = extract_workers
        self.storage_options = dict(storage_options) if storage_options else None
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_backoff)
        self._readers: dict[int, _OlmoEarthZoneReader] = {}
        self._readers_lock = threading.Lock()

    def _reader(self, zone: int) -> _OlmoEarthZoneReader:
        """Return (lazily building) the reader for one UTM zone group."""
        with self._readers_lock:
            reader = self._readers.get(zone)
            if reader is None:
                reader = _OlmoEarthZoneReader(
                    self.url,
                    zone=zone,
                    dims=self.dims,
                    cache_dir=self.cache_dir,
                    anon=self.anon,
                    nodata=self.nodata,
                    prefetch_workers=self.prefetch_workers,
                    extract_workers=self.extract_workers,
                    endpoint_url=self.endpoint_url,
                    storage_options=self.storage_options,
                    max_retries=self.max_retries,
                    retry_backoff=self.retry_backoff,
                )
                self._readers[zone] = reader
            return reader

    def fetch(
        self,
        bbox: tuple[float, float, float, float],
        crs: str,
        year: int,
        out_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        return self.fetch_batch([FetchRequest(bbox, crs, year, out_shape)])[0]

    def fetch_batch(self, requests: list[FetchRequest]) -> list[np.ndarray]:
        """Route each request to its UTM-zone reader and re-assemble in input order.

        Zones are processed sequentially; each zone's sub-batch runs through
        :meth:`ZarrMosaicBackend.fetch_batch` with its own thread pools.
        """
        if not requests:
            return []

        n = len(requests)
        by_zone: dict[int, list[int]] = {}
        for i, req in enumerate(requests):
            by_zone.setdefault(resolve_zone(req.bbox, req.crs), []).append(i)

        zones = sorted(by_zone)
        logger.info(
            f"OlmoEarth fetch_batch: {n} requests across {len(zones)} UTM zones "
            f"{[zone_group(z) for z in zones]}"
        )

        results: list[np.ndarray | None] = [None] * n
        for zone in zones:
            idxs = by_zone[zone]
            zone_results = self._reader(zone).fetch_batch([requests[i] for i in idxs])
            for i, arr in zip(idxs, zone_results):
                results[i] = arr

        n_all_nan = sum(1 for arr in results if arr is not None and np.isnan(arr).all())
        if n_all_nan:
            logger.warning(
                f"OlmoEarth fetch_batch: {n_all_nan}/{n} footprints came back entirely "
                "nodata (all-NaN vectors). The store covers 2023-2025 globally but "
                "2017-2022 only for selected countries; footprints outside the "
                "backfilled regions, or outside their zone group's extent, have no data."
            )
        return results
