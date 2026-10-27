import gc
import importlib.util
from pathlib import Path
import threading

import geopandas as gpd
from loguru import logger
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from tests.test_data import ExampleGELOSDataSet  # noqa: F401 — resolves class_path in YAML
from tests.utils import create_test_geojson
import xarray as xr
import yaml

from gelos.analysis import run_analysis
from gelos.cloud_embeddings import runner as cloud_runner
from gelos.cloud_embeddings.aggregate import flatten_pool, mean_pool
from gelos.cloud_embeddings.backends import zarr_mosaic
from gelos.cloud_embeddings.backends.alphaearth import AlphaEarthBackend
from gelos.cloud_embeddings.backends.base import FetchRequest
from gelos.cloud_embeddings.backends.olmoearth import (
    OlmoEarthBackend,
    _OlmoEarthZoneReader,
    dequantize,
    resolve_zone,
)
from gelos.cloud_embeddings.footprints import compute_footprint_manifest
from gelos.cloud_embeddings.patches import resolve_patch_indices, resolve_timestep_indices
from gelos.cloud_embeddings.runner import generate_cloud_embeddings
from gelos.cloud_embeddings.year import resolve_chip_year
from gelos.generation import generate_embeddings

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _rewrite_tiff_with_transform(path: Path, transform, crs: str) -> None:
    """Rewrite an existing tiff in-place with a new transform + CRS, preserving data."""
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile
    profile.update(transform=transform, crs=crs)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


# ---------------------------------------------------------------------------
# Shared patch resolution util (mirror of tests/test_spectral_bands.py::test_resolve_*)
# ---------------------------------------------------------------------------


def test_resolve_patches_center_2x2():
    spec = {"patches": "center_2x2"}
    assert resolve_patch_indices(spec, n_rows=6, n_cols=6) == [(2, 2), (2, 3), (3, 2), (3, 3)]


def test_resolve_patches_center_1x4():
    spec = {"patches": "center_1x4"}
    indices = resolve_patch_indices(spec, n_rows=6, n_cols=6)
    # Centered horizontal line: single row, consecutive cols → contiguous row-major.
    assert indices == [(2, 1), (2, 2), (2, 3), (2, 4)]


def test_resolve_patches_explicit_list():
    spec = {"patches": [[0, 0], [2, 3]]}
    assert resolve_patch_indices(spec, n_rows=6, n_cols=6) == [(0, 0), (2, 3)]


def test_resolve_patches_out_of_range_raises():
    spec = {"patches": [[10, 10]]}
    with pytest.raises(ValueError):
        resolve_patch_indices(spec, n_rows=6, n_cols=6)


def test_resolve_timesteps_all():
    assert resolve_timestep_indices({"timesteps": "all"}, T=4) == [0, 1, 2, 3]


def test_resolve_timesteps_out_of_range_raises():
    with pytest.raises(ValueError):
        resolve_timestep_indices({"timesteps": [5]}, T=4)


# ---------------------------------------------------------------------------
# Footprint manifest
# ---------------------------------------------------------------------------


def test_footprint_manifest_utm_bboxes(tmp_path):
    """North-up UTM transform → exact bbox values for center_2x2 patches."""
    sensors = {"S2L2A": len(ExampleGELOSDataSet.S2L2A_BAND_NAMES)}
    n_timesteps = {"S2L2A": 1}
    create_test_geojson(tmp_path, n_samples=1, sensors=sensors, n_timesteps=n_timesteps, img_size=96)

    s2_path = tmp_path / "S2L2A_000001_00.tif"
    _rewrite_tiff_with_transform(s2_path, from_origin(500000, 4000000, 30, 30), "EPSG:32618")

    bands = {"S2L2A": ["blue", "green", "red"]}
    dataset = ExampleGELOSDataSet(data_root=tmp_path, bands=bands)

    manifest = compute_footprint_manifest(
        dataset,
        reference_sensor="S2L2A",
        patch_size=16,
        strategy_specs={"center_2x2": {"patches": "center_2x2"}},
    )

    assert len(manifest) == 4
    assert set(manifest["strategy"].unique()) == {"center_2x2"}
    assert (manifest["crs"] == "EPSG:32618").all()

    p22 = manifest[(manifest["patch_row"] == 2) & (manifest["patch_col"] == 2)].iloc[0]
    # patch (2, 2): col→x range [500000 + 2*480, 500000 + 3*480], row→y range [4000000 - 3*480, 4000000 - 2*480]
    assert p22["bbox_minx"] == 500000 + 2 * 16 * 30
    assert p22["bbox_maxx"] == 500000 + 3 * 16 * 30
    assert p22["bbox_maxy"] == 4000000 - 2 * 16 * 30
    assert p22["bbox_miny"] == 4000000 - 3 * 16 * 30
    gc.collect()


def test_footprint_manifest_multi_strategy(tmp_path):
    sensors = {"S2L2A": len(ExampleGELOSDataSet.S2L2A_BAND_NAMES)}
    n_timesteps = {"S2L2A": 1}
    create_test_geojson(tmp_path, n_samples=1, sensors=sensors, n_timesteps=n_timesteps, img_size=96)
    _rewrite_tiff_with_transform(
        tmp_path / "S2L2A_000001_00.tif", from_origin(500000, 4000000, 30, 30), "EPSG:32618"
    )

    dataset = ExampleGELOSDataSet(data_root=tmp_path, bands={"S2L2A": ["blue", "green", "red"]})
    manifest = compute_footprint_manifest(
        dataset,
        reference_sensor="S2L2A",
        patch_size=16,
        strategy_specs={
            "center_2x2": {"patches": "center_2x2"},
            "center_1x1": {"patches": "center_1x1"},
        },
    )
    # 4 patches for 2x2 + 1 patch for 1x1, single chip
    assert len(manifest) == 5
    assert set(manifest["strategy"].unique()) == {"center_2x2", "center_1x1"}
    gc.collect()


def test_footprint_manifest_h_not_divisible_raises(tmp_path):
    sensors = {"S2L2A": len(ExampleGELOSDataSet.S2L2A_BAND_NAMES)}
    n_timesteps = {"S2L2A": 1}
    create_test_geojson(tmp_path, n_samples=1, sensors=sensors, n_timesteps=n_timesteps, img_size=96)
    _rewrite_tiff_with_transform(
        tmp_path / "S2L2A_000001_00.tif", from_origin(500000, 4000000, 30, 30), "EPSG:32618"
    )

    dataset = ExampleGELOSDataSet(data_root=tmp_path, bands={"S2L2A": ["blue", "green", "red"]})
    with pytest.raises(ValueError, match="divisible"):
        compute_footprint_manifest(
            dataset,
            reference_sensor="S2L2A",
            patch_size=17,  # 96 % 17 != 0
            strategy_specs={"s": {"patches": [[0, 0]]}},
        )
    gc.collect()


# ---------------------------------------------------------------------------
# Year strategy
# ---------------------------------------------------------------------------


class _StubDatasetForYear:
    def __init__(self, gdf, paths_by_index):
        self.gdf = gdf
        self._paths_by_index = paths_by_index

    def _get_file_paths(self, index, sensor):
        return self._paths_by_index[index]


def test_resolve_year_from_filename():
    ds = _StubDatasetForYear(
        gdf=pd.DataFrame({"year": [2022]}),
        paths_by_index={0: [Path("S2L2A_2023_000001_00.tif")]},
    )
    cfg = {"type": "from_filename", "sensor": "S2L2A", "pattern": r"_(\d{4})_"}
    assert resolve_chip_year(ds, 0, cfg) == 2023


def test_resolve_year_from_column():
    ds = _StubDatasetForYear(
        gdf=pd.DataFrame({"year": [2024]}),
        paths_by_index={0: [Path("anything.tif")]},
    )
    cfg = {"type": "from_column", "column": "year"}
    assert resolve_chip_year(ds, 0, cfg) == 2024


def test_resolve_year_unknown_strategy_raises():
    ds = _StubDatasetForYear(gdf=pd.DataFrame({"year": [2024]}), paths_by_index={0: []})
    with pytest.raises(ValueError, match="unknown year_strategy"):
        resolve_chip_year(ds, 0, {"type": "from_planets"})


def test_resolve_year_from_filename_no_match_raises():
    ds = _StubDatasetForYear(
        gdf=pd.DataFrame({"year": [2022]}),
        paths_by_index={0: [Path("noyear.tif")]},
    )
    with pytest.raises(ValueError, match="did not match"):
        resolve_chip_year(
            ds, 0, {"type": "from_filename", "sensor": "S2L2A", "pattern": r"_(\d{4})_"}
        )


# ---------------------------------------------------------------------------
# Mean pool
# ---------------------------------------------------------------------------


def test_mean_pool_ones():
    arr = np.ones((10, 10, 4), dtype=np.float32)
    result = mean_pool(arr)
    assert result.shape == (4,)
    assert np.allclose(result, 1.0)


def test_mean_pool_with_nans():
    arr = np.ones((4, 4, 2), dtype=np.float32)
    arr[0, 0, 0] = np.nan
    result = mean_pool(arr)
    assert result.shape == (2,)
    assert np.allclose(result, 1.0)


def test_mean_pool_all_nan_returns_nan():
    arr = np.full((4, 4, 1), np.nan, dtype=np.float32)
    with pytest.warns(RuntimeWarning):
        result = mean_pool(arr)
    assert np.isnan(result[0])


def test_mean_pool_rejects_2d():
    with pytest.raises(ValueError, match="3D"):
        mean_pool(np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# Flatten pool
# ---------------------------------------------------------------------------


def test_flatten_pool_shape():
    arr = np.zeros((4, 4, 2), dtype=np.float32)
    result = flatten_pool(arr)
    assert result.shape == (32,)


def test_flatten_pool_preserves_nan():
    arr = np.ones((2, 3, 2), dtype=np.float32)
    arr[1, 2, 0] = np.nan
    result = flatten_pool(arr)
    # row-major C order: index = 1*(3*2) + 2*2 + 0 = 10
    assert np.isnan(result[10])
    assert not np.isnan(result[:10]).any()
    assert not np.isnan(result[11:]).any()


def test_flatten_pool_rejects_2d():
    with pytest.raises(ValueError, match="3D"):
        flatten_pool(np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# AlphaEarth backend (synthetic local Zarr — no network)
# ---------------------------------------------------------------------------


def _write_synthetic_zarr(
    path: Path,
    years: tuple[int, ...] = (2017, 2018, 2019),
    H: int = 32,
    W: int = 32,
    C: int = 64,
    nodata_corner: bool = False,
    chunks_yx: tuple[int, int] | None = None,
) -> None:
    """Write a synthetic AEF-shaped Zarr store at ``path``.

    Each band ``b`` is filled with the int8 value ``b - 30`` (spread across the
    int8 range) so per-band fetches are deterministic. ``y`` is descending to
    match the production mosaic layout. When ``chunks_yx`` is given, the
    embedding array is written with that (y, x) chunk shape — drives the
    chunk-grouping test.
    """
    data = np.empty((len(years), C, H, W), dtype=np.int8)
    for b in range(C):
        data[:, b, :, :] = np.int8(b - 30)
    if nodata_corner:
        # Mark the top-left pixel (y_idx=0, x_idx=0) of year 0 as nodata
        data[0, :, 0, 0] = -128
    ds = xr.Dataset(
        {"embedding": (("time", "band", "y", "x"), data)},
        coords={
            "time": list(years),
            "band": np.arange(C),
            "y": np.linspace(43.0, 42.0, H),  # descending
            "x": np.linspace(-72.0, -71.0, W),
        },
    )
    to_zarr_kwargs: dict = {"mode": "w", "zarr_format": 3}
    if chunks_yx is not None:
        to_zarr_kwargs["encoding"] = {
            "embedding": {"chunks": (1, C, chunks_yx[0], chunks_yx[1])}
        }
    ds.to_zarr(path, **to_zarr_kwargs)


def test_alphaearth_missing_zarr_raises_actionable_import_error(monkeypatch):
    """Missing 'zarr' surfaces at __init__ with a 'pip install gelos[alphaearth]' hint."""
    import importlib.util as iu

    real_find_spec = iu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "zarr":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(iu, "find_spec", fake_find_spec)
    with pytest.raises(ImportError, match=r"gelos\[alphaearth\]"):
        AlphaEarthBackend()


def test_alphaearth_cache_key_namespaced():
    backend = AlphaEarthBackend()
    key = backend._cache_key("EPSG:4326", 2023, (-72.0, 42.0, -71.0, 43.0))
    assert key.startswith("mosaic_")


def test_alphaearth_year_to_time_index(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, years=(2017, 2018, 2019))
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    assert backend._year_to_index(2017) == 0
    assert backend._year_to_index(2018) == 1
    assert backend._year_to_index(2019) == 2
    gc.collect()


def test_alphaearth_year_out_of_range_raises(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, years=(2017, 2018, 2019))
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    with pytest.raises(ValueError, match="out of range"):
        backend.fetch((-71.7, 42.2, -71.5, 42.5), crs="EPSG:4326", year=2030)
    gc.collect()


def test_alphaearth_bbox_no_intersect_raises(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    with pytest.raises(ValueError, match="does not intersect"):
        backend.fetch((100.0, 50.0, 100.5, 50.5), crs="EPSG:4326", year=2018)
    gc.collect()


def test_alphaearth_fetch_returns_hwc_float(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    bbox = (-71.7, 42.2, -71.5, 42.5)
    arr = backend.fetch(bbox, crs="EPSG:4326", year=2018)

    assert arr.dtype == np.float32
    assert arr.ndim == 3
    assert arr.shape[-1] == 64
    assert arr.shape[0] > 0 and arr.shape[1] > 0
    for b in range(64):
        np.testing.assert_allclose(arr[..., b], float(b - 30), atol=1e-5)
    gc.collect()


def test_alphaearth_nodata_becomes_nan(tmp_path):
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, nodata_corner=True)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    # Cover the entire grid so the nodata corner pixel is included
    arr = backend.fetch((-72.0, 42.0, -71.0, 43.0), crs="EPSG:4326", year=2017)
    # y is descending → arr[0, ...] is largest y == top; x ascending → arr[:, 0, :] is leftmost
    assert np.isnan(arr[0, 0, :]).all()
    gc.collect()


def test_alphaearth_cache_hit_short_circuits_open(tmp_path, monkeypatch):
    """Second fetch with same key reads from disk and does not re-open the Zarr."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path)
    cache_dir = tmp_path / "cache"
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False, cache_dir=cache_dir)

    bbox = (-71.7, 42.2, -71.5, 42.5)
    first = backend.fetch(bbox, crs="EPSG:4326", year=2018)

    cached_files = list(cache_dir.glob("mosaic_*.tif"))
    assert len(cached_files) == 1, f"expected one cached tif, got {cached_files}"

    def boom(self):
        raise AssertionError("data_array must not be accessed on cache hit")

    monkeypatch.setattr(AlphaEarthBackend, "data_array", property(boom))
    second = backend.fetch(bbox, crs="EPSG:4326", year=2018)

    assert second.shape == first.shape
    np.testing.assert_allclose(
        np.nan_to_num(second, nan=0.0), np.nan_to_num(first, nan=0.0), atol=1e-5
    )
    gc.collect()


def test_alphaearth_fetch_with_out_shape(tmp_path):
    """fetch(..., out_shape=(8, 8)) resamples the slab to exactly that H, W."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    arr = backend.fetch(
        (-71.7, 42.2, -71.5, 42.5), crs="EPSG:4326", year=2018, out_shape=(8, 8)
    )
    assert arr.shape == (8, 8, 64)
    gc.collect()


def test_alphaearth_consolidated_fallback(tmp_path):
    """Synthetic zarr lacks consolidated metadata → backend falls back to consolidated=False."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path)  # no consolidation
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    ds = backend.dataset
    assert "embedding" in ds.data_vars
    gc.collect()


def test_alphaearth_chunks_intersecting_single(tmp_path):
    """Bbox inside one chunk returns a single (cy, cx) pair."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)

    # 32x32 grid, 8x8 chunks → 4x4 chunk layout; y descends 43→42, x ascends -72→-71.
    # bbox fits entirely inside chunk (0, 0): px_y ∈ [1, 4], px_x ∈ [0, 4].
    chunks = backend._chunks_intersecting_bbox((-72.0, 42.85, -71.85, 42.95))
    assert chunks == {(0, 0)}
    gc.collect()


def test_alphaearth_chunks_intersecting_straddle(tmp_path):
    """Bbox straddling chunk boundaries returns every chunk it touches."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)

    # Spans columns from chunk 0 into chunk 1 *and* rows from chunk 1 into
    # chunk 2 → 4 chunks total: {(1,0),(1,1),(2,0),(2,1)}.
    chunks = backend._chunks_intersecting_bbox((-71.78, 42.4, -71.5, 42.6))
    assert len(chunks) >= 2
    assert {(cy, cx) for cy in (1, 2) for cx in (0, 1)} <= chunks
    gc.collect()


def test_alphaearth_fetch_batch_matches_fetch(tmp_path):
    """Multi-request fetch_batch returns identical values to per-request fetch."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)

    requests = [
        # Two bboxes in chunk (0, 0) → grouped together in fetch_batch
        FetchRequest(bbox=(-72.0, 42.85, -71.85, 42.95), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.85, 42.85, -71.78, 42.95), crs="EPSG:4326", year=2018),
        # One in a middle chunk
        FetchRequest(bbox=(-71.5, 42.45, -71.4, 42.55), crs="EPSG:4326", year=2018),
        # One that straddles a chunk boundary
        FetchRequest(bbox=(-71.85, 42.7, -71.7, 42.85), crs="EPSG:4326", year=2018),
        # Same bbox, different year → must materialize separately
        FetchRequest(bbox=(-71.5, 42.45, -71.4, 42.55), crs="EPSG:4326", year=2019),
    ]
    batch = backend.fetch_batch(requests)
    individual = [backend.fetch(r.bbox, r.crs, r.year, out_shape=r.out_shape) for r in requests]
    assert len(batch) == len(individual) == len(requests)
    for i, (b, ind) in enumerate(zip(batch, individual)):
        assert b.shape == ind.shape, f"request {i}: shape mismatch {b.shape} vs {ind.shape}"
        np.testing.assert_allclose(b, ind, equal_nan=True, atol=1e-5)
    gc.collect()


def test_alphaearth_fetch_batch_minimizes_loads(tmp_path, monkeypatch):
    """Four bboxes inside one Zarr chunk trigger exactly one .load() call.

    Pins the chunk-streaming invariant: one ``.load()`` per *unique*
    ``(year, cy, cx)``, not per group of co-resident bboxes.
    """
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)

    load_count = {"n": 0}
    original_load = xr.DataArray.load

    def counting_load(self, *args, **kwargs):
        load_count["n"] += 1
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "load", counting_load)

    # All four bboxes fall within chunk (0, 0) — px_y ∈ [0,7], px_x ∈ [0,7].
    requests = [
        FetchRequest(bbox=(-72.0, 42.85, -71.93, 42.95), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.93, 42.85, -71.85, 42.95), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-72.0, 42.78, -71.93, 42.85), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.93, 42.78, -71.85, 42.85), crs="EPSG:4326", year=2018),
    ]
    backend.fetch_batch(requests)
    assert load_count["n"] == 1, f"expected 1 .load() call, got {load_count['n']}"
    gc.collect()


def test_alphaearth_fetch_batch_dedupes_across_overlap(tmp_path, monkeypatch):
    """Overlapping requests with different chunk sets dedupe to the union, not the sum.

    Req A sits inside chunk (0, 0). Req B straddles (0, 0) + (0, 1). The
    old ``_chunk_box``-based grouping would have issued two group loads,
    re-fetching (0, 0) inside Req B's union slab. The chunk-streaming
    inversion fetches (0, 0) and (0, 1) exactly once each — total 2.
    """
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)

    load_count = {"n": 0}
    original_load = xr.DataArray.load

    def counting_load(self, *args, **kwargs):
        load_count["n"] += 1
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "load", counting_load)

    requests = [
        FetchRequest(bbox=(-72.0, 42.85, -71.93, 42.95), crs="EPSG:4326", year=2018),
        # maxx=-71.7 lands in chunk column 1 → straddles (0,0)+(0,1)
        FetchRequest(bbox=(-71.93, 42.85, -71.7, 42.95), crs="EPSG:4326", year=2018),
    ]
    backend.fetch_batch(requests)
    assert load_count["n"] == 2, f"expected 2 unique chunk loads, got {load_count['n']}"
    gc.collect()


def test_alphaearth_fetch_batch_straddle_correctness(tmp_path):
    """A request straddling two chunks returns values identical to single-fetch."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)

    straddle_bbox = (-71.93, 42.85, -71.7, 42.95)  # spans (0,0)+(0,1)
    # Add a second uncached request to force the multi-uncached inversion
    # path (the single-uncached short-circuit takes a different code branch).
    requests = [
        FetchRequest(bbox=straddle_bbox, crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.5, 42.45, -71.4, 42.55), crs="EPSG:4326", year=2018),
    ]
    batch = backend.fetch_batch(requests)
    individual = backend.fetch(straddle_bbox, crs="EPSG:4326", year=2018)
    assert batch[0].shape == individual.shape
    np.testing.assert_allclose(batch[0], individual, equal_nan=True, atol=1e-5)
    gc.collect()


def test_alphaearth_fetch_batch_concurrent_pool(tmp_path):
    """Same mixed scenario as matches_fetch but with small pools forcing contention.

    Without this, the default 16-fetch / 8-extract pools mean every other
    test runs effectively single-threaded (1 chunk per fetch, all extracts
    serialized behind it). Shrinking the pools and reusing the mixed
    same-chunk / straddle / multi-year workload guarantees real concurrent
    submits and proves the orchestrator doesn't race or deadlock.
    """
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    backend = AlphaEarthBackend(
        url=str(zarr_path), anon=False, prefetch_workers=2, extract_workers=2
    )

    requests = [
        FetchRequest(bbox=(-72.0, 42.85, -71.85, 42.95), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.85, 42.85, -71.78, 42.95), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.5, 42.45, -71.4, 42.55), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.85, 42.7, -71.7, 42.85), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.5, 42.45, -71.4, 42.55), crs="EPSG:4326", year=2019),
    ]
    batch = backend.fetch_batch(requests)
    individual = [backend.fetch(r.bbox, r.crs, r.year, out_shape=r.out_shape) for r in requests]
    assert len(batch) == len(individual) == len(requests)
    for i, (b, ind) in enumerate(zip(batch, individual)):
        assert b.shape == ind.shape, f"request {i}: shape mismatch {b.shape} vs {ind.shape}"
        np.testing.assert_allclose(b, ind, equal_nan=True, atol=1e-5)
    gc.collect()


def test_alphaearth_fetch_batch_all_cached_skips_open(tmp_path, monkeypatch):
    """fetch_batch of all-cached requests never touches data_array."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path)
    cache_dir = tmp_path / "cache"
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False, cache_dir=cache_dir)

    bboxes = [
        (-71.7, 42.2, -71.5, 42.5),
        (-71.5, 42.5, -71.3, 42.7),
        (-71.9, 42.8, -71.7, 42.95),
    ]
    # Prime the cache via single fetches
    primed = [backend.fetch(b, crs="EPSG:4326", year=2018) for b in bboxes]
    assert len(list(cache_dir.glob("mosaic_*.tif"))) == 3

    def boom(self):
        raise AssertionError("data_array must not be accessed when all requests are cached")

    monkeypatch.setattr(AlphaEarthBackend, "data_array", property(boom))

    requests = [FetchRequest(bbox=b, crs="EPSG:4326", year=2018) for b in bboxes]
    results = backend.fetch_batch(requests)
    assert len(results) == 3
    for r, p in zip(results, primed):
        np.testing.assert_allclose(
            np.nan_to_num(r, nan=0.0), np.nan_to_num(p, nan=0.0), atol=1e-5
        )
    gc.collect()


def test_alphaearth_fetch_batch_empty_returns_empty(tmp_path):
    """fetch_batch([]) is a no-op returning an empty list — does not open the dataset."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    assert backend.fetch_batch([]) == []
    # data_array should remain unopened
    assert backend._da is None
    gc.collect()


# ---------------------------------------------------------------------------
# AlphaEarth edge-nodata fix (padded selection + bbox-pinned output grid)
# ---------------------------------------------------------------------------


def test_alphaearth_utm_out_shape_has_no_edge_nodata(tmp_path):
    """Regression: UTM fetch with out_shape must not fabricate nodata at edges.

    Pre-fix, _select_bbox clipped the source slab strictly to the request
    bbox, so nearest-neighbor reprojection filled edge target pixels (whose
    nearest source-pixel center fell just outside the slab) with -128 → NaN.
    The synthetic zarr has full coverage, so ANY NaN here is fabricated.
    """
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    # lon -72…-71 sits in UTM zone 19N
    bbox_utm = tuple(transform_bounds("EPSG:4326", "EPSG:32619", -71.7, 42.2, -71.5, 42.5))
    arr = backend.fetch(bbox_utm, crs="EPSG:32619", year=2018, out_shape=(16, 16))
    assert arr.shape == (16, 16, 64)
    n_nan = int(np.isnan(arr[..., 0]).sum())
    assert not np.isnan(arr).any(), f"false edge nodata: {n_nan} NaN cells in band 0"
    gc.collect()


def test_alphaearth_out_shape_grid_anchored_to_bbox(tmp_path):
    """The cached tif's bounds equal the request bbox (from_bounds pinning).

    The padded source selection must not widen the output footprint: the
    16x16 output grid covers exactly the requested chip footprint.
    """
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path)
    cache_dir = tmp_path / "cache"
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False, cache_dir=cache_dir)
    bbox_utm = tuple(transform_bounds("EPSG:4326", "EPSG:32619", -71.7, 42.2, -71.5, 42.5))
    backend.fetch(bbox_utm, crs="EPSG:32619", year=2018, out_shape=(16, 16))

    cached_files = list(cache_dir.glob("mosaic_*.tif"))
    assert len(cached_files) == 1, f"expected one cached tif, got {cached_files}"
    with rasterio.open(cached_files[0]) as src:
        bounds = src.bounds
        assert src.width == 16 and src.height == 16
    assert bounds.left == pytest.approx(bbox_utm[0], abs=1e-6)
    assert bounds.bottom == pytest.approx(bbox_utm[1], abs=1e-6)
    assert bounds.right == pytest.approx(bbox_utm[2], abs=1e-6)
    assert bounds.top == pytest.approx(bbox_utm[3], abs=1e-6)
    gc.collect()


def test_alphaearth_genuine_nodata_preserved(tmp_path):
    """Padding must not fabricate data over genuine mosaic gaps.

    The synthetic zarr marks the top-left source pixel (lon -72.0, lat 43.0)
    of year 2017 as nodata. A UTM fetch covering that corner must keep NaN at
    the corresponding output cells and stay NaN-free away from them.
    """
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, nodata_corner=True)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    bbox_utm = tuple(transform_bounds("EPSG:4326", "EPSG:32619", -72.0, 42.9, -71.9, 43.0))
    arr = backend.fetch(bbox_utm, crs="EPSG:32619", year=2017, out_shape=(16, 16))
    # Output row 0 = north (lat 43), col 0 = west (lon -72): the nodata
    # source pixel maps to the top-left output cells.
    assert np.isnan(arr[0, 0, :]).all(), "genuine nodata at the corner was papered over"
    # The single nodata source pixel covers only a small top-left block of
    # the 16x16 grid; everything away from that corner must be data.
    assert not np.isnan(arr[6:, :, :]).any()
    assert not np.isnan(arr[:, 6:, :]).any()
    gc.collect()


def test_alphaearth_pad_at_mosaic_boundary(tmp_path):
    """Bbox flush against the mosaic's outer edge: the pad is clipped, not fatal.

    Padding beyond maxy=43.0 (the synthetic mosaic's top edge) must be
    silently clipped by .sel / chunk-index clamping — no 'does not intersect'
    and no fabricated nodata for the fully-covered footprint.
    """
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    bbox_utm = tuple(transform_bounds("EPSG:4326", "EPSG:32619", -71.6, 42.9, -71.5, 43.0))
    arr = backend.fetch(bbox_utm, crs="EPSG:32619", year=2018, out_shape=(16, 16))
    assert arr.shape == (16, 16, 64)
    assert not np.isnan(arr).any()
    gc.collect()


def test_alphaearth_pad_crosses_chunk_boundary(tmp_path):
    """The padded bbox must drive chunk loading, not just slab slicing.

    32x32 grid with 16x16 chunks: the request's WGS84 envelope ends at
    lon -71.49, inside chunk column 0 (last center kept: pixel 15 at
    -71.516), while the one-pixel pad reaches pixel 16 (-71.484) in chunk
    column 1. If chunk intersection used the unpadded bbox, chunk (0, 1)
    would never load, _select_bbox would silently clip the pad away, and the
    right output column would come back NaN.
    """
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(16, 16))
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    bbox_utm = tuple(transform_bounds("EPSG:4326", "EPSG:32619", -71.7, 42.7, -71.49, 42.8))
    arr = backend.fetch(bbox_utm, crs="EPSG:32619", year=2018, out_shape=(16, 16))
    assert arr.shape == (16, 16, 64)
    n_nan = int(np.isnan(arr[..., 0]).sum())
    assert not np.isnan(arr).any(), f"pad did not cross chunk boundary: {n_nan} NaN cells"
    gc.collect()


def test_alphaearth_cache_key_snapshot():
    """Pin the pre-refactor AEF cache-key format byte-for-byte.

    Computed once with the original (pre-ZarrMosaicBackend) implementation:
    sha256("EPSG:32619|2018|1.0|2.0|3.0|4.0|(16, 16)|pad1")[:8] == "29ad6864".
    Existing on-disk ``_footprint_cache`` entries must stay valid.
    """
    key = AlphaEarthBackend()._cache_key("EPSG:32619", 2018, (1.0, 2.0, 3.0, 4.0), (16, 16))
    assert key == "mosaic_EPSG32619_2018_1.000_2.000_3.000_4.000_29ad6864.tif"


# ---------------------------------------------------------------------------
# OlmoEarth backend (synthetic per-UTM-zone sharded Zarr v3 — no network)
# ---------------------------------------------------------------------------

OLMO_X0 = 499840.0
OLMO_Y0 = 4001280.0
OLMO_ZONE19_X0 = 300000.0


def _write_synthetic_olmoearth_zarr(
    path: Path,
    zones: tuple[int, ...] = (18,),
    years: tuple[int, ...] = (2023, 2024, 2025),
    H: int = 64,
    W: int = 64,
    C: int = 128,
    x0: float = OLMO_X0,
    y0: float = OLMO_Y0,
    res: float = 10.0,
    nodata_corner: bool = False,
    inner_chunks_yx: tuple[int, int] = (16, 16),
    shard_yx: tuple[int, int] = (32, 32),
    proj_code_override: str | None = None,
    nodata_year_index: int | None = None,
    index_bands: bool = False,
) -> None:
    """Write a synthetic OlmoEarth-shaped store: one ``utmNN`` group per zone.

    Mirrors the real layout: ``embeddings`` dims ``(time, band, y, x)``, int8,
    no ``band`` coordinate, int32 ``time`` years, descending ``y`` pixel
    centres, ``proj:code`` attrs, and a sharded zarr v3 encoding whose inner
    chunk (``inner_chunks_yx``, 64 bands) is smaller than the shard
    (``shard_yx``, all bands). Band ``b`` holds int8 ``b - 60`` (all valid, a
    mix of signs). Zone 19 (when requested) gets ``x0 = OLMO_ZONE19_X0`` so
    bboxes are unambiguous. ``nodata_corner`` sets the top-left pixel of year
    0 to -128; ``nodata_year_index`` blanks a whole year. ``index_bands``
    overwrites band 0 with the pixel's row index and band 1 with its column
    index (``H, W <= 127``) so tests can tell *which* source pixels a window
    sampled.
    """
    for k, zone in enumerate(zones):
        zx0 = x0 if zone == zones[0] else OLMO_ZONE19_X0
        data = np.empty((len(years), C, H, W), dtype=np.int8)
        for b in range(C):
            data[:, b, :, :] = np.int8(b - 60)
        if index_bands:
            data[:, 0, :, :] = np.arange(H, dtype=np.int8)[:, None]
            data[:, 1, :, :] = np.arange(W, dtype=np.int8)[None, :]
        if nodata_corner:
            data[0, :, 0, 0] = -128
        if nodata_year_index is not None:
            data[nodata_year_index] = -128
        ds = xr.Dataset(
            {"embeddings": (("time", "band", "y", "x"), data)},
            coords={
                "time": np.array(years, dtype="int32"),
                "y": y0 - res / 2 - res * np.arange(H),  # descending centres
                "x": zx0 + res / 2 + res * np.arange(W),
            },
            attrs={
                "proj:code": proj_code_override or f"EPSG:326{zone:02d}",
                "spatial:transform": [res, 0.0, zx0, 0.0, -res, y0],
            },
        )
        ds.to_zarr(
            path,
            mode="w" if k == 0 else "a",
            zarr_format=3,
            group=f"utm{zone:02d}",
            encoding={
                "embeddings": {
                    "chunks": (1, 64, inner_chunks_yx[0], inner_chunks_yx[1]),
                    "shards": (1, C, shard_yx[0], shard_yx[1]),
                }
            },
        )


def _olmo_expected_band_values(C: int = 128) -> np.ndarray:
    return dequantize(np.arange(C, dtype=np.int8) - np.int8(60))


def test_olmoearth_registered_in_backends():
    assert "olmoearth" in cloud_runner.BACKENDS
    assert cloud_runner.BACKENDS["olmoearth"] is OlmoEarthBackend
    assert cloud_runner._backend_accepts_kwarg(OlmoEarthBackend, "cache_dir")


def test_olmoearth_dequantize_formula():
    q = np.array([-128, -127, -64, 0, 64, 127], dtype=np.int8)
    out = dequantize(q)
    assert out.dtype == np.float32
    assert np.isnan(out[0])
    expected = np.array(
        [-((127 / 127.5) ** 2), -((64 / 127.5) ** 2), 0.0, (64 / 127.5) ** 2, (127 / 127.5) ** 2]
    )
    np.testing.assert_allclose(out[1:], expected, atol=1e-6)
    assert out[1] == pytest.approx(-0.9922, abs=1e-3)
    # Float input with NaN already at nodata (what the base class passes).
    f = np.array([np.nan, 64.0, -64.0], dtype=np.float32)
    out_f = dequantize(f)
    assert np.isnan(out_f[0])
    np.testing.assert_allclose(out_f[1:], [(64 / 127.5) ** 2, -((64 / 127.5) ** 2)], atol=1e-6)


def test_olmoearth_zone_resolution():
    bbox_utm = (500000.0, 4000000.0, 502880.0, 4002880.0)
    assert resolve_zone(bbox_utm, "EPSG:32618") == 18
    assert resolve_zone(bbox_utm, "EPSG:32718") == 18
    # WGS84 bboxes: centre longitude drives the zone.
    assert resolve_zone((-74.1, 40.6, -73.9, 40.8), "EPSG:4326") == 18
    assert resolve_zone((179.8, 10.0, 180.0, 10.2), "EPSG:4326") == 60
    assert resolve_zone((-180.0, 10.0, -179.8, 10.2), "EPSG:4326") == 1
    # Non-UTM projected CRS: centre is reprojected (Web Mercator over NYC).
    bbox_3857 = tuple(transform_bounds("EPSG:4326", "EPSG:3857", -74.1, 40.6, -73.9, 40.8))
    assert resolve_zone(bbox_3857, "EPSG:3857") == 18


def test_olmoearth_invalid_dims_raises():
    with pytest.raises(ValueError, match="dims"):
        OlmoEarthBackend(dims=100)


def test_olmoearth_chunk_cache_dir_rejected(tmp_path):
    with pytest.raises(ValueError, match="chunk_cache_dir"):
        OlmoEarthBackend(chunk_cache_dir=tmp_path / "chunks")


def test_olmoearth_missing_zarr_raises_actionable_import_error(monkeypatch):
    import importlib.util as iu

    real_find_spec = iu.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "zarr":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(iu, "find_spec", fake_find_spec)
    with pytest.raises(ImportError, match=r"gelos\[olmoearth-embeddings\]"):
        OlmoEarthBackend()


def test_olmoearth_fetch_utm_passthrough_dequantized(tmp_path):
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store)
    backend = OlmoEarthBackend(url=str(store))
    bbox = (OLMO_X0, OLMO_Y0 - 160, OLMO_X0 + 160, OLMO_Y0)
    arr = backend.fetch(bbox, crs="EPSG:32618", year=2024)
    assert arr.shape == (16, 16, 128)
    assert arr.dtype == np.float32
    assert not np.isnan(arr).any()
    assert np.abs(arr).max() <= 1.0
    expected = _olmo_expected_band_values(128)
    for b in range(128):
        np.testing.assert_allclose(arr[..., b], expected[b], atol=1e-6)
    gc.collect()


def test_olmoearth_dims_64_reads_first_bands(tmp_path):
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store)
    backend = OlmoEarthBackend(url=str(store), dims=64)
    bbox = (OLMO_X0, OLMO_Y0 - 160, OLMO_X0 + 160, OLMO_Y0)
    arr = backend.fetch(bbox, crs="EPSG:32618", year=2024)
    assert arr.shape == (16, 16, 64)
    expected = _olmo_expected_band_values(128)[:64]
    for b in range(64):
        np.testing.assert_allclose(arr[..., b], expected[b], atol=1e-6)
    # Inner chunk (16x16), not the 32x32 shard: pins prefer_inner_chunks.
    reader = backend._reader(18)
    assert reader._chunk_layout[:2] == (16, 16)
    assert reader.band_slice == (0, 64)
    gc.collect()


def test_olmoearth_year_lookup_and_missing_year(tmp_path):
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store, years=(2023, 2024, 2025))
    backend = OlmoEarthBackend(url=str(store))
    reader = backend._reader(18)
    assert reader._year_to_index(2023) == 0
    assert reader._year_to_index(2024) == 1
    assert reader._year_to_index(2025) == 2
    bbox = (OLMO_X0, OLMO_Y0 - 160, OLMO_X0 + 160, OLMO_Y0)
    with pytest.raises(ValueError, match="out of range") as excinfo:
        backend.fetch(bbox, crs="EPSG:32618", year=2019)
    assert "2023" in str(excinfo.value)
    gc.collect()


def test_olmoearth_nodata_becomes_nan(tmp_path):
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store, nodata_corner=True)
    backend = OlmoEarthBackend(url=str(store))
    bbox = (OLMO_X0, OLMO_Y0 - 160, OLMO_X0 + 160, OLMO_Y0)
    arr = backend.fetch(bbox, crs="EPSG:32618", year=2023)
    assert arr.shape == (16, 16, 128)
    assert np.isnan(arr[0, 0, :]).all()
    assert not np.isnan(arr[1:, :, :]).any()
    assert not np.isnan(arr[:, 1:, :]).any()
    gc.collect()


def test_olmoearth_wgs84_request_with_out_shape(tmp_path):
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store)
    backend = OlmoEarthBackend(url=str(store))
    bbox_utm = (OLMO_X0 + 240, OLMO_Y0 - 400, OLMO_X0 + 400, OLMO_Y0 - 240)
    bbox_wgs84 = tuple(transform_bounds("EPSG:32618", "EPSG:4326", *bbox_utm))
    arr = backend.fetch(bbox_wgs84, crs="EPSG:4326", year=2024, out_shape=(8, 8))
    assert arr.shape == (8, 8, 128)
    assert not np.isnan(arr).any()
    expected = _olmo_expected_band_values(128)
    np.testing.assert_allclose(arr[..., 5], expected[5], atol=1e-6)
    gc.collect()


def test_olmoearth_southern_crs_reprojects(tmp_path):
    """A southern-hemisphere UTM request (EPSG:327NN) is routed to utmNN and reprojected."""
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store)
    backend = OlmoEarthBackend(url=str(store))
    bbox_n = (OLMO_X0 + 240, OLMO_Y0 - 400, OLMO_X0 + 400, OLMO_Y0 - 240)
    # EPSG:32718 differs from EPSG:32618 by a 10,000,000 m false northing.
    bbox_s = tuple(transform_bounds("EPSG:32618", "EPSG:32718", *bbox_n))
    assert bbox_s[1] == pytest.approx(bbox_n[1] + 10_000_000, abs=1e-3)
    north = backend.fetch(bbox_n, crs="EPSG:32618", year=2024, out_shape=(8, 8))
    south = backend.fetch(bbox_s, crs="EPSG:32718", year=2024, out_shape=(8, 8))
    assert south.shape == (8, 8, 128)
    assert not np.isnan(south).any()
    np.testing.assert_allclose(south, north, atol=1e-6)
    assert list(backend._readers) == [18]
    gc.collect()


def test_olmoearth_cache_roundtrip_dequantized(tmp_path, monkeypatch):
    """Cache files stay int8 on disk; the cache-read path dequantizes like the fresh path."""
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store, nodata_corner=True)
    cache_dir = tmp_path / "cache"
    backend = OlmoEarthBackend(url=str(store), cache_dir=cache_dir)
    bbox = (OLMO_X0, OLMO_Y0 - 160, OLMO_X0 + 160, OLMO_Y0)
    first = backend.fetch(bbox, crs="EPSG:32618", year=2023)

    cached_files = list(cache_dir.glob("*.tif"))
    assert len(cached_files) == 1, f"expected one cached tif, got {cached_files}"
    assert cached_files[0].name.startswith("olmoearth_utm18_d128_EPSG32618_2023_")
    with rasterio.open(cached_files[0]) as src:
        assert src.count == 128
        assert set(src.dtypes) == {"int8"}  # one dtype entry per band
        assert src.nodata == -128

    def boom(self):
        raise AssertionError("data_array must not be accessed on cache hit")

    monkeypatch.setattr(_OlmoEarthZoneReader, "data_array", property(boom))
    second = backend.fetch(bbox, crs="EPSG:32618", year=2023)
    assert second.shape == first.shape
    assert second.dtype == np.float32
    np.testing.assert_allclose(second, first, equal_nan=True, atol=1e-6)
    assert np.isnan(second[0, 0, :]).all()
    assert np.abs(np.nan_to_num(second)).max() <= 1.0
    gc.collect()


def test_olmoearth_multi_zone_batch_preserves_order(tmp_path):
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store, zones=(18, 19))
    backend = OlmoEarthBackend(url=str(store))
    z18 = lambda k: (OLMO_X0 + 160 * k, OLMO_Y0 - 160, OLMO_X0 + 160 * (k + 1), OLMO_Y0)  # noqa: E731
    z19 = lambda k: (  # noqa: E731
        OLMO_ZONE19_X0 + 160 * k,
        OLMO_Y0 - 320,
        OLMO_ZONE19_X0 + 160 * (k + 1),
        OLMO_Y0 - 160,
    )
    requests = [
        FetchRequest(bbox=z18(0), crs="EPSG:32618", year=2024),
        FetchRequest(bbox=z19(0), crs="EPSG:32619", year=2024),
        FetchRequest(bbox=z18(1), crs="EPSG:32618", year=2025),
        FetchRequest(bbox=z19(1), crs="EPSG:32619", year=2023),
        FetchRequest(bbox=z18(2), crs="EPSG:32618", year=2024),
    ]
    batch = backend.fetch_batch(requests)
    assert len(batch) == 5
    assert sorted(backend._readers) == [18, 19]
    individual = [backend.fetch(r.bbox, r.crs, r.year, out_shape=r.out_shape) for r in requests]
    for i, (b, ind) in enumerate(zip(batch, individual)):
        assert b.shape == ind.shape == (16, 16, 128), f"request {i}"
        np.testing.assert_allclose(b, ind, equal_nan=True, atol=1e-6)
    gc.collect()


def test_olmoearth_fetch_batch_minimizes_inner_chunk_loads(tmp_path, monkeypatch):
    """Four bboxes inside one 16x16 inner chunk trigger exactly one .load() call.

    The bboxes keep a one-pixel margin from the chunk edges: every OlmoEarth
    request is resampled onto a footprint-pinned grid, so its source
    selection is padded by one pixel (see ``_pad_bbox_src``) and a bbox flush
    against the chunk boundary would legitimately pull in the neighbour.
    """
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store)
    backend = OlmoEarthBackend(url=str(store))

    load_count = {"n": 0}
    original_load = xr.DataArray.load

    def counting_load(self, *args, **kwargs):
        load_count["n"] += 1
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(xr.DataArray, "load", counting_load)

    x0, y0 = OLMO_X0, OLMO_Y0
    requests = [
        FetchRequest(bbox=(x0 + 10, y0 - 70, x0 + 70, y0 - 10), crs="EPSG:32618", year=2024),
        FetchRequest(bbox=(x0 + 90, y0 - 70, x0 + 150, y0 - 10), crs="EPSG:32618", year=2024),
        FetchRequest(bbox=(x0 + 10, y0 - 150, x0 + 70, y0 - 90), crs="EPSG:32618", year=2024),
        FetchRequest(bbox=(x0 + 90, y0 - 150, x0 + 150, y0 - 90), crs="EPSG:32618", year=2024),
    ]
    results = backend.fetch_batch(requests)
    assert all(r.shape == (6, 6, 128) for r in results)
    assert load_count["n"] == 1, f"expected 1 .load() call, got {load_count['n']}"
    gc.collect()


# ---------------------------------------------------------------------------
# OlmoEarth mean path (no out_shape): window pinned to the chip footprint
# ---------------------------------------------------------------------------


def _requantize(v: np.ndarray) -> np.ndarray:
    """Invert ``dequantize`` for non-negative codes: ``(q / 127.5) ** 2 -> q``."""
    return np.rint(np.sqrt(np.asarray(v, dtype=np.float64)) * 127.5).astype(int)


def test_olmoearth_mean_path_window_pinned_to_footprint(tmp_path):
    """A request without ``out_shape`` (``pool: mean``) over a footprint whose edges
    sit half a pixel (5 m) off the 10 m zone grid comes back as exactly
    ``extent / 10`` samples per axis — 16x16 for 160 m — drawn from 16 consecutive
    source rows/cols the footprint covers, and ``mean_pool`` averages exactly those
    pixels. Unpinned, ``.sel`` would pass through the 17x17 pixels whose centres fall
    in the bbox (a 170 m window hanging half a pixel outside the footprint)."""
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store, index_bands=True)
    backend = OlmoEarthBackend(url=str(store))
    minx, maxy = OLMO_X0 + 45.0, OLMO_Y0 - 45.0  # edges at 5 mod 10 m
    bbox = (minx, maxy - 160.0, minx + 160.0, maxy)
    arr = backend.fetch(bbox, crs="EPSG:32618", year=2024)

    assert arr.shape == (16, 16, 128)
    assert not np.isnan(arr).any()
    rows = _requantize(arr[:, :, 0])
    cols = _requantize(arr[:, :, 1])
    # Each output row samples one source row, each output column one source column.
    assert (rows == rows[:, :1]).all() and (cols == cols[:1, :]).all()
    first_row, first_col = int(rows[0, 0]), int(cols[0, 0])
    # The footprint spans source rows/cols 4..20 (17 pixels at half-pixel offset);
    # the pinned window picks 16 consecutive ones, starting at 4 or 5.
    assert rows[:, 0].tolist() == list(range(first_row, first_row + 16))
    assert cols[0, :].tolist() == list(range(first_col, first_col + 16))
    assert 4 <= first_row <= 5 and 4 <= first_col <= 5

    pooled = mean_pool(arr)
    assert pooled.shape == (128,)
    exp_rows = dequantize(np.arange(first_row, first_row + 16, dtype=np.int8))
    exp_cols = dequantize(np.arange(first_col, first_col + 16, dtype=np.int8))
    np.testing.assert_allclose(pooled[0], exp_rows.mean(), atol=1e-6)
    np.testing.assert_allclose(pooled[1], exp_cols.mean(), atol=1e-6)
    np.testing.assert_allclose(pooled[2:], _olmo_expected_band_values(128)[2:], atol=1e-6)
    gc.collect()


def test_olmoearth_mean_path_zone_and_reproject_branches_agree(tmp_path):
    """Same half-pixel-offset footprint without ``out_shape`` via the zone CRS
    (EPSG:326NN, no reprojection needed) and via the southern CRS (EPSG:327NN,
    reproject branch): both are pinned to the same 16x16 footprint grid."""
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store)
    backend = OlmoEarthBackend(url=str(store))
    minx, maxy = OLMO_X0 + 45.0, OLMO_Y0 - 45.0
    bbox_n = (minx, maxy - 160.0, minx + 160.0, maxy)
    bbox_s = tuple(transform_bounds("EPSG:32618", "EPSG:32718", *bbox_n))
    north = backend.fetch(bbox_n, crs="EPSG:32618", year=2024)
    south = backend.fetch(bbox_s, crs="EPSG:32718", year=2024)
    assert north.shape == south.shape == (16, 16, 128)
    assert not np.isnan(north).any() and not np.isnan(south).any()
    np.testing.assert_allclose(south, north, atol=1e-6)
    assert list(backend._readers) == [18]
    gc.collect()


def test_olmoearth_mean_path_nan_aware(tmp_path):
    """Nodata inside a pinned window stays NaN and ``mean_pool`` ignores it."""
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store, nodata_corner=True)
    backend = OlmoEarthBackend(url=str(store))
    bbox = (OLMO_X0, OLMO_Y0 - 160, OLMO_X0 + 160, OLMO_Y0)
    arr = backend.fetch(bbox, crs="EPSG:32618", year=2023)
    assert arr.shape == (16, 16, 128)
    assert np.isnan(arr[0, 0, :]).all()
    assert np.isnan(arr).sum() == 128
    pooled = mean_pool(arr)
    assert np.isfinite(pooled).all()
    np.testing.assert_allclose(pooled, _olmo_expected_band_values(128), atol=1e-6)
    gc.collect()


def _spy_open_zarr(monkeypatch) -> list[dict]:
    """Record the kwargs of every ``xr.open_zarr`` call made by the backends."""
    captured: list[dict] = []
    real_open_zarr = xr.open_zarr

    def spy(*args, **kwargs):
        captured.append(dict(kwargs))
        return real_open_zarr(*args, **kwargs)

    monkeypatch.setattr(xr, "open_zarr", spy)
    return captured


def test_olmoearth_zone_reader_opens_without_dask(tmp_path, monkeypatch):
    """Zone groups are opened with ``chunks=None`` (no dask graph), values unchanged.

    The real zone array has ~50M inner chunks; a dask-wrapped open pays ~a
    minute of graph optimisation per windowed read. ``da.chunks is None`` is
    asserted on the loaded variable, and — because the test image has no dask,
    so that alone would pass trivially — the open kwargs are spied as well.
    """
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store)
    captured = _spy_open_zarr(monkeypatch)
    backend = OlmoEarthBackend(url=str(store))
    bbox = (OLMO_X0, OLMO_Y0 - 160, OLMO_X0 + 160, OLMO_Y0)
    arr = backend.fetch(bbox, crs="EPSG:32618", year=2024)

    reader = backend._reader(18)
    assert reader.use_dask is False
    assert captured, "xr.open_zarr was never called"
    assert all("chunks" in kw and kw["chunks"] is None for kw in captured), captured
    assert reader.data_array.chunks is None
    # Inner-chunk batching still resolves through encoding["chunks"] without dask.
    assert reader._chunk_layout[:2] == (16, 16)

    assert arr.shape == (16, 16, 128)
    expected = _olmo_expected_band_values(128)
    for b in range(128):
        np.testing.assert_allclose(arr[..., b], expected[b], atol=1e-6)
    gc.collect()


def test_alphaearth_opens_with_default_dask_chunking(tmp_path, monkeypatch):
    """AlphaEarth keeps xarray's default chunking: ``chunks`` is never passed."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, chunks_yx=(8, 8))
    captured = _spy_open_zarr(monkeypatch)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    assert backend.use_dask is True
    arr = backend.fetch((-71.7, 42.2, -71.5, 42.5), crs="EPSG:4326", year=2018)
    assert captured, "xr.open_zarr was never called"
    assert all("chunks" not in kw for kw in captured), captured
    assert arr.shape[-1] == 64
    gc.collect()


def test_olmoearth_proj_code_mismatch_raises(tmp_path):
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store, proj_code_override="EPSG:32619")
    backend = OlmoEarthBackend(url=str(store))
    bbox = (OLMO_X0, OLMO_Y0 - 160, OLMO_X0 + 160, OLMO_Y0)
    with pytest.raises(ValueError, match="proj:code"):
        backend.fetch(bbox, crs="EPSG:32618", year=2024)
    gc.collect()


def test_olmoearth_all_nodata_batch_warns(tmp_path):
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store, nodata_year_index=1)
    backend = OlmoEarthBackend(url=str(store))
    bbox = (OLMO_X0, OLMO_Y0 - 160, OLMO_X0 + 160, OLMO_Y0)

    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        arr = backend.fetch(bbox, crs="EPSG:32618", year=2024)
    finally:
        logger.remove(sink_id)

    assert arr.shape == (16, 16, 128)
    assert np.isnan(arr).all()
    warnings_text = "\n".join(captured)
    assert "1/1 footprints" in warnings_text
    assert "2017-2022" in warnings_text
    gc.collect()


# ---------------------------------------------------------------------------
# Transient-error retry around store opens and chunk loads (ZarrMosaicBackend._with_retries)
# ---------------------------------------------------------------------------

_HAS_AIOHTTP = importlib.util.find_spec("aiohttp") is not None


def _connection_reset() -> BaseException:
    return ConnectionResetError(104, "Connection reset by peer")


def _aiohttp_payload_error() -> BaseException:
    import aiohttp

    # Exactly what the real store produces mid-download (see smoke logs).
    exc = aiohttp.ClientPayloadError(
        "Response payload is not completed: <ContentLengthError: 400, message="
        "'Not enough data to satisfy content length header (received 16384 of 93888 bytes).'>"
    )
    exc.__cause__ = ConnectionResetError(104, "Connection reset by peer")
    return exc


_TRANSIENT_FACTORIES = [
    pytest.param(_connection_reset, id="ConnectionResetError"),
    pytest.param(
        _aiohttp_payload_error,
        id="aiohttp.ClientPayloadError",
        marks=pytest.mark.skipif(not _HAS_AIOHTTP, reason="aiohttp not installed"),
    ),
]


class _FlakyLoad:
    """Monkeypatch target for ``xr.DataArray.load`` that fails the first ``n_fail`` calls.

    The failure counter is shared across the prefetch pool's worker threads,
    so it is guarded by a lock; after ``n_fail`` failures every call succeeds
    (or, with ``n_fail=None``, every call fails).
    """

    def __init__(self, make_exc, n_fail: int | None) -> None:
        self.make_exc = make_exc
        self.n_fail = n_fail
        self.calls = 0
        self._lock = threading.Lock()
        self._original = xr.DataArray.load

    def install(self, monkeypatch) -> None:
        # A plain function (not a callable instance) so attribute lookup on a
        # DataArray binds it as a method.
        monkeypatch.setattr(xr.DataArray, "load", lambda da, *a, **kw: self(da, *a, **kw))

    def __call__(self, da, *args, **kwargs):
        with self._lock:
            self.calls += 1
            fail = self.n_fail is None or self.calls <= self.n_fail
        if fail:
            raise self.make_exc()
        return self._original(da, *args, **kwargs)


def _patch_retry_sleep(monkeypatch) -> list[float]:
    """Replace the backoff sleep with a recorder so retry tests run instantly."""
    delays: list[float] = []
    monkeypatch.setattr(zarr_mosaic, "_sleep", lambda d: delays.append(d))
    return delays


def _capture_warnings():
    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    return captured, sink_id


@pytest.mark.parametrize("make_exc", _TRANSIENT_FACTORIES)
def test_zarr_mosaic_retries_transient_load_errors(tmp_path, monkeypatch, make_exc):
    """A chunk load that fails transiently N times then succeeds yields the same
    results as a clean run, logs one warning per retry, and backs off exponentially."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    requests = [
        FetchRequest(bbox=(-72.0, 42.85, -71.85, 42.95), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.5, 42.45, -71.4, 42.55), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.85, 42.7, -71.7, 42.85), crs="EPSG:4326", year=2018),
        FetchRequest(bbox=(-71.5, 42.45, -71.4, 42.55), crs="EPSG:4326", year=2019),
    ]
    clean = AlphaEarthBackend(url=str(zarr_path), anon=False).fetch_batch(requests)

    flaky = _FlakyLoad(make_exc, n_fail=2)
    flaky.install(monkeypatch)
    delays = _patch_retry_sleep(monkeypatch)
    backend = zarr_mosaic.ZarrMosaicBackend(
        url=str(zarr_path), anon=False, max_retries=4, retry_backoff=0.5
    )
    captured, sink_id = _capture_warnings()
    try:
        results = backend.fetch_batch(requests)
    finally:
        logger.remove(sink_id)

    assert len(results) == len(clean)
    for got, exp in zip(results, clean):
        assert got.shape == exp.shape
        np.testing.assert_array_equal(got, exp)

    # The two failures may hit one chunk twice (retry 1/4 then 2/4) or two
    # chunks once each (retry 1/4 twice) — the prefetch pool is concurrent.
    retry_lines = [m for m in captured if "load failed" in m and "retry " in m]
    assert len(retry_lines) == 2, captured
    assert all("/4 in " in m for m in retry_lines)
    assert all(type(make_exc()).__name__ in m for m in retry_lines)
    assert len(delays) == 2
    # Each delay is 0.5 * 2**attempt (attempt 0 or 1) plus up to 10 % jitter.
    assert all(0.5 <= d <= 0.55 or 1.0 <= d <= 1.1 for d in delays), delays
    gc.collect()


def test_zarr_mosaic_retry_exhaustion_reraises_original(tmp_path, monkeypatch):
    """When every attempt fails the original exception (not a wrapper) propagates,
    after exactly max_retries sleeps and max_retries + 1 load attempts."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    flaky = _FlakyLoad(_connection_reset, n_fail=None)
    flaky.install(monkeypatch)
    delays = _patch_retry_sleep(monkeypatch)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)
    backend.max_retries = 2
    backend.retry_backoff = 0.25

    captured, sink_id = _capture_warnings()
    try:
        with pytest.raises(ConnectionResetError, match="reset by peer") as excinfo:
            # Single uncached request -> notebook fast path, one chunk.
            backend.fetch((-72.0, 42.85, -71.93, 42.95), "EPSG:4326", 2018)
    finally:
        logger.remove(sink_id)

    assert excinfo.value.errno == 104
    assert flaky.calls == 3
    # Exponential backoff on a single chunk: 0.25, then 0.5 (each + <=10 % jitter).
    assert len(delays) == 2
    assert 0.25 <= delays[0] <= 0.275 and 0.5 <= delays[1] <= 0.55, delays
    retry_lines = [m for m in captured if "load failed" in m and "retry " in m]
    assert [("retry 1/2" in m, "retry 2/2" in m) for m in retry_lines] == [
        (True, False),
        (False, True),
    ]
    assert any("giving up after 3 attempt(s)" in m for m in captured), captured
    gc.collect()


@pytest.mark.parametrize(
    "make_exc",
    [
        pytest.param(lambda: KeyError("c/0/0/0"), id="KeyError"),
        pytest.param(lambda: FileNotFoundError("https://host/store/c/0/0"), id="404"),
    ],
)
def test_zarr_mosaic_non_transient_errors_not_retried(tmp_path, monkeypatch, make_exc):
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    flaky = _FlakyLoad(make_exc, n_fail=None)
    flaky.install(monkeypatch)
    delays = _patch_retry_sleep(monkeypatch)
    backend = AlphaEarthBackend(url=str(zarr_path), anon=False)

    with pytest.raises(type(make_exc())):
        backend.fetch((-72.0, 42.85, -71.93, 42.95), "EPSG:4326", 2018)
    assert flaky.calls == 1
    assert delays == []
    gc.collect()


def test_zarr_mosaic_max_retries_zero_disables_retry(tmp_path, monkeypatch):
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    flaky = _FlakyLoad(_connection_reset, n_fail=1)
    flaky.install(monkeypatch)
    delays = _patch_retry_sleep(monkeypatch)
    backend = zarr_mosaic.ZarrMosaicBackend(url=str(zarr_path), anon=False, max_retries=0)
    with pytest.raises(ConnectionResetError):
        backend.fetch((-72.0, 42.85, -71.93, 42.95), "EPSG:4326", 2018)
    assert flaky.calls == 1 and delays == []


def test_zarr_mosaic_retry_options_validated():
    with pytest.raises(ValueError, match="max_retries"):
        zarr_mosaic.ZarrMosaicBackend(url="x.zarr", max_retries=-1)
    with pytest.raises(ValueError, match="retry_backoff"):
        zarr_mosaic.ZarrMosaicBackend(url="x.zarr", retry_backoff=-0.1)
    with pytest.raises(ValueError, match="max_retries"):
        OlmoEarthBackend(max_retries=-1)


def _http_status_error(status: int) -> BaseException:
    """What fsspec's HTTP filesystem raises (unwrapped) for a non-2xx response."""
    import aiohttp
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    # A real RequestInfo: aiohttp's __str__ formats request_info.real_url, which
    # the backend's retry log lines exercise via f"{exc}".
    url = URL("https://data.source.coop/olmoearth/embeddings.zarr/utm34/y/c/8")
    info = aiohttp.RequestInfo(url, "GET", CIMultiDictProxy(CIMultiDict()), url)
    return aiohttp.ClientResponseError(request_info=info, history=(), status=status)


def test_is_transient_error_classification():
    is_transient = zarr_mosaic._is_transient_error
    assert is_transient(ConnectionResetError(104, "reset"))
    assert is_transient(TimeoutError("read timed out"))
    # Wrapped transient error is recognised through the cause chain.
    wrapper = RuntimeError("store read failed")
    wrapper.__cause__ = ConnectionResetError(104, "reset")
    assert is_transient(wrapper)
    assert not is_transient(KeyError("k"))
    assert not is_transient(FileNotFoundError("404"))
    assert not is_transient(PermissionError("403"))
    assert not is_transient(OSError("generic"))
    if _HAS_AIOHTTP:
        assert is_transient(_aiohttp_payload_error())
        # HTTP statuses: 429 and 5xx are the proxy misbehaving (503 observed
        # live while opening a zone group); 4xx are configuration errors.
        assert is_transient(_http_status_error(503))
        assert is_transient(_http_status_error(502))
        assert is_transient(_http_status_error(429))
        assert not is_transient(_http_status_error(404))
        assert not is_transient(_http_status_error(403))
        # ... and the same through a zarr/fsspec wrapper.
        wrapped_503 = RuntimeError("store open failed")
        wrapped_503.__cause__ = _http_status_error(503)
        assert is_transient(wrapped_503)
        wrapped_404 = RuntimeError("store open failed")
        wrapped_404.__cause__ = _http_status_error(404)
        assert not is_transient(wrapped_404)


class _FlakyOpenZarr:
    """Monkeypatch target for ``xr.open_zarr`` that fails the first ``n_fail`` calls."""

    def __init__(self, make_exc, n_fail: int | None) -> None:
        self.make_exc = make_exc
        self.n_fail = n_fail
        self.calls = 0
        self._original = xr.open_zarr

    def install(self, monkeypatch) -> None:
        monkeypatch.setattr(xr, "open_zarr", self)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.n_fail is None or self.calls <= self.n_fail:
            raise self.make_exc()
        return self._original(*args, **kwargs)


_OPEN_BBOX = (-72.0, 42.85, -71.93, 42.95)


@pytest.mark.skipif(not _HAS_AIOHTTP, reason="aiohttp not installed")
def test_zarr_mosaic_open_retries_transient_http_errors(tmp_path, monkeypatch):
    """A 503 from ``xr.open_zarr`` (eager coordinate read) is retried like a chunk
    read: the open succeeds after N failures, the fetch is correct, one warning per
    retry, exponential delays."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    clean = AlphaEarthBackend(url=str(zarr_path), anon=False).fetch(_OPEN_BBOX, "EPSG:4326", 2018)

    flaky = _FlakyOpenZarr(lambda: _http_status_error(503), n_fail=2)
    flaky.install(monkeypatch)
    delays = _patch_retry_sleep(monkeypatch)
    backend = zarr_mosaic.ZarrMosaicBackend(
        url=str(zarr_path), anon=False, max_retries=4, retry_backoff=0.5
    )
    captured, sink_id = _capture_warnings()
    try:
        got = backend.fetch(_OPEN_BBOX, "EPSG:4326", 2018)
    finally:
        logger.remove(sink_id)

    np.testing.assert_array_equal(got, clean)
    assert flaky.calls == 3
    retry_lines = [m for m in captured if "store open at" in m and "retry " in m]
    assert len(retry_lines) == 2, captured
    assert "retry 1/4" in retry_lines[0] and "retry 2/4" in retry_lines[1]
    assert all("ClientResponseError" in m and "503" in m for m in retry_lines)
    assert len(delays) == 2
    assert 0.5 <= delays[0] <= 0.55 and 1.0 <= delays[1] <= 1.1, delays
    gc.collect()


@pytest.mark.skipif(not _HAS_AIOHTTP, reason="aiohttp not installed")
def test_zarr_mosaic_open_retry_exhaustion_reraises_and_leaves_backend_usable(
    tmp_path, monkeypatch
):
    """Exhausting the open retry re-raises the original 503 and caches nothing, so
    the same backend opens and reads fine once the network recovers."""
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    clean = AlphaEarthBackend(url=str(zarr_path), anon=False).fetch(_OPEN_BBOX, "EPSG:4326", 2018)

    flaky = _FlakyOpenZarr(lambda: _http_status_error(503), n_fail=None)
    flaky.install(monkeypatch)
    delays = _patch_retry_sleep(monkeypatch)
    backend = zarr_mosaic.ZarrMosaicBackend(
        url=str(zarr_path), anon=False, max_retries=2, retry_backoff=0.25
    )
    import aiohttp

    captured, sink_id = _capture_warnings()
    try:
        with pytest.raises(aiohttp.ClientResponseError) as excinfo:
            backend.fetch(_OPEN_BBOX, "EPSG:4326", 2018)
    finally:
        logger.remove(sink_id)
    assert excinfo.value.status == 503
    assert flaky.calls == 3 and len(delays) == 2
    assert any("store open at" in m and "giving up after 3 attempt(s)" in m for m in captured)
    assert backend._dataset is None and backend._da is None

    # "Network recovers": the very next fetch opens the store and reads correctly.
    flaky.n_fail = 0
    got = backend.fetch(_OPEN_BBOX, "EPSG:4326", 2018)
    np.testing.assert_array_equal(got, clean)
    assert flaky.calls == 4 and len(delays) == 2
    gc.collect()


@pytest.mark.parametrize(
    "make_exc",
    [
        pytest.param(
            lambda: FileNotFoundError("https://host/store.zarr/zarr.json"), id="fsspec-404"
        ),
        pytest.param(
            lambda: _http_status_error(404),
            id="ClientResponseError-404",
            marks=pytest.mark.skipif(not _HAS_AIOHTTP, reason="aiohttp not installed"),
        ),
    ],
)
def test_zarr_mosaic_open_404_not_retried(tmp_path, monkeypatch, make_exc):
    zarr_path = tmp_path / "synthetic.zarr"
    _write_synthetic_zarr(zarr_path, H=32, W=32, chunks_yx=(8, 8))
    flaky = _FlakyOpenZarr(make_exc, n_fail=None)
    flaky.install(monkeypatch)
    delays = _patch_retry_sleep(monkeypatch)
    backend = zarr_mosaic.ZarrMosaicBackend(url=str(zarr_path), anon=False, max_retries=4)
    with pytest.raises(type(make_exc())):
        backend.fetch(_OPEN_BBOX, "EPSG:4326", 2018)
    # One consolidated=True attempt; FileNotFoundError (an OSError) also triggers
    # the consolidated=False fallback, which fails the same way - still no retry.
    assert flaky.calls <= 2 and delays == []
    assert backend._dataset is None
    gc.collect()


@pytest.mark.skipif(not _HAS_AIOHTTP, reason="aiohttp not installed")
def test_olmoearth_zone_open_retries_transient_http_errors(tmp_path, monkeypatch):
    """The OlmoEarth zone readers inherit the open retry (the live 503 was raised
    while opening ``utm34``'s coordinate arrays)."""
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store)
    flaky = _FlakyOpenZarr(lambda: _http_status_error(503), n_fail=1)
    flaky.install(monkeypatch)
    delays = _patch_retry_sleep(monkeypatch)
    backend = OlmoEarthBackend(url=str(store), max_retries=3, retry_backoff=0.01)
    captured, sink_id = _capture_warnings()
    try:
        arr = backend.fetch((OLMO_X0, OLMO_Y0 - 80, OLMO_X0 + 80, OLMO_Y0), "EPSG:32618", 2024)
    finally:
        logger.remove(sink_id)
    assert arr.shape == (8, 8, 128)
    np.testing.assert_allclose(arr[0, 0], _olmo_expected_band_values(), atol=1e-6)
    assert flaky.calls == 2 and len(delays) == 1
    assert any("OlmoEarth utm18 store open at" in m and "retry 1/3" in m for m in captured)
    gc.collect()


def test_olmoearth_passes_retry_options_to_zone_readers(tmp_path, monkeypatch):
    """OlmoEarthBackend(max_retries=..., retry_backoff=...) reaches every zone reader
    and a flaky OlmoEarth read recovers with correct dequantized values."""
    store = tmp_path / "olmo.zarr"
    _write_synthetic_olmoearth_zarr(store)
    backend = OlmoEarthBackend(url=str(store), max_retries=3, retry_backoff=0.01)
    reader = backend._reader(18)
    assert reader.max_retries == 3 and reader.retry_backoff == 0.01

    flaky = _FlakyLoad(_connection_reset, n_fail=1)
    flaky.install(monkeypatch)
    delays = _patch_retry_sleep(monkeypatch)
    arr = backend.fetch((OLMO_X0, OLMO_Y0 - 80, OLMO_X0 + 80, OLMO_Y0), "EPSG:32618", 2024)
    assert arr.shape == (8, 8, 128)
    np.testing.assert_allclose(arr[0, 0], _olmo_expected_band_values(), atol=1e-6)
    assert flaky.calls == 2 and len(delays) == 1
    gc.collect()


# ---------------------------------------------------------------------------
# Dispatcher (gelos.generation.generate_embeddings) routing
# ---------------------------------------------------------------------------


def test_dispatch_both_blocks_raises(tmp_path):
    yaml_path = tmp_path / "both.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "data_version": "v0",
                "data": {"class_path": "anything", "init_args": {}},
                "model": {"class_path": "anything"},
                "cloud_embedding": {"backend": "stub"},
            }
        )
    )
    with pytest.raises(ValueError, match="cloud_embedding.*model"):
        generate_embeddings(
            yaml_path,
            raw_data_dir=tmp_path,
            embedding_dir=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# Stub backend + end-to-end through gelos.generation.generate_embeddings
# ---------------------------------------------------------------------------


class _StubBackend:
    """Deterministic backend that returns a 16x16x{dim} array seeded by bbox.

    Accepts arbitrary kwargs (incl. ``cache_dir`` injected by the runner) so
    the dispatcher's auto-injection doesn't crash test-only backends.
    """

    def __init__(self, embedding_dim: int = 64, **kwargs) -> None:
        self.embedding_dim = embedding_dim
        self.init_kwargs = kwargs

    def fetch(self, bbox, crs, year, out_shape=None):
        seed = abs(hash((tuple(bbox), str(crs), int(year)))) % (2**31)
        rng = np.random.default_rng(seed)
        H, W = out_shape if out_shape is not None else (16, 16)
        return rng.random((H, W, self.embedding_dim)).astype(np.float32)


@pytest.fixture()
def cloud_run_dirs(tmp_path, monkeypatch):
    """Set up directories + chip tracker with year + lulc columns; register stub backend."""
    monkeypatch.setitem(cloud_runner.BACKENDS, "stub", _StubBackend)

    data_version = "v0.50.1"
    raw_data_dir = tmp_path / "raw"
    embedding_dir = tmp_path / "interim"
    processed_data_dir = tmp_path / "processed"
    figures_dir = tmp_path / "figures"

    versioned_data = raw_data_dir / data_version
    versioned_data.mkdir(parents=True)

    sensors = {
        "S2L2A": len(ExampleGELOSDataSet.S2L2A_BAND_NAMES),
        "S1RTC": len(ExampleGELOSDataSet.S1RTC_BAND_NAMES),
        "DEM": len(ExampleGELOSDataSet.DEM_BAND_NAMES),
    }
    n_timesteps = {"S2L2A": 1, "S1RTC": 1, "DEM": 1}
    create_test_geojson(versioned_data, n_samples=2, sensors=sensors, n_timesteps=n_timesteps, img_size=96)

    tracker = versioned_data / "gelos_chip_tracker.geojson"
    gdf = gpd.read_file(tracker)
    gdf["lulc"] = [1, 2]
    gdf["year"] = [2023, 2024]
    gdf.to_file(tracker, driver="GeoJSON")

    return {
        "data_version": data_version,
        "raw_data_dir": raw_data_dir,
        "embedding_dir": embedding_dir,
        "processed_data_dir": processed_data_dir,
        "figures_dir": figures_dir,
    }


def test_dispatch_cloud_e2e(cloud_run_dirs):
    """generate_embeddings (the unified entry point) routes a cloud yaml through the runner."""
    yaml_path = Path(__file__).parent / "fixtures" / "example_cloud_embedding_config.yaml"

    generate_embeddings(
        yaml_path,
        raw_data_dir=cloud_run_dirs["raw_data_dir"],
        embedding_dir=cloud_run_dirs["embedding_dir"],
    )

    config_stem = yaml_path.stem
    data_version = cloud_run_dirs["data_version"]
    output_dir = cloud_run_dirs["embedding_dir"] / data_version / config_stem

    strat_dir = output_dir / "center_2x2"
    assert strat_dir.is_dir(), f"missing strategy dir {strat_dir}"
    parquets = sorted(strat_dir.glob("*_embedding.parquet"))
    assert len(parquets) == 2

    for pq in parquets:
        df = pd.read_parquet(pq)
        assert "embedding" in df.columns
        assert "file_id" in df.columns
        emb = np.asarray(df["embedding"].iloc[0].tolist())
        assert emb.shape == (4, 64)

    flatten_dir = output_dir / "center_2x2_flatten"
    assert flatten_dir.is_dir(), f"missing strategy dir {flatten_dir}"
    flatten_parquets = sorted(flatten_dir.glob("*_embedding.parquet"))
    assert len(flatten_parquets) == 2
    for pq in flatten_parquets:
        df = pd.read_parquet(pq)
        emb = np.asarray(df["embedding"].iloc[0].tolist())
        assert emb.shape == (4, 16 * 16 * 64)

    assert (output_dir / ".embeddings_complete").exists()
    assert (output_dir / "_footprint_cache").is_dir()

    # Re-run should be a no-op (marker present)
    generate_embeddings(
        yaml_path,
        raw_data_dir=cloud_run_dirs["raw_data_dir"],
        embedding_dir=cloud_run_dirs["embedding_dir"],
    )

    # Analysis consumes the parquets without modification
    run_analysis(
        yaml_path,
        raw_data_dir=cloud_run_dirs["raw_data_dir"],
        embedding_dir=cloud_run_dirs["embedding_dir"],
        processed_data_dir=cloud_run_dirs["processed_data_dir"],
        figures_base_dir=cloud_run_dirs["figures_dir"],
    )

    processed_root = cloud_run_dirs["processed_data_dir"] / data_version / config_stem
    cached = list(processed_root.rglob("*_embeddings.npy"))
    assert cached, f"expected cached _embeddings.npy under {processed_root}"
    gc.collect()


def test_footprint_manifest_cached_across_runs(cloud_run_dirs, monkeypatch):
    """Second run reuses ``_footprint_manifest.parquet`` instead of recomputing."""
    yaml_path = Path(__file__).parent / "fixtures" / "example_cloud_embedding_config.yaml"

    generate_cloud_embeddings(
        yaml_path,
        raw_data_dir=cloud_run_dirs["raw_data_dir"],
        embedding_dir=cloud_run_dirs["embedding_dir"],
    )

    output_dir = (
        cloud_run_dirs["embedding_dir"] / cloud_run_dirs["data_version"] / yaml_path.stem
    )
    cache_file = output_dir / "_footprint_manifest.parquet"
    assert cache_file.exists()

    # Drop the marker to force a second pass; the cache should short-circuit
    # compute_footprint_manifest entirely.
    (output_dir / ".embeddings_complete").unlink()

    def boom(*args, **kwargs):
        raise AssertionError("compute_footprint_manifest must not run on cache hit")

    monkeypatch.setattr(cloud_runner, "compute_footprint_manifest", boom)
    generate_cloud_embeddings(
        yaml_path,
        raw_data_dir=cloud_run_dirs["raw_data_dir"],
        embedding_dir=cloud_run_dirs["embedding_dir"],
    )
    gc.collect()


def test_runner_callable_directly(cloud_run_dirs):
    """generate_cloud_embeddings can still be called directly (notebook usage)."""
    yaml_path = Path(__file__).parent / "fixtures" / "example_cloud_embedding_config.yaml"

    generate_cloud_embeddings(
        yaml_path,
        raw_data_dir=cloud_run_dirs["raw_data_dir"],
        embedding_dir=cloud_run_dirs["embedding_dir"],
    )

    output_dir = (
        cloud_run_dirs["embedding_dir"]
        / cloud_run_dirs["data_version"]
        / yaml_path.stem
    )
    assert (output_dir / ".embeddings_complete").exists()
    gc.collect()


def test_runner_fallback_for_backend_without_fetch_batch(cloud_run_dirs):
    """Stub backend lacks fetch_batch; runner falls back to per-request fetch."""
    # Pin the contract: the stub used by the e2e suite must not implement fetch_batch
    # so the runner's hasattr branch keeps exercising the fallback path.
    assert not hasattr(_StubBackend, "fetch_batch")

    yaml_path = Path(__file__).parent / "fixtures" / "example_cloud_embedding_config.yaml"
    generate_cloud_embeddings(
        yaml_path,
        raw_data_dir=cloud_run_dirs["raw_data_dir"],
        embedding_dir=cloud_run_dirs["embedding_dir"],
    )
    output_dir = (
        cloud_run_dirs["embedding_dir"] / cloud_run_dirs["data_version"] / yaml_path.stem
    )
    assert (output_dir / ".embeddings_complete").exists()
    gc.collect()


# ---------------------------------------------------------------------------
# OlmoEarth backend end-to-end through the runner (real backend, synthetic store)
# ---------------------------------------------------------------------------


def test_olmoearth_e2e_through_runner(cloud_run_dirs, tmp_path):
    """OlmoEarthBackend through generate_cloud_embeddings: cache_dir injection, both pools."""
    raw_dir = cloud_run_dirs["raw_data_dir"] / cloud_run_dirs["data_version"]
    # Georeference both chips in UTM 18N: 96 px x 30 m = 2880 m starting at
    # (500000, 4000000); center_2x2 patches at 16 px cover rows/cols 2-3.
    for name in ("S2L2A_000001_00.tif", "S2L2A_000002_00.tif"):
        _rewrite_tiff_with_transform(
            raw_dir / name, from_origin(500000, 4000000, 30, 30), "EPSG:32618"
        )

    # 320 px x 10 m store spanning x 499840-503040, y 3997440-4000640: encloses
    # the centre patches (x 500960-501920, y 3998080-3999040) with margin.
    store = tmp_path / "olmo_e2e.zarr"
    _write_synthetic_olmoearth_zarr(store, H=320, W=320, y0=4000640.0)

    fixture = Path(__file__).parent / "fixtures" / "example_olmoearth_embeddings_config.yaml"
    cfg = yaml.safe_load(fixture.read_text())
    assert cfg["cloud_embedding"]["backend"] == "olmoearth"
    cfg["cloud_embedding"]["init_args"] = {"url": str(store), "dims": 64}
    yaml_path = tmp_path / "olmoearth_e2e.yaml"
    yaml_path.write_text(yaml.safe_dump(cfg))

    generate_cloud_embeddings(
        yaml_path,
        raw_data_dir=cloud_run_dirs["raw_data_dir"],
        embedding_dir=cloud_run_dirs["embedding_dir"],
    )

    output_dir = cloud_run_dirs["embedding_dir"] / cloud_run_dirs["data_version"] / yaml_path.stem
    expected = _olmo_expected_band_values(128)[:64]

    parquets = sorted((output_dir / "center_2x2").glob("*_embedding.parquet"))
    assert len(parquets) == 2
    for pq in parquets:
        emb = np.asarray(pd.read_parquet(pq)["embedding"].iloc[0].tolist())
        assert emb.shape == (4, 64)
        assert np.isfinite(emb).all()
        np.testing.assert_allclose(emb, np.broadcast_to(expected, (4, 64)), atol=1e-6)

    flatten_parquets = sorted((output_dir / "center_2x2_flatten").glob("*_embedding.parquet"))
    assert len(flatten_parquets) == 2
    for pq in flatten_parquets:
        emb = np.asarray(pd.read_parquet(pq)["embedding"].iloc[0].tolist())
        assert emb.shape == (4, 16 * 16 * 64)
        assert np.isfinite(emb).all()

    assert (output_dir / ".embeddings_complete").exists()
    cache_files = list((output_dir / "_footprint_cache").glob("olmoearth_utm18_d64_*.tif"))
    # Both chips share one footprint but carry different tracker years (2023,
    # 2024): 4 patches x 2 years x 2 strategies (mean: out_shape=None, flatten:
    # 16x16) = 16 unique keys.
    assert len(cache_files) == 16, f"unexpected cache files: {sorted(p.name for p in cache_files)}"
    assert all("_2023_" in p.name or "_2024_" in p.name for p in cache_files)
    gc.collect()
