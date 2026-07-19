import gc
from pathlib import Path

import geopandas as gpd
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
from gelos.cloud_embeddings.backends.alphaearth import AlphaEarthBackend
from gelos.cloud_embeddings.backends.base import FetchRequest
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
# Shared patch resolution util (mirror of tests/test_raw_pixels.py::test_resolve_*)
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
