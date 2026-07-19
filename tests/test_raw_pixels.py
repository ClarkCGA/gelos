import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.test_data import ExampleGELOSDataSet  # noqa: F401 — resolve class_path in YAML
from tests.utils import create_test_geojson
import torch

from gelos.analysis import run_analysis
from gelos.generation import generate_embeddings
from gelos.raw_pixels import RawPixelEmbeddingTask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_task(tmp_path: Path, patch_size: int = 16, strategies=None) -> RawPixelEmbeddingTask:
    return RawPixelEmbeddingTask(
        output_dir=str(tmp_path),
        extraction_strategies=strategies or {},
        patch_size=patch_size,
    )


# ---------------------------------------------------------------------------
# Patch / timestep resolution
# ---------------------------------------------------------------------------


def test_resolve_center_2x2(tmp_path):
    task = make_task(tmp_path)
    spec = {"patches": "center_2x2", "timesteps": [0]}
    indices = task._resolve_patch_indices(spec, n_rows=6, n_cols=6)
    assert indices == [(2, 2), (2, 3), (3, 2), (3, 3)]


def test_resolve_center_1x4(tmp_path):
    task = make_task(tmp_path)
    spec = {"patches": "center_1x4", "timesteps": [0]}
    indices = task._resolve_patch_indices(spec, n_rows=6, n_cols=6)
    # r0 = (6 - 1) // 2 = 2, c0 = (6 - 4) // 2 = 1 → centered horizontal line
    assert indices == [(2, 1), (2, 2), (2, 3), (2, 4)]
    # Row-major contiguity: single row, consecutive columns.
    assert len({r for r, _ in indices}) == 1
    cols = [c for _, c in indices]
    assert cols == list(range(cols[0], cols[0] + len(cols)))


def test_resolve_center_1x8_col_overflow_raises(tmp_path):
    task = make_task(tmp_path)
    spec = {"patches": "center_1x8", "timesteps": [0]}
    with pytest.raises(ValueError):
        task._resolve_patch_indices(spec, n_rows=6, n_cols=6)


def test_resolve_center_8x1_row_overflow_raises(tmp_path):
    task = make_task(tmp_path)
    spec = {"patches": "center_8x1", "timesteps": [0]}
    with pytest.raises(ValueError):
        task._resolve_patch_indices(spec, n_rows=6, n_cols=6)


def test_resolve_center_1x1_odd_grid(tmp_path):
    task = make_task(tmp_path)
    spec = {"patches": "center_1x1", "timesteps": [0]}
    indices = task._resolve_patch_indices(spec, n_rows=7, n_cols=7)
    assert indices == [(3, 3)]


def test_resolve_center_1x1_even_grid(tmp_path):
    task = make_task(tmp_path)
    spec = {"patches": "center_1x1", "timesteps": [0]}
    # top-left = ((6 - 1) // 2, (6 - 1) // 2) = (2, 2)
    assert task._resolve_patch_indices(spec, n_rows=6, n_cols=6) == [(2, 2)]


def test_resolve_explicit_list(tmp_path):
    task = make_task(tmp_path)
    spec = {"patches": [[0, 0], [2, 3]], "timesteps": [0]}
    assert task._resolve_patch_indices(spec, n_rows=6, n_cols=6) == [(0, 0), (2, 3)]


def test_resolve_patches_out_of_range_raises(tmp_path):
    task = make_task(tmp_path)
    spec = {"patches": [[10, 10]], "timesteps": [0]}
    with pytest.raises(ValueError):
        task._resolve_patch_indices(spec, n_rows=6, n_cols=6)


def test_resolve_center_too_large_raises(tmp_path):
    task = make_task(tmp_path)
    spec = {"patches": "center_8x8", "timesteps": [0]}
    with pytest.raises(ValueError):
        task._resolve_patch_indices(spec, n_rows=6, n_cols=6)


def test_resolve_timesteps_all(tmp_path):
    task = make_task(tmp_path)
    assert task._resolve_timestep_indices({"timesteps": "all"}, T=4) == [0, 1, 2, 3]


def test_resolve_timesteps_explicit(tmp_path):
    task = make_task(tmp_path)
    assert task._resolve_timestep_indices({"timesteps": [0, 2]}, T=4) == [0, 2]


def test_resolve_timesteps_out_of_range_raises(tmp_path):
    task = make_task(tmp_path)
    with pytest.raises(ValueError):
        task._resolve_timestep_indices({"timesteps": [5]}, T=4)


# ---------------------------------------------------------------------------
# Extraction shapes
# ---------------------------------------------------------------------------


def test_extract_shape_tensor(tmp_path):
    task = make_task(tmp_path, patch_size=16)
    image = torch.zeros((2, 3, 4, 96, 96))
    spec = {"patches": "center_2x2", "timesteps": [0]}
    tokens = task._extract_raw_pixels("s", image, spec)
    assert tuple(tokens.shape) == (2, 4, 3 * 16 * 16)


def test_extract_shape_dict(tmp_path):
    task = make_task(tmp_path, patch_size=16)
    image = {
        "S2L2A": torch.zeros((2, 6, 4, 96, 96)),
        "S1RTC": torch.zeros((2, 2, 4, 96, 96)),
        "DEM": torch.zeros((2, 1, 4, 96, 96)),
    }
    spec = {"patches": "center_2x2", "timesteps": [0]}
    tokens = task._extract_raw_pixels("s", image, spec)
    assert tuple(tokens.shape) == (2, 4, (6 + 2 + 1) * 16 * 16)


# ---------------------------------------------------------------------------
# Extraction values
# ---------------------------------------------------------------------------


def test_extract_values_tensor(tmp_path):
    task = make_task(tmp_path, patch_size=2)
    # [B=1, C=1, T=1, H=4, W=4] with sequential values
    image = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4)
    spec = {"patches": [[0, 0]], "timesteps": [0]}
    tokens = task._extract_raw_pixels("s", image, spec)
    # top-left 2x2 patch flattens to [0, 1, 4, 5]
    assert torch.equal(tokens[0, 0], torch.tensor([0.0, 1.0, 4.0, 5.0]))

    spec = {"patches": [[1, 1]], "timesteps": [0]}
    tokens = task._extract_raw_pixels("s", image, spec)
    # bottom-right 2x2 patch flattens to [10, 11, 14, 15]
    assert torch.equal(tokens[0, 0], torch.tensor([10.0, 11.0, 14.0, 15.0]))


def test_extract_values_dict_channel_order(tmp_path):
    task = make_task(tmp_path, patch_size=2)
    image = {
        "A": torch.full((1, 1, 1, 4, 4), 1.0),
        "B": torch.full((1, 1, 1, 4, 4), 2.0),
        "C": torch.full((1, 1, 1, 4, 4), 3.0),
    }
    spec = {"patches": [[0, 0]], "timesteps": [0]}
    tokens = task._extract_raw_pixels("s", image, spec)
    # flat_dim per sensor = 1*2*2 = 4 → sensor A occupies [0:4], B [4:8], C [8:12]
    assert torch.equal(tokens[0, 0, 0:4], torch.full((4,), 1.0))
    assert torch.equal(tokens[0, 0, 4:8], torch.full((4,), 2.0))
    assert torch.equal(tokens[0, 0, 8:12], torch.full((4,), 3.0))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_dict_sensor_shape_mismatch_raises(tmp_path):
    task = make_task(tmp_path, patch_size=16)
    image = {
        "S2L2A": torch.zeros((1, 3, 4, 96, 96)),
        "S1RTC": torch.zeros((1, 2, 2, 96, 96)),  # T mismatch
    }
    spec = {"patches": "center_1x1", "timesteps": [0]}
    with pytest.raises(ValueError, match="S1RTC"):
        task._extract_raw_pixels("mismatch", image, spec)


def test_h_not_divisible_raises(tmp_path):
    task = make_task(tmp_path, patch_size=16)
    image = torch.zeros((1, 3, 1, 90, 96))
    with pytest.raises(ValueError):
        task._patch_grid(image)


def test_w_not_divisible_raises(tmp_path):
    task = make_task(tmp_path, patch_size=16)
    image = torch.zeros((1, 3, 1, 96, 90))
    with pytest.raises(ValueError):
        task._patch_grid(image)


# ---------------------------------------------------------------------------
# predict_step writes parquet
# ---------------------------------------------------------------------------


def test_predict_step_writes_parquet_tensor(tmp_path):
    strategies = {
        "center_1x1_t0": {"patches": "center_1x1", "timesteps": [0]},
    }
    task = make_task(tmp_path, patch_size=16, strategies=strategies)
    batch = {
        "image": torch.zeros((2, 3, 1, 96, 96)),
        "filename": np.array(["000001", "000002"], dtype=str),
        "file_id": torch.tensor([1, 2]),
    }
    task.predict_step(batch)

    out_dir = tmp_path / "center_1x1_t0"
    assert out_dir.exists()
    parquet_files = sorted(out_dir.glob("*_embedding.parquet"))
    assert len(parquet_files) == 2

    df = pd.read_parquet(parquet_files[0])
    assert "embedding" in df.columns
    assert "file_id" in df.columns
    emb = np.asarray(df["embedding"].iloc[0].tolist())
    assert emb.shape == (1, 3 * 16 * 16)
    gc.collect()


def test_predict_step_writes_parquet_dict(tmp_path):
    strategies = {
        "center_2x2_t0": {"patches": "center_2x2", "timesteps": [0]},
    }
    task = make_task(tmp_path, patch_size=16, strategies=strategies)
    batch = {
        "image": {
            "S2L2A": torch.zeros((1, 6, 1, 96, 96)),
            "DEM": torch.zeros((1, 1, 1, 96, 96)),
        },
        "filename": np.array(["000001"], dtype=str),
        "file_id": torch.tensor([1]),
    }
    task.predict_step(batch)

    out_dir = tmp_path / "center_2x2_t0"
    parquet_files = list(out_dir.glob("*_embedding.parquet"))
    assert len(parquet_files) == 1
    df = pd.read_parquet(parquet_files[0])
    emb = np.asarray(df["embedding"].iloc[0].tolist())
    assert emb.shape == (4, (6 + 1) * 16 * 16)
    assert int(df["file_id"].iloc[0]) == 1
    gc.collect()


@pytest.fixture()
def raw_pixel_run_dirs(tmp_path):
    """Build raw_data_dir/<data_version>/ with dummy chips and return directory paths."""
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
    n_timesteps = {"S2L2A": 4, "S1RTC": 4, "DEM": 1}
    create_test_geojson(versioned_data, n_samples=2, sensors=sensors, n_timesteps=n_timesteps, img_size=96)
    # Tests rely on `lulc` labels for the analysis path; geojson here doesn't include them,
    # but run_analysis only needs them if category-dependent steps run. Inject a simple lulc.
    import geopandas as gpd

    tracker = versioned_data / "gelos_chip_tracker.geojson"
    gdf = gpd.read_file(tracker)
    gdf["lulc"] = [1, 2]
    gdf.to_file(tracker, driver="GeoJSON")

    return {
        "data_version": data_version,
        "raw_data_dir": raw_data_dir,
        "embedding_dir": embedding_dir,
        "processed_data_dir": processed_data_dir,
        "figures_dir": figures_dir,
    }


def test_raw_pixel_generation_e2e(raw_pixel_run_dirs):
    yaml_path = Path(__file__).parent / "fixtures" / "example_raw_pixel_config.yaml"

    generate_embeddings(
        yaml_path,
        raw_data_dir=raw_pixel_run_dirs["raw_data_dir"],
        embedding_dir=raw_pixel_run_dirs["embedding_dir"],
    )

    config_stem = yaml_path.stem
    data_version = raw_pixel_run_dirs["data_version"]
    output_dir = raw_pixel_run_dirs["embedding_dir"] / data_version / config_stem

    for strategy in ("center_2x2_t0", "all_timesteps_center_patch"):
        strat_dir = output_dir / strategy
        assert strat_dir.is_dir(), f"missing strategy dir {strat_dir}"
        parquets = list(strat_dir.glob("*_embedding.parquet"))
        assert len(parquets) == 2, f"expected 2 parquets in {strat_dir}, got {len(parquets)}"

        df = pd.read_parquet(parquets[0])
        assert "embedding" in df.columns
        emb = np.asarray(df["embedding"].iloc[0].tolist())
        assert emb.ndim == 2 and emb.shape[0] == 4  # 4 tokens each

    assert (output_dir / ".embeddings_complete").exists()

    run_analysis(
        yaml_path,
        raw_data_dir=raw_pixel_run_dirs["raw_data_dir"],
        embedding_dir=raw_pixel_run_dirs["embedding_dir"],
        processed_data_dir=raw_pixel_run_dirs["processed_data_dir"],
        figures_base_dir=raw_pixel_run_dirs["figures_dir"],
    )

    processed_root = raw_pixel_run_dirs["processed_data_dir"] / data_version / config_stem
    cached = list(processed_root.rglob("*_embeddings.npy"))
    assert cached, f"expected cached _embeddings.npy under {processed_root}"


def test_predict_step_multiple_strategies(tmp_path):
    strategies = {
        "center_1x1_t0": {"patches": "center_1x1", "timesteps": [0]},
        "center_1x1_all": {"patches": "center_1x1", "timesteps": "all"},
    }
    task = make_task(tmp_path, patch_size=16, strategies=strategies)
    batch = {
        "image": torch.zeros((1, 3, 4, 96, 96)),
        "filename": np.array(["000001"], dtype=str),
        "file_id": torch.tensor([1]),
    }
    task.predict_step(batch)

    assert (tmp_path / "center_1x1_t0" / "000001_embedding.parquet").exists()
    assert (tmp_path / "center_1x1_all" / "000001_embedding.parquet").exists()

    df = pd.read_parquet(tmp_path / "center_1x1_all" / "000001_embedding.parquet")
    emb = np.asarray(df["embedding"].iloc[0].tolist())
    # 4 timesteps × 1 patch = 4 tokens
    assert emb.shape == (4, 3 * 16 * 16)
    gc.collect()
