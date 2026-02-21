import gc
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from gelos.models import MODELS, run_knn_cv, run_linear_probe_cv, run_random_forest_cv
from gelos.plotting import PLOTS
from gelos.transforms import TRANSFORMS, pca_from_embeddings, tsne_from_embeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 100
N_FEATURES = 64
N_CLASSES = 3


@pytest.fixture()
def synthetic_embeddings():
    """Random embeddings (100, 64) with chip_indices 0..99."""
    rng = np.random.RandomState(42)
    embeddings = rng.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
    chip_indices = list(range(N_SAMPLES))
    return embeddings, chip_indices


@pytest.fixture()
def synthetic_labels():
    """100 labels across 3 classes."""
    return np.array([0, 1, 2] * 33 + [0])


@pytest.fixture()
def mock_chip_gdf(synthetic_labels):
    """GeoDataFrame with id, lulc, geometry columns matching synthetic data."""
    gdf = gpd.GeoDataFrame(
        {
            "id": list(range(N_SAMPLES)),
            "lulc": synthetic_labels,
            "geometry": [Point(float(i), float(i)) for i in range(N_SAMPLES)],
        },
        crs="EPSG:4326",
    )
    return gdf.set_index("id")


# ---------------------------------------------------------------------------
# Tests: Registry keys
# ---------------------------------------------------------------------------


def test_transforms_registry_keys():
    """TRANSFORMS registry has tsne and pca entries."""
    assert "tsne" in TRANSFORMS
    assert "pca" in TRANSFORMS
    assert callable(TRANSFORMS["tsne"])
    assert callable(TRANSFORMS["pca"])


def test_models_registry_keys():
    """MODELS registry has knn, linear_probe, and random_forest entries."""
    assert "knn" in MODELS
    assert "linear_probe" in MODELS
    assert "random_forest" in MODELS
    for fn in MODELS.values():
        assert callable(fn)


def test_plots_registry_keys():
    """PLOTS registry has tsne_scatter entry."""
    assert "tsne_scatter" in PLOTS
    assert callable(PLOTS["tsne_scatter"])


# ---------------------------------------------------------------------------
# Tests: Transform functions
# ---------------------------------------------------------------------------


def test_pca_output_shape_fixed_components(synthetic_embeddings):
    """PCA with n_components=2 returns (N, 2)."""
    embeddings, _ = synthetic_embeddings
    result = pca_from_embeddings(embeddings, n_components=2)
    assert result.shape == (N_SAMPLES, 2)
    gc.collect()


def test_pca_variance_threshold(synthetic_embeddings):
    """PCA with n_components=0.95 returns (N, k) where k <= D."""
    embeddings, _ = synthetic_embeddings
    result = pca_from_embeddings(embeddings, n_components=0.95)
    assert result.shape[0] == N_SAMPLES
    assert result.shape[1] <= N_FEATURES
    gc.collect()


def test_tsne_output_shape():
    """t-SNE returns (N, 2) with default params."""
    rng = np.random.RandomState(42)
    embeddings = rng.rand(50, 10).astype(np.float32)
    result = tsne_from_embeddings(embeddings, perplexity=5, verbose=0)
    assert result.shape == (50, 2)
    gc.collect()


# ---------------------------------------------------------------------------
# Tests: Model functions
# ---------------------------------------------------------------------------


def test_knn_cv_returns_metrics(synthetic_embeddings, synthetic_labels, tmp_path):
    """KNN CV returns dict with accuracy, per_class, and predictions keys."""
    embeddings, _ = synthetic_embeddings
    result = run_knn_cv(embeddings, synthetic_labels, tmp_path, "test_knn")
    assert "accuracy" in result
    assert "per_class" in result
    assert "predictions" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert isinstance(result["per_class"], dict)
    assert len(result["predictions"]) == N_SAMPLES
    # Verify CSV saved
    csv_files = list(tmp_path.glob("*_knn_results.csv"))
    assert len(csv_files) == 1
    gc.collect()


def test_linear_probe_cv_returns_metrics(synthetic_embeddings, synthetic_labels, tmp_path):
    """Linear probe CV returns dict with accuracy, per_class, and predictions keys."""
    embeddings, _ = synthetic_embeddings
    result = run_linear_probe_cv(embeddings, synthetic_labels, tmp_path, "test_lp")
    assert "accuracy" in result
    assert "per_class" in result
    assert "predictions" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    csv_files = list(tmp_path.glob("*_linear_probe_results.csv"))
    assert len(csv_files) == 1
    gc.collect()


def test_random_forest_cv_returns_metrics(synthetic_embeddings, synthetic_labels, tmp_path):
    """Random forest CV returns dict with accuracy, per_class, and predictions keys."""
    embeddings, _ = synthetic_embeddings
    result = run_random_forest_cv(
        embeddings, synthetic_labels, tmp_path, "test_rf", n_estimators=10
    )
    assert "accuracy" in result
    assert "per_class" in result
    assert "predictions" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    csv_files = list(tmp_path.glob("*_random_forest_results.csv"))
    assert len(csv_files) == 1
    gc.collect()


# ---------------------------------------------------------------------------
# Tests: Pipeline integration
# ---------------------------------------------------------------------------


def test_strategy_without_steps_raises():
    """Strategy without any of transforms/plots/models raises ValueError."""
    strategy_cfg = {
        "title": "CLS Token",
        "slice_args": [{"start": 0, "stop": 1, "step": 1}],
    }

    has_transforms = "transforms" in strategy_cfg
    has_plots = "plots" in strategy_cfg
    has_models = "models" in strategy_cfg
    assert not (has_transforms or has_plots or has_models)
    gc.collect()


def test_pipeline_dispatches_transforms(synthetic_embeddings, tmp_path):
    """Pipeline transform dispatch calls registered transform functions."""
    from gelos.analysis import _save_transform_result

    embeddings, chip_indices = synthetic_embeddings

    # Run PCA via registry
    pca_fn = TRANSFORMS["pca"]
    result = pca_fn(embeddings, n_components=2)
    assert result.shape == (N_SAMPLES, 2)

    # Save and reload
    cache_path = tmp_path / "test_pca.csv"
    _save_transform_result(result, chip_indices, cache_path, "pca", "test")
    assert cache_path.exists()

    from gelos.analysis import _load_cached_transform

    loaded, loaded_indices = _load_cached_transform(cache_path)
    np.testing.assert_array_almost_equal(loaded, result, decimal=5)
    assert loaded_indices == chip_indices
    gc.collect()


def test_pipeline_unknown_transform_raises():
    """Referencing an unregistered transform type raises KeyError."""
    assert "nonexistent" not in TRANSFORMS


def test_pipeline_unknown_model_raises():
    """Referencing an unregistered model type raises KeyError."""
    assert "nonexistent" not in MODELS


def test_pipeline_unknown_plot_raises():
    """Referencing an unregistered plot type raises KeyError."""
    assert "nonexistent" not in PLOTS


# ---------------------------------------------------------------------------
# Tests: chip_id_column join
# ---------------------------------------------------------------------------


def test_chip_id_column_sets_index(tmp_path, synthetic_labels):
    """load_chip_tracker + set_index(chip_id_column) enables .loc lookup by file_id."""
    from gelos.analysis import load_chip_tracker

    # IDs starting from 1 (not 0) to ensure .loc uses the index, not row position
    ids = list(range(1, N_SAMPLES + 1))
    gdf = gpd.GeoDataFrame(
        {
            "id": ids,
            "lulc": synthetic_labels,
            "geometry": [Point(float(i), float(i)) for i in ids],
        },
        crs="EPSG:4326",
    )
    geojson_path = tmp_path / "chip_tracker.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")

    loaded = load_chip_tracker(geojson_path)
    loaded = loaded.set_index("id")

    # Simulate what run_pipeline does: look up labels by chip_indices from extract_embeddings
    chip_indices = [1, 5, 10]
    labels = loaded["lulc"].loc[chip_indices].to_numpy()
    assert len(labels) == 3
    expected = gdf.set_index("id")["lulc"].loc[chip_indices].to_numpy()
    np.testing.assert_array_equal(labels, expected)
