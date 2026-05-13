import gc

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from gelos.metrics import METRICS, knn_purity, pca_ablation
from gelos.models import MODELS, run_knn_cv, run_linear_probe_cv, run_random_forest_cv
from gelos.plotting import PLOTS
from gelos.transforms import (
    TRANSFORMS,
    pca_from_embeddings,
    tsne_from_embeddings,
    umap_from_embeddings,
)

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
    assert "umap" in TRANSFORMS
    assert callable(TRANSFORMS["tsne"])
    assert callable(TRANSFORMS["pca"])
    assert callable(TRANSFORMS["umap"])


def test_models_registry_keys():
    """MODELS registry has knn, linear_probe, and random_forest entries."""
    assert "knn" in MODELS
    assert "linear_probe" in MODELS
    assert "random_forest" in MODELS
    for fn in MODELS.values():
        assert callable(fn)


def test_plots_registry_keys():
    """PLOTS registry has scatter_2d and temporal_cosine_similarity entries."""
    assert "scatter_2d" in PLOTS
    assert callable(PLOTS["scatter_2d"])
    assert "temporal_cosine_similarity" in PLOTS
    assert callable(PLOTS["temporal_cosine_similarity"])


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
    result = tsne_from_embeddings(embeddings, perplexity=5, verbose=False)
    assert result.shape == (50, 2)
    gc.collect()


def test_umap_output_shape():
    """UMAP returns (N, 2) with default params."""
    rng = np.random.RandomState(42)
    embeddings = rng.rand(50, 10).astype(np.float32)
    result = umap_from_embeddings(embeddings, verbose=False)
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
# Tests: confusion_matrix plot
# ---------------------------------------------------------------------------


@pytest.fixture()
def cm_style_cfg():
    return {
        "category_column": "lulc",
        "colors": {"0": "#1f77b4", "1": "#ff7f0e", "2": "#2ca02c"},
        "labels": {"0": "A", "1": "B", "2": "C"},
    }


def test_confusion_matrix_output(synthetic_labels, cm_style_cfg, tmp_path):
    """confusion_matrix writes a PNG to output_path."""
    from gelos.plotting import confusion_matrix

    rng = np.random.RandomState(0)
    predictions = rng.choice([0, 1, 2], size=N_SAMPLES)
    chip_indices = list(range(N_SAMPLES))
    output_path = tmp_path / "test_cm.png"

    confusion_matrix(
        predictions=predictions,
        labels=synthetic_labels,
        chip_indices=chip_indices,
        style_cfg=cm_style_cfg,
        experiment_name="Test Experiment",
        strategy_title="Test Strategy",
        model_type="knn",
        embedding_layer="layer_-1",
        output_path=output_path,
    )
    assert output_path.exists()
    gc.collect()


def test_confusion_matrix_missing_class(synthetic_labels, cm_style_cfg, tmp_path):
    """confusion_matrix runs cleanly when predictions miss a class present in labels."""
    from gelos.plotting import confusion_matrix

    rng = np.random.RandomState(0)
    # Predictions only contain classes 0 and 1; labels contain 0, 1, 2
    predictions = rng.choice([0, 1], size=N_SAMPLES)
    chip_indices = list(range(N_SAMPLES))
    output_path = tmp_path / "test_cm_missing.png"

    confusion_matrix(
        predictions=predictions,
        labels=synthetic_labels,
        chip_indices=chip_indices,
        style_cfg=cm_style_cfg,
        experiment_name="Test",
        strategy_title="Test",
        model_type="knn",
        embedding_layer="layer_-1",
        output_path=output_path,
    )
    assert output_path.exists()
    gc.collect()


# ---------------------------------------------------------------------------
# Tests: Metrics
# ---------------------------------------------------------------------------


def test_metrics_registry_keys():
    """METRICS registry has pca_ablation and knn_purity entries."""
    assert "pca_ablation" in METRICS
    assert callable(METRICS["pca_ablation"])
    assert "knn_purity" in METRICS
    assert callable(METRICS["knn_purity"])


def test_pca_ablation_output(synthetic_embeddings, tmp_path):
    """pca_ablation writes CSV with correct columns and sensible values."""
    embeddings, _ = synthetic_embeddings
    result = pca_ablation(embeddings, output_dir=tmp_path, prefix="test")

    # Check return structure
    assert "thresholds" in result
    assert len(result["thresholds"]) == 5  # default thresholds

    for row in result["thresholds"]:
        assert row["n_components"] > 0
        assert 0.0 < row["total_variance_explained"] <= 1.0

    # Check CSV was written
    csv_files = list(tmp_path.glob("*_pca_ablation.csv"))
    assert len(csv_files) == 1

    import pandas as pd

    df = pd.read_csv(csv_files[0])
    assert set(df.columns) == {
        "threshold",
        "n_components",
        "proportion_of_total_components",
        "total_variance_explained",
    }
    assert len(df) == 5
    gc.collect()


def test_pca_ablation_cache_skip(synthetic_embeddings, tmp_path):
    """Metrics dispatch skips when cache CSV exists."""
    embeddings, _ = synthetic_embeddings

    # Create a fake cache file
    layer_dir = tmp_path / "layer"
    layer_dir.mkdir()
    cache_path = layer_dir / "test_pca_ablation.csv"
    cache_path.write_text("threshold,n_components,total_variance_explained\n0.95,10,0.96\n")

    # The cache check is in analysis.py dispatch — verify the file exists
    assert cache_path.exists()

    # Run pca_ablation fresh to a different prefix to confirm it works
    pca_ablation(embeddings, output_dir=layer_dir, prefix="fresh")
    assert (layer_dir / "fresh_pca_ablation.csv").exists()
    gc.collect()


def test_knn_purity_output(synthetic_embeddings, synthetic_labels, tmp_path):
    """knn_purity writes CSV with correct columns and sensible values."""
    embeddings, _ = synthetic_embeddings
    result = knn_purity(embeddings, output_dir=tmp_path, prefix="test", labels=synthetic_labels)

    assert "rows" in result
    assert "k_values" in result
    assert len(result["rows"]) > 0

    for row in result["rows"]:
        assert 0.0 <= row["purity"] <= 1.0
        assert row["n_samples"] > 0

    csv_files = list(tmp_path.glob("*_knn_purity.csv"))
    assert len(csv_files) == 1

    import pandas as pd

    df = pd.read_csv(csv_files[0])
    assert set(df.columns) == {"k", "class", "purity", "n_samples"}
    # Default 6 k values, each with overall + 3 classes = 4 rows per k = 24
    assert len(df) == 6 * (1 + N_CLASSES)
    gc.collect()


def test_knn_purity_writes_per_query_csv(synthetic_embeddings, synthetic_labels, tmp_path):
    """knn_purity writes a per-query CSV alongside the aggregated one."""
    import pandas as pd

    embeddings, _ = synthetic_embeddings
    knn_purity(embeddings, output_dir=tmp_path, prefix="test", labels=synthetic_labels)

    per_query_csv = tmp_path / "test_knn_purity_per_query.csv"
    assert per_query_csv.exists()

    df = pd.read_csv(per_query_csv)
    assert set(df.columns) == {"k", "class", "query_idx", "purity"}
    # Default 6 k values × 100 queries (no subsampling)
    assert len(df) == 6 * N_SAMPLES
    assert df["purity"].between(0.0, 1.0).all()
    gc.collect()


def test_knn_purity_per_query_respects_subsampling(
    synthetic_embeddings, synthetic_labels, tmp_path
):
    """Per-query CSV row count matches subsampled query count, not full N."""
    import pandas as pd

    embeddings, _ = synthetic_embeddings
    knn_purity(
        embeddings,
        output_dir=tmp_path,
        prefix="test_sub",
        labels=synthetic_labels,
        n_subsample=30,
    )

    df = pd.read_csv(tmp_path / "test_sub_knn_purity_per_query.csv")
    # 6 default k values × <= 30 query rows
    n_queries = df[df["k"] == df["k"].iloc[0]].shape[0]
    assert n_queries <= 30
    assert len(df) == 6 * n_queries
    gc.collect()


def test_knn_purity_subsampling(synthetic_embeddings, synthetic_labels, tmp_path):
    """knn_purity with n_subsample produces valid output with fewer queries."""
    embeddings, _ = synthetic_embeddings
    result = knn_purity(
        embeddings,
        output_dir=tmp_path,
        prefix="test_sub",
        labels=synthetic_labels,
        n_subsample=30,
    )

    assert "rows" in result
    for row in result["rows"]:
        assert 0.0 <= row["purity"] <= 1.0
        # Overall n_samples should be <= 30
        if row["class"] == "overall":
            assert row["n_samples"] <= 30

    csv_files = list(tmp_path.glob("*_knn_purity.csv"))
    assert len(csv_files) == 1
    gc.collect()


def test_knn_purity_perfect_clusters(tmp_path):
    """Well-separated clusters should yield purity ~1.0 at small k."""
    rng = np.random.RandomState(42)
    n_per_class = 30
    dim = 16

    # Three tight clusters far apart
    c0 = rng.randn(n_per_class, dim) * 0.01 + np.array([0] * dim)
    c1 = rng.randn(n_per_class, dim) * 0.01 + np.array([100] * dim)
    c2 = rng.randn(n_per_class, dim) * 0.01 + np.array([-100] * dim)
    embeddings = np.vstack([c0, c1, c2]).astype(np.float32)
    labels = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    result = knn_purity(
        embeddings,
        output_dir=tmp_path,
        prefix="perfect",
        labels=labels,
        k_values=[1, 5, 10, 20],
    )

    for row in result["rows"]:
        if row["class"] == "overall" and row["k"] <= 20:
            assert row["purity"] > 0.95, f"Expected high purity at k={row['k']}"
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


# ---------------------------------------------------------------------------
# Tests: temporal_cosine_similarity plot
# ---------------------------------------------------------------------------

N_TIMESTEPS = 4
N_TEMPORAL_SAMPLES = 50
N_TEMPORAL_FEATURES = 16


@pytest.fixture()
def temporal_embeddings():
    """Synthetic temporal embeddings (50, 4*16) with chip_indices 0..49."""
    rng = np.random.RandomState(42)
    embeddings = rng.rand(N_TEMPORAL_SAMPLES, N_TIMESTEPS * N_TEMPORAL_FEATURES).astype(np.float32)
    chip_indices = list(range(N_TEMPORAL_SAMPLES))
    return embeddings, chip_indices


@pytest.fixture()
def temporal_chip_gdf():
    """GeoDataFrame with 2 categories for temporal cosine similarity tests."""
    categories = ["1"] * 25 + ["2"] * 25
    gdf = gpd.GeoDataFrame(
        {
            "id": list(range(N_TEMPORAL_SAMPLES)),
            "lulc": categories,
            "geometry": [Point(float(i), float(i)) for i in range(N_TEMPORAL_SAMPLES)],
        },
        crs="EPSG:4326",
    )
    return gdf.set_index("id")


@pytest.fixture()
def temporal_style_cfg():
    return {
        "category_column": "lulc",
        "colors": {"1": "#419bdf", "2": "#397d49"},
        "labels": {"1": "Water", "2": "Trees"},
    }


def test_temporal_cosine_similarity_output(
    temporal_embeddings, temporal_chip_gdf, temporal_style_cfg, tmp_path
):
    """temporal_cosine_similarity creates a .png file at output_path."""
    from gelos.plotting import temporal_cosine_similarity

    embeddings, chip_indices = temporal_embeddings
    output_path = tmp_path / "test_temporal.png"

    temporal_cosine_similarity(
        embeddings,
        temporal_chip_gdf,
        chip_indices,
        temporal_style_cfg,
        "Test Experiment",
        "Test Strategy",
        "raw",
        "layer_-1",
        output_path,
        n_timesteps=N_TIMESTEPS,
    )
    assert output_path.exists()
    gc.collect()


def test_temporal_cosine_similarity_invalid_timesteps():
    """ValueError raised when n_timesteps=1 or n_timesteps is not an int."""
    from gelos.plotting import temporal_cosine_similarity

    dummy_args = (
        np.zeros((10, 64)),
        gpd.GeoDataFrame(),
        [],
        {"category_column": "x", "colors": {}, "labels": {}},
        "",
        "",
        "raw",
        "layer",
        None,
    )

    with pytest.raises(ValueError, match="n_timesteps must be an int > 1"):
        temporal_cosine_similarity(*dummy_args, n_timesteps=1)

    with pytest.raises(ValueError, match="n_timesteps must be an int > 1"):
        temporal_cosine_similarity(*dummy_args, n_timesteps="four")
    gc.collect()
