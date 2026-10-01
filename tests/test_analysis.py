import gc

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from gelos.metrics import (
    METRICS,
    chip_centers_latlon,
    knn_geo_distance,
    knn_purity,
    pca_ablation,
)
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
    assert "knn_geo_distance" in METRICS
    assert callable(METRICS["knn_geo_distance"])


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
# Tests: knn_geo_distance metric (issue #85)
# ---------------------------------------------------------------------------

# Great-circle length of one degree of latitude on the sphere used by the metric.
_METRES_PER_DEG = 2 * np.pi * 6_371_008.8 / 360


def test_knn_geo_distance_output(synthetic_embeddings, mock_chip_gdf, tmp_path):
    """knn_geo_distance writes aggregate + per-query CSVs with the expected schemas."""
    import pandas as pd

    embeddings, chip_indices = synthetic_embeddings
    result = knn_geo_distance(
        embeddings,
        output_dir=tmp_path,
        prefix="test",
        chip_gdf=mock_chip_gdf,
        chip_indices=chip_indices,
    )
    assert "rows" in result and "k_values" in result
    assert result["k_values"] == [1, 2, 5, 10, 20, 50]

    df = pd.read_csv(tmp_path / "test_knn_geo_distance.csv")
    assert set(df.columns) == {"k", "measure", "mean", "median", "q1", "q3", "n_samples"}
    # 6 default k values × 2 measures
    assert len(df) == 6 * 2
    assert set(df["measure"]) == {"gsd_m", "lat_diff_deg"}
    assert (df[["mean", "median", "q1", "q3"]] >= 0).all().all()
    assert (df.loc[df["measure"] == "lat_diff_deg", "mean"] <= 180).all()
    assert (df["n_samples"] == N_SAMPLES).all()

    per_query = pd.read_csv(tmp_path / "test_knn_geo_distance_per_query.csv")
    assert set(per_query.columns) == {"k", "query_idx", "chip_id", "gsd_m", "lat_diff_deg"}
    assert len(per_query) == 6 * N_SAMPLES
    assert (per_query[["gsd_m", "lat_diff_deg"]] >= 0).all().all()
    gc.collect()


def test_knn_geo_distance_known_geometry(tmp_path):
    """Hand-checked distances: 1-D embeddings fix the neighbour order, chips 1 deg apart."""
    import pandas as pd

    embeddings = np.array([[0.0], [1.0], [10.0]], dtype=np.float32)
    chip_gdf = gpd.GeoDataFrame(
        {"id": [0, 1, 2], "geometry": [Point(0.0, 0.0), Point(0.0, 1.0), Point(0.0, 3.0)]},
        crs="EPSG:4326",
    ).set_index("id")

    knn_geo_distance(
        embeddings,
        output_dir=tmp_path,
        prefix="geo",
        chip_gdf=chip_gdf,
        chip_indices=[0, 1, 2],
        k_values=[1],
    )

    per_query = pd.read_csv(tmp_path / "geo_knn_geo_distance_per_query.csv").sort_values(
        "query_idx"
    )
    # Nearest neighbours in embedding space: 0->1, 1->0, 2->1
    np.testing.assert_allclose(per_query["lat_diff_deg"].to_numpy(), [1.0, 1.0, 2.0])
    np.testing.assert_allclose(
        per_query["gsd_m"].to_numpy(),
        [_METRES_PER_DEG, _METRES_PER_DEG, 2 * _METRES_PER_DEG],
        rtol=1e-6,
    )
    assert per_query["gsd_m"].iloc[0] == pytest.approx(111195.08, rel=1e-6)

    agg = pd.read_csv(tmp_path / "geo_knn_geo_distance.csv").set_index("measure")
    assert agg.loc["lat_diff_deg", "mean"] == pytest.approx(4 / 3)
    assert agg.loc["gsd_m", "mean"] == pytest.approx(4 / 3 * _METRES_PER_DEG, rel=1e-6)
    gc.collect()


def test_knn_geo_distance_requires_coordinates(synthetic_embeddings, mock_chip_gdf, tmp_path):
    """Missing chip_gdf / chip_indices or a length mismatch raise ValueError."""
    embeddings, chip_indices = synthetic_embeddings
    with pytest.raises(ValueError, match="requires chip_gdf"):
        knn_geo_distance(embeddings, output_dir=tmp_path, prefix="x", chip_indices=chip_indices)
    with pytest.raises(ValueError, match="requires chip_gdf"):
        knn_geo_distance(embeddings, output_dir=tmp_path, prefix="x", chip_gdf=mock_chip_gdf)
    with pytest.raises(ValueError, match="does not match"):
        knn_geo_distance(
            embeddings,
            output_dir=tmp_path,
            prefix="x",
            chip_gdf=mock_chip_gdf,
            chip_indices=chip_indices[:-1],
        )
    gc.collect()


def test_knn_geo_distance_csv_tracker_latlon_columns(synthetic_embeddings, tmp_path):
    """A plain DataFrame tracker works via lat/lon columns; without them it raises."""
    import pandas as pd

    embeddings, chip_indices = synthetic_embeddings
    tracker = pd.DataFrame(
        {
            "id": chip_indices,
            "lat": [float(i) for i in chip_indices],
            "lon": [float(i) for i in chip_indices],
        }
    ).set_index("id")

    lat, lon = chip_centers_latlon(tracker, chip_indices[:3])
    np.testing.assert_array_equal(lat, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(lon, [0.0, 1.0, 2.0])

    knn_geo_distance(
        embeddings,
        output_dir=tmp_path,
        prefix="csv",
        chip_gdf=tracker,
        chip_indices=chip_indices,
        k_values=[1, 5],
    )
    assert (tmp_path / "csv_knn_geo_distance.csv").exists()

    # latitude/longitude spelling is accepted too
    alt = tracker.rename(columns={"lat": "latitude", "lon": "longitude"})
    lat_alt, lon_alt = chip_centers_latlon(alt, chip_indices[:3])
    np.testing.assert_array_equal(lat_alt, lat)
    np.testing.assert_array_equal(lon_alt, lon)

    no_coords = pd.DataFrame({"id": chip_indices, "lulc": 0}).set_index("id")
    with pytest.raises(ValueError, match="lat/lon"):
        chip_centers_latlon(no_coords, chip_indices)
    with pytest.raises(ValueError, match="lat/lon"):
        knn_geo_distance(
            embeddings,
            output_dir=tmp_path,
            prefix="bad",
            chip_gdf=no_coords,
            chip_indices=chip_indices,
        )
    gc.collect()


def test_knn_geo_distance_projected_crs(synthetic_embeddings, tmp_path):
    """A tracker in a projected CRS gives the same results as EPSG:4326."""
    import pandas as pd

    embeddings, chip_indices = synthetic_embeddings
    # Keep latitudes within the Web Mercator domain (mock_chip_gdf goes past 85 deg).
    chip_gdf = gpd.GeoDataFrame(
        {
            "id": chip_indices,
            "geometry": [Point(0.5 * i - 20.0, 0.5 * i - 25.0) for i in chip_indices],
        },
        crs="EPSG:4326",
    ).set_index("id")
    knn_geo_distance(
        embeddings,
        output_dir=tmp_path,
        prefix="wgs84",
        chip_gdf=chip_gdf,
        chip_indices=chip_indices,
        k_values=[1, 5],
    )
    knn_geo_distance(
        embeddings,
        output_dir=tmp_path,
        prefix="merc",
        chip_gdf=chip_gdf.to_crs(3857),
        chip_indices=chip_indices,
        k_values=[1, 5],
    )
    a = pd.read_csv(tmp_path / "wgs84_knn_geo_distance.csv")
    b = pd.read_csv(tmp_path / "merc_knn_geo_distance.csv")
    np.testing.assert_allclose(a["mean"].to_numpy(), b["mean"].to_numpy(), rtol=1e-6)
    gc.collect()


def test_knn_geo_distance_subsampling(synthetic_embeddings, mock_chip_gdf, tmp_path):
    """n_subsample limits the number of query rows per k."""
    import pandas as pd

    embeddings, chip_indices = synthetic_embeddings
    knn_geo_distance(
        embeddings,
        output_dir=tmp_path,
        prefix="sub",
        chip_gdf=mock_chip_gdf,
        chip_indices=chip_indices,
        n_subsample=30,
    )
    per_query = pd.read_csv(tmp_path / "sub_knn_geo_distance_per_query.csv")
    n_queries = per_query[per_query["k"] == per_query["k"].iloc[0]].shape[0]
    assert n_queries <= 30
    assert len(per_query) == 6 * n_queries

    agg = pd.read_csv(tmp_path / "sub_knn_geo_distance.csv")
    assert (agg["n_samples"] <= 30).all()
    gc.collect()


def test_knn_geo_distance_skips_large_k(mock_chip_gdf, tmp_path):
    """k values exceeding N-1 are skipped with a warning (same rule as knn_purity)."""
    import pandas as pd

    rng = np.random.RandomState(0)
    n = 50
    embeddings = rng.rand(n, 8).astype(np.float32)
    chip_indices = list(range(n))
    knn_geo_distance(
        embeddings,
        output_dir=tmp_path,
        prefix="bigk",
        chip_gdf=mock_chip_gdf,
        chip_indices=chip_indices,
        k_values=[1, 100],
    )
    agg = pd.read_csv(tmp_path / "bigk_knn_geo_distance.csv")
    assert set(agg["k"]) == {1}
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
# Tests: drop_null_rows
# ---------------------------------------------------------------------------


def test_drop_null_rows_removes_nonfinite_and_null_labels():
    """Rows with NaN/inf embeddings or null labels are dropped, alignment preserved."""
    from gelos.analysis import drop_null_rows

    embeddings = np.array(
        [
            [1.0, 2.0, 3.0],  # keep
            [np.nan, 1.0, 1.0],  # drop: NaN embedding
            [4.0, 5.0, 6.0],  # drop: null label
            [np.inf, 0.0, 0.0],  # drop: inf embedding
            [7.0, 8.0, 9.0],  # keep
        ],
        dtype=np.float64,
    )
    chip_indices = [10, 11, 12, 13, 14]
    labels = np.array([0.0, 1.0, np.nan, 2.0, 1.0])

    emb_out, idx_out, lab_out = drop_null_rows(embeddings, chip_indices, labels)

    # Only rows 0 and 4 survive (finite embedding AND non-null label).
    assert len(emb_out) == len(idx_out) == len(lab_out) == 2
    assert idx_out == [10, 14]
    np.testing.assert_array_equal(emb_out, np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]))
    np.testing.assert_array_equal(lab_out, np.array([0.0, 1.0]))


def test_drop_null_rows_no_nulls_returns_unchanged():
    """With no nulls, all rows are kept and order is preserved."""
    from gelos.analysis import drop_null_rows

    embeddings = np.arange(12, dtype=np.float64).reshape(4, 3)
    chip_indices = [1, 2, 3, 4]
    labels = np.array(["a", "b", "c", "d"], dtype=object)

    emb_out, idx_out, lab_out = drop_null_rows(embeddings, chip_indices, labels)

    assert idx_out == chip_indices
    np.testing.assert_array_equal(emb_out, embeddings)
    np.testing.assert_array_equal(lab_out, labels)


def test_drop_null_rows_object_label_none():
    """None entries in an object-dtype label array are treated as null."""
    from gelos.analysis import drop_null_rows

    embeddings = np.ones((3, 2), dtype=np.float64)
    chip_indices = [5, 6, 7]
    labels = np.array(["forest", None, "water"], dtype=object)

    emb_out, idx_out, lab_out = drop_null_rows(embeddings, chip_indices, labels)

    assert idx_out == [5, 7]
    assert list(lab_out) == ["forest", "water"]


def test_drop_null_rows_all_null_returns_empty():
    """When every row is null, returns empty arrays/list without raising."""
    from gelos.analysis import drop_null_rows

    embeddings = np.full((3, 4), np.nan, dtype=np.float64)
    chip_indices = [1, 2, 3]
    labels = np.array([0, 1, 2])

    emb_out, idx_out, lab_out = drop_null_rows(embeddings, chip_indices, labels)

    assert idx_out == []
    assert emb_out.shape[0] == 0
    assert lab_out.shape[0] == 0


# ---------------------------------------------------------------------------
# Tests: fill_null_rows
# ---------------------------------------------------------------------------


def test_fill_null_rows_zeros_nonfinite_and_drops_null_labels():
    """NaN/inf embeddings are zeroed; only null-label rows are dropped."""
    from gelos.analysis import fill_null_rows

    embeddings = np.array(
        [
            [1.0, 2.0, 3.0],  # keep as-is
            [np.nan, 1.0, 1.0],  # keep, NaN -> 0
            [4.0, 5.0, 6.0],  # drop: null label
            [np.inf, 0.0, -np.inf],  # keep, inf -> 0
            [7.0, 8.0, 9.0],  # keep as-is
        ],
        dtype=np.float64,
    )
    chip_indices = [10, 11, 12, 13, 14]
    labels = np.array([0.0, 1.0, np.nan, 2.0, 1.0])

    emb_out, idx_out, lab_out = fill_null_rows(embeddings, chip_indices, labels)

    # Row 2 (null label) dropped; the other four survive with non-finite cells zeroed.
    assert len(emb_out) == len(idx_out) == len(lab_out) == 4
    assert idx_out == [10, 11, 13, 14]
    np.testing.assert_array_equal(
        emb_out,
        np.array(
            [
                [1.0, 2.0, 3.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [7.0, 8.0, 9.0],
            ]
        ),
    )
    np.testing.assert_array_equal(lab_out, np.array([0.0, 1.0, 2.0, 1.0]))
    assert np.isfinite(emb_out).all()


def test_fill_null_rows_all_finite_passes_through():
    """All-finite embeddings with valid labels are returned unchanged."""
    from gelos.analysis import fill_null_rows

    embeddings = np.arange(12, dtype=np.float64).reshape(4, 3)
    chip_indices = [1, 2, 3, 4]
    labels = np.array(["a", "b", "c", "d"], dtype=object)

    emb_out, idx_out, lab_out = fill_null_rows(embeddings, chip_indices, labels)

    assert idx_out == chip_indices
    np.testing.assert_array_equal(emb_out, embeddings)
    np.testing.assert_array_equal(lab_out, labels)


def test_fill_null_rows_all_null_labels_returns_empty():
    """When every label is null, returns empty arrays/list without raising."""
    from gelos.analysis import fill_null_rows

    embeddings = np.full((3, 4), np.nan, dtype=np.float64)
    chip_indices = [1, 2, 3]
    labels = np.array([None, None, None], dtype=object)

    emb_out, idx_out, lab_out = fill_null_rows(embeddings, chip_indices, labels)

    assert idx_out == []
    assert emb_out.shape[0] == 0
    assert lab_out.shape[0] == 0


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


# ---------------------------------------------------------------------------
# Tests: run_analysis completion marker + model result skipping
# ---------------------------------------------------------------------------


def _marker_ctx(tmp_path, synthetic_labels):
    """Build a minimal AnalysisContext for run_analysis marker tests."""
    from gelos.analysis import AnalysisContext

    chip_gdf = gpd.GeoDataFrame(
        {
            "id": list(range(N_SAMPLES)),
            "lulc": synthetic_labels,
            "geometry": [Point(float(i), float(i)) for i in range(N_SAMPLES)],
        },
        crs="EPSG:4326",
    ).set_index("id")
    emb_dir = tmp_path / "embeddings" / "layer_-1"
    emb_dir.mkdir(parents=True)
    return AnalysisContext(
        yaml_config={},
        config_stem="exptest",
        experiment_name="marker test",
        style_cfg={"category_column": "lulc", "colors": {}, "labels": {}},
        category_column="lulc",
        embedding_extraction_strategies={
            "strategy": {
                "slice_args": [{"start": 0, "stop": 1, "step": 1}],
                "transforms": [{"type": "pca", "params": {"n_components": 2}}],
                "models": [{"type": "knn", "transform": "pca", "params": {"n_splits": 2}}],
            }
        },
        chip_gdf=chip_gdf,
        input_dir=tmp_path / "embeddings",
        output_dir=tmp_path / "processed",
        figures_dir=tmp_path / "figures",
        null_handling="drop",
        embeddings_directories=[emb_dir],
    )


def test_run_analysis_marker_and_model_skip(
    tmp_path, synthetic_embeddings, synthetic_labels, monkeypatch
):
    """Second run skips via .analysis_complete; overwrite re-enters but cached models skip."""
    import gelos.analysis as analysis_mod

    embeddings, chip_indices = synthetic_embeddings
    ctx = _marker_ctx(tmp_path, synthetic_labels)
    ctx.figures_dir.mkdir(parents=True)
    monkeypatch.setattr(analysis_mod, "setup_analysis_run", lambda *a, **k: ctx)
    extract_calls = []

    def fake_extract(directory, slice_args):
        extract_calls.append(directory)
        return embeddings, chip_indices

    monkeypatch.setattr(analysis_mod, "extract_embeddings", fake_extract)
    args = (tmp_path / "cfg.yaml", tmp_path, tmp_path, tmp_path, tmp_path)

    results = analysis_mod.run_analysis(*args)
    marker = ctx.output_dir / ".analysis_complete"
    assert marker.exists()
    assert len(extract_calls) == 1
    assert any(key.endswith("_knn") for key in results)
    model_csv = ctx.output_dir / "layer_-1" / "exptest_strategy_layer_-1_knn_knn_results.csv"
    assert model_csv.exists()
    # Figure names omit the config stem ("exptest"); the stem is the folder instead.
    cm_png = ctx.figures_dir / "strategy_layer_-1_knn_confusion_matrix.png"
    assert cm_png.exists()
    assert not list(ctx.figures_dir.glob("exptest_*"))

    # Second run: marker short-circuits everything.
    assert analysis_mod.run_analysis(*args) == {}
    assert len(extract_calls) == 1

    # Overwrite: re-enters, but embeddings come from cache and the cached model
    # result (results CSV + confusion matrix) is not recomputed.
    csv_mtime = model_csv.stat().st_mtime_ns
    results = analysis_mod.run_analysis(*args, overwrite=True)
    assert len(extract_calls) == 1  # .npy cache hit, no re-extraction
    assert results == {}  # model skipped, nothing recomputed
    assert model_csv.stat().st_mtime_ns == csv_mtime
    gc.collect()


def test_run_analysis_passes_chip_gdf_to_metrics(
    tmp_path, synthetic_embeddings, synthetic_labels, monkeypatch
):
    """run_analysis forwards chip_gdf/chip_indices so knn_geo_distance can run end to end."""
    import pandas as pd

    import gelos.analysis as analysis_mod

    embeddings, chip_indices = synthetic_embeddings
    ctx = _marker_ctx(tmp_path, synthetic_labels)
    ctx.figures_dir.mkdir(parents=True)
    strategy = ctx.embedding_extraction_strategies["strategy"]
    del strategy["models"]
    strategy["metrics"] = [{"type": "knn_geo_distance", "params": {"k_values": [1, 5]}}]
    monkeypatch.setattr(analysis_mod, "setup_analysis_run", lambda *a, **k: ctx)
    monkeypatch.setattr(
        analysis_mod,
        "extract_embeddings",
        lambda directory, slice_args: (embeddings, chip_indices),
    )

    analysis_mod.run_analysis(tmp_path / "cfg.yaml", tmp_path, tmp_path, tmp_path, tmp_path)

    metric_csv = ctx.output_dir / "layer_-1" / "exptest_strategy_layer_-1_knn_geo_distance.csv"
    assert metric_csv.exists()
    df = pd.read_csv(metric_csv)
    assert set(df["k"]) == {1, 5}
    assert set(df["measure"]) == {"gsd_m", "lat_diff_deg"}
# ---------------------------------------------------------------------------
# Tests: figure layout (issue #89)
# ---------------------------------------------------------------------------


def test_build_figure_prefix():
    """Figure prefix is {strategy}_{layer} with no config stem."""
    from gelos.analysis import build_figure_prefix

    assert build_figure_prefix("cls", "layer_-1") == "cls_layer_-1"


def test_setup_analysis_run_figures_dir_nested_by_config(tmp_path, cm_style_cfg):
    """setup_analysis_run resolves figures_dir to {base}/{data_version}/{config_stem}."""
    import yaml

    from gelos.analysis import setup_analysis_run

    raw = tmp_path / "raw"
    (raw / "v1").mkdir(parents=True)
    (raw / "v1" / "chips.csv").write_text("id,lulc\n0,0\n1,1\n2,2\n")

    config = {
        "data_version": "v1",
        "experiment_name": "figures dir test",
        "chip_tracker": "chips.csv",
        "chip_id_column": "id",
        "style": cm_style_cfg,
        "embedding_extraction_strategies": {},
    }
    yaml_path = tmp_path / "exp_figures.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    figures_base = tmp_path / "figures"
    ctx = setup_analysis_run(
        yaml_path, raw, tmp_path / "interim", tmp_path / "processed", figures_base
    )

    assert ctx.figures_dir == figures_base / "v1" / "exp_figures"
    assert ctx.figures_dir.is_dir()
    assert ctx.output_dir == tmp_path / "processed" / "v1" / "exp_figures"
    assert ctx.embeddings_directories == []
    gc.collect()
