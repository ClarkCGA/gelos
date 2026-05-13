import gc

import numpy as np
import pandas as pd
import pytest

from gelos.comp_metrics import (
    COMP_METRICS,
    cosine_distance,
    knn_purity_comparison,
    knn_purity_per_query_comparison,
    pca_ablation_comparison,
    wasserstein_distance,
)
from gelos.comp_plots import (
    COMP_PLOTS,
    distance_matrix,
    knn_purity_distribution_plot,
    knn_purity_plot,
    pca_ablation_table,
)
from gelos.comparison import ComparisonExperiment, setup_comparison

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 50
N_FEATURES = 32


@pytest.fixture()
def two_experiment_embeddings():
    """Two synthetic embedding sets with labels."""
    rng = np.random.RandomState(42)
    emb_a = rng.rand(N_SAMPLES, N_FEATURES).astype(np.float32)
    emb_b = rng.rand(N_SAMPLES, N_FEATURES).astype(np.float32) + 0.5
    return [("Experiment A", emb_a), ("Experiment B", emb_b)]


# ---------------------------------------------------------------------------
# Tests: Registry keys
# ---------------------------------------------------------------------------


def test_comp_metrics_registry_keys():
    """COMP_METRICS registry has expected keys and callables."""
    expected = {
        "pca_ablation_comparison",
        "cosine_distance",
        "wasserstein_distance",
        "knn_purity_comparison",
        "knn_purity_per_query_comparison",
    }
    assert expected <= set(COMP_METRICS.keys())
    for fn in COMP_METRICS.values():
        assert callable(fn)


def test_comp_plots_registry_keys():
    """COMP_PLOTS registry has expected keys and callables."""
    expected = {
        "pca_ablation_table",
        "distance_matrix",
        "knn_purity_plot",
        "knn_purity_distribution_plot",
    }
    assert expected <= set(COMP_PLOTS.keys())
    for fn in COMP_PLOTS.values():
        assert callable(fn)


# ---------------------------------------------------------------------------
# Tests: Comparison metrics
# ---------------------------------------------------------------------------


def test_cosine_distance(two_experiment_embeddings, tmp_path):
    """Cosine distance returns a symmetric matrix with diagonal ~1.0."""
    result = cosine_distance(two_experiment_embeddings, output_dir=tmp_path, prefix="test")
    assert "labels" in result
    assert "matrix" in result
    assert "df" in result

    matrix = result["matrix"]
    assert matrix.shape == (2, 2)

    # Diagonal should be 1.0 (self-similarity)
    np.testing.assert_almost_equal(matrix[0, 0], 1.0, decimal=5)
    np.testing.assert_almost_equal(matrix[1, 1], 1.0, decimal=5)

    # Symmetric
    np.testing.assert_almost_equal(matrix[0, 1], matrix[1, 0], decimal=5)

    # CSV saved
    csv_files = list(tmp_path.glob("*_cosine_distance.csv"))
    assert len(csv_files) == 1
    gc.collect()


def test_wasserstein_distance(two_experiment_embeddings, tmp_path):
    """Wasserstein distance returns a symmetric matrix with diagonal = 0."""
    result = wasserstein_distance(two_experiment_embeddings, output_dir=tmp_path, prefix="test")
    assert "labels" in result
    assert "matrix" in result

    matrix = result["matrix"]
    assert matrix.shape == (2, 2)

    # Diagonal should be 0.0 (self-distance)
    np.testing.assert_almost_equal(matrix[0, 0], 0.0, decimal=5)
    np.testing.assert_almost_equal(matrix[1, 1], 0.0, decimal=5)

    # Symmetric
    np.testing.assert_almost_equal(matrix[0, 1], matrix[1, 0], decimal=5)

    # Off-diagonal should be positive (embeddings differ)
    assert matrix[0, 1] > 0

    # CSV saved
    csv_files = list(tmp_path.glob("*_wasserstein_distance.csv"))
    assert len(csv_files) == 1
    gc.collect()


def test_pca_ablation_comparison(tmp_path):
    """pca_ablation_comparison merges per-experiment CSVs into one."""
    # Create fake per-experiment PCA ablation CSVs at deterministic paths
    exp_dir = tmp_path / "v3" / "config_a" / "layer_11"
    exp_dir.mkdir(parents=True)
    df_a = pd.DataFrame(
        {
            "threshold": [0.9, 0.95],
            "n_components": [5, 10],
            "total_variance_explained": [0.91, 0.96],
        }
    )
    df_a.to_csv(exp_dir / "config_a_cls_layer_11_pca_ablation.csv", index=False)

    experiments = [
        ComparisonExperiment(
            data_version="v3", config="config_a", strategy="cls", layer="layer_11", label="Exp A"
        ),
    ]

    output_dir = tmp_path / "comparisons"
    output_dir.mkdir()

    result = pca_ablation_comparison(
        [(exp.label, None) for exp in experiments],
        processed_data_dir=tmp_path,
        output_dir=output_dir,
        prefix="test",
        experiments=experiments,
    )
    assert "comparison_df" in result
    assert not result["comparison_df"].empty
    assert "experiment" in result["comparison_df"].columns

    csv_files = list(output_dir.glob("*_pca_ablation_comparison.csv"))
    assert len(csv_files) == 1
    gc.collect()


# ---------------------------------------------------------------------------
# Tests: Comparison plots
# ---------------------------------------------------------------------------


def test_distance_matrix_plot(two_experiment_embeddings, tmp_path):
    """distance_matrix creates a PNG file."""
    metric_result = cosine_distance(two_experiment_embeddings, output_dir=tmp_path, prefix="test")
    output_path = tmp_path / "test_distance_matrix.png"
    distance_matrix(metric_result, output_path=output_path)
    assert output_path.exists()
    gc.collect()


def test_pca_ablation_table_plot(tmp_path):
    """pca_ablation_table creates a PNG file from comparison data."""
    df = pd.DataFrame(
        {
            "experiment": ["A", "A", "B", "B"],
            "threshold": [0.9, 0.95, 0.9, 0.95],
            "n_components": [5, 10, 8, 15],
            "proportion_of_total_components": [0.05, 0.10, 0.08, 0.15],
            "total_variance_explained": [0.91, 0.96, 0.92, 0.97],
        }
    )
    output_path = tmp_path / "test_pca_table.png"
    pca_ablation_table({"comparison_df": df}, output_path=output_path)
    assert output_path.exists()
    gc.collect()


def test_knn_purity_comparison(tmp_path):
    """knn_purity_comparison merges per-experiment CSVs into one."""
    exp_dir = tmp_path / "v3" / "config_a" / "layer_11"
    exp_dir.mkdir(parents=True)
    df_a = pd.DataFrame(
        {
            "k": [1, 1, 5, 5],
            "class": ["overall", "0", "overall", "0"],
            "purity": [0.8, 0.9, 0.7, 0.75],
            "n_samples": [100, 50, 100, 50],
        }
    )
    df_a.to_csv(exp_dir / "config_a_cls_layer_11_knn_purity.csv", index=False)

    experiments = [
        ComparisonExperiment(
            data_version="v3", config="config_a", strategy="cls", layer="layer_11", label="Exp A"
        ),
    ]

    output_dir = tmp_path / "comparisons"
    output_dir.mkdir()

    result = knn_purity_comparison(
        [(exp.label, None) for exp in experiments],
        processed_data_dir=tmp_path,
        output_dir=output_dir,
        prefix="test",
        experiments=experiments,
    )
    assert "comparison_df" in result
    assert not result["comparison_df"].empty
    assert "experiment" in result["comparison_df"].columns

    csv_files = list(output_dir.glob("*_knn_purity_comparison.csv"))
    assert len(csv_files) == 1
    gc.collect()


def test_knn_purity_plot_output(tmp_path):
    """knn_purity_plot creates a PNG file from comparison data."""
    df = pd.DataFrame(
        {
            "k": [1, 1, 1, 5, 5, 5, 1, 1, 1, 5, 5, 5],
            "class": ["overall", "0", "1"] * 4,
            "purity": [0.8, 0.9, 0.7, 0.75, 0.85, 0.65, 0.82, 0.88, 0.76, 0.78, 0.84, 0.72],
            "n_samples": [100, 50, 50] * 4,
            "experiment": ["A"] * 6 + ["B"] * 6,
        }
    )
    output_path = tmp_path / "test_knn_purity.png"
    knn_purity_plot({"comparison_df": df}, output_path=output_path)
    assert output_path.exists()
    gc.collect()


def test_knn_purity_per_query_comparison(tmp_path):
    """knn_purity_per_query_comparison merges per-experiment per-query CSVs."""
    exp_dir = tmp_path / "v3" / "config_a" / "layer_11"
    exp_dir.mkdir(parents=True)
    df_a = pd.DataFrame(
        {
            "k": [1, 1, 5, 5],
            "class": ["0", "1", "0", "1"],
            "query_idx": [0, 1, 0, 1],
            "purity": [1.0, 0.0, 0.6, 0.4],
        }
    )
    df_a.to_csv(exp_dir / "config_a_cls_layer_11_knn_purity_per_query.csv", index=False)

    experiments = [
        ComparisonExperiment(
            data_version="v3", config="config_a", strategy="cls", layer="layer_11", label="Exp A"
        ),
    ]

    output_dir = tmp_path / "comparisons"
    output_dir.mkdir()

    result = knn_purity_per_query_comparison(
        [(exp.label, None) for exp in experiments],
        processed_data_dir=tmp_path,
        output_dir=output_dir,
        prefix="test",
        experiments=experiments,
    )
    assert "comparison_df" in result
    assert not result["comparison_df"].empty
    assert "experiment" in result["comparison_df"].columns

    csv_files = list(output_dir.glob("*_knn_purity_per_query_comparison.csv"))
    assert len(csv_files) == 1
    gc.collect()


def test_knn_purity_distribution_plot_output(tmp_path):
    """knn_purity_distribution_plot creates a PNG file from per-query data."""
    rng = np.random.RandomState(0)
    rows = []
    for exp in ["A", "B"]:
        for cls in ["0", "1"]:
            for k in [1, 5]:
                for q_idx in range(20):
                    rows.append(
                        {
                            "experiment": exp,
                            "class": cls,
                            "k": k,
                            "query_idx": q_idx,
                            "purity": float(rng.rand()),
                        }
                    )
    df = pd.DataFrame(rows)
    output_path = tmp_path / "test_knn_purity_distribution.png"
    knn_purity_distribution_plot({"comparison_df": df}, output_path=output_path)
    assert output_path.exists()
    gc.collect()


# ---------------------------------------------------------------------------
# Tests: Comparison setup
# ---------------------------------------------------------------------------


def test_setup_comparison_missing_path(tmp_path):
    """setup_comparison warns when an experiment's embeddings don't exist."""
    import yaml

    config = {
        "comparison_name": "Test Comparison",
        "experiments": [
            {
                "data_version": "v3",
                "config": "s2l2a_prithvi",
                "strategy": "cls_token",
                "layer": "encoder_layer_11",
                "label": "S2 CLS L11",
            },
        ],
        "comp_metrics": [],
        "comp_plots": [],
    }

    yaml_path = tmp_path / "test_comparison.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()

    ctx = setup_comparison(yaml_path, processed_dir, figures_dir)
    assert len(ctx.experiments) == 1
    assert ctx.comparison_name == "Test Comparison"
    assert ctx.output_dir.exists()
    assert ctx.figures_dir.exists()
    gc.collect()
