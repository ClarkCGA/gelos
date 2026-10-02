import gc

import numpy as np
import pandas as pd
import pytest

from gelos.comp_metrics import (
    COMP_METRICS,
    cosine_distance,
    knn_geo_distance_comparison,
    knn_purity_comparison,
    knn_purity_per_query_comparison,
    pca_ablation_comparison,
    per_chip_similarity_to_control,
    wasserstein_distance,
)
from gelos.comp_plots import (
    COMP_PLOTS,
    _strip_common_prefix,
    distance_matrix,
    knn_gsd_plot,
    knn_lat_diff_plot,
    knn_purity_distribution_plot,
    knn_purity_plot,
    knn_purity_violin_distribution_plot,
    pca_ablation_table,
    per_class_ecdf_plot,
    per_class_similarity_distribution_plot,
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
        "knn_geo_distance_comparison",
        "per_chip_similarity_to_control",
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
        "knn_gsd_plot",
        "knn_lat_diff_plot",
        "knn_purity_distribution_plot",
        "knn_purity_violin_distribution_plot",
        "per_class_similarity_distribution_plot",
        "per_class_ecdf_plot",
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


# ---------------------------------------------------------------------------
# Tests: kNN geographic distance comparison + plots (issue #85)
# ---------------------------------------------------------------------------


def _make_geo_distance_df(experiments, ks=(1, 5)):
    """Build an aggregated knn_geo_distance_comparison df (both measures) for any experiments."""
    rows = []
    for e_idx, exp in enumerate(experiments):
        for k in ks:
            for measure, base in (("gsd_m", 50_000.0), ("lat_diff_deg", 0.5)):
                mean = base * (1 + 0.1 * e_idx + 0.05 * k)
                rows.append(
                    {
                        "k": k,
                        "measure": measure,
                        "mean": mean,
                        "median": mean * 0.9,
                        "q1": mean * 0.5,
                        "q3": mean * 1.5,
                        "n_samples": 100,
                        "experiment": exp,
                    }
                )
    return pd.DataFrame(rows)


def test_knn_geo_distance_comparison(tmp_path):
    """knn_geo_distance_comparison merges per-experiment CSVs and tags the experiment."""
    exp_dir = tmp_path / "v3" / "config_a" / "layer_11"
    exp_dir.mkdir(parents=True)
    df_a = _make_geo_distance_df(["unused"]).drop(columns="experiment")
    df_a.to_csv(exp_dir / "config_a_cls_layer_11_knn_geo_distance.csv", index=False)

    experiments = [
        ComparisonExperiment(
            data_version="v3", config="config_a", strategy="cls", layer="layer_11", label="Exp A"
        ),
        ComparisonExperiment(
            data_version="v3", config="config_b", strategy="cls", layer="layer_11", label="Exp B"
        ),  # no CSV on disk -> skipped with a warning
    ]

    output_dir = tmp_path / "comparisons"
    output_dir.mkdir()

    result = knn_geo_distance_comparison(
        [(exp.label, None) for exp in experiments],
        processed_data_dir=tmp_path,
        output_dir=output_dir,
        prefix="test",
        experiments=experiments,
    )
    merged = result["comparison_df"]
    assert not merged.empty
    assert set(merged["experiment"]) == {"Exp A"}
    assert {"k", "measure", "mean", "median", "q1", "q3", "n_samples"} <= set(merged.columns)
    assert len(merged) == len(df_a)

    csv_files = list(output_dir.glob("*_knn_geo_distance_comparison.csv"))
    assert len(csv_files) == 1
    gc.collect()


def test_knn_geo_distance_comparison_no_data(tmp_path):
    """No per-experiment CSVs -> empty comparison_df, nothing written."""
    experiments = [
        ComparisonExperiment(
            data_version="v3", config="config_a", strategy="cls", layer="layer_11", label="Exp A"
        ),
    ]
    result = knn_geo_distance_comparison(
        [("Exp A", None)],
        processed_data_dir=tmp_path,
        output_dir=tmp_path,
        prefix="test",
        experiments=experiments,
    )
    assert result["comparison_df"].empty
    assert not list(tmp_path.glob("*_knn_geo_distance_comparison.csv"))
    gc.collect()


@pytest.mark.parametrize("plot_fn", [knn_gsd_plot, knn_lat_diff_plot])
def test_knn_geo_distance_plot_output(plot_fn, tmp_path):
    """Both geo-distance plots write a PNG, with and without the IQR band / log-x."""
    df = _make_geo_distance_df(["A", "B"])
    plain = tmp_path / f"{plot_fn.__name__}.png"
    plot_fn({"comparison_df": df}, output_path=plain)
    assert plain.exists()

    styled = tmp_path / f"{plot_fn.__name__}_iqr_log.png"
    plot_fn({"comparison_df": df}, output_path=styled, show_iqr=True, log_x=True)
    assert styled.exists()
    gc.collect()


@pytest.mark.parametrize("plot_fn", [knn_gsd_plot, knn_lat_diff_plot])
def test_knn_geo_distance_plot_empty_df(plot_fn, tmp_path):
    """Empty comparison_df returns early without writing a file."""
    output_path = tmp_path / f"{plot_fn.__name__}_empty.png"
    plot_fn({"comparison_df": pd.DataFrame()}, output_path=output_path)
    assert not output_path.exists()
    gc.collect()


def test_knn_geo_distance_plot_missing_measure(tmp_path):
    """A df with only gsd_m rows -> knn_lat_diff_plot writes nothing, knn_gsd_plot does."""
    df = _make_geo_distance_df(["A", "B"])
    gsd_only = df[df["measure"] == "gsd_m"]

    lat_path = tmp_path / "lat_diff_missing.png"
    knn_lat_diff_plot({"comparison_df": gsd_only}, output_path=lat_path)
    assert not lat_path.exists()

    gsd_path = tmp_path / "gsd_present.png"
    knn_gsd_plot({"comparison_df": gsd_only}, output_path=gsd_path)
    assert gsd_path.exists()
    gc.collect()


def test_knn_geo_distance_plot_scales_and_single_axis(tmp_path, monkeypatch):
    """gsd plot shows km (mean/1000), lat plot shows degrees; both are single-panel."""
    from gelos import comp_plots

    figs = []
    monkeypatch.setattr(comp_plots.plt, "close", lambda fig, *a, **k: figs.append(fig))
    try:
        df = _make_geo_distance_df(["A"], ks=(1, 5))
        knn_gsd_plot({"comparison_df": df}, output_path=tmp_path / "gsd.png")
        knn_lat_diff_plot({"comparison_df": df}, output_path=tmp_path / "lat.png")

        gsd_fig, lat_fig = figs
        assert len(gsd_fig.axes) == 1 and len(lat_fig.axes) == 1

        gsd_line = gsd_fig.axes[0].get_lines()[0]
        expected_km = df[df["measure"] == "gsd_m"].sort_values("k")["mean"].to_numpy() / 1000
        np.testing.assert_allclose(gsd_line.get_ydata(), expected_km)

        lat_line = lat_fig.axes[0].get_lines()[0]
        expected_deg = df[df["measure"] == "lat_diff_deg"].sort_values("k")["mean"].to_numpy()
        np.testing.assert_allclose(lat_line.get_ydata(), expected_deg)
    finally:
        monkeypatch.undo()
        import matplotlib.pyplot as plt

        plt.close("all")
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


def _make_per_query_df(experiments, classes=("0", "1"), ks=(1, 5), n_queries=20, seed=0):
    """Build a synthetic per-query purity df for the violin plot tests."""
    rng = np.random.RandomState(seed)
    rows = []
    for exp in experiments:
        for cls in classes:
            for k in ks:
                for q_idx in range(n_queries):
                    rows.append(
                        {
                            "experiment": exp,
                            "class": cls,
                            "k": k,
                            "query_idx": q_idx,
                            "purity": float(rng.rand()),
                        }
                    )
    return pd.DataFrame(rows)


def test_knn_purity_violin_distribution_plot_output(tmp_path):
    """Two experiments -> split-violin path saves a PNG."""
    df = _make_per_query_df(["A", "B"], seed=0)
    output_path = tmp_path / "test_knn_purity_violin_distribution.png"
    knn_purity_violin_distribution_plot({"comparison_df": df}, output_path=output_path)
    assert output_path.exists()
    gc.collect()


def test_knn_purity_violin_distribution_plot_single_experiment(tmp_path):
    """One experiment -> single-violin fallback saves a PNG."""
    df = _make_per_query_df(["A"], seed=1)
    output_path = tmp_path / "test_knn_purity_violin_single.png"
    knn_purity_violin_distribution_plot({"comparison_df": df}, output_path=output_path)
    assert output_path.exists()
    gc.collect()


def test_knn_purity_violin_distribution_plot_three_experiments(tmp_path):
    """Three experiments -> side-by-side single-violin fallback saves a PNG."""
    df = _make_per_query_df(["A", "B", "C"], seed=2)
    output_path = tmp_path / "test_knn_purity_violin_three.png"
    knn_purity_violin_distribution_plot({"comparison_df": df}, output_path=output_path)
    assert output_path.exists()
    gc.collect()


def test_knn_purity_violin_distribution_plot_force_single_split(tmp_path):
    """split=False forces single violins even with two experiments."""
    df = _make_per_query_df(["A", "B"], seed=3)
    output_path = tmp_path / "test_knn_purity_violin_force_single.png"
    knn_purity_violin_distribution_plot(
        {"comparison_df": df}, output_path=output_path, split=False
    )
    assert output_path.exists()
    gc.collect()


def test_knn_purity_violin_distribution_plot_degenerate(tmp_path):
    """Constant/single-sample purity exercises the small-sample guard without raising."""
    rows = []
    # One experiment, constant purity (np.ptp == 0) and a single-sample group.
    for cls, n in [("0", 5), ("1", 1)]:
        for q_idx in range(n):
            rows.append(
                {"experiment": "A", "class": cls, "k": 1, "query_idx": q_idx, "purity": 0.5}
            )
    df = pd.DataFrame(rows)
    output_path = tmp_path / "test_knn_purity_violin_degenerate.png"
    knn_purity_violin_distribution_plot({"comparison_df": df}, output_path=output_path)
    assert output_path.exists()
    gc.collect()


def test_knn_purity_violin_distribution_plot_empty(tmp_path):
    """Empty comparison_df returns early without writing a file."""
    output_path = tmp_path / "test_knn_purity_violin_empty.png"
    knn_purity_violin_distribution_plot({"comparison_df": pd.DataFrame()}, output_path=output_path)
    assert not output_path.exists()
    gc.collect()


def test_knn_purity_violin_distribution_plot_split_pairs_two_pairs(tmp_path):
    """Two pairs (4 experiments) -> two split violins per k, PNG written."""
    df = _make_per_query_df(["A1", "A2", "B1", "B2"], seed=4)
    output_path = tmp_path / "test_knn_purity_violin_split_pairs_two.png"
    knn_purity_violin_distribution_plot(
        {"comparison_df": df},
        output_path=output_path,
        split_pairs=[["A1", "A2"], ["B1", "B2"]],
    )
    assert output_path.exists()
    gc.collect()


def test_knn_purity_violin_distribution_plot_split_pairs_one_pair_one_unpaired(tmp_path):
    """One pair + one unpaired experiment -> split + single coexist, PNG written."""
    df = _make_per_query_df(["A1", "A2", "C"], seed=5)
    output_path = tmp_path / "test_knn_purity_violin_split_pairs_mixed.png"
    knn_purity_violin_distribution_plot(
        {"comparison_df": df},
        output_path=output_path,
        split_pairs=[["A1", "A2"]],
    )
    assert output_path.exists()
    gc.collect()


def test_knn_purity_violin_distribution_plot_split_pairs_missing_experiment(tmp_path):
    """A pair naming a missing experiment warns and is skipped; PNG still written."""
    from loguru import logger

    df = _make_per_query_df(["A", "B"], seed=6)
    output_path = tmp_path / "test_knn_purity_violin_split_pairs_missing.png"

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        knn_purity_violin_distribution_plot(
            {"comparison_df": df},
            output_path=output_path,
            split_pairs=[["A", "ghost"]],
        )
    finally:
        logger.remove(sink_id)

    assert any("ghost" in msg for msg in messages)
    # The invalid pair is skipped; A and B fall through as unpaired singles.
    assert output_path.exists()
    gc.collect()


def test_knn_purity_violin_distribution_plot_split_pairs_overrides_split_flag(tmp_path):
    """split_pairs takes precedence whether split is True or False; PNG written."""
    df = _make_per_query_df(["A1", "A2", "B1", "B2"], seed=7)

    out_true = tmp_path / "test_knn_purity_violin_split_pairs_override_true.png"
    knn_purity_violin_distribution_plot(
        {"comparison_df": df},
        output_path=out_true,
        split=True,
        split_pairs=[["A1", "A2"], ["B1", "B2"]],
    )
    assert out_true.exists()

    out_false = tmp_path / "test_knn_purity_violin_split_pairs_override_false.png"
    knn_purity_violin_distribution_plot(
        {"comparison_df": df},
        output_path=out_false,
        split=False,
        split_pairs=[["A1", "A2"], ["B1", "B2"]],
    )
    assert out_false.exists()
    gc.collect()


def test_knn_purity_violin_distribution_plot_split_pairs_empty(tmp_path):
    """split_pairs=[] -> all experiments rendered as single violins, no crash."""
    df = _make_per_query_df(["A", "B", "C"], seed=8)
    output_path = tmp_path / "test_knn_purity_violin_split_pairs_empty.png"
    knn_purity_violin_distribution_plot(
        {"comparison_df": df},
        output_path=output_path,
        split_pairs=[],
    )
    assert output_path.exists()
    gc.collect()


# ---------------------------------------------------------------------------
# Tests: fixed kNN plot layout (issue #84)
# ---------------------------------------------------------------------------


def _make_purity_df(experiments, ks=(1, 5)):
    """Build an aggregated knn_purity_comparison df (classes overall/0/1) for any experiments."""
    rows = []
    for e_idx, exp in enumerate(experiments):
        for k in ks:
            for cls in ("overall", "0", "1"):
                rows.append(
                    {
                        "k": k,
                        "class": cls,
                        "purity": 0.5 + 0.05 * e_idx + (0.1 if cls == "0" else 0.0) - 0.01 * k,
                        "n_samples": 100 if cls == "overall" else 50,
                        "experiment": exp,
                    }
                )
    return pd.DataFrame(rows)


# Facet-grid plots (one subplot per class); these also take the shared-axes test.
_KNN_FACET_PLOT_CASES = [
    (knn_purity_plot, _make_purity_df),
    (knn_purity_distribution_plot, _make_per_query_df),
    (knn_purity_violin_distribution_plot, _make_per_query_df),
]

# Every fixed-layout kNN plot, including the single-panel geo-distance plots (issue #85).
_KNN_PLOT_CASES = _KNN_FACET_PLOT_CASES + [
    (knn_gsd_plot, _make_geo_distance_df),
    (knn_lat_diff_plot, _make_geo_distance_df),
]

# 60-char names starting with different letters: they share no prefix, so
# _strip_common_prefix strips nothing and each is rendered in full (no truncation).
_KNN_LONG_NAMES = [(f"{c}xperiment_" + "long_name_" * 6)[:60] for c in "ABCD"]
_KNN_NAME_SETS = [["A"], ["A", "B"], ["A", "B", "C", "D"], _KNN_LONG_NAMES]


def _capture_knn_figures(monkeypatch):
    """Patch ``plt.close`` in comp_plots to collect figures instead of closing them."""
    import gelos.comp_plots as comp_plots

    figs = []
    monkeypatch.setattr(comp_plots.plt, "close", lambda fig, *a, **k: figs.append(fig))
    return figs


def _release_knn_figures(monkeypatch):
    """Undo the ``plt.close`` patch before closing everything to avoid figure leaks."""
    import matplotlib.pyplot as plt

    monkeypatch.undo()
    plt.close("all")
    gc.collect()


def test_strip_common_prefix_no_truncation():
    """_strip_common_prefix strips a shared prefix but never truncates labels."""
    long_label = "x" * 120
    assert _strip_common_prefix([long_label]) == {long_label: long_label}
    assert len(_strip_common_prefix([long_label])[long_label]) == 120
    # Shared prefix ending on a word boundary is stripped whole.
    assert _strip_common_prefix(["model_a_run_1", "model_a_run_2"]) == {
        "model_a_run_1": "1",
        "model_a_run_2": "2",
    }
    # Shared prefix ending mid-word backs off to the previous word boundary.
    assert _strip_common_prefix(["model_alpha", "model_amber"]) == {
        "model_alpha": "alpha",
        "model_amber": "amber",
    }
    assert _strip_common_prefix(["same", "same"]) == {"same": "same"}
    assert _strip_common_prefix([]) == {}


@pytest.mark.parametrize(("plot_fn", "make_df"), _KNN_PLOT_CASES)
def test_knn_plot_fixed_size_and_axes(plot_fn, make_df, tmp_path, monkeypatch):
    """PNG pixel size, figure size and axes positions do not depend on the legend content."""
    import matplotlib.image as mpimg

    figs = _capture_knn_figures(monkeypatch)
    try:
        shapes, sizes, positions = [], [], []
        for i, names in enumerate(_KNN_NAME_SETS):
            path = tmp_path / f"{plot_fn.__name__}_{i}.png"
            plot_fn({"comparison_df": make_df(names)}, output_path=path)
            assert path.exists()
            fig = figs[-1]
            shapes.append(mpimg.imread(path).shape[:2])
            sizes.append(tuple(fig.get_size_inches()))
            positions.append([ax.get_position().bounds for ax in fig.axes])
            assert len(fig.legends) == 1
            assert len(fig.legends[0].get_texts()) == len(names)
        assert len(figs) == len(_KNN_NAME_SETS)
        assert len(set(shapes)) == 1, shapes
        assert len(set(sizes)) == 1, sizes
        assert all(p == positions[0] for p in positions), positions
    finally:
        _release_knn_figures(monkeypatch)


@pytest.mark.parametrize(("plot_fn", "make_df"), _KNN_PLOT_CASES)
def test_knn_plot_legend_inside_figure(plot_fn, make_df, tmp_path, monkeypatch):
    """Four long-name legend entries are rendered verbatim, inside the canvas and below the xlabels."""
    figs = _capture_knn_figures(monkeypatch)
    try:
        path = tmp_path / f"{plot_fn.__name__}_long.png"
        plot_fn({"comparison_df": make_df(_KNN_LONG_NAMES)}, output_path=path)
        fig = figs[-1]
        assert [t.get_text() for t in fig.legends[0].get_texts()] == _KNN_LONG_NAMES
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_bbox = fig.legends[0].get_window_extent(renderer)
        fig_bbox = fig.bbox
        assert legend_bbox.x0 >= 0 and legend_bbox.y0 >= 0
        assert legend_bbox.x1 <= fig_bbox.width and legend_bbox.y1 <= fig_bbox.height
        # Lowest axes row (smallest y0 in figure fraction) must sit above the legend.
        lowest_y0 = min(ax.get_position().y0 for ax in fig.axes)
        lowest_axes = [ax for ax in fig.axes if ax.get_position().y0 == lowest_y0]
        xlabel_bottom = min(ax.get_tightbbox(renderer).y0 for ax in lowest_axes)
        assert legend_bbox.y1 <= xlabel_bottom
    finally:
        _release_knn_figures(monkeypatch)


@pytest.mark.parametrize("plot_fn", [knn_gsd_plot, knn_lat_diff_plot])
@pytest.mark.parametrize("kwargs", [{}, {"show_iqr": True, "log_x": True}])
def test_knn_geo_distance_plot_axes_inside_canvas(plot_fn, kwargs, tmp_path, monkeypatch):
    """With 5-digit km values the y-axis decoration (ticks + ylabel) stays inside the canvas."""
    from gelos import comp_plots

    df = _make_geo_distance_df(["A", "B", "C", "D"], ks=(1, 5, 10, 50))
    gsd = df["measure"] == "gsd_m"
    df.loc[gsd, ["mean", "median", "q1", "q3"]] *= 500  # base 25 000 000 m -> 5-digit km
    figs = _capture_knn_figures(monkeypatch)
    try:
        path = tmp_path / f"{plot_fn.__name__}_wide_ticks.png"
        plot_fn({"comparison_df": df}, output_path=path, **kwargs)
        fig = figs[-1]
        assert fig.get_figwidth() == pytest.approx(
            comp_plots._KNN_LEFT_IN + 6.0 + comp_plots._KNN_RIGHT_IN
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            bbox = ax.get_tightbbox(renderer)
            assert bbox.x0 >= 0, bbox
            assert bbox.x1 <= fig.bbox.width, bbox
    finally:
        _release_knn_figures(monkeypatch)


@pytest.mark.parametrize(("plot_fn", "make_df"), _KNN_FACET_PLOT_CASES)
def test_knn_plot_shared_axes(plot_fn, make_df, tmp_path, monkeypatch):
    """Facets share both x and y axes."""
    figs = _capture_knn_figures(monkeypatch)
    try:
        path = tmp_path / f"{plot_fn.__name__}_shared.png"
        plot_fn({"comparison_df": make_df(["A", "B"])}, output_path=path)
        fig = figs[-1]
        assert len(fig.axes) >= 2
        ax0, ax1 = fig.axes[0], fig.axes[1]
        assert ax0.get_shared_x_axes().joined(ax0, ax1)
        assert ax0.get_shared_y_axes().joined(ax0, ax1)
    finally:
        _release_knn_figures(monkeypatch)


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


# ---------------------------------------------------------------------------
# Tests: per_chip_similarity_to_control
# ---------------------------------------------------------------------------


def _write_experiment_cache(
    processed_data_dir,
    exp: ComparisonExperiment,
    embeddings: np.ndarray,
    chip_indices: np.ndarray,
) -> None:
    """Write ``{prefix}_embeddings.npy`` and ``{prefix}_chip_indices.npy`` for an exp."""
    from gelos.analysis import build_prefix

    prefix = build_prefix(exp.config, exp.strategy, exp.layer)
    layer_dir = processed_data_dir / exp.data_version / exp.config / exp.layer
    layer_dir.mkdir(parents=True, exist_ok=True)
    np.save(layer_dir / f"{prefix}_embeddings.npy", embeddings)
    np.save(layer_dir / f"{prefix}_chip_indices.npy", chip_indices)


def _write_chip_tracker_csv(path, ids, classes) -> None:
    pd.DataFrame({"id": ids, "lulc": classes}).to_csv(path, index=False)


def test_per_chip_similarity_to_control(tmp_path):
    """per_chip_similarity_to_control aligns chips and writes both CSVs."""
    rng = np.random.RandomState(0)
    chip_ids = np.array([10, 11, 12, 13])
    classes = ["A", "A", "B", "B"]

    control_emb = rng.rand(4, 8).astype(np.float32)
    ablation_emb = control_emb + 0.1 * rng.randn(4, 8).astype(np.float32)

    control_exp = ComparisonExperiment(
        data_version="v1",
        config="cfg",
        strategy="cls",
        layer="layer_11",
        label="Control",
    )
    ablation_exp = ComparisonExperiment(
        data_version="v1",
        config="cfg_abl",
        strategy="cls",
        layer="layer_11",
        label="Ablation",
    )
    _write_experiment_cache(tmp_path, control_exp, control_emb, chip_ids)
    _write_experiment_cache(tmp_path, ablation_exp, ablation_emb, chip_ids)

    tracker_path = tmp_path / "tracker.csv"
    _write_chip_tracker_csv(tracker_path, chip_ids, classes)

    output_dir = tmp_path / "comparisons"
    output_dir.mkdir()

    result = per_chip_similarity_to_control(
        [("Control", None), ("Ablation", None)],
        processed_data_dir=tmp_path,
        output_dir=output_dir,
        prefix="test",
        experiments=[control_exp, ablation_exp],
        control_label="Control",
        chip_tracker_path=tracker_path,
        chip_id_column="id",
        category_column="lulc",
    )

    assert "comparison_df" in result and "summary_df" in result
    long_df = result["comparison_df"]
    assert set(long_df.columns) == {"experiment", "file_id", "class", "cosine_similarity"}
    # 2 experiments × 4 chips
    assert len(long_df) == 8

    control_sims = long_df[long_df["experiment"] == "Control"]["cosine_similarity"].to_numpy()
    np.testing.assert_allclose(control_sims, 1.0, atol=1e-6)

    summary = result["summary_df"]
    # 2 experiments × 2 classes
    assert len(summary) == 4
    assert {"experiment", "class", "n", "median", "ks_vs_control_stat"} <= set(summary.columns)

    assert (output_dir / "test_per_chip_similarity_to_control.csv").exists()
    assert (output_dir / "test_per_chip_similarity_to_control_summary.csv").exists()
    gc.collect()


def test_per_chip_similarity_to_control_missing_control(tmp_path):
    """Unknown control_label raises ValueError."""
    exp = ComparisonExperiment(
        data_version="v1",
        config="cfg",
        strategy="cls",
        layer="layer_11",
        label="Ablation",
    )
    with pytest.raises(ValueError, match="control_label"):
        per_chip_similarity_to_control(
            [("Ablation", None)],
            processed_data_dir=tmp_path,
            output_dir=tmp_path,
            prefix="test",
            experiments=[exp],
            control_label="Control",  # not in experiments
            chip_tracker_path=tmp_path / "tracker.csv",
            chip_id_column="id",
            category_column="lulc",
        )
    gc.collect()


def test_per_chip_similarity_to_control_partial_overlap(tmp_path):
    """Chips not in both runs are dropped; result size = intersection × n_experiments."""
    rng = np.random.RandomState(1)
    control_ids = np.array([1, 2, 3, 4])
    ablation_ids = np.array([3, 4, 5, 6])  # 2-chip intersection
    classes_lookup = {1: "A", 2: "A", 3: "B", 4: "B", 5: "A", 6: "B"}

    control_emb = rng.rand(4, 6).astype(np.float32)
    ablation_emb = rng.rand(4, 6).astype(np.float32)

    control_exp = ComparisonExperiment(
        data_version="v1",
        config="cfg",
        strategy="cls",
        layer="layer_11",
        label="Control",
    )
    ablation_exp = ComparisonExperiment(
        data_version="v1",
        config="cfg_abl",
        strategy="cls",
        layer="layer_11",
        label="Ablation",
    )
    _write_experiment_cache(tmp_path, control_exp, control_emb, control_ids)
    _write_experiment_cache(tmp_path, ablation_exp, ablation_emb, ablation_ids)

    tracker_path = tmp_path / "tracker.csv"
    all_ids = sorted(classes_lookup.keys())
    _write_chip_tracker_csv(tracker_path, all_ids, [classes_lookup[i] for i in all_ids])

    output_dir = tmp_path / "comparisons"
    output_dir.mkdir()

    result = per_chip_similarity_to_control(
        [("Control", None), ("Ablation", None)],
        processed_data_dir=tmp_path,
        output_dir=output_dir,
        prefix="test",
        experiments=[control_exp, ablation_exp],
        control_label="Control",
        chip_tracker_path=tracker_path,
        chip_id_column="id",
        category_column="lulc",
    )

    long_df = result["comparison_df"]
    # 2 experiments × 2-chip intersection
    assert len(long_df) == 4
    # The intersection is {3, 4} — each experiment should contain exactly those ids
    for exp_label in ("Control", "Ablation"):
        sub = long_df[long_df["experiment"] == exp_label]
        assert sorted(sub["file_id"].tolist()) == [3, 4]
    gc.collect()


def test_per_class_similarity_distribution_plot_output(tmp_path):
    """per_class_similarity_distribution_plot creates a PNG from a long-form df."""
    rng = np.random.RandomState(0)
    rows = []
    for exp in ["Control", "Blue", "NIR"]:
        for cls in ["0", "1"]:
            for _ in range(30):
                rows.append(
                    {
                        "experiment": exp,
                        "file_id": rng.randint(0, 10_000),
                        "class": cls,
                        "cosine_similarity": (
                            1.0 if exp == "Control" else float(rng.uniform(0.5, 1.0))
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    output_path = tmp_path / "test_per_class_similarity_distribution.png"
    per_class_similarity_distribution_plot(
        {"comparison_df": df},
        output_path=output_path,
        control_label="Control",
    )
    assert output_path.exists()
    gc.collect()


def test_per_class_ecdf_plot_output(tmp_path):
    """per_class_ecdf_plot creates a PNG from a long-form df."""
    rng = np.random.RandomState(0)
    rows = []
    for exp in ["Control", "Blue", "NIR"]:
        for cls in ["0", "1"]:
            for _ in range(30):
                rows.append(
                    {
                        "experiment": exp,
                        "file_id": rng.randint(0, 10_000),
                        "class": cls,
                        "cosine_similarity": (
                            1.0 if exp == "Control" else float(rng.uniform(0.5, 1.0))
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    output_path = tmp_path / "test_per_class_ecdf.png"
    per_class_ecdf_plot(
        {"comparison_df": df},
        output_path=output_path,
        control_label="Control",
    )
    assert output_path.exists()
    gc.collect()


def test_per_class_ecdf_plot_empty_df(tmp_path):
    """Empty comparison_df returns early and writes nothing."""
    output_path = tmp_path / "test_per_class_ecdf_empty.png"
    per_class_ecdf_plot(
        {"comparison_df": pd.DataFrame()},
        output_path=output_path,
        control_label="Control",
    )
    assert not output_path.exists()
    gc.collect()


def test_per_class_ecdf_plot_control_only(tmp_path):
    """Only the control series present -> plot_df empty, returns early."""
    rows = [
        {"experiment": "Control", "file_id": i, "class": "0", "cosine_similarity": 1.0}
        for i in range(10)
    ]
    df = pd.DataFrame(rows)
    output_path = tmp_path / "test_per_class_ecdf_control_only.png"
    per_class_ecdf_plot(
        {"comparison_df": df},
        output_path=output_path,
        control_label="Control",
    )
    assert not output_path.exists()
    gc.collect()
