from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance as _wasserstein_1d
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from gelos.analysis import build_prefix


def _resolve_metric_csv(exp, processed_data_dir: Path, metric_name: str) -> Path:
    """Resolve the deterministic CSV path for a per-experiment metric."""
    exp_prefix = build_prefix(exp.config, exp.strategy, exp.layer)
    return (
        processed_data_dir
        / exp.data_version
        / exp.config
        / exp.layer
        / f"{exp_prefix}_{metric_name}.csv"
    )


def pca_ablation_comparison(
    experiment_embeddings: list[tuple[str, np.ndarray | None]],
    processed_data_dir: Path,
    output_dir: Path,
    prefix: str,
    experiments: list | None = None,
    **kwargs,
) -> dict:
    """Join per-experiment PCA ablation CSVs into a single comparison table.

    Each experiment's ``{prefix}_pca_ablation.csv`` must already exist
    (produced by the analysis-stage ``pca_ablation`` metric).

    Args:
        experiment_embeddings: List of (label, embeddings) tuples (embeddings unused here).
        processed_data_dir: Root processed directory to resolve per-experiment CSVs.
        output_dir: Directory to write the comparison CSV.
        prefix: File name prefix for the comparison output.
        experiments: List of :class:`ComparisonExperiment` objects for path resolution.

    Returns:
        Dict with ``comparison_df`` key holding the merged DataFrame.
    """
    if experiments is None:
        raise ValueError("pca_ablation_comparison requires 'experiments' to resolve CSV paths")

    frames = []
    for exp in experiments:
        csv_path = _resolve_metric_csv(exp, processed_data_dir, "pca_ablation")
        if not csv_path.exists():
            logger.warning(f"no pca_ablation CSV found for '{exp.label}' at {csv_path}, skipping")
            continue
        df = pd.read_csv(csv_path)
        df["experiment"] = exp.label
        frames.append(df)

    if not frames:
        logger.warning("no PCA ablation data found for any experiment")
        return {"comparison_df": pd.DataFrame()}

    merged = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / f"{prefix}_pca_ablation_comparison.csv"
    merged.to_csv(csv_path, index=False)
    logger.info(f"saved PCA ablation comparison to {csv_path}")
    return {"comparison_df": merged}


def cosine_distance(
    experiment_embeddings: list[tuple[str, np.ndarray]],
    output_dir: Path,
    prefix: str,
    **kwargs,
) -> dict:
    """Compute pairwise cosine similarity between experiment mean embeddings.

    Args:
        experiment_embeddings: List of (label, embeddings) tuples.
        output_dir: Directory to write the distance matrix CSV.
        prefix: File name prefix for the output CSV.

    Returns:
        Dict with ``labels``, ``matrix`` (np.ndarray), and ``df`` (DataFrame) keys.
    """
    labels = [label for label, _ in experiment_embeddings]
    means = np.array([emb.mean(axis=0) for _, emb in experiment_embeddings])

    sim_matrix = cosine_similarity(means)

    df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    csv_path = output_dir / f"{prefix}_cosine_distance.csv"
    df.to_csv(csv_path)
    logger.info(f"saved cosine distance matrix to {csv_path}")

    return {"labels": labels, "matrix": sim_matrix, "df": df}


def wasserstein_distance(
    experiment_embeddings: list[tuple[str, np.ndarray]],
    output_dir: Path,
    prefix: str,
    n_pca_components: int = 10,
    **kwargs,
) -> dict:
    """Compute pairwise Wasserstein distance on PCA-reduced embeddings.

    Reduces each experiment's embeddings via PCA, then computes the mean
    1D Wasserstein distance across the first ``n_pca_components`` dimensions.

    Args:
        experiment_embeddings: List of (label, embeddings) tuples.
        output_dir: Directory to write the distance matrix CSV.
        prefix: File name prefix for the output CSV.
        n_pca_components: Number of PCA components to reduce to before computing distances.

    Returns:
        Dict with ``labels``, ``matrix`` (np.ndarray), and ``df`` (DataFrame) keys.
    """
    labels = [label for label, _ in experiment_embeddings]
    n_exp = len(labels)

    # PCA-reduce each experiment
    reduced = []
    for label, emb in experiment_embeddings:
        n_comp = min(n_pca_components, emb.shape[1], emb.shape[0])
        pca = PCA(n_components=n_comp, random_state=42)
        reduced.append(pca.fit_transform(emb))

    # Compute pairwise mean Wasserstein distance across dimensions
    dist_matrix = np.zeros((n_exp, n_exp))
    for i in range(n_exp):
        for j in range(i + 1, n_exp):
            n_dims = min(reduced[i].shape[1], reduced[j].shape[1])
            dim_dists = [
                _wasserstein_1d(reduced[i][:, d], reduced[j][:, d]) for d in range(n_dims)
            ]
            mean_dist = float(np.mean(dim_dists))
            dist_matrix[i, j] = mean_dist
            dist_matrix[j, i] = mean_dist

    df = pd.DataFrame(dist_matrix, index=labels, columns=labels)
    csv_path = output_dir / f"{prefix}_wasserstein_distance.csv"
    df.to_csv(csv_path)
    logger.info(f"saved Wasserstein distance matrix to {csv_path}")

    return {"labels": labels, "matrix": dist_matrix, "df": df}


def knn_purity_comparison(
    experiment_embeddings: list[tuple[str, np.ndarray | None]],
    processed_data_dir: Path,
    output_dir: Path,
    prefix: str,
    experiments: list | None = None,
    **kwargs,
) -> dict:
    """Join per-experiment KNN purity CSVs into a single comparison table.

    Each experiment's ``{prefix}_knn_purity.csv`` must already exist
    (produced by the analysis-stage ``knn_purity`` metric).

    Args:
        experiment_embeddings: List of (label, embeddings) tuples (embeddings unused here).
        processed_data_dir: Root processed directory to resolve per-experiment CSVs.
        output_dir: Directory to write the comparison CSV.
        prefix: File name prefix for the comparison output.
        experiments: List of :class:`ComparisonExperiment` objects for path resolution.

    Returns:
        Dict with ``comparison_df`` key holding the merged DataFrame.
    """
    if experiments is None:
        raise ValueError("knn_purity_comparison requires 'experiments' to resolve CSV paths")

    frames = []
    for exp in experiments:
        csv_path = _resolve_metric_csv(exp, processed_data_dir, "knn_purity")
        if not csv_path.exists():
            logger.warning(f"no knn_purity CSV found for '{exp.label}' at {csv_path}, skipping")
            continue
        df = pd.read_csv(csv_path)
        df["experiment"] = exp.label
        frames.append(df)

    if not frames:
        logger.warning("no KNN purity data found for any experiment")
        return {"comparison_df": pd.DataFrame()}

    merged = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / f"{prefix}_knn_purity_comparison.csv"
    merged.to_csv(csv_path, index=False)
    logger.info(f"saved KNN purity comparison to {csv_path}")
    return {"comparison_df": merged}


pca_ablation_comparison.requires_embeddings = False
cosine_distance.requires_embeddings = True
wasserstein_distance.requires_embeddings = True
knn_purity_comparison.requires_embeddings = False

COMP_METRICS: dict[str, callable] = {
    "pca_ablation_comparison": pca_ablation_comparison,
    "cosine_distance": cosine_distance,
    "wasserstein_distance": wasserstein_distance,
    "knn_purity_comparison": knn_purity_comparison,
}
