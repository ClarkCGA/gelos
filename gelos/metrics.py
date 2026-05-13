from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def pca_ablation(
    embeddings: np.ndarray,
    output_dir: Path,
    prefix: str,
    variance_thresholds: list[float] | None = None,
    **kwargs,
) -> dict:
    """Run PCA at multiple variance thresholds and record component counts.

    For each threshold, fits PCA and records the number of components needed
    to explain that fraction of variance.

    Args:
        embeddings: Input array of shape (N, D).
        output_dir: Directory to write the result CSV.
        prefix: File name prefix for the output CSV.
        variance_thresholds: List of cumulative variance fractions to test.

    Returns:
        Dict with ``thresholds`` list of per-threshold results.
    """
    if variance_thresholds is None:
        variance_thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]

    rows = []
    for threshold in variance_thresholds:
        pca = PCA(n_components=threshold, random_state=42)
        pca.fit(embeddings)
        N, D = embeddings.shape
        rows.append(
            {
                "threshold": threshold,
                "n_components": pca.n_components_,
                "proportion_of_total_components": pca.n_components_ / D,
                "total_variance_explained": float(pca.explained_variance_ratio_.sum()),
            }
        )
        logger.info(
            f"PCA ablation: threshold={threshold}, "
            f"n_components={pca.n_components_}, "
            f"proportion_of_total_components={pca.n_components_ / D}, "
            f"variance={pca.explained_variance_ratio_.sum():.4f}"
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{prefix}_pca_ablation.csv"
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"saved PCA ablation results to {csv_path}")

    return {"thresholds": rows}


def knn_purity(
    embeddings: np.ndarray,
    output_dir: Path,
    prefix: str,
    labels: np.ndarray | None = None,
    k_values: list[int] | None = None,
    n_subsample: int | None = None,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """Compute KNN class purity at multiple k values.

    For each chip, measures what fraction of its k nearest neighbors share
    the same class. Reports per-class and overall purity at each k.

    Args:
        embeddings: Input array of shape (N, D).
        output_dir: Directory to write the result CSV.
        prefix: File name prefix for the output CSV.
        labels: Class labels of shape (N,). Required.
        k_values: List of k values to evaluate.
        n_subsample: If set, use a stratified subsample of query points.
        random_state: Random state for reproducibility.

    Returns:
        Dict with ``rows`` list of per-k/per-class results and ``k_values``.
    """
    if labels is None:
        raise ValueError("knn_purity requires labels")

    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 50]

    k_values = sorted(k_values)
    max_k = max(k_values)
    n_samples = embeddings.shape[0]

    # Determine query indices (optionally subsample)
    rng = np.random.RandomState(random_state)
    if n_subsample is not None and n_subsample < n_samples:
        # Stratified subsample
        unique_classes = np.unique(labels)
        query_indices = []
        per_class_n = max(1, n_subsample // len(unique_classes))
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            n_take = min(per_class_n, len(cls_indices))
            query_indices.extend(rng.choice(cls_indices, size=n_take, replace=False))
        query_indices = np.array(sorted(query_indices))
    else:
        query_indices = np.arange(n_samples)

    # Fit on full embeddings, query with max_k+1 to exclude self
    nn = NearestNeighbors(n_neighbors=min(max_k + 1, n_samples), algorithm="brute", n_jobs=-1)
    nn.fit(embeddings)
    _, neighbor_indices = nn.kneighbors(embeddings[query_indices])

    # Exclude self (first column)
    neighbor_indices = neighbor_indices[:, 1:]
    query_labels = labels[query_indices]

    rows = []
    per_query_rows = []
    for k in k_values:
        if k > neighbor_indices.shape[1]:
            logger.warning(f"k={k} exceeds available neighbors, skipping")
            continue

        k_neighbors = neighbor_indices[:, :k]
        k_neighbor_labels = labels[k_neighbors]
        # Purity: fraction of neighbors sharing the query's class
        matches = k_neighbor_labels == query_labels[:, np.newaxis]
        per_query_purity = matches.mean(axis=1)

        for q_idx, (cls, p) in enumerate(zip(query_labels, per_query_purity)):
            per_query_rows.append(
                {
                    "k": k,
                    "class": str(cls),
                    "query_idx": int(q_idx),
                    "purity": float(p),
                }
            )

        # Overall purity
        overall = float(per_query_purity.mean())
        rows.append(
            {
                "k": k,
                "class": "overall",
                "purity": overall,
                "n_samples": len(query_indices),
            }
        )

        # Per-class purity
        unique_classes = np.unique(query_labels)
        for cls in unique_classes:
            mask = query_labels == cls
            cls_purity = float(per_query_purity[mask].mean())
            rows.append(
                {
                    "k": k,
                    "class": str(cls),
                    "purity": cls_purity,
                    "n_samples": int(mask.sum()),
                }
            )

        logger.info(f"KNN purity: k={k}, overall={overall:.4f}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{prefix}_knn_purity.csv"
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"saved KNN purity results to {csv_path}")

    per_query_df = pd.DataFrame(per_query_rows)
    per_query_csv_path = output_dir / f"{prefix}_knn_purity_per_query.csv"
    per_query_df.to_csv(per_query_csv_path, index=False)
    logger.info(f"saved KNN per-query purity to {per_query_csv_path}")

    return {"rows": rows, "k_values": k_values}


METRICS: dict[str, callable] = {
    "pca_ablation": pca_ablation,
    "knn_purity": knn_purity,
}
