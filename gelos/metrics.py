from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


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
        rows.append(
            {
                "threshold": threshold,
                "n_components": pca.n_components_,
                "total_variance_explained": float(pca.explained_variance_ratio_.sum()),
            }
        )
        logger.info(
            f"PCA ablation: threshold={threshold}, "
            f"n_components={pca.n_components_}, "
            f"variance={pca.explained_variance_ratio_.sum():.4f}"
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{prefix}_pca_ablation.csv"
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"saved PCA ablation results to {csv_path}")

    return {"thresholds": rows}


METRICS: dict[str, callable] = {
    "pca_ablation": pca_ablation,
}
