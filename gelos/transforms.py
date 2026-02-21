from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def tsne_from_embeddings(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    perplexity: int = 50,
    max_iter: int = 1000,
    verbose: int = 1,
    **kwargs,
) -> np.ndarray:
    """Run t-SNE on embeddings.

    Args:
        embeddings: Input array of shape (N, D).
        n_components: Number of output dimensions.
        random_state: Random state for reproducibility.
        perplexity: t-SNE perplexity parameter.
        max_iter: Maximum number of iterations.
        verbose: Verbosity level.

    Returns:
        Transformed array of shape (N, n_components).
    """
    logger.info(
        f"running t-SNE: perplexity={perplexity}, n_components={n_components}, "
        f"max_iter={max_iter}"
    )
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=max_iter,
        verbose=verbose,
    )
    return tsne.fit_transform(embeddings)


def pca_from_embeddings(
    embeddings: np.ndarray,
    n_components: int | float = 0.95,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Run PCA on embeddings.

    Args:
        embeddings: Input array of shape (N, D).
        n_components: Number of components (int) or variance threshold (float).
        random_state: Random state for reproducibility.

    Returns:
        Transformed array of shape (N, K) where K is the number of components.
    """
    logger.info(f"running PCA: n_components={n_components}")
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(embeddings)


def save_tsne_as_csv(
    embeddings_tsne: np.ndarray,
    chip_indices: list[int],
    model_title: str,
    extraction_strategy: str,
    embedding_layer: str,
    output_dir: str | Path = None,
) -> None:
    """Save t-SNE coordinates to a CSV file with standardized column names."""
    model_title_lower = model_title.replace(" ", "").lower()
    extraction_strategy_lower = extraction_strategy.replace(" ", "").lower()
    embedding_layer_lower = embedding_layer.replace("_", "").lower()
    csv_path = (
        output_dir
        / f"{model_title_lower}_{extraction_strategy_lower}_{embedding_layer_lower}_tsne.csv"
    )

    embeddings_df = pd.DataFrame(
        {
            "id": chip_indices,
            f"{model_title_lower}_{extraction_strategy_lower}_tsne_x": embeddings_tsne[:, 0],
            f"{model_title_lower}_{extraction_strategy_lower}_tsne_y": embeddings_tsne[:, 1],
        }
    ).set_index("id")
    embeddings_df.to_csv(csv_path)


TRANSFORMS: dict[str, callable] = {
    "tsne": tsne_from_embeddings,
    "pca": pca_from_embeddings,
}
