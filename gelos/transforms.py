from __future__ import annotations

from loguru import logger
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


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
        **kwargs: Additional keyword arguments passed to `sklearn.manifold.TSNE`.

    Returns:
        Transformed array of shape (N, n_components).
    """
    logger.info(
        f"running t-SNE: perplexity={perplexity}, n_components={n_components}, max_iter={max_iter}"
    )
    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        max_iter=max_iter,
        verbose=verbose,
    )
    return tsne.fit_transform(embeddings)


# TODO: continue updating to match UMAP params in scikitlearn
def umap_from_embeddings(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 5,
    min_dist: float = 0.0,
    metric: str = "euclidean",
    random_state: int = 42,
    verbose: int = 1,
    **kwargs,
) -> np.ndarray:
    """Run UMAP on embeddings.

    Args:
        embeddings: Input array of shape (N, D).
        n_components: Number of output dimensions.
        n_neighbors: Size of the local neighborhood (smaller = more local structure).
        min_dist: how tightly UMAP is allowed to pack points together.
        metric: distance metric used for transformation.
        random_state: Random state for reproducibility.
        verbose: Verbosity level.
        **kwargs: Additional keyword arguments passed to `umap.UMAP`.

    Returns:
        Transformed array of shape (N, n_components).
    """
    logger.info(f"running UMAP: n_neighbors={n_neighbors}, n_components={n_components}")
    umap_transformer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=verbose,
    )
    return umap_transformer.fit_transform(embeddings)


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
        **kwargs: Additional keyword arguments passed to `sklearn.decomposition.PCA`.

    Returns:
        Transformed array of shape (N, K) where K is the number of components.
    """
    logger.info(f"running PCA: n_components={n_components}")
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(embeddings)


TRANSFORMS: dict[str, callable] = {
    "tsne": tsne_from_embeddings,
    "pca": pca_from_embeddings,
    "umap": umap_from_embeddings,
}

TRANSFORM_TITLES: dict[str, str] = {
    "tsne": "t-SNE",
    "pca": "PCA",
    "umap": "UMAP",
}
