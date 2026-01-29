from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def tsne_from_embeddings(embeddings: np.array) -> np.array:
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000, verbose=1)
    embeddings_tsne = tsne.fit_transform(embeddings)
    return embeddings_tsne


def save_tsne_as_csv(
    embeddings_tsne: np.array,
    chip_indices: list[int],
    model_title: str,
    extraction_strategy: str,
    embedding_layer: str,
    output_dir: str | Path = None,
) -> None:
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
