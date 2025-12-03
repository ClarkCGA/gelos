import geopandas as gpd
import os
import random
import pandas as pd
from pathlib import Path
import pyarrow as pa
from matplotlib.patches import Patch
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import numpy as np
import yaml
from gelos import config
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gelos.config import PROJ_ROOT, PROCESSED_DATA_DIR, DATA_VERSION, RAW_DATA_DIR
from gelos.config import REPORTS_DIR, FIGURES_DIR

legend_patches = [
    Patch(color=color, label=name)
    for name, color in [
        ("Water", "#419bdf"),
        ("Trees", "#397d49"),
        ("Crops", "#e49635"),
        ("Built Area", "#c4281b"),
        ("Bare Ground", "#a59b8f"),
        ("Rangeland", "#e3e2c3"),
    ]
]


def sample_files(directory: str | Path, sample_size: int, *, seed: int | None = None) -> list[Path]:
    rng = random.Random(seed)
    directory = Path(directory)

    files = [Path(entry.path) for entry in os.scandir(directory) if entry.is_file()]
    if sample_size >= len(files):
        files.sort()
        return files
    else:
        files = rng.sample(files, sample_size)
        files.sort()
        return files

def select_embedding_indices(
        embeddings_column: pa.lib.ListArray, 
        slice_args: list[dict[str, int]]
        ) -> pa.lib.ListArray:
    array = embeddings_column
    for arg in slice_args:
        array = pa.compute.list_slice(
            array,
            start=arg['start'],
            stop=arg["stop"],
            step=arg["step"]
            )
        array = pa.compute.list_flatten(array)
    return array

def extract_embeddings_from_directory(
        directory: Path | str, 
        n_sample: int = 100000, 
        chip_indices: list[int] = None, 
        slice_args: list[dict[str, int]] = [{"start": 0, "stop": None, "step": 1}]
        ) -> tuple[np.array]:
    # extract embeddings in numpy format from geoparquet
    if chip_indices:
        files = [directory / f"{str(id).zfill(6)}_embedding.parquet" for id in chip_indices]
    else:
        files = sample_files(directory, n_sample, seed=42)
    dataset = ds.dataset(files, format="parquet")
    scanner = dataset.scanner(columns=["embedding", "file_id"])
    # OPTIMIZE: This would be faster batched, but the order of embeddings must be preserved
    # embedding generation should save chip id as a colummn of the parquet output
    emb_chunks, id_chunks = [], []
    batches = scanner.to_batches()
    for batch in tqdm(batches, desc="Processing embeddings"):
        sliced = select_embedding_indices(batch.column("embedding"), slice_args)
        flattened = pa.compute.list_flatten(sliced, recursive=True)
        emb_np = flattened.to_numpy(zero_copy_only=False).reshape(len(batch), -1)
        emb_chunks.append(emb_np)
        id_chunks.append(batch.column("file_id").to_numpy())
    embeddings = np.vstack(emb_chunks)
    chip_indices = np.concatenate(id_chunks).astype(int).tolist()
    return embeddings, chip_indices

def tsne_from_embeddings(embeddings: np.array) -> np.array:
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000, verbose=1)
    embeddings_tsne = tsne.fit_transform(embeddings)
    return embeddings_tsne

def plot_from_tsne(
        embeddings_tsne: np.array,
        chip_gdf: gpd.GeoDataFrame,
        model_title: str,
        extraction_strategy: str,
        embedding_layer: str,
        legend_patches: list[Patch],
        chip_indices: list[int],
        axis_lim: int = 90,
        output_dir: str | Path = None
        ) -> None:
    """
    plot a tSNE transform of embeddings colored according to land cover
    """
    colors = chip_gdf.loc[chip_indices]['color']
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_tsne[:, 1], -embeddings_tsne[:, 0], c=colors, s=2)
    plt.suptitle(f"t-SNE Visualization of GELOS Embeddings for {model_title}", fontsize=14)
    plt.title(extraction_strategy)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    if axis_lim:
        plt.xlim([-axis_lim, axis_lim])
        plt.ylim([-axis_lim, axis_lim])
    plt.legend(handles=legend_patches, loc="upper left", fontsize=10, framealpha=0.9)

    if output_dir:
        model_title = model_title.replace(" ", "").lower()
        extraction_strategy = extraction_strategy.replace(" ", "").lower()
        embedding_layer = embedding_layer.replace("_", "").lower()
        plt.savefig(output_dir / f"{model_title}_{extraction_strategy}_{embedding_layer}_tsneplot.png", dpi=600, bbox_inches="tight")
    else:
        plt.show()

def save_tsne_as_csv(
        embeddings_tsne: np.array,
        chip_indices: list[int],
        model_title: str,
        extraction_strategy: str,
        embedding_layer: str,
        output_dir: str | Path = None
) -> None:
    model_title = model_title.replace(" ", "").lower()
    extraction_strategy = extraction_strategy.replace(" ", "").lower()
    embedding_layer = embedding_layer.replace("_", "").lower()
    embeddings_df = pd.DataFrame({
        "id" : chip_indices,
        f"{model_title}_{extraction_strategy}_tsne_x" : embeddings_tsne[:, 0],
        f"{model_title}_{extraction_strategy}_tsne_y" : embeddings_tsne[:, 1],
    }).set_index("id")
    embeddings_df.to_csv(output_dir / f"{model_title}_{extraction_strategy}_{embedding_layer}_tsnetable.csv")

yaml_config_directory = PROJ_ROOT / 'gelos' / 'configs'

output_dir = PROCESSED_DATA_DIR / DATA_VERSION / model_name
data_root = RAW_DATA_DIR / DATA_VERSION
chip_gdf = gpd.read_file(data_root / 'gelos_chip_tracker.geojson')
reports_dir = REPORTS_DIR / DATA_VERSION
reports_dir.mkdir(exist_ok=True, parents=True)
figures_dir = FIGURES_DIR / DATA_VERSION
figures_dir.mkdir(exist_ok=True, parents=True)

for yaml_filepath in yaml_config_directory.glob("*.yaml"):
    with open(yaml_filepath, "r") as f:
        yaml_config = yaml.safe_load(f)
    print(yaml.dump(yaml_config))
    model_name = yaml_config['model']['init_args']['model']
    model_title = yaml_config['model']['title']
    embedding_extraction_strategies = yaml_config['embedding_extraction_strategies']

    # add variables to yaml config so it can be passed to classes
    yaml_config['data']['init_args']['data_root'] = data_root
    yaml_config['model']['init_args']['output_dir'] = output_dir
    embeddings_directories = [item for item in output_dir.iterdir() if item.is_dir()]

    for embeddings_directory in embeddings_directories:

        embedding_layer = embeddings_directory.stem

        for extraction_strategy, slice_args in embedding_extraction_strategies.items():

            embeddings, chip_indices = extract_embeddings_from_directory(
                embeddings_directory,
                n_sample = 100000,
                slice_args=slice_args
                )

            embeddings_tsne = tsne_from_embeddings(embeddings)

            save_tsne_as_csv(
                embeddings_tsne,
                chip_indices,
                model_title,
                extraction_strategy,
                embedding_layer,
                output_dir
                )

            plot_from_tsne(
                embeddings_tsne,
                chip_gdf,
                model_title,
                extraction_strategy,
                embedding_layer,
                legend_patches,
                chip_indices,
                axis_lim = None,
                output_dir = figures_dir
                )