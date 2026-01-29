from pathlib import Path
from typing import Optional

import geopandas as gpd
from loguru import logger
import pandas as pd
import typer
import yaml

from gelos.config import CONFIG_DIR, FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from gelos.embedding_extraction import extract_embeddings
from gelos.embedding_generation import perturb_args_to_string
from gelos.plotting import legend_patches, plot_from_tsne
from gelos.tsne_transform import save_tsne_as_csv, tsne_from_embeddings

app = typer.Typer()


def transform_embeddings(yaml_path: Path) -> None:
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    logger.info(f"processing {yaml_path}")

    data_version = yaml_config["data_version"]
    model_name = yaml_config["model"]["init_args"]["model"]
    model_title = yaml_config["model"]["title"]
    embedding_extraction_strategies = yaml_config["embedding_extraction_strategies"]
    perturb = yaml_config["data"]["init_args"].get("perturb_bands", None)
    perturb_string = perturb_args_to_string(perturb)
    output_dir = PROCESSED_DATA_DIR / data_version / model_name / perturb_string

    data_root = RAW_DATA_DIR / data_version
    chip_gdf = gpd.read_file(data_root / "gelos_chip_tracker.geojson")
    figures_dir = FIGURES_DIR / data_version
    figures_dir.mkdir(exist_ok=True, parents=True)

    embeddings_directories = [item for item in output_dir.iterdir() if item.is_dir()]

    for embeddings_directory in embeddings_directories:
        embedding_layer = embeddings_directory.stem

        for extraction_strategy, slice_args in embedding_extraction_strategies.items():
            model_title_lower = model_title.replace(" ", "").lower()
            extraction_strategy_lower = extraction_strategy.replace(" ", "").lower()
            embedding_layer_lower = embedding_layer.replace("_", "").lower()

            csv_path = (
                output_dir
                / f"{model_title_lower}_{extraction_strategy_lower}_{embedding_layer_lower}_tsne.csv"
            )

            if csv_path.exists() and 1 == 0:
                logger.info(f"{str(csv_path)} already exists, loading embeddings from file")
                loaded_embeddings = pd.read_csv(csv_path)
                embeddings_tsne = loaded_embeddings[
                    [
                        f"{model_title_lower}_{extraction_strategy_lower}_tsne_x",
                        f"{model_title_lower}_{extraction_strategy_lower}_tsne_y",
                    ]
                ].to_numpy()
                chip_indices = loaded_embeddings["id"].to_numpy()
            else:
                logger.info(f"extracting embeddings into {str(csv_path)}")

                embeddings, chip_indices = extract_embeddings(
                    embeddings_directory, slice_args=slice_args
                )

                embeddings_tsne = tsne_from_embeddings(embeddings)

                logger.info(f"tnse transform finished, extracting embeddings into {str(csv_path)}")

                save_tsne_as_csv(
                    embeddings_tsne,
                    chip_indices,
                    model_title,
                    extraction_strategy,
                    embedding_layer,
                    output_dir,
                )

            logger.info("plotting...")
            plot_from_tsne(
                embeddings_tsne,
                chip_gdf,
                model_title,
                extraction_strategy,
                embedding_layer,
                legend_patches,
                chip_indices,
                axis_lim=None,
                output_dir=figures_dir,
            )


@app.command()
def main(
    yaml_path: Optional[Path] = typer.Option(
        None, "--yaml-path", "-y", help="Path to a single yaml config to process."
    ),
):
    """
    Generate embeddings from a model and data specified in a yaml config.

    If --yaml-path is provided, only that yaml will be processed.
    Otherwise, all yamls in the default config directory will be processed.
    """
    if yaml_path:
        yaml_paths = [Path(yaml_path)]
    else:
        yaml_paths = list(
            CONFIG_DIR.glob("*noperturb*.yaml*")
        )  # only do tsne transforms for non-perturbed

    logger.info(f"yamls to process: {yaml_paths}")
    for yaml_path in yaml_paths:
        transform_embeddings(yaml_path)


if __name__ == "__main__":
    app()
