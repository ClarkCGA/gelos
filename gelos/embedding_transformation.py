from pathlib import Path
from typing import Optional

import geopandas as gpd
from loguru import logger
from matplotlib.patches import Patch
import pandas as pd
import typer
import yaml

from gelos.embedding_extraction import extract_embeddings
from gelos.embedding_generation import perturb_args_to_string
from gelos.plotting import plot_from_tsne
from gelos.tsne_transform import save_tsne_as_csv, tsne_from_embeddings

app = typer.Typer()


def transform_embeddings(
    yaml_path: Path,
    raw_data_dir: Path,
    processed_data_dir: Path,
    figures_dir: Path,
    legend_patches: dict[str, Patch],
) -> None:
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    logger.info(f"processing {yaml_path}")

    data_version = yaml_config["data_version"]
    model_name = yaml_config["model"]["init_args"]["model"]
    model_title = yaml_config["model"]["title"]
    embedding_extraction_strategies = yaml_config["embedding_extraction_strategies"]
    perturb = yaml_config["data"]["init_args"].get("perturb_bands", None)
    perturb_string = perturb_args_to_string(perturb)
    output_dir = processed_data_dir / data_version / model_name / perturb_string

    data_root = raw_data_dir / data_version
    chip_gdf = gpd.read_file(data_root / "gelos_chip_tracker.geojson")
    figures_dir = figures_dir / data_version
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

            if csv_path.exists():
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
                category_column,
                color_dict,
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
    raw_data_dir: Path = typer.Option(
        ..., "--raw-data-dir", "-r", help="Root directory for raw data."
    ),
    processed_data_dir: Path = typer.Option(
        ..., "--processed-data-dir", "-p", help="Root directory for processed outputs."
    ),
    figures_dir: Path = typer.Option(
        ..., "--figures-dir", "-f", help="Root directory for generated figures."
    ),
    config_dir: Optional[Path] = typer.Option(
        None,
        "--config-dir",
        "-c",
        help="Directory containing YAML configs (used when --yaml-path is not set).",
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
        if not config_dir:
            raise typer.BadParameter("--config-dir is required when --yaml-path is not provided.")
        yaml_paths = list(
            Path(config_dir).glob("*.yaml")
        )  

    logger.info(f"yamls to process: {yaml_paths}")
    for yaml_path in yaml_paths:
        transform_embeddings(
            yaml_path,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            figures_dir=figures_dir,
        )


if __name__ == "__main__":
    app()
