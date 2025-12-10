
from gelos.embedding_extraction import extract_embeddings
from gelos.tsne_transform import tsne_from_embeddings, save_tsne_as_csv
from gelos.plotting import plot_from_tsne, legend_patches
from gelos.embedding_generation import perturb_args_to_string
import geopandas as gpd
import yaml
from gelos.config import PROJ_ROOT, PROCESSED_DATA_DIR, DATA_VERSION, RAW_DATA_DIR
from gelos.config import REPORTS_DIR, FIGURES_DIR
from pathlib import Path
import typer
from loguru import logger
app = typer.Typer()
from typing import Optional

def transform_embeddings(yaml_path: Path) -> None:

    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    print(yaml.dump(yaml_config))

    data_root = RAW_DATA_DIR / DATA_VERSION
    chip_gdf = gpd.read_file(data_root / 'gelos_chip_tracker.geojson')
    figures_dir = FIGURES_DIR / DATA_VERSION
    figures_dir.mkdir(exist_ok=True, parents=True)

    model_name = yaml_config['model']['init_args']['model']
    model_title = yaml_config['model']['title']
    embedding_extraction_strategies = yaml_config['embedding_extraction_strategies']
    perturb = yaml_config['data']['init_args'].get('perturb_bands', None)
    perturb_string = perturb_args_to_string(perturb)
    output_dir = PROCESSED_DATA_DIR / DATA_VERSION / model_name / perturb_string

    embeddings_directories = [item for item in output_dir.iterdir() if item.is_dir()]

    for embeddings_directory in embeddings_directories:

        embedding_layer = embeddings_directory.stem

        for extraction_strategy, slice_args in embedding_extraction_strategies.items():


            model_title_lower = model_title.replace(" ", "").lower()
            extraction_strategy_lower = extraction_strategy.replace(" ", "").lower()
            embedding_layer_lower = embedding_layer.replace("_", "").lower()

            csv_path = output_dir / f"{model_title_lower}_{perturb_string}_{extraction_strategy_lower}_{embedding_layer_lower}_tnse.csv"
            if csv_path.exists():
                print(f"{str(csv_path)} already exists, skipping embedding extraction")
                continue
            print(f"extracting embeddings into {str(csv_path)}")

            embeddings, chip_indices = extract_embeddings(
                embeddings_directory,
                slice_args=slice_args
                )

            embeddings_tsne = tsne_from_embeddings(embeddings)

            print(f"tnse transform finished, extracting embeddings into {str(csv_path)} and plotting")

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
@app.command()
def main(
    yaml_path: Optional[Path] = typer.Option(
        None, "--yaml-path", "-y", help="Path to a single yaml config to process."
    )
):
    """
    Generate embeddings from a model and data specified in a yaml config.

    If --yaml-path is provided, only that yaml will be processed.
    Otherwise, all yamls in the default config directory will be processed.
    """
    if yaml_path:
        yaml_paths = [Path(yaml_path)]
    else:
        yaml_config_directory = PROJ_ROOT / "gelos" / "configs"
        yaml_paths = list(yaml_config_directory.glob("*.yaml"))

    logger.info(f"yamls to process: {yaml_paths}")
    for yaml_path in yaml_paths:
        transform_embeddings(yaml_path)


if __name__ == "__main__":
    app()