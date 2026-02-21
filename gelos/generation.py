from pathlib import Path
from typing import Optional

from lightning.pytorch import Trainer
from loguru import logger
from terratorch.tasks import EmbeddingGenerationTask
import torch
import typer
import yaml
from jsonargparse import ArgumentParser
from gelos.gelosdatamodule import GELOSDataModule

app = typer.Typer()


class LenientEmbeddingGenerationTask(EmbeddingGenerationTask):
    def check_file_ids(self, file_ids, x):
        return


def perturb_args_to_string(perturb):
    """
    Generates a string containing pertub args for file and folder naming

    :param perturb: dict(str, str) of band perturbation args
    """
    if perturb is None:
        return "noperturb"
    perturb_list = []
    for k, v in perturb.items():
        perturb_list.append(k.lower())
        for b, a in v.items():
            perturb_list.append(b.lower())
            perturb_list.append(str(a))
    perturb_list.append("perturb")
    perturb_string = "".join(perturb_list)
    return perturb_string


def generate_embeddings(
    yaml_path: Path,
    raw_data_dir: Path,
    processed_data_dir: Path,
) -> None:
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    print(yaml.dump(yaml_config))

    config_stem = yaml_path.stem
    data_version = yaml_config["data_version"]

    output_dir = processed_data_dir / config_stem
    output_dir.mkdir(exist_ok=True, parents=True)
    data_root = raw_data_dir / data_version
    marker_file = output_dir / ".embeddings_complete"
    if (marker_file).exists():
        print("embeddings already complete, skipping...")
        return

    parser = ArgumentParser()
    parser.add_class_arguments(GELOSDataModule, "data")
    parser.add_class_arguments(LenientEmbeddingGenerationTask, "model")

    data_init_args = yaml_config['data']['init_args']
    data_init_args['data_root'] = str(data_root)
    model_init_args = yaml_config['model']['init_args']
    model_init_args['output_dir'] = str(output_dir)

    cfg = parser.parse_object({"data": data_init_args, "model": model_init_args})
    init = parser.instantiate_classes(cfg)
    gelos_datamodule = init.data
    task = init.model
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(accelerator=device, devices=1)

    trainer.predict(model=task, datamodule=gelos_datamodule)
    marker_file.touch()
    print("marking embeddings as complete")


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
        yaml_paths = list(Path(config_dir).glob("*.yaml"))

    logger.info(f"yamls to process: {yaml_paths}")
    for yaml_path in yaml_paths:
        generate_embeddings(
            yaml_path,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
        )


if __name__ == "__main__":
    app()
