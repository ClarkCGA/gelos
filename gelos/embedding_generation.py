from pathlib import Path
from typing import Optional

from lightning.pytorch import Trainer
from lightning.pytorch.cli import instantiate_class
from loguru import logger
from terratorch.tasks import EmbeddingGenerationTask
import torch
import typer
import yaml

from gelos.config import PROCESSED_DATA_DIR, CONFIG_DIR, RAW_DATA_DIR
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


def generate_embeddings(yaml_path: Path) -> None:
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    print(yaml.dump(yaml_config))

    model_name = yaml_config["model"]["init_args"]["model"]
    data_version = yaml_config["data_version"]
    perturb = yaml_config["data"]["init_args"].get("perturb_bands", None)
    perturb_string = perturb_args_to_string(perturb)

    output_dir = PROCESSED_DATA_DIR / data_version / model_name / perturb_string
    output_dir.mkdir(exist_ok=True, parents=True)
    data_root = RAW_DATA_DIR / data_version
    marker_file = output_dir / ".embeddings_complete"
    if (marker_file).exists():
        print("embeddings already complete, skipping...")
        return

    # add variables to yaml config so it can be passed to classes
    yaml_config["data"]["init_args"]["data_root"] = data_root
    yaml_config["model"]["init_args"]["output_dir"] = output_dir

    # instantiate transform classes if they exist
    if "transform" in yaml_config["data"]["init_args"].keys():
        yaml_config["data"]["init_args"]["transform"] = [
            instantiate_class(args=(), init=class_path)
            for class_path in yaml_config["data"]["init_args"]["transform"]
        ]

    gelos_datamodule = GELOSDataModule(**yaml_config["data"]["init_args"])
    task = LenientEmbeddingGenerationTask(**yaml_config["model"]["init_args"])

    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(accelerator=device, devices=1)

    trainer.predict(model=task, datamodule=gelos_datamodule)
    marker_file.touch()
    print("marking embeddings as complete")


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
        yaml_paths = list(CONFIG_DIR.glob("*.yaml"))

    logger.info(f"yamls to process: {yaml_paths}")
    for yaml_path in yaml_paths:
        generate_embeddings(yaml_path)


if __name__ == "__main__":
    app()
