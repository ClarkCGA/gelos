from pathlib import Path
from typing import Any, Optional

from lightning.pytorch import Trainer
from lightning.pytorch.cli import instantiate_class
from loguru import logger
from terratorch.tasks import EmbeddingGenerationTask
import torch
import typer
import yaml

from gelos.gelosdatamodule import GELOSDataModule

try:
    import gelos.backbones.olmoearth_backbone  # noqa: F401 — registers OlmoEarth factories
except ImportError:
    pass

app = typer.Typer()


class LenientEmbeddingGenerationTask(EmbeddingGenerationTask):
    def check_file_ids(self, file_ids, x):
        return

    @torch.no_grad()
    def predict_step(self, batch: dict) -> Any:
        """Thread per-batch acquisition dates into a date-aware backbone.

        Stock terratorch ``predict_step``/``get_embeddings`` only forward
        ``batch["image"]`` to the backbone, so the outer ``"timestamps"`` key
        cannot reach it through the normal call path. We pop it here and stash it
        on the backbone (or its ``.encoder`` when wrapped by ``TemporalWrapper``)
        via ``set_batch_timestamps``, then clear it in ``finally`` so the stash
        never outlives one batch. The override is a no-op for backbones lacking
        the setter (e.g. Prithvi, TerraMind), keeping the task generic.

        ``@torch.no_grad()`` mirrors the parent ``predict_step``.
        """
        timestamps = batch.pop("timestamps", None)
        backbone = getattr(self, "model", None)
        # OlmoEarth backbone is either self.model or self.model.encoder if wrapped.
        target = getattr(backbone, "encoder", None)
        if hasattr(backbone, "set_batch_timestamps"):
            setter_obj = backbone
        elif hasattr(target, "set_batch_timestamps"):
            setter_obj = target
        else:
            setter_obj = None
        if timestamps is not None and setter_obj is not None:
            setter_obj.set_batch_timestamps(timestamps)
        try:
            return super().predict_step(batch)
        finally:
            if setter_obj is not None:
                setter_obj.clear_batch_timestamps()


def instantiate_recursive(node: Any) -> Any:
    """Walk a parsed YAML node and instantiate any ``{class_path, init_args}`` dicts bottom-up.

    Bypasses jsonargparse introspection, which breaks on classes whose
    ``__annotations__`` are rewritten by metaclasses (e.g. albumentations'
    ``ValidatedTransformMeta``) and on group-name collisions with init params.
    Any dict containing a ``class_path`` key is treated as an instantiation
    spec; everything else is passed through untouched.
    """
    if isinstance(node, dict):
        if "class_path" in node:
            init_args = instantiate_recursive(node.get("init_args") or {})
            return instantiate_class(
                (), {"class_path": node["class_path"], "init_args": init_args}
            )
        return {k: instantiate_recursive(v) for k, v in node.items()}
    if isinstance(node, list):
        return [instantiate_recursive(item) for item in node]
    return node


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


def setup_embedding_run(
    yaml_path: Path,
    raw_data_dir: Path,
    embedding_dir: Path,
) -> tuple[GELOSDataModule, Any, Trainer, Path]:
    """Parse a YAML config and instantiate all objects needed for one embedding run.

    Useful in notebooks where you want to inspect or modify the datamodule, task,
    or trainer before calling ``trainer.predict(model=task, datamodule=datamodule)``.

    Args:
        yaml_path: Path to the YAML experiment config.
        raw_data_dir: Root directory for raw data.
        embedding_dir: Root directory for embedding outputs.

    Returns:
        Tuple of (datamodule, task, trainer, output_dir).
    """
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    logger.info(f"config:\n{yaml.dump(yaml_config)}")

    config_stem = yaml_path.stem
    data_version = yaml_config["data_version"]

    output_dir = embedding_dir / data_version / config_stem
    output_dir.mkdir(exist_ok=True, parents=True)
    data_root = raw_data_dir / data_version

    yaml_config["data"]["init_args"]["data_root"] = str(data_root)
    yaml_config["model"].setdefault("init_args", {})
    yaml_config["model"]["init_args"]["output_dir"] = str(output_dir)

    if "raw_pixel_extraction" in yaml_config:
        yaml_config["model"]["init_args"]["extraction_strategies"] = yaml_config[
            "raw_pixel_extraction"
        ]

    datamodule: GELOSDataModule = instantiate_recursive(yaml_config["data"])

    model_class_path = yaml_config["model"].get("class_path", "")
    if "EmbeddingGenerationTask" in model_class_path:
        # Override the task class with the lenient subclass but reuse the
        # recursively-instantiated init_args from the YAML.
        model_init_args = instantiate_recursive(yaml_config["model"].get("init_args") or {})
        task = LenientEmbeddingGenerationTask(**model_init_args)
    else:
        task = instantiate_recursive(yaml_config["model"])

    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(accelerator=device, devices=1)

    return datamodule, task, trainer, output_dir


def generate_embeddings(
    yaml_path: Path,
    raw_data_dir: Path,
    embedding_dir: Path,
    overwrite: bool = False,
) -> None:
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    has_cloud = "cloud_embedding" in yaml_config
    has_model = "model" in yaml_config
    if has_cloud and has_model:
        raise ValueError(
            f"{yaml_path}: YAML has both 'cloud_embedding' and 'model' blocks; "
            "choose one — they're mutually exclusive generation paths."
        )
    if has_cloud:
        from gelos.cloud_embeddings.runner import generate_cloud_embeddings

        return generate_cloud_embeddings(
            yaml_path, raw_data_dir, embedding_dir, overwrite=overwrite
        )

    datamodule, task, trainer, output_dir = setup_embedding_run(
        yaml_path, raw_data_dir, embedding_dir
    )

    marker_file = output_dir / ".embeddings_complete"
    if marker_file.exists() and not overwrite:
        logger.info("embeddings already complete, skipping...")
        return
    elif marker_file.exists() and overwrite:
        logger.info("starting overwrite of existing embeddings...")

    trainer.predict(model=task, datamodule=datamodule)
    marker_file.touch()
    logger.info("marking embeddings as complete")


@app.command()
def main(
    yaml_path: Optional[Path] = typer.Option(
        None, "--yaml-path", "-y", help="Path to a single yaml config to process."
    ),
    raw_data_dir: Path = typer.Option(
        "/app/data/raw", "--raw-data-dir", "-r", help="Root directory for raw data."
    ),
    embedding_dir: Path = typer.Option(
        "/app/data/interim", "--embedding-dir", "-p", help="Root directory for embedding outputs."
    ),
    config_dir: Optional[Path] = typer.Option(
        "/app/configs",
        "--config-dir",
        "-c",
        help="Directory containing YAML configs (used when --yaml-path is not set).",
    ),
    overwrite: Optional[bool] = typer.Option(
        False, "--overwrite", help="Overwrite existing embeddings"
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
        yaml_paths = list(Path(config_dir).glob("*.yaml"))

    logger.info(f"yamls to process: {yaml_paths}")
    for yaml_path in yaml_paths:
        generate_embeddings(
            yaml_path,
            raw_data_dir=raw_data_dir,
            embedding_dir=embedding_dir,
        )


if __name__ == "__main__":
    app()
