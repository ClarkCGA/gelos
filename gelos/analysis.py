from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import typer
import yaml
from loguru import logger
from matplotlib.patches import Patch

from gelos.extraction import extract_embeddings
from gelos.models import MODELS
from gelos.plotting import PLOTS
from gelos.transforms import TRANSFORMS

app = typer.Typer()


def load_chip_tracker(path: Path) -> pd.DataFrame:
    """Load a chip tracker file as a DataFrame, dispatching on file extension.

    Supports .geojson/.json (via geopandas) and .csv (via pandas).
    """
    suffix = path.suffix.lower()
    if suffix in (".geojson", ".json"):
        return gpd.read_file(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported chip tracker format '{suffix}'. Use .geojson, .json, or .csv")


def _build_style_from_config(style_cfg: dict) -> tuple[str, dict, list[Patch]]:
    """Extract category_column, color_dict, and legend_patches from the style config section."""
    category_column = style_cfg["category_column"]
    color_dict = style_cfg["colors"]
    legend_patches = [
        Patch(color=color, label=style_cfg["labels"][k]) for k, color in color_dict.items()
    ]
    return category_column, color_dict, legend_patches


def _save_transform_result(
    result: np.ndarray,
    chip_indices: list[int],
    cache_path: Path,
    transform_type: str,
    prefix: str,
) -> None:
    """Save transform output to CSV for caching."""
    cols = {
        f"{prefix}_{transform_type}_{i}": result[:, i] for i in range(result.shape[1])
    }
    df = pd.DataFrame({"id": chip_indices, **cols})
    df.to_csv(cache_path, index=False)
    logger.info(f"saved {transform_type} result to {cache_path}")


def _load_cached_transform(cache_path: Path) -> tuple[np.ndarray, list[int]]:
    """Load a cached transform result from CSV."""
    df = pd.read_csv(cache_path)
    chip_indices = df["id"].tolist()
    data_cols = [c for c in df.columns if c != "id"]
    return df[data_cols].to_numpy(), chip_indices


def run_pipeline(
    yaml_path: Path,
    raw_data_dir: Path,
    processed_data_dir: Path,
    figures_dir: Path,
) -> dict:
    """Run the config-driven embedding pipeline.

    Parses the YAML config, resolves paths, then for each embedding layer
    and extraction strategy: extracts embeddings and dispatches through
    the configured transforms, plots, and models.

    Args:
        yaml_path: Path to the YAML experiment config.
        raw_data_dir: Root directory for raw data.
        processed_data_dir: Root directory for processed outputs.
        figures_dir: Root directory for generated figures.

    Returns:
        Nested dict of results keyed by ``{layer}_{strategy}_{step_type}``.
    """
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    logger.info(f"processing {yaml_path}")

    config_stem = yaml_path.stem
    style_cfg = yaml_config["style"]
    category_column, _, _ = _build_style_from_config(style_cfg)

    data_version = yaml_config["data_version"]
    experiment_name = yaml_config["experiment_name"]
    embedding_extraction_strategies = yaml_config["embedding_extraction_strategies"]
    output_dir = processed_data_dir / config_stem

    data_root = raw_data_dir / data_version
    chip_tracker_file = yaml_config["chip_tracker"]
    chip_id_column = yaml_config["chip_id_column"]
    chip_gdf = load_chip_tracker(data_root / chip_tracker_file)
    chip_gdf = chip_gdf.set_index(chip_id_column)
    figures_dir.mkdir(exist_ok=True, parents=True)

    embeddings_directories = [item for item in output_dir.iterdir() if item.is_dir()]
    all_results = {}

    for embeddings_directory in embeddings_directories:
        embedding_layer = embeddings_directory.stem

        for strategy_key, strategy_cfg in embedding_extraction_strategies.items():
            slice_args = strategy_cfg["slice_args"]
            strategy_title = strategy_cfg.get("title", strategy_key)
            prefix = f"{config_stem}_{strategy_key}_{embedding_layer}"

            # --- Extract embeddings ---
            logger.info(
                f"extracting embeddings: layer={embedding_layer}, "
                f"strategy={strategy_key}"
            )
            embeddings, chip_indices = extract_embeddings(
                embeddings_directory, slice_args=slice_args
            )

            # --- Run transforms (default: t-SNE if none specified) ---
            transform_results: dict[str, np.ndarray] = {"raw": embeddings}
            default_transforms = [{"type": "tsne"}]
            for t_cfg in strategy_cfg.get("transforms", default_transforms):
                t_type = t_cfg["type"]
                t_params = t_cfg.get("params", {})
                layer_dir = output_dir / embedding_layer
                cache_path = layer_dir / f"{prefix}_{t_type}.csv"

                if t_type not in TRANSFORMS:
                    raise KeyError(
                        f"transform '{t_type}' not found in registry. "
                        f"Available: {list(TRANSFORMS.keys())}"
                    )

                if cache_path.exists():
                    logger.info(f"{cache_path} exists, loading cached {t_type} result")
                    cached_data, _ = _load_cached_transform(cache_path)
                    transform_results[t_type] = cached_data
                else:
                    t_fn = TRANSFORMS[t_type]
                    result = t_fn(embeddings, **t_params)
                    transform_results[t_type] = result
                    _save_transform_result(
                        result, chip_indices, cache_path, t_type, prefix
                    )

            # --- Run plots (default: t-SNE scatter if none specified) ---
            default_plots = [{"type": "tsne_scatter", "transform": "tsne"}]
            for p_cfg in strategy_cfg.get("plots", default_plots):
                p_type = p_cfg["type"]
                p_params = p_cfg.get("params", {})
                source = p_cfg.get("transform", "raw")

                if p_type not in PLOTS:
                    raise KeyError(
                        f"plot '{p_type}' not found in registry. "
                        f"Available: {list(PLOTS.keys())}"
                    )
                if source not in transform_results:
                    logger.warning(
                        f"plot '{p_type}' references transform '{source}' which "
                        f"was not run, skipping"
                    )
                    continue

                data = transform_results[source]
                output_path = figures_dir / f"{prefix}_{p_type}.png"
                logger.info(f"plotting {p_type} for {strategy_key}")
                p_fn = PLOTS[p_type]
                p_fn(
                    data,
                    chip_gdf,
                    chip_indices,
                    style_cfg,
                    output_path,
                    experiment_name,
                    strategy_title,
                    embedding_layer,
                    **p_params,
                )

            # --- Run models (none by default) ---
            for m_cfg in strategy_cfg.get("models", []):
                m_type = m_cfg["type"]
                m_params = m_cfg.get("params", {})
                source = m_cfg.get("transform", "raw")

                if m_type not in MODELS:
                    raise KeyError(
                        f"model '{m_type}' not found in registry. "
                        f"Available: {list(MODELS.keys())}"
                    )
                if source not in transform_results:
                    logger.warning(
                        f"model '{m_type}' references transform '{source}' which "
                        f"was not run, skipping"
                    )
                    continue

                data = transform_results[source]
                labels = chip_gdf[category_column].loc[chip_indices].to_numpy()
                run_name = f"{prefix}_{m_type}"
                logger.info(f"running model {m_type} for {strategy_key}")
                m_fn = MODELS[m_type]
                layer_dir = output_dir / embedding_layer
                result = m_fn(
                    data, labels, output_dir=layer_dir, run_name=run_name, **m_params
                )
                all_results[f"{prefix}_{m_type}"] = result

    return all_results


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
    Analyze embeddings using transforms, plots, and models specified in a yaml config.

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
        run_pipeline(
            yaml_path,
            raw_data_dir=raw_data_dir,
            processed_data_dir=processed_data_dir,
            figures_dir=figures_dir,
        )


if __name__ == "__main__":
    app()
