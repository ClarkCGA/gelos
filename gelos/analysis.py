from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
from loguru import logger
import numpy as np
import pandas as pd
import typer
import yaml

from gelos.extraction import extract_embeddings
from gelos.metrics import METRICS
from gelos.models import MODELS
from gelos.plotting import PLOTS, build_style_from_config, confusion_matrix
from gelos.transforms import TRANSFORMS

app = typer.Typer()


@dataclass
class AnalysisContext:
    """Resolved state for one analysis run, returned by :func:`setup_analysis_run`.

    Useful in notebooks where you want to inspect paths, chip metadata, or
    embedding directories before running transforms, plots, or models.
    """

    yaml_config: dict
    config_stem: str
    experiment_name: str
    style_cfg: dict
    category_column: str
    embedding_extraction_strategies: dict
    chip_gdf: gpd.GeoDataFrame
    input_dir: Path
    output_dir: Path
    figures_dir: Path
    null_handling: str = "drop"
    embeddings_directories: list[Path] = field(default_factory=list)


def build_prefix(config_stem: str, strategy_key: str, embedding_layer: str) -> str:
    """Build the deterministic file-name prefix for cached artifacts.

    This is the single source of truth for the ``{config}_{strategy}_{layer}``
    convention used by analysis and comparison stages.
    """
    return f"{config_stem}_{strategy_key}_{embedding_layer}"


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
        raise ValueError(
            f"Unsupported chip tracker format '{suffix}'. Use .geojson, .json, or .csv"
        )


def _save_transform_result(
    result: np.ndarray,
    chip_indices: list[int],
    cache_path: Path,
    transform_type: str,
    prefix: str,
) -> None:
    """Save transform output to CSV for caching."""
    cols = {f"dim_{i}": result[:, i] for i in range(result.shape[1])}
    df = pd.DataFrame({"id": chip_indices, **cols})
    df.to_csv(cache_path, index=False)
    logger.info(f"saved {transform_type} result to {cache_path}")


def _load_cached_transform(cache_path: Path) -> tuple[np.ndarray, list[int]]:
    """Load a cached transform result from CSV."""
    df = pd.read_csv(cache_path)
    chip_indices = df["id"].tolist()
    data_cols = [c for c in df.columns if c != "id"]
    return df[data_cols].to_numpy(), chip_indices


def drop_null_rows(
    embeddings: np.ndarray,
    chip_indices: list[int],
    labels: np.ndarray,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Drop rows with non-finite embeddings or null labels, keeping all three aligned.

    A row is dropped when its embedding vector contains any non-finite value
    (``NaN``/``inf`` — e.g. the AlphaEarth nodata sentinel ``-128`` dequantized to
    ``NaN``) or when its resolved label is null. Dropping (rather than imputing)
    avoids introducing bias and keeps sklearn transforms/metrics/models from
    raising ``ValueError: Input contains NaN``.

    Args:
        embeddings: ``(N, D)`` embedding matrix.
        chip_indices: Length-``N`` list of chip ids aligned with ``embeddings``.
        labels: Length-``N`` array of resolved labels aligned with ``embeddings``.

    Returns:
        Filtered ``(embeddings, chip_indices, labels)`` with the same row order,
        all three reduced to the kept rows. If no rows survive, returns empty
        arrays/list so the caller can skip gracefully instead of crashing sklearn.
    """
    finite_mask = np.isfinite(embeddings).all(axis=1)
    label_mask = ~pd.isna(labels)
    keep = finite_mask & label_mask

    n_total = len(chip_indices)
    n_kept = int(keep.sum())
    if n_kept < n_total:
        n_nonfinite = int((~finite_mask).sum())
        n_null_label = int((finite_mask & ~label_mask).sum())
        logger.info(
            f"dropping {n_total - n_kept}/{n_total} rows with nulls "
            f"({n_nonfinite} non-finite embedding, {n_null_label} null label)"
        )

    if n_kept == 0:
        logger.error("all rows dropped after null filtering; nothing to analyze for this strategy")

    filtered_indices = [c for c, k in zip(chip_indices, keep) if k]
    return embeddings[keep], filtered_indices, labels[keep]


def fill_null_rows(
    embeddings: np.ndarray,
    chip_indices: list[int],
    labels: np.ndarray,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Replace non-finite embedding values with 0; drop rows with null labels only.

    Unlike :func:`drop_null_rows`, this keeps rows whose embeddings contain
    ``NaN``/``inf`` by imputing those individual cells to ``0`` (e.g. the
    AlphaEarth nodata sentinel ``-128`` dequantized to ``NaN``). Rows whose
    resolved label is null are still dropped, since a missing label leaves
    nothing to train or evaluate against.

    Args:
        embeddings: ``(N, D)`` embedding matrix.
        chip_indices: Length-``N`` list of chip ids aligned with ``embeddings``.
        labels: Length-``N`` array of resolved labels aligned with ``embeddings``.

    Returns:
        Filtered ``(embeddings, chip_indices, labels)`` with non-finite embedding
        cells zeroed and null-label rows removed, all three kept aligned. If no
        rows survive the label filter, returns empty arrays/list.
    """
    n_zeroed = int((~np.isfinite(embeddings)).sum())
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    label_mask = ~pd.isna(labels)
    n_total = len(chip_indices)
    n_kept = int(label_mask.sum())
    n_null_label = n_total - n_kept

    if n_zeroed:
        logger.info(f"zeroed {n_zeroed} non-finite embedding values")
    if n_null_label:
        logger.info(f"dropping {n_null_label}/{n_total} rows with null label")
    if n_kept == 0:
        logger.error("all rows dropped after null filtering; nothing to analyze for this strategy")

    filtered_indices = [c for c, k in zip(chip_indices, label_mask) if k]
    return embeddings[label_mask], filtered_indices, labels[label_mask]


def setup_analysis_run(
    yaml_path: Path,
    raw_data_dir: Path,
    embedding_dir: Path,
    processed_data_dir: Path,
    figures_base_dir: Path,
) -> AnalysisContext:
    """Parse a YAML config and resolve all paths and objects needed for one analysis run.

    Useful in notebooks where you want to inspect chip metadata, embedding
    directories, or style config before running transforms, plots, or models.

    Args:
        yaml_path: Path to the YAML experiment config.
        raw_data_dir: Root directory for raw data.
        embedding_dir: Root directory for embeddings.
        processed_data_dir: Root directory for processed outputs.
        figures_base_dir: Root directory for generated figures.

    Returns:
        :class:`AnalysisContext` with all resolved paths and loaded objects.
    """
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    logger.info(f"processing {yaml_path}")

    config_stem = yaml_path.stem
    style_cfg = yaml_config["style"]
    category_column, _, _ = build_style_from_config(style_cfg)

    data_version = yaml_config["data_version"]
    experiment_name = yaml_config["experiment_name"]
    embedding_extraction_strategies = yaml_config["embedding_extraction_strategies"]

    # How to handle non-finite embedding values: "drop" the affected rows
    # (default, backward-compatible) or "zero" them in place. Configs that
    # predate this option omit it and fall back to "drop".
    null_handling = yaml_config.get("null_handling", "drop")
    if null_handling not in ("drop", "zero"):
        raise ValueError(f"null_handling must be 'drop' or 'zero', got '{null_handling}'")
    output_dir = processed_data_dir / data_version / config_stem
    input_dir = embedding_dir / data_version / config_stem

    data_root = raw_data_dir / data_version
    chip_tracker_file = yaml_config["chip_tracker"]
    chip_id_column = yaml_config["chip_id_column"]
    chip_gdf = load_chip_tracker(data_root / chip_tracker_file)
    chip_gdf = chip_gdf.set_index(chip_id_column)
    figures_dir = figures_base_dir / data_version
    figures_dir.mkdir(exist_ok=True, parents=True)

    if not input_dir.exists():
        logger.error(
            f"embedding directory {input_dir} does not exist. "
            f"Run generation first to produce embeddings."
        )
        embeddings_directories = []
    else:
        # Skip internal scratch dirs (e.g. cloud-embedding `_footprint_cache/`).
        embeddings_directories = [
            item
            for item in input_dir.iterdir()
            if item.is_dir() and not item.name.startswith(("_", "."))
        ]

    return AnalysisContext(
        yaml_config=yaml_config,
        config_stem=config_stem,
        experiment_name=experiment_name,
        style_cfg=style_cfg,
        category_column=category_column,
        embedding_extraction_strategies=embedding_extraction_strategies,
        chip_gdf=chip_gdf,
        input_dir=input_dir,
        output_dir=output_dir,
        figures_dir=figures_dir,
        null_handling=null_handling,
        embeddings_directories=embeddings_directories,
    )


def run_analysis(
    yaml_path: Path,
    raw_data_dir: Path,
    embedding_dir: Path,
    processed_data_dir: Path,
    figures_base_dir: Path,
    overwrite: bool = False,
) -> dict:
    """Run the config-driven embedding pipeline.

    Parses the YAML config, resolves paths, then for each embedding layer
    and extraction strategy: extracts embeddings and dispatches through
    the configured transforms, plots, and models.

    A ``.analysis_complete`` marker is written to the config's output
    directory after a full pass; when it exists the run is skipped entirely
    unless ``overwrite`` is set. ``overwrite`` bypasses only this marker —
    the per-step caches (extracted embeddings, transform CSVs, metrics,
    plots, model results) still skip individually, so a re-entered run only
    computes what is missing. Delete the cached outputs for a full recompute.

    Args:
        yaml_path: Path to the YAML experiment config.
        raw_data_dir: Root directory for raw data.
        embedding_dir: Root directory for embeddings.
        processed_data_dir: Root directory for processed outputs.
        figures_base_dir: Root directory for generated figures.
        overwrite: Re-enter a run marked complete (see above).

    Returns:
        Nested dict of results keyed by ``{layer}_{strategy}_{step_type}``.
    """
    ctx = setup_analysis_run(
        yaml_path, raw_data_dir, embedding_dir, processed_data_dir, figures_base_dir
    )

    marker_file = ctx.output_dir / ".analysis_complete"
    if marker_file.exists() and not overwrite:
        logger.info("analysis already complete, skipping...")
        return {}
    elif marker_file.exists() and overwrite:
        logger.info("re-entering completed analysis (cached steps still skip)...")

    if not ctx.embeddings_directories:
        return {}

    all_results = {}

    for embeddings_directory in ctx.embeddings_directories:
        embedding_layer = embeddings_directory.stem

        for strategy_key, strategy_cfg in ctx.embedding_extraction_strategies.items():
            slice_args = strategy_cfg["slice_args"]
            strategy_title = strategy_cfg.get("title", strategy_key)
            prefix = build_prefix(ctx.config_stem, strategy_key, embedding_layer)

            # --- Validate strategy has at least one analysis step ---
            has_transforms = "transforms" in strategy_cfg
            has_plots = "plots" in strategy_cfg
            has_models = "models" in strategy_cfg
            if not (has_transforms or has_plots or has_models):
                logger.warning(
                    f"strategy '{strategy_key}' has no 'transforms', 'plots', or 'models' defined"
                )
            # --- Extract embeddings ---
            layer_dir = ctx.output_dir / embedding_layer
            emb_cache = layer_dir / f"{prefix}_embeddings.npy"
            idx_cache = layer_dir / f"{prefix}_chip_indices.npy"

            if emb_cache.exists() and idx_cache.exists():
                logger.info(f"loading cached embeddings from {emb_cache}")
                embeddings = np.load(emb_cache)
                chip_indices = np.load(idx_cache).tolist()
            else:
                logger.info(
                    f"extracting embeddings: layer={embedding_layer}, strategy={strategy_key}"
                )
                embeddings, chip_indices = extract_embeddings(
                    embeddings_directory, slice_args=slice_args
                )
                layer_dir.mkdir(exist_ok=True, parents=True)
                np.save(emb_cache, embeddings)
                np.save(idx_cache, np.array(chip_indices))
                logger.info(f"cached embeddings to {emb_cache}")

            # --- Resolve labels and drop null rows (covers cached + fresh paths) ---
            # Resolved here (before transforms) so a single mask keeps embeddings,
            # chip_indices, and labels aligned for every downstream step. The cache
            # stores raw extracted embeddings that may contain NaN, so filtering must
            # happen post-load.
            labels = ctx.chip_gdf[ctx.category_column].loc[chip_indices].to_numpy()
            null_handling = getattr(ctx, "null_handling", "drop")
            if null_handling == "zero":
                embeddings, chip_indices, labels = fill_null_rows(embeddings, chip_indices, labels)
            else:
                embeddings, chip_indices, labels = drop_null_rows(embeddings, chip_indices, labels)
            if len(chip_indices) == 0:
                logger.warning(
                    f"no valid rows for strategy '{strategy_key}' after null filtering, skipping"
                )
                continue

            # --- Run transforms ---
            transform_results: dict[str, np.ndarray] = {"raw": embeddings}
            for t_cfg in strategy_cfg.get("transforms", []):
                t_type = t_cfg["type"]
                t_params = t_cfg.get("params", {})
                layer_dir = ctx.output_dir / embedding_layer
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
                    layer_dir.mkdir(exist_ok=True, parents=True)
                    _save_transform_result(result, chip_indices, cache_path, t_type, prefix)

            # --- Run metrics ---
            for met_cfg in strategy_cfg.get("metrics", []):
                met_type = met_cfg["type"]
                met_params = met_cfg.get("params", {})

                if met_type not in METRICS:
                    raise KeyError(
                        f"metric '{met_type}' not found in registry. "
                        f"Available: {list(METRICS.keys())}"
                    )

                cache_path = layer_dir / f"{prefix}_{met_type}.csv"
                if cache_path.exists():
                    logger.info(f"cached {met_type} result exists at {cache_path}, skipping")
                else:
                    met_fn = METRICS[met_type]
                    met_fn(
                        embeddings,
                        output_dir=layer_dir,
                        prefix=prefix,
                        labels=labels,
                        **met_params,
                    )

            # --- Run plots ---
            for p_cfg in strategy_cfg.get("plots", []):
                p_type = p_cfg["type"]
                p_params = p_cfg.get("params", {})
                t_type = p_cfg.get("transform", "raw")

                if p_type not in PLOTS:
                    raise KeyError(
                        f"plot '{p_type}' not found in registry. Available: {list(PLOTS.keys())}"
                    )
                if t_type not in transform_results:
                    logger.warning(
                        f"plot '{p_type}' references transform '{t_type}' which "
                        f"was not run, skipping"
                    )
                    continue

                data = transform_results[t_type]
                output_path = ctx.figures_dir / f"{prefix}_{t_type}_{p_type}.png"
                if output_path.exists():
                    logger.info(
                        f"plot {p_type} for {strategy_key} with transform: {t_type}"
                        "already exists - skipping"
                    )
                    continue
                logger.info(f"plotting {p_type} for {strategy_key} with transform: {t_type}")
                p_fn = PLOTS[p_type]
                p_fn(
                    data,
                    ctx.chip_gdf,
                    chip_indices,
                    ctx.style_cfg,
                    ctx.experiment_name,
                    strategy_title,
                    t_type,
                    embedding_layer,
                    output_path,
                    **p_params,
                )
                logger.info(f"plot saved to {output_path}")

            # --- Run models ---
            for m_cfg in strategy_cfg.get("models", []):
                m_type = m_cfg["type"]
                m_params = m_cfg.get("params", {})
                source = m_cfg.get("transform", "raw")

                if m_type not in MODELS:
                    raise KeyError(
                        f"model '{m_type}' not found in registry. Available: {list(MODELS.keys())}"
                    )
                if source not in transform_results:
                    logger.warning(
                        f"model '{m_type}' references transform '{source}' which "
                        f"was not run, skipping"
                    )
                    continue

                data = transform_results[source]
                run_name = f"{prefix}_{m_type}"
                layer_dir = ctx.output_dir / embedding_layer
                # Every MODELS entry writes {run_name}_{registry_key}_results.csv
                # (via _save_results_csv), so the cached result path is known
                # before running.
                results_csv = layer_dir / f"{run_name}_{m_type}_results.csv"
                cm_path = ctx.figures_dir / f"{prefix}_{m_type}_confusion_matrix.png"
                if results_csv.exists() and cm_path.exists():
                    logger.info(f"model {m_type} results exist at {results_csv} - skipping")
                    continue
                logger.info(f"running model {m_type} for {strategy_key}")
                m_fn = MODELS[m_type]
                result = m_fn(data, labels, output_dir=layer_dir, run_name=run_name, **m_params)

                if result.get("predictions") is not None:
                    confusion_matrix(
                        predictions=result["predictions"],
                        labels=labels,
                        chip_indices=chip_indices,
                        style_cfg=ctx.style_cfg,
                        experiment_name=ctx.experiment_name,
                        strategy_title=strategy_title,
                        model_type=m_type,
                        embedding_layer=embedding_layer,
                        output_path=cm_path,
                    )
                    logger.info(f"confusion matrix saved to {cm_path}")

                all_results[f"{prefix}_{m_type}"] = result

    ctx.output_dir.mkdir(exist_ok=True, parents=True)
    marker_file.touch()
    logger.info("marking analysis as complete")
    return all_results


@app.command()
def main(
    yaml_path: Optional[Path] = typer.Option(
        None, "--yaml-path", "-y", help="Path to a single yaml config to process."
    ),
    raw_data_dir: Path = typer.Option(
        "/app/data/raw", "--raw-data-dir", "-r", help="Root directory for raw data."
    ),
    embedding_dir: Path = typer.Option(
        "/app/data/interim", "--embedding-dir", "-e", help="Root directory for embedding outputs."
    ),
    processed_data_dir: Path = typer.Option(
        "/app/data/processed",
        "--processed-data-dir",
        "-p",
        help="Root directory for processed outputs.",
    ),
    figures_base_dir: Path = typer.Option(
        "/app/reports/figures",
        "--figures-base-dir",
        "-f",
        help="Root directory for generated figures.",
    ),
    config_dir: Optional[Path] = typer.Option(
        "/app/configs",
        "--config-dir",
        "-c",
        help="Directory containing YAML configs (used when --yaml-path is not set).",
    ),
    overwrite: Optional[bool] = typer.Option(
        False,
        "--overwrite",
        help="Re-enter runs marked complete (.analysis_complete). Per-step caches still "
        "skip, so only missing artifacts are recomputed; delete outputs for a full redo.",
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
        yaml_paths = list(Path(config_dir).glob("*.yaml"))

    logger.info(f"yamls to process: {yaml_paths}")
    for yaml_path in yaml_paths:
        run_analysis(
            yaml_path,
            raw_data_dir=raw_data_dir,
            embedding_dir=embedding_dir,
            processed_data_dir=processed_data_dir,
            figures_base_dir=figures_base_dir,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    app()
