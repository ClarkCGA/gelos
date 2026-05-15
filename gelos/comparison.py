from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
import numpy as np
import typer
import yaml

from gelos.analysis import build_prefix
from gelos.comp_metrics import COMP_METRICS
from gelos.comp_plots import COMP_PLOTS

app = typer.Typer()


@dataclass
class ComparisonExperiment:
    """One experiment entry in a comparison config."""

    data_version: str
    config: str
    strategy: str
    layer: str
    label: str
    color: str | None = None


@dataclass
class ComparisonContext:
    """Resolved state for one comparison run."""

    comparison_name: str
    config_stem: str
    experiments: list[ComparisonExperiment]
    comp_metrics: list[dict]
    comp_plots: list[dict]
    output_dir: Path
    figures_dir: Path
    class_labels: dict[str, str]
    experiment_colors: dict[str, str]


def _resolve_embedding_path(exp: ComparisonExperiment, processed_data_dir: Path) -> Path:
    """Resolve the cached embedding .npy path for an experiment entry."""
    prefix = build_prefix(exp.config, exp.strategy, exp.layer)
    return (
        processed_data_dir / exp.data_version / exp.config / exp.layer / f"{prefix}_embeddings.npy"
    )


def setup_comparison(
    yaml_path: Path,
    processed_data_dir: Path,
    figures_base_dir: Path,
) -> ComparisonContext:
    """Parse a comparison YAML config and resolve output directories.

    Args:
        yaml_path: Path to the comparison YAML config.
        processed_data_dir: Root directory for processed outputs.
        figures_base_dir: Root directory for generated figures.

    Returns:
        :class:`ComparisonContext` with resolved paths and parsed experiments.
    """
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    logger.info(f"processing comparison config: {yaml_path}")

    config_stem = yaml_path.stem
    comparison_name = yaml_config["comparison_name"]

    experiments = []
    for exp_cfg in yaml_config["experiments"]:
        experiments.append(
            ComparisonExperiment(
                data_version=exp_cfg["data_version"],
                config=exp_cfg["config"],
                strategy=exp_cfg["strategy"],
                layer=exp_cfg["layer"],
                label=exp_cfg["label"],
                color=exp_cfg.get("color"),
            )
        )
    experiment_colors = {exp.label: exp.color for exp in experiments if exp.color}

    comp_metrics = yaml_config.get("comp_metrics", [])
    comp_plots = yaml_config.get("comp_plots", [])

    # Normalize class label keys to strings so they can be looked up against the
    # CSV ``class`` column (which is string-typed on read).
    raw_labels = yaml_config.get("class_labels", {}) or {}
    class_labels = {str(k): str(v) for k, v in raw_labels.items()}

    output_dir = processed_data_dir / "comparisons" / config_stem
    output_dir.mkdir(exist_ok=True, parents=True)
    figures_dir = figures_base_dir / "comparisons" / config_stem
    figures_dir.mkdir(exist_ok=True, parents=True)

    return ComparisonContext(
        comparison_name=comparison_name,
        config_stem=config_stem,
        experiments=experiments,
        comp_metrics=comp_metrics,
        comp_plots=comp_plots,
        output_dir=output_dir,
        figures_dir=figures_dir,
        class_labels=class_labels,
        experiment_colors=experiment_colors,
    )


def run_comparison(
    yaml_path: Path,
    processed_data_dir: Path,
    figures_base_dir: Path,
) -> dict:
    """Run the comparison pipeline.

    Loads cached embeddings for each experiment, dispatches through
    ``COMP_METRICS`` and ``COMP_PLOTS`` registries.

    Args:
        yaml_path: Path to the comparison YAML config.
        processed_data_dir: Root directory for processed outputs.
        figures_base_dir: Root directory for generated figures.

    Returns:
        Dict of metric results keyed by metric type.
    """
    ctx = setup_comparison(yaml_path, processed_data_dir, figures_base_dir)

    # Determine whether any metric requires loading embeddings
    any_needs_embeddings = any(
        getattr(COMP_METRICS.get(m["type"]), "requires_embeddings", True) for m in ctx.comp_metrics
    )

    # Build experiment lists: always build labels-only, load arrays only when needed
    labels_only: list[tuple[str, None]] = [(exp.label, None) for exp in ctx.experiments]
    loaded: list[tuple[str, np.ndarray]] | None = None

    if any_needs_embeddings:
        loaded = []
        for exp in ctx.experiments:
            emb_path = _resolve_embedding_path(exp, processed_data_dir)
            if not emb_path.exists():
                logger.warning(f"embeddings not found for '{exp.label}' at {emb_path}, skipping")
                continue
            emb = np.load(emb_path)
            loaded.append((exp.label, emb))
            logger.info(f"loaded embeddings for '{exp.label}': shape={emb.shape}")

        if len(loaded) < 2:
            logger.warning(
                f"need at least 2 experiments with embeddings for comparison, got {len(loaded)}"
            )
            return {}

        # Validate embedding dimensionality
        dim = loaded[0][1].shape[1]
        for label, emb in loaded[1:]:
            if emb.shape[1] != dim:
                raise ValueError(
                    f"embedding dimension mismatch: '{loaded[0][0]}' has {dim} dims, "
                    f"'{label}' has {emb.shape[1]} dims"
                )
    else:
        logger.info("no metrics require embeddings, skipping embedding loading")

    # Dispatch metrics
    metric_results: dict[str, dict] = {}
    for met_cfg in ctx.comp_metrics:
        met_type = met_cfg["type"]
        met_params = met_cfg.get("params", {})

        if met_type not in COMP_METRICS:
            raise KeyError(
                f"comparison metric '{met_type}' not found in registry. "
                f"Available: {list(COMP_METRICS.keys())}"
            )

        met_fn = COMP_METRICS[met_type]
        needs_emb = getattr(met_fn, "requires_embeddings", True)
        experiment_data = loaded if needs_emb else labels_only

        result = met_fn(
            experiment_data,
            processed_data_dir=processed_data_dir,
            output_dir=ctx.output_dir,
            prefix=ctx.config_stem,
            experiments=ctx.experiments,
            **met_params,
        )
        metric_results[met_type] = result
        logger.info(f"completed comparison metric: {met_type}")

    # Dispatch plots
    for plot_cfg in ctx.comp_plots:
        p_type = plot_cfg["type"]
        p_params = plot_cfg.get("params", {})
        source_metric = plot_cfg.get("metric", p_type.replace("_table", "").replace("_plot", ""))

        if p_type not in COMP_PLOTS:
            raise KeyError(
                f"comparison plot '{p_type}' not found in registry. "
                f"Available: {list(COMP_PLOTS.keys())}"
            )

        if source_metric not in metric_results:
            logger.warning(
                f"plot '{p_type}' references metric '{source_metric}' which was not run, skipping"
            )
            continue

        output_path = ctx.figures_dir / f"{ctx.config_stem}_{p_type}.png"
        p_fn = COMP_PLOTS[p_type]
        p_fn(
            metric_results[source_metric],
            output_path=output_path,
            class_labels=ctx.class_labels,
            experiment_colors=ctx.experiment_colors,
            **p_params,
        )
        logger.info(f"comparison plot saved to {output_path}")

    return metric_results


@app.command()
def main(
    yaml_path: Optional[Path] = typer.Option(
        None, "--yaml-path", "-y", help="Path to a single comparison yaml config."
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
        "/app/configs/comparisons",
        "--config-dir",
        "-c",
        help="Directory containing comparison YAML configs.",
    ),
):
    """Run cross-config comparison using metrics and plots from a YAML config."""
    if yaml_path:
        yaml_paths = [Path(yaml_path)]
    else:
        yaml_paths = list(Path(config_dir).glob("*.yaml"))

    logger.info(f"comparison yamls to process: {yaml_paths}")
    for yp in yaml_paths:
        run_comparison(yp, processed_data_dir, figures_base_dir)


if __name__ == "__main__":
    app()
