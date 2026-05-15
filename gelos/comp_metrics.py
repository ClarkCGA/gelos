from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance as _wasserstein_1d
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from gelos.analysis import build_prefix, load_chip_tracker


def _resolve_metric_csv(exp, processed_data_dir: Path, metric_name: str) -> Path:
    """Resolve the deterministic CSV path for a per-experiment metric."""
    exp_prefix = build_prefix(exp.config, exp.strategy, exp.layer)
    return (
        processed_data_dir
        / exp.data_version
        / exp.config
        / exp.layer
        / f"{exp_prefix}_{metric_name}.csv"
    )


def pca_ablation_comparison(
    experiment_embeddings: list[tuple[str, np.ndarray | None]],
    processed_data_dir: Path,
    output_dir: Path,
    prefix: str,
    experiments: list | None = None,
    **kwargs,
) -> dict:
    """Join per-experiment PCA ablation CSVs into a single comparison table.

    Each experiment's ``{prefix}_pca_ablation.csv`` must already exist
    (produced by the analysis-stage ``pca_ablation`` metric).

    Args:
        experiment_embeddings: List of (label, embeddings) tuples (embeddings unused here).
        processed_data_dir: Root processed directory to resolve per-experiment CSVs.
        output_dir: Directory to write the comparison CSV.
        prefix: File name prefix for the comparison output.
        experiments: List of :class:`ComparisonExperiment` objects for path resolution.

    Returns:
        Dict with ``comparison_df`` key holding the merged DataFrame.
    """
    if experiments is None:
        raise ValueError("pca_ablation_comparison requires 'experiments' to resolve CSV paths")

    frames = []
    for exp in experiments:
        csv_path = _resolve_metric_csv(exp, processed_data_dir, "pca_ablation")
        if not csv_path.exists():
            logger.warning(f"no pca_ablation CSV found for '{exp.label}' at {csv_path}, skipping")
            continue
        df = pd.read_csv(csv_path)
        df["experiment"] = exp.label
        frames.append(df)

    if not frames:
        logger.warning("no PCA ablation data found for any experiment")
        return {"comparison_df": pd.DataFrame()}

    merged = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / f"{prefix}_pca_ablation_comparison.csv"
    merged.to_csv(csv_path, index=False)
    logger.info(f"saved PCA ablation comparison to {csv_path}")
    return {"comparison_df": merged}


def cosine_distance(
    experiment_embeddings: list[tuple[str, np.ndarray]],
    output_dir: Path,
    prefix: str,
    **kwargs,
) -> dict:
    """Compute pairwise cosine similarity between experiment mean embeddings.

    Args:
        experiment_embeddings: List of (label, embeddings) tuples.
        output_dir: Directory to write the distance matrix CSV.
        prefix: File name prefix for the output CSV.

    Returns:
        Dict with ``labels``, ``matrix`` (np.ndarray), and ``df`` (DataFrame) keys.
    """
    labels = [label for label, _ in experiment_embeddings]
    means = np.array([emb.mean(axis=0) for _, emb in experiment_embeddings])

    sim_matrix = cosine_similarity(means)

    df = pd.DataFrame(sim_matrix, index=labels, columns=labels)
    csv_path = output_dir / f"{prefix}_cosine_distance.csv"
    df.to_csv(csv_path)
    logger.info(f"saved cosine distance matrix to {csv_path}")

    return {"labels": labels, "matrix": sim_matrix, "df": df}


def wasserstein_distance(
    experiment_embeddings: list[tuple[str, np.ndarray]],
    output_dir: Path,
    prefix: str,
    n_pca_components: int = 10,
    **kwargs,
) -> dict:
    """Compute pairwise Wasserstein distance on PCA-reduced embeddings.

    Reduces each experiment's embeddings via PCA, then computes the mean
    1D Wasserstein distance across the first ``n_pca_components`` dimensions.

    Args:
        experiment_embeddings: List of (label, embeddings) tuples.
        output_dir: Directory to write the distance matrix CSV.
        prefix: File name prefix for the output CSV.
        n_pca_components: Number of PCA components to reduce to before computing distances.

    Returns:
        Dict with ``labels``, ``matrix`` (np.ndarray), and ``df`` (DataFrame) keys.
    """
    labels = [label for label, _ in experiment_embeddings]
    n_exp = len(labels)

    # PCA-reduce each experiment
    reduced = []
    for label, emb in experiment_embeddings:
        n_comp = min(n_pca_components, emb.shape[1], emb.shape[0])
        pca = PCA(n_components=n_comp, random_state=42)
        reduced.append(pca.fit_transform(emb))

    # Compute pairwise mean Wasserstein distance across dimensions
    dist_matrix = np.zeros((n_exp, n_exp))
    for i in range(n_exp):
        for j in range(i + 1, n_exp):
            n_dims = min(reduced[i].shape[1], reduced[j].shape[1])
            dim_dists = [
                _wasserstein_1d(reduced[i][:, d], reduced[j][:, d]) for d in range(n_dims)
            ]
            mean_dist = float(np.mean(dim_dists))
            dist_matrix[i, j] = mean_dist
            dist_matrix[j, i] = mean_dist

    df = pd.DataFrame(dist_matrix, index=labels, columns=labels)
    csv_path = output_dir / f"{prefix}_wasserstein_distance.csv"
    df.to_csv(csv_path)
    logger.info(f"saved Wasserstein distance matrix to {csv_path}")

    return {"labels": labels, "matrix": dist_matrix, "df": df}


def knn_purity_comparison(
    experiment_embeddings: list[tuple[str, np.ndarray | None]],
    processed_data_dir: Path,
    output_dir: Path,
    prefix: str,
    experiments: list | None = None,
    **kwargs,
) -> dict:
    """Join per-experiment KNN purity CSVs into a single comparison table.

    Each experiment's ``{prefix}_knn_purity.csv`` must already exist
    (produced by the analysis-stage ``knn_purity`` metric).

    Args:
        experiment_embeddings: List of (label, embeddings) tuples (embeddings unused here).
        processed_data_dir: Root processed directory to resolve per-experiment CSVs.
        output_dir: Directory to write the comparison CSV.
        prefix: File name prefix for the comparison output.
        experiments: List of :class:`ComparisonExperiment` objects for path resolution.

    Returns:
        Dict with ``comparison_df`` key holding the merged DataFrame.
    """
    if experiments is None:
        raise ValueError("knn_purity_comparison requires 'experiments' to resolve CSV paths")

    frames = []
    for exp in experiments:
        csv_path = _resolve_metric_csv(exp, processed_data_dir, "knn_purity")
        if not csv_path.exists():
            logger.warning(f"no knn_purity CSV found for '{exp.label}' at {csv_path}, skipping")
            continue
        df = pd.read_csv(csv_path)
        df["experiment"] = exp.label
        frames.append(df)

    if not frames:
        logger.warning("no KNN purity data found for any experiment")
        return {"comparison_df": pd.DataFrame()}

    merged = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / f"{prefix}_knn_purity_comparison.csv"
    merged.to_csv(csv_path, index=False)
    logger.info(f"saved KNN purity comparison to {csv_path}")
    return {"comparison_df": merged}


def knn_purity_per_query_comparison(
    experiment_embeddings: list[tuple[str, np.ndarray | None]],
    processed_data_dir: Path,
    output_dir: Path,
    prefix: str,
    experiments: list | None = None,
    **kwargs,
) -> dict:
    """Join per-experiment KNN per-query purity CSVs into a comparison table.

    Each experiment's ``{prefix}_knn_purity_per_query.csv`` must already exist
    (produced by the analysis-stage ``knn_purity`` metric). One row per
    (query, k) per experiment — the distribution that the aggregated
    comparison collapses to a mean.

    Args:
        experiment_embeddings: List of (label, embeddings) tuples (embeddings unused here).
        processed_data_dir: Root processed directory to resolve per-experiment CSVs.
        output_dir: Directory to write the comparison CSV.
        prefix: File name prefix for the comparison output.
        experiments: List of :class:`ComparisonExperiment` objects for path resolution.

    Returns:
        Dict with ``comparison_df`` key holding the merged DataFrame.
    """
    if experiments is None:
        raise ValueError(
            "knn_purity_per_query_comparison requires 'experiments' to resolve CSV paths"
        )

    frames = []
    for exp in experiments:
        csv_path = _resolve_metric_csv(exp, processed_data_dir, "knn_purity_per_query")
        if not csv_path.exists():
            logger.warning(
                f"no knn_purity_per_query CSV found for '{exp.label}' at {csv_path}, skipping"
            )
            continue
        df = pd.read_csv(csv_path)
        df["experiment"] = exp.label
        frames.append(df)

    if not frames:
        logger.warning("no KNN per-query purity data found for any experiment")
        return {"comparison_df": pd.DataFrame()}

    merged = pd.concat(frames, ignore_index=True)
    csv_path = output_dir / f"{prefix}_knn_purity_per_query_comparison.csv"
    merged.to_csv(csv_path, index=False)
    logger.info(f"saved KNN per-query purity comparison to {csv_path}")
    return {"comparison_df": merged}


def _load_experiment_with_indices(
    exp,
    processed_data_dir: Path,
) -> tuple[str, np.ndarray, np.ndarray] | None:
    """Resolve and load both ``{prefix}_embeddings.npy`` and ``{prefix}_chip_indices.npy``.

    Mirrors :func:`gelos.comparison._resolve_embedding_path` but additionally
    requires the chip-indices cache so callers can align per-chip across
    experiments. Returns ``None`` (and logs) if either file is missing.
    """
    prefix = build_prefix(exp.config, exp.strategy, exp.layer)
    layer_dir = processed_data_dir / exp.data_version / exp.config / exp.layer
    emb_path = layer_dir / f"{prefix}_embeddings.npy"
    idx_path = layer_dir / f"{prefix}_chip_indices.npy"

    if not emb_path.exists():
        logger.warning(f"embeddings not found for '{exp.label}' at {emb_path}, skipping")
        return None
    if not idx_path.exists():
        logger.warning(
            f"chip_indices cache not found for '{exp.label}' at {idx_path}; "
            f"re-run analysis to produce it"
        )
        return None

    embeddings = np.load(emb_path)
    chip_indices = np.load(idx_path)
    return exp.label, embeddings, chip_indices


def per_chip_similarity_to_control(
    experiment_embeddings: list[tuple[str, np.ndarray | None]],
    processed_data_dir: Path,
    output_dir: Path,
    prefix: str,
    experiments: list | None = None,
    *,
    control_label: str,
    chip_tracker_path: str | Path,
    chip_id_column: str,
    category_column: str,
    **kwargs,
) -> dict:
    """Per-chip cosine similarity between each ablation embedding and its control.

    For every ablation experiment, pairs each chip with the same chip in the
    control experiment (via ``{prefix}_chip_indices.npy``) and computes the
    row-wise cosine similarity. Each row is also tagged with the chip's
    category via the chip-tracker lookup, so downstream plotting can break
    the distribution down per-class.

    Args:
        experiment_embeddings: Unused — the metric re-loads embeddings
            alongside chip_indices via :func:`_load_experiment_with_indices`.
        processed_data_dir: Root processed directory for cache resolution.
        output_dir: Directory to write the long-form and summary CSVs.
        prefix: File name prefix for the output CSVs.
        experiments: List of :class:`ComparisonExperiment` for path resolution.
        control_label: Label that identifies the control experiment in
            ``experiments``. Must match exactly one entry.
        chip_tracker_path: Path to the chip tracker (.geojson/.json/.csv)
            providing the class label per chip.
        chip_id_column: Column in the chip tracker holding the chip id used
            in the cached ``{prefix}_chip_indices.npy`` files.
        category_column: Column in the chip tracker holding the per-chip
            class label.

    Returns:
        Dict with ``comparison_df`` (long-form per-chip rows) and
        ``summary_df`` (per ``experiment, class`` aggregates).
    """
    if experiments is None:
        raise ValueError(
            "per_chip_similarity_to_control requires 'experiments' to resolve cache paths"
        )

    control_exp = next((e for e in experiments if e.label == control_label), None)
    if control_exp is None:
        raise ValueError(
            f"control_label '{control_label}' did not match any experiment label "
            f"(available: {[e.label for e in experiments]})"
        )

    # Load chip tracker once and index by chip id for class lookup
    tracker_path = Path(chip_tracker_path)
    if not tracker_path.exists():
        raise FileNotFoundError(f"chip tracker not found at {tracker_path}")
    chip_df = load_chip_tracker(tracker_path).set_index(chip_id_column)
    if category_column not in chip_df.columns:
        raise KeyError(
            f"category_column '{category_column}' not in chip tracker "
            f"(columns: {list(chip_df.columns)})"
        )

    # Load every experiment up front so we can compute a global intersection
    # of chip_ids — per-chip cosine similarity is only meaningful for chips
    # present in every run being compared.
    loaded_experiments: list[tuple[str, np.ndarray, np.ndarray]] = []
    control_loaded: tuple[str, np.ndarray, np.ndarray] | None = None
    for exp in experiments:
        loaded = _load_experiment_with_indices(exp, processed_data_dir)
        if loaded is None:
            continue
        label, emb, ids = loaded
        loaded_experiments.append(loaded)
        if exp.label == control_label:
            control_loaded = loaded

    if control_loaded is None:
        logger.warning(
            f"could not load control experiment '{control_label}', "
            f"per_chip_similarity_to_control aborting"
        )
        return {"comparison_df": pd.DataFrame(), "summary_df": pd.DataFrame()}

    _, control_emb, control_ids = control_loaded
    control_dim = control_emb.shape[1]

    shared_ids = control_ids
    for label, emb, ids in loaded_experiments:
        if emb.shape[1] != control_dim:
            raise ValueError(
                f"embedding dimension mismatch: '{control_label}' has {control_dim} dims, "
                f"'{label}' has {emb.shape[1]} dims"
            )
        shared_ids = np.intersect1d(shared_ids, ids)

    if shared_ids.size == 0:
        logger.warning("no chips shared across all experiments, skipping")
        return {"comparison_df": pd.DataFrame(), "summary_df": pd.DataFrame()}

    # Resolve class labels once for the shared chips
    missing = set(shared_ids.tolist()).difference(chip_df.index)
    if missing:
        raise KeyError(
            f"chip tracker missing {len(missing)} of the shared chip ids "
            f"(first few: {list(missing)[:5]})"
        )
    shared_classes = chip_df.loc[shared_ids, category_column].to_numpy().astype(str)

    # Align control embeddings to shared_ids once
    _, _, ctrl_idx_into_shared = np.intersect1d(control_ids, shared_ids, return_indices=True)
    # ctrl_idx_into_shared gives positions in control_ids of shared_ids, in
    # the order of intersect1d's first arg (control_ids). Re-sort to match
    # shared_ids' canonical (sorted) order.
    order = np.argsort(control_ids[ctrl_idx_into_shared])
    control_aligned = control_emb[ctrl_idx_into_shared[order]]

    frames: list[pd.DataFrame] = []
    for label, emb, ids in loaded_experiments:
        n_dropped = len(ids) - len(shared_ids)
        if n_dropped:
            logger.info(
                f"'{label}': dropped {n_dropped} chips not in global intersection "
                f"({len(shared_ids)} retained)"
            )
        _, _, idx_into_exp = np.intersect1d(ids, shared_ids, return_indices=True)
        order = np.argsort(ids[idx_into_exp])
        a = emb[idx_into_exp[order]]
        sims = np.einsum("ij,ij->i", a, control_aligned) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(control_aligned, axis=1)
        )
        frames.append(
            pd.DataFrame(
                {
                    "experiment": label,
                    "file_id": shared_ids,
                    "class": shared_classes,
                    "cosine_similarity": sims,
                }
            )
        )

    if not frames:
        logger.warning("no experiments produced per-chip similarities")
        return {"comparison_df": pd.DataFrame(), "summary_df": pd.DataFrame()}

    long_df = pd.concat(frames, ignore_index=True)
    long_csv = output_dir / f"{prefix}_per_chip_similarity_to_control.csv"
    long_df.to_csv(long_csv, index=False)
    logger.info(f"saved per-chip similarity table to {long_csv}")

    # Build per-(experiment, class) summary with KS-vs-control where applicable
    control_by_class: dict[str, np.ndarray] = {
        cls: grp["cosine_similarity"].to_numpy()
        for cls, grp in long_df[long_df["experiment"] == control_label].groupby("class")
    }

    summary_rows = []
    for (exp_label, cls), grp in long_df.groupby(["experiment", "class"]):
        values = grp["cosine_similarity"].to_numpy()
        ks_stat: float | None = None
        ks_p: float | None = None
        ctrl_values = control_by_class.get(cls)
        if (
            exp_label != control_label
            and ctrl_values is not None
            and ctrl_values.size > 0
            and values.size > 0
        ):
            res = ks_2samp(values, ctrl_values)
            ks_stat = float(res.statistic)
            ks_p = float(res.pvalue)
        summary_rows.append(
            {
                "experiment": exp_label,
                "class": cls,
                "n": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "ks_vs_control_stat": ks_stat,
                "ks_vs_control_p": ks_p,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / f"{prefix}_per_chip_similarity_to_control_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"saved per-chip similarity summary to {summary_csv}")

    return {"comparison_df": long_df, "summary_df": summary_df}


pca_ablation_comparison.requires_embeddings = False
cosine_distance.requires_embeddings = True
wasserstein_distance.requires_embeddings = True
knn_purity_comparison.requires_embeddings = False
knn_purity_per_query_comparison.requires_embeddings = False
per_chip_similarity_to_control.requires_embeddings = False

COMP_METRICS: dict[str, callable] = {
    "pca_ablation_comparison": pca_ablation_comparison,
    "cosine_distance": cosine_distance,
    "wasserstein_distance": wasserstein_distance,
    "knn_purity_comparison": knn_purity_comparison,
    "knn_purity_per_query_comparison": knn_purity_per_query_comparison,
    "per_chip_similarity_to_control": per_chip_similarity_to_control,
}
