from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Mean Earth radius (IUGG) used for great-circle (haversine) distances.
EARTH_RADIUS_M = 6_371_008.8


def _query_neighbors(embeddings: np.ndarray, query_indices: np.ndarray, max_k: int) -> np.ndarray:
    """Return the ``max_k`` nearest neighbours (self excluded) of each query row.

    Fits a brute-force Euclidean :class:`NearestNeighbors` on the full
    ``embeddings`` and queries ``embeddings[query_indices]`` with ``max_k + 1``
    neighbours, dropping the first column (the query point itself). Shared by
    :func:`knn_purity` and :func:`knn_geo_distance` so both metrics always agree
    on the neighbour sets.

    Returns:
        Integer array of shape ``(len(query_indices), min(max_k, N - 1))``.
    """
    n_samples = embeddings.shape[0]
    nn = NearestNeighbors(n_neighbors=min(max_k + 1, n_samples), algorithm="brute", n_jobs=-1)
    nn.fit(embeddings)
    _, neighbor_indices = nn.kneighbors(embeddings[query_indices])
    # Exclude self (first column)
    return neighbor_indices[:, 1:]


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres between (lat1, lon1) and (lat2, lon2).

    Inputs are decimal degrees and broadcast against each other (e.g. ``(Q, 1)``
    against ``(Q, k)``). Uses a sphere of radius :data:`EARTH_RADIUS_M`.
    """
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(a, dtype=np.float64)) for a in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def chip_centers_latlon(chip_gdf, chip_indices) -> tuple[np.ndarray, np.ndarray]:
    """Look up the (lat, lon) centre of each chip in ``chip_indices``.

    ``chip_gdf`` must be indexed by chip id (as ``setup_analysis_run`` does).
    For a GeoDataFrame the centre is the midpoint of each geometry's bounds
    after reprojecting to EPSG:4326 (identical to the point for Point
    geometries; avoids geopandas' geographic-CRS centroid warning for footprint
    polygons). For a plain DataFrame (CSV tracker) the ``lat``/``lon`` or
    ``latitude``/``longitude`` columns are used.

    Returns:
        ``(lat, lon)`` float64 arrays aligned with ``chip_indices``.

    Raises:
        ValueError: If no geometry or coordinate columns are available.
    """
    sub = chip_gdf.loc[list(chip_indices)]
    geometry = getattr(sub, "geometry", None)
    if geometry is not None and hasattr(sub, "crs"):
        if sub.crs is not None and not sub.crs.equals("EPSG:4326"):
            sub = sub.to_crs(4326)
        b = sub.geometry.bounds
        lon = ((b["minx"] + b["maxx"]) / 2).to_numpy(dtype=np.float64)
        lat = ((b["miny"] + b["maxy"]) / 2).to_numpy(dtype=np.float64)
        return lat, lon

    for lat_col, lon_col in (("lat", "lon"), ("latitude", "longitude")):
        if lat_col in sub.columns and lon_col in sub.columns:
            lat = sub[lat_col].to_numpy(dtype=np.float64)
            lon = sub[lon_col].to_numpy(dtype=np.float64)
            return lat, lon

    raise ValueError(
        "chip tracker has no geometry and no lat/lon (or latitude/longitude) columns; "
        f"available columns: {list(sub.columns)}"
    )


def pca_ablation(
    embeddings: np.ndarray,
    output_dir: Path,
    prefix: str,
    variance_thresholds: list[float] | None = None,
    **kwargs,
) -> dict:
    """Run PCA at multiple variance thresholds and record component counts.

    For each threshold, fits PCA and records the number of components needed
    to explain that fraction of variance.

    Args:
        embeddings: Input array of shape (N, D).
        output_dir: Directory to write the result CSV.
        prefix: File name prefix for the output CSV.
        variance_thresholds: List of cumulative variance fractions to test.

    Returns:
        Dict with ``thresholds`` list of per-threshold results.
    """
    if variance_thresholds is None:
        variance_thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]

    rows = []
    for threshold in variance_thresholds:
        pca = PCA(n_components=threshold, random_state=42)
        pca.fit(embeddings)
        N, D = embeddings.shape
        rows.append(
            {
                "threshold": threshold,
                "n_components": pca.n_components_,
                "proportion_of_total_components": pca.n_components_ / D,
                "total_variance_explained": float(pca.explained_variance_ratio_.sum()),
            }
        )
        logger.info(
            f"PCA ablation: threshold={threshold}, "
            f"n_components={pca.n_components_}, "
            f"proportion_of_total_components={pca.n_components_ / D}, "
            f"variance={pca.explained_variance_ratio_.sum():.4f}"
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{prefix}_pca_ablation.csv"
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"saved PCA ablation results to {csv_path}")

    return {"thresholds": rows}


def knn_purity(
    embeddings: np.ndarray,
    output_dir: Path,
    prefix: str,
    labels: np.ndarray | None = None,
    k_values: list[int] | None = None,
    n_subsample: int | None = None,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """Compute KNN class purity at multiple k values.

    For each chip, measures what fraction of its k nearest neighbors share
    the same class. Reports per-class and overall purity at each k.

    Args:
        embeddings: Input array of shape (N, D).
        output_dir: Directory to write the result CSV.
        prefix: File name prefix for the output CSV.
        labels: Class labels of shape (N,). Required.
        k_values: List of k values to evaluate.
        n_subsample: If set, use a stratified subsample of query points.
        random_state: Random state for reproducibility.

    Returns:
        Dict with ``rows`` list of per-k/per-class results and ``k_values``.
    """
    if labels is None:
        raise ValueError("knn_purity requires labels")

    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 50]

    k_values = sorted(k_values)
    max_k = max(k_values)
    n_samples = embeddings.shape[0]

    # Determine query indices (optionally subsample)
    rng = np.random.RandomState(random_state)
    if n_subsample is not None and n_subsample < n_samples:
        # Stratified subsample
        unique_classes = np.unique(labels)
        query_indices = []
        per_class_n = max(1, n_subsample // len(unique_classes))
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            n_take = min(per_class_n, len(cls_indices))
            query_indices.extend(rng.choice(cls_indices, size=n_take, replace=False))
        query_indices = np.array(sorted(query_indices))
    else:
        query_indices = np.arange(n_samples)

    # Fit on full embeddings, query with max_k+1 to exclude self
    neighbor_indices = _query_neighbors(embeddings, query_indices, max_k)
    query_labels = labels[query_indices]

    rows = []
    per_query_rows = []
    for k in k_values:
        if k > neighbor_indices.shape[1]:
            logger.warning(f"k={k} exceeds available neighbors, skipping")
            continue

        k_neighbors = neighbor_indices[:, :k]
        k_neighbor_labels = labels[k_neighbors]
        # Purity: fraction of neighbors sharing the query's class
        matches = k_neighbor_labels == query_labels[:, np.newaxis]
        per_query_purity = matches.mean(axis=1)

        for q_idx, (cls, p) in enumerate(zip(query_labels, per_query_purity)):
            per_query_rows.append(
                {
                    "k": k,
                    "class": str(cls),
                    "query_idx": int(q_idx),
                    "purity": float(p),
                }
            )

        # Overall purity
        overall = float(per_query_purity.mean())
        rows.append(
            {
                "k": k,
                "class": "overall",
                "purity": overall,
                "n_samples": len(query_indices),
            }
        )

        # Per-class purity
        unique_classes = np.unique(query_labels)
        for cls in unique_classes:
            mask = query_labels == cls
            cls_purity = float(per_query_purity[mask].mean())
            rows.append(
                {
                    "k": k,
                    "class": str(cls),
                    "purity": cls_purity,
                    "n_samples": int(mask.sum()),
                }
            )

        logger.info(f"KNN purity: k={k}, overall={overall:.4f}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{prefix}_knn_purity.csv"
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"saved KNN purity results to {csv_path}")

    per_query_df = pd.DataFrame(per_query_rows)
    per_query_csv_path = output_dir / f"{prefix}_knn_purity_per_query.csv"
    per_query_df.to_csv(per_query_csv_path, index=False)
    logger.info(f"saved KNN per-query purity to {per_query_csv_path}")

    return {"rows": rows, "k_values": k_values}


def knn_geo_distance(
    embeddings: np.ndarray,
    output_dir: Path,
    prefix: str,
    labels: np.ndarray | None = None,
    chip_gdf=None,
    chip_indices=None,
    k_values: list[int] | None = None,
    n_subsample: int | None = None,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """Compute the mean geographic distance to each chip's k nearest embedding neighbours.

    For each query chip and each k, finds the k nearest neighbours in embedding
    space (Euclidean, self excluded — the same neighbour sets as
    :func:`knn_purity`) and averages two ground-distance measures over them:

    - ``gsd_m``: great-circle (haversine) distance between chip centres in metres.
    - ``lat_diff_deg``: absolute latitude difference in decimal degrees.

    The metric is class-agnostic, so it also works for datasets without
    chip-level labels. Chip centres come from :func:`chip_centers_latlon`.

    Args:
        embeddings: Input array of shape (N, D).
        output_dir: Directory to write the result CSVs.
        prefix: File name prefix for the output CSVs.
        labels: Unused; accepted for dispatch compatibility.
        chip_gdf: Chip tracker indexed by chip id (GeoDataFrame or DataFrame
            with ``lat``/``lon`` columns). Required.
        chip_indices: Length-N chip ids aligned with ``embeddings``. Required.
        k_values: List of k values to evaluate.
        n_subsample: If set, use a uniform random subsample of query points.
        random_state: Random state for reproducibility.

    Returns:
        Dict with ``rows`` (aggregate rows, one per ``(k, measure)``) and ``k_values``.
    """
    if chip_gdf is None or chip_indices is None:
        raise ValueError("knn_geo_distance requires chip_gdf and chip_indices")
    if len(chip_indices) != embeddings.shape[0]:
        raise ValueError(
            f"chip_indices length ({len(chip_indices)}) does not match "
            f"embeddings rows ({embeddings.shape[0]})"
        )

    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 50]

    k_values = sorted(k_values)
    max_k = max(k_values)
    n_samples = embeddings.shape[0]

    lat, lon = chip_centers_latlon(chip_gdf, chip_indices)
    n_bad = int((~np.isfinite(lat) | ~np.isfinite(lon)).sum())
    if n_bad:
        logger.warning(
            f"{n_bad} chips have non-finite coordinates; they are ignored in the per-query means"
        )

    rng = np.random.RandomState(random_state)
    if n_subsample is not None and n_subsample < n_samples:
        query_indices = np.sort(rng.choice(n_samples, size=n_subsample, replace=False))
    else:
        query_indices = np.arange(n_samples)

    neighbor_indices = _query_neighbors(embeddings, query_indices, max_k)
    lat_q = lat[query_indices][:, np.newaxis]
    lon_q = lon[query_indices][:, np.newaxis]
    chip_ids = np.asarray(chip_indices)[query_indices]

    rows = []
    per_query_rows = []
    for k in k_values:
        if k > neighbor_indices.shape[1]:
            logger.warning(f"k={k} exceeds available neighbors, skipping")
            continue

        nb = neighbor_indices[:, :k]
        gsd = np.nanmean(_haversine_m(lat_q, lon_q, lat[nb], lon[nb]), axis=1)
        lat_diff = np.nanmean(np.abs(lat[nb] - lat_q), axis=1)

        for q_idx, (chip_id, g, d) in enumerate(zip(chip_ids, gsd, lat_diff)):
            per_query_rows.append(
                {
                    "k": k,
                    "query_idx": int(q_idx),
                    "chip_id": chip_id,
                    "gsd_m": float(g),
                    "lat_diff_deg": float(d),
                }
            )

        for measure, values in (("gsd_m", gsd), ("lat_diff_deg", lat_diff)):
            rows.append(
                {
                    "k": k,
                    "measure": measure,
                    "mean": float(np.nanmean(values)),
                    "median": float(np.nanmedian(values)),
                    "q1": float(np.nanquantile(values, 0.25)),
                    "q3": float(np.nanquantile(values, 0.75)),
                    "n_samples": len(query_indices),
                }
            )

        logger.info(
            f"KNN geo distance: k={k}, mean_gsd={np.nanmean(gsd) / 1000:.2f} km, "
            f"mean_lat_diff={np.nanmean(lat_diff):.4f} deg"
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"{prefix}_knn_geo_distance.csv"
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(csv_path, index=False)
    logger.info(f"saved KNN geo distance results to {csv_path}")

    per_query_df = pd.DataFrame(per_query_rows)
    per_query_csv_path = output_dir / f"{prefix}_knn_geo_distance_per_query.csv"
    per_query_df.to_csv(per_query_csv_path, index=False)
    logger.info(f"saved KNN per-query geo distance to {per_query_csv_path}")

    return {"rows": rows, "k_values": k_values}


METRICS: dict[str, callable] = {
    "pca_ablation": pca_ablation,
    "knn_purity": knn_purity,
    "knn_geo_distance": knn_geo_distance,
}
