from collections.abc import Iterable, Iterator
import inspect
from pathlib import Path
from typing import Any

import geopandas as gpd
from loguru import logger
import numpy as np
from tqdm import tqdm
import yaml

from gelos.cloud_embeddings.aggregate import POOLS, write_chip_parquet
from gelos.cloud_embeddings.backends.alphaearth import AlphaEarthBackend
from gelos.cloud_embeddings.backends.base import FetchRequest
from gelos.cloud_embeddings.backends.olmoearth import OlmoEarthBackend
from gelos.cloud_embeddings.footprints import compute_footprint_manifest
from gelos.cloud_embeddings.year import resolve_chip_year
from gelos.generation import instantiate_recursive

BACKENDS: dict[str, type] = {
    "alphaearth": AlphaEarthBackend,
    "olmoearth": OlmoEarthBackend,
}

DEFAULT_WINDOW_CHIPS = 256


def _backend_accepts_kwarg(backend_cls: type, kwarg: str) -> bool:
    params = inspect.signature(backend_cls.__init__).parameters
    if kwarg in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _chunked(iterable: Iterable, n: int) -> Iterator[list]:
    """Yield successive ``n``-sized lists from ``iterable``.

    Manual chunker because :func:`itertools.batched` is 3.12+ and the project
    pins Python 3.11.
    """
    buf: list = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def _build_backend(cloud_cfg: dict[str, Any], default_cache_dir: Path):
    name = cloud_cfg["backend"]
    if name not in BACKENDS:
        raise KeyError(
            f"cloud-embedding backend '{name}' not registered. Available: {list(BACKENDS.keys())}"
        )
    backend_cls = BACKENDS[name]
    init_args = dict(cloud_cfg.get("init_args") or {})
    if _backend_accepts_kwarg(backend_cls, "cache_dir") and "cache_dir" not in init_args:
        init_args["cache_dir"] = default_cache_dir
    # chunk_cache_dir is opt-in only: fsspec's SimpleCacheFileSystem._cat_file
    # rejects negative `start` values, which zarr v3's sharding codec emits for
    # shard-index reads (SuffixByteRequest). Auto-enabling the wrapper would
    # break any sharded store. Users can still opt in via YAML init_args.
    return backend_cls(**init_args)


def _setup_dataset(yaml_config: dict, raw_data_dir: Path):
    data_version = yaml_config["data_version"]
    data_root = raw_data_dir / data_version
    yaml_config["data"]["init_args"]["data_root"] = str(data_root)
    datamodule = instantiate_recursive(yaml_config["data"])
    datamodule.setup(stage="predict")
    return datamodule.dataset


def generate_cloud_embeddings(
    yaml_path: Path,
    raw_data_dir: Path,
    embedding_dir: Path,
    overwrite: bool = False,
) -> None:
    """Run one cloud-embedding generation pass from a YAML config.

    Mirrors :func:`gelos.generation.generate_embeddings`: writes per-chip
    Parquet files under ``embedding_dir/data_version/config_stem/<strategy>``
    plus a top-level ``.embeddings_complete`` marker. Backends that accept a
    ``cache_dir`` kwarg receive ``embedding_dir/data_version/config_stem/_footprint_cache``
    by default; per-config ``init_args.cache_dir`` wins when specified.

    The footprint manifest is cached at
    ``embedding_dir/data_version/config_stem/_footprint_manifest.parquet`` on
    first compute and reused on subsequent runs (delete the file to invalidate).
    """
    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    logger.info(f"config:\n{yaml.dump(yaml_config)}")

    config_stem = yaml_path.stem
    data_version = yaml_config["data_version"]

    output_dir = embedding_dir / data_version / config_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / "_footprint_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    marker_file = output_dir / ".embeddings_complete"
    if marker_file.exists() and not overwrite:
        logger.info("cloud embeddings already complete, skipping...")
        return
    if marker_file.exists() and overwrite:
        logger.info("starting overwrite of existing cloud embeddings...")

    logger.info("setting up dataset...")
    dataset = _setup_dataset(yaml_config, raw_data_dir)
    logger.info(f"dataset ready: {len(dataset)} chips")

    cloud_cfg = yaml_config["cloud_embedding"]
    reference_sensor = cloud_cfg["reference_sensor"]
    patch_size = cloud_cfg["patch_size"]
    year_strategy_cfg = cloud_cfg["year_strategy"]
    extraction_strategies = cloud_cfg["extraction_strategies"]

    logger.info(f"building backend '{cloud_cfg['backend']}'")
    backend = _build_backend(cloud_cfg, default_cache_dir=cache_dir)
    backend_has_batch = hasattr(backend, "fetch_batch")
    window_chips = int(cloud_cfg.get("window_chips", DEFAULT_WINDOW_CHIPS))

    manifest_cache_path = output_dir / "_footprint_manifest.parquet"
    if manifest_cache_path.exists():
        logger.info(f"loading cached footprint manifest from {manifest_cache_path}")
        manifest = gpd.read_parquet(manifest_cache_path)
    else:
        logger.info(
            f"computing footprint manifest "
            f"(reference_sensor={reference_sensor}, patch_size={patch_size})"
        )
        manifest = compute_footprint_manifest(
            dataset,
            reference_sensor=reference_sensor,
            patch_size=patch_size,
            strategy_specs=extraction_strategies,
        )
        manifest.to_parquet(manifest_cache_path)
        logger.info(f"cached footprint manifest to {manifest_cache_path}")
    logger.info(
        f"footprint manifest: {len(manifest)} rows across {len(extraction_strategies)} strategies"
    )

    for strategy_name, strategy_spec in extraction_strategies.items():
        pool_name = strategy_spec.get("pool", "mean")
        if pool_name not in POOLS:
            raise KeyError(
                f"strategy '{strategy_name}': unknown pool '{pool_name}'. "
                f"Available: {sorted(POOLS)}"
            )
        pool_fn = POOLS[pool_name]
        out_shape = (patch_size, patch_size) if pool_name == "flatten" else None
        strategy_rows = manifest[manifest["strategy"] == strategy_name]
        strategy_out_dir = output_dir / strategy_name
        strategy_out_dir.mkdir(parents=True, exist_ok=True)
        chip_groups = list(strategy_rows.groupby("chip_index"))
        logger.info(
            f"strategy '{strategy_name}' (pool={pool_name}): "
            f"{len(chip_groups)} chips × {len(strategy_rows) // max(len(chip_groups), 1)} patches"
        )
        with tqdm(total=len(chip_groups), desc=f"{strategy_name} chips", unit="chip") as pbar:
            for window in _chunked(chip_groups, window_chips):
                window_sorted = [
                    (
                        chip_index,
                        chip_rows.sort_values(["patch_row", "patch_col"]),
                        resolve_chip_year(dataset, int(chip_index), year_strategy_cfg),
                    )
                    for chip_index, chip_rows in window
                ]
                requests = [
                    FetchRequest(
                        bbox=(
                            float(row["bbox_minx"]),
                            float(row["bbox_miny"]),
                            float(row["bbox_maxx"]),
                            float(row["bbox_maxy"]),
                        ),
                        crs=row["crs"],
                        year=year,
                        out_shape=out_shape,
                    )
                    for _, sorted_rows, year in window_sorted
                    for _, row in sorted_rows.iterrows()
                ]
                if backend_has_batch:
                    arrays = backend.fetch_batch(requests)
                else:
                    arrays = [
                        backend.fetch(r.bbox, r.crs, r.year, out_shape=r.out_shape)
                        for r in requests
                    ]

                cursor = 0
                for chip_index, sorted_rows, _year in window_sorted:
                    n_patches = len(sorted_rows)
                    chip_arrays = arrays[cursor : cursor + n_patches]
                    cursor += n_patches
                    patch_vectors = np.stack([pool_fn(arr) for arr in chip_arrays])
                    filename = sorted_rows["filename"].iloc[0]
                    file_id = sorted_rows["file_id"].iloc[0]
                    out_path = write_chip_parquet(
                        filename, file_id, patch_vectors, strategy_out_dir
                    )
                    logger.info(f"wrote {out_path}")
                pbar.update(len(window))

    marker_file.touch()
    logger.info("marking cloud embeddings as complete")
