import os
from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
from tqdm import tqdm


def sample_files(
    directory: str | Path, sample_size: int, *, seed: int | None = None
) -> list[Path]:
    rng = random.Random(seed)
    directory = Path(directory)

    files = [Path(entry.path) for entry in os.scandir(directory) if entry.is_file()]
    if not sample_size or sample_size >= len(files):
        files.sort()
        return files
    else:
        files = rng.sample(files, sample_size)
        files.sort()
        return files


def select_embedding_indices(
    embeddings_column: pa.lib.ListArray, slice_args: list[dict[str, int]]
) -> pa.lib.ListArray:
    array = embeddings_column
    for arg in slice_args:
        array = pa.compute.list_slice(
            array, start=arg["start"], stop=arg["stop"], step=arg["step"]
        )
        array = pa.compute.list_flatten(array)
    return array


def extract_embeddings(
    directory: Path | str,
    n_sample: int = None,
    chip_indices: list[int] | None = None,
    slice_args: list[dict[str, int]] | None = None,
) -> tuple[np.ndarray, list[int]]:
    # extract embeddings in numpy format from geoparquet
    # TODO: Take embedding shapes and automatically determine slicing args
    if slice_args is None:
        slice_args = [{"start": 0, "stop": None, "step": 1}]
    if chip_indices:
        logger.info(f"filtering embeddings by {len(chip_indices)} file_ids via column")
        files = sample_files(directory, None, seed=42)
        row_filter = ds.field("file_id").isin(chip_indices)
    else:
        files = sample_files(directory, n_sample, seed=42)
        row_filter = None
    dataset = ds.dataset(files, format="parquet")
    scanner = dataset.scanner(columns=["embedding", "file_id"], filter=row_filter)
    emb_chunks, id_chunks = [], []
    batches = scanner.to_batches()
    for batch in tqdm(batches, desc="Processing embeddings"):
        sliced = select_embedding_indices(batch.column("embedding"), slice_args)
        flattened = pa.compute.list_flatten(sliced, recursive=True)
        emb_np = flattened.to_numpy(zero_copy_only=False).reshape(len(batch), -1)
        emb_chunks.append(emb_np)
        id_chunks.append(batch.column("file_id").to_numpy())
    embeddings = np.vstack(emb_chunks)
    chip_indices = np.concatenate(id_chunks).astype(int).tolist()
    return embeddings, chip_indices
