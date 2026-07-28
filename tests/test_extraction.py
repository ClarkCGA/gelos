"""Tests for embedding slicing in gelos.extraction.select_embedding_indices."""

import numpy as np
import pyarrow as pa

from gelos.extraction import select_embedding_indices

DIM = 3


def token(chip: int, index: int) -> list[float]:
    """A recognizable embedding vector: chip*1000 + token index, repeated."""
    return [float(chip * 1000 + index)] * DIM


def flat_column(n_chips: int, n_tokens: int) -> pa.Array:
    """Flat layout: each chip row is a list of n_tokens token vectors."""
    return pa.array([[token(c, i) for i in range(n_tokens)] for c in range(n_chips)])


def nested_column(n_chips: int, n_steps: int, n_tokens: int) -> pa.Array:
    """Nested layout (chip -> timestep -> token), as stored for TerraMind."""
    return pa.array(
        [
            [[token(c, t * n_tokens + i) for i in range(n_tokens)] for t in range(n_steps)]
            for c in range(n_chips)
        ]
    )


def selected_ids(sliced: pa.Array, n_chips: int) -> np.ndarray:
    """Mimic extract_embeddings: flatten fully, reshape per chip, recover ids."""
    flattened = pa.compute.list_flatten(sliced, recursive=True)
    values = flattened.to_numpy(zero_copy_only=False).reshape(n_chips, -1)
    per_token = values.reshape(n_chips, -1, DIM)
    assert (per_token == per_token[:, :, :1]).all(), "token vectors got interleaved"
    return per_token[:, :, 0].astype(int) % 1000


def test_single_level_slice():
    column = flat_column(2, 8)
    sliced = select_embedding_indices(column, [{"start": 1, "stop": 7, "step": 2}])
    assert selected_ids(sliced, 2).tolist() == [[1, 3, 5], [1, 3, 5]]


def test_nested_level_slice():
    # TerraMind-style layout: slice timestep 1 at the outer level, then a
    # contiguous run of 4 tokens within it.
    column = nested_column(2, 4, 36)
    sliced = select_embedding_indices(
        column,
        [
            {"start": 1, "stop": 2, "step": 1},
            {"start": 13, "stop": 17, "step": 1},
        ],
    )
    assert selected_ids(sliced, 2).tolist() == [[49, 50, 51, 52]] * 2


def test_strided_slice_across_timesteps():
    # Flat time-major layout: center patch (index 14) of each of 4 timesteps.
    column = flat_column(2, 4 * 36)
    sliced = select_embedding_indices(column, [{"start": 14, "stop": None, "step": 36}])
    assert selected_ids(sliced, 2).tolist() == [[14, 50, 86, 122]] * 2
