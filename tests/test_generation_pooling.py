import numpy as np
import pyarrow.parquet as pq
import pytest
from terratorch.models.utils import TemporalWrapper
import torch

from gelos.generation import LenientEmbeddingGenerationTask


class _TokenEncoder(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        batch_size = image.shape[0]
        tokens = torch.zeros(batch_size, 197, 3)
        tokens[:, 0, :] = -1000
        tokens[:, 1:, :] = torch.arange(batch_size).view(batch_size, 1, 1) + 1
        return [tokens]


def _task() -> LenientEmbeddingGenerationTask:
    return LenientEmbeddingGenerationTask.__new__(LenientEmbeddingGenerationTask)


def test_vit_spatial_mean_keeps_one_mean_vector_per_timestep(tmp_path):
    task = _task()
    wrapped_model = TemporalWrapper(_TokenEncoder(), pooling="keep")
    image = torch.zeros(1, 2, 4, 8, 8)

    layer_output = wrapped_model(image)[0]
    assert layer_output.shape == (1, 4, 197, 3)

    embedding = task.pool_embedding(layer_output[0], "vit_spatial_mean", has_cls=True)
    expected = torch.arange(1, 5, dtype=torch.float32).view(4, 1).expand(4, 3)
    torch.testing.assert_close(embedding, expected)

    task.write_parquet(embedding, "chip_0001.tif", {"file_id": torch.tensor(1)}, tmp_path)
    table = pq.read_table(tmp_path / "chip_0001_embedding.parquet")
    saved_embedding = np.asarray(table["embedding"].to_pylist()[0])
    assert saved_embedding.shape == (4, 3)
    np.testing.assert_array_equal(saved_embedding, expected.numpy())


def test_vit_spatial_mean_infers_cls_from_odd_token_count():
    embedding = torch.arange(3 * 3 * 2, dtype=torch.float32).reshape(3, 3, 2)

    pooled = _task().pool_embedding(embedding, "vit_spatial_mean", has_cls=None)

    torch.testing.assert_close(pooled, embedding[:, 1:, :].mean(dim=1))


def test_vit_spatial_mean_rejects_output_without_temporal_wrapper():
    embedding = torch.zeros(197, 3)

    with pytest.raises(ValueError, match="timesteps, tokens, channels"):
        _task().pool_embedding(embedding, "vit_spatial_mean", has_cls=True)


def test_vit_spatial_mean_rejects_cls_only_embedding():
    embedding = torch.zeros(4, 1, 3)

    with pytest.raises(ValueError, match="only a CLS token"):
        _task().pool_embedding(embedding, "vit_spatial_mean", has_cls=True)
