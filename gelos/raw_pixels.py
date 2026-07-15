from itertools import product
from pathlib import Path
from typing import Any

from einops import rearrange
import geopandas as gpd
import numpy as np
import torch
from torch import nn
from torchgeo.trainers import BaseTask

from gelos.cloud_embeddings.patches import resolve_patch_indices, resolve_timestep_indices


class RawPixelEmbeddingTask(BaseTask):
    """Passthrough Lightning task that writes raw normalized pixels as 'embeddings'.

    Mirrors the output layout of ``terratorch.tasks.EmbeddingGenerationTask`` so
    baseline runs flow through the existing analysis/comparison stages
    unchanged. Normalization is expected to have already run inside
    ``GELOSDataModule.on_after_batch_transfer`` via its ``aug`` pipeline.
    """

    def __init__(
        self,
        output_dir: str = "embeddings",
        extraction_strategies: dict[str, dict[str, Any]] | None = None,
        patch_size: int = 16,
        embed_file_key: str = "filename",
    ) -> None:
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.extraction_strategies = extraction_strategies or {}
        self.patch_size = patch_size
        self.embed_file_key = embed_file_key
        super().__init__()

    def configure_callbacks(self):
        return []

    def configure_models(self) -> None:
        self.model = nn.Identity()

    def _patch_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, C, T, H, W]`` into a ``[B, T, nr, nc, C*ph*pw]`` patch grid."""
        if tensor.ndim != 5:
            raise ValueError(
                f"expected 5D tensor [B, C, T, H, W], got {tensor.ndim}D shape {tuple(tensor.shape)}"
            )
        _, _, _, H, W = tensor.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"H ({H}) and W ({W}) must be divisible by patch_size ({self.patch_size})"
            )
        return rearrange(
            tensor,
            "b c t (nr ph) (nc pw) -> b t nr nc (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

    def _resolve_patch_indices(
        self, spec: dict, n_rows: int, n_cols: int
    ) -> list[tuple[int, int]]:
        return resolve_patch_indices(spec, n_rows, n_cols)

    def _resolve_timestep_indices(self, spec: dict, T: int) -> list[int]:
        return resolve_timestep_indices(spec, T)

    def _extract_raw_pixels(
        self,
        name: str,
        image: torch.Tensor | dict[str, torch.Tensor],
        spec: dict,
    ) -> torch.Tensor:
        if isinstance(image, dict):
            if not image:
                raise ValueError(f"strategy '{name}': empty image dict")
            first_sensor = next(iter(image))
            ref_shape = tuple(image[first_sensor].shape[-3:])
            for sensor, t in image.items():
                if tuple(t.shape[-3:]) != ref_shape:
                    raise ValueError(
                        f"strategy '{name}': sensor '{sensor}' has T,H,W={tuple(t.shape[-3:])} "
                        f"but '{first_sensor}' has {ref_shape}"
                    )
            grids = [self._patch_grid(image[sensor]) for sensor in image.keys()]
            grid = torch.cat(grids, dim=-1)
        else:
            grid = self._patch_grid(image)

        _, T, n_rows, n_cols, _ = grid.shape
        patch_indices = self._resolve_patch_indices(spec, n_rows, n_cols)
        timestep_indices = self._resolve_timestep_indices(spec, T)

        tokens = [
            grid[:, t_idx, r, c, :] for t_idx, (r, c) in product(timestep_indices, patch_indices)
        ]
        return torch.stack(tokens, dim=1)

    @torch.no_grad()
    def predict_step(self, batch: dict, *args, **kwargs) -> None:
        image = batch["image"]
        filenames = batch[self.embed_file_key]
        file_ids = batch.get("file_id")

        for name, spec in self.extraction_strategies.items():
            tokens = self._extract_raw_pixels(name, image, spec)
            out_dir = self.output_path / name
            out_dir.mkdir(parents=True, exist_ok=True)
            B = tokens.shape[0]
            for b in range(B):
                filename = self._resolve_filename(filenames, b)
                file_id = self._resolve_file_id(file_ids, b)
                self.write_parquet(tokens[b], filename, file_id, out_dir)

    @staticmethod
    def _resolve_filename(filenames, b: int) -> str:
        if isinstance(filenames, (list, tuple)):
            return str(filenames[b])
        if isinstance(filenames, np.ndarray):
            return str(filenames[b])
        if isinstance(filenames, torch.Tensor):
            return str(filenames[b].item())
        return str(filenames)

    @staticmethod
    def _resolve_file_id(file_ids, b: int):
        if file_ids is None:
            return None
        if isinstance(file_ids, torch.Tensor):
            return file_ids[b].item()
        if isinstance(file_ids, np.ndarray):
            value = file_ids[b]
            return value.item() if value.ndim == 0 else value.tolist()
        if isinstance(file_ids, (list, tuple)):
            return file_ids[b]
        return file_ids

    def write_parquet(
        self,
        embedding: torch.Tensor,
        filename: str,
        file_id: Any,
        dir_path: Path,
    ) -> None:
        filename = Path(filename).stem
        out_path = dir_path / f"{filename}_embedding.parquet"
        arr = embedding.detach().cpu().numpy()

        row = {"embedding": arr.tolist()}
        if file_id is not None:
            row["file_id"] = file_id

        df = gpd.GeoDataFrame([row])
        df.to_parquet(out_path, index=False)


__all__ = ["RawPixelEmbeddingTask"]
