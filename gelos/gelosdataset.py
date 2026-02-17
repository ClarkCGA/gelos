from abc import abstractmethod
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
from terratorch.datasets.transforms import MultimodalTransforms
import torch
from torchgeo.datasets import NonGeoDataset


class MultimodalToTensor:
    def __init__(self, modalities):
        self.modalities = modalities

    def __call__(self, d):
        new_dict = {}
        for k, v in d.items():
            new_dict[k] = torch.from_numpy(v)
        return new_dict


class GELOSDataSet(NonGeoDataset):
    """
    Abstract base class for GELOS datasets.

    Defines the contract for the embedding pipeline: output dict must contain
    ``image``, ``filename``, and ``file_id`` keys. Provides reusable logic for
    band validation, perturbation, band repeating, transform dispatch, and
    output formatting.

    Subclasses must implement:
        - ``__len__``
        - ``_get_file_paths``
        - ``_load_file``
        - ``_get_sample_id``
    """

    def __init__(
        self,
        bands: dict[str, list[str]],
        all_band_names: dict[str, list[str]],
        means: dict[str, dict[str, float]] | None = None,
        stds: dict[str, dict[str, float]] | None = None,
        transform: A.Compose | None = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] | None = None,
        perturb_bands: dict[str, list[str]] | None = None,
        perturb_alpha: float = 1,
    ) -> None:

        super().__init__(
            bands=bands,
            all_band_names=all_band_names,
            means=means,
            stds=stds,
            concat_bands=concat_bands,
            repeat_bands=repeat_bands,
            perturb_bands=perturb_bands,
            perturb_alpha=perturb_alpha,
        )

        assert set(self.bands.keys()).issubset(set(self.all_band_names.keys())), (
            f"Please choose a subset of valid sensors: {self.all_band_names.keys()}"
        )

        self.band_indices = {
            sens: [self.all_band_names[sens].index(band) for band in self.bands[sens]]
            for sens in self.bands.keys()
        }

        if self.perturb_bands:
            self.perturb_band_indices = {
                sens: [self.all_band_names[sens].index(band) for band in self.perturb_bands[sens]]
                for sens in self.perturb_bands.keys()
            }

        # Adjust transforms based on the number of sensors
        if transform is None:
            self.transform = MultimodalToTensor(self.bands.keys())
        else:
            transform = {s: transform for s in self.bands.keys()}
            self.transform = MultimodalTransforms(transform, shared=False)

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    @abstractmethod
    def _get_file_paths(self, index: int, sensor: str) -> list[Path]:
        """Return file paths for the given sample index and sensor."""
        ...

    @abstractmethod
    def _load_file(self, path: Path, band_indices: list[int]) -> np.ndarray:
        """Load a single file and return array with shape [H, W, C]."""
        ...

    @abstractmethod
    def _get_sample_id(self, index: int) -> tuple[str, Any]:
        """Return (filename_string, file_id) for the sample at index."""
        ...

    def __getitem__(self, index: int) -> dict:
        output = {}

        for sensor in self.bands.keys():
            image = self._load_sensor_images(index, sensor)
            output[sensor] = image.astype(np.float32)

        if self.repeat_bands:
            for sensor, repeats in self.repeat_bands.items():
                output[sensor] = np.tile(output[sensor], (repeats, 1, 1, 1))

        if self.perturb_bands:
            for sensor, perturb_band_dict in self.perturb_bands.items():
                band_dict = {
                    self.bands[sensor].index(band): alpha
                    for band, alpha in perturb_band_dict.items()
                }
                output = self._perturb_bands(output, sensor, band_dict)

        if self.transform:
            output = self.transform(output)

        # Format image output
        if len(self.bands.keys()) == 1:
            sensor = list(output.keys())[0]
            output["image"] = output.pop(sensor)
        elif self.concat_bands:
            data = [output.pop(m) for m in self.bands.keys() if m in output]
            output["image"] = torch.cat(data, dim=1)
        else:
            output["image"] = {m: output.pop(m) for m in self.bands.keys() if m in output}

        # Add required pipeline metadata
        filename, file_id = self._get_sample_id(index)
        output["filename"] = np.array(filename, dtype=str)
        output["file_id"] = file_id

        return output

    def _load_sensor_images(self, index: int, sensor: str) -> np.ndarray:
        """Load and stack sensor images into [T, H, W, C] array."""
        file_paths = self._get_file_paths(index, sensor)
        band_indices = self.band_indices[sensor]
        sensor_images = [self._load_file(path, band_indices) for path in file_paths]
        return np.stack(sensor_images, axis=0)

    def _perturb_bands(self, output, sensor, band_dict):
        for band_index, alpha in band_dict.items():
            loc = self.means[sensor][band_index]
            scale = self.stds[sensor][band_index]

            size = output[sensor][:, :, :, band_index].shape
            noise = np.random.normal(loc=loc, scale=scale, size=size)
            original_band = output[sensor][:, :, :, band_index]
            combined_noise = (noise * alpha) + (original_band * (1 - alpha))
            output[sensor][:, :, :, band_index] = combined_noise
        return output
