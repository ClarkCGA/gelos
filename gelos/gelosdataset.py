from abc import abstractmethod
from pathlib import Path
from typing import Any

import albumentations as A
from einops import rearrange
import numpy as np
from terratorch.datasets.transforms import MultimodalTransforms
import torch
from torchgeo.datasets import NonGeoDataset


class MultimodalToTensor:
    """
    Default Transform
    Stacking and unstacking in with terratorch.datasets.transforms ALSO rearranges from [T, H, W, C] to [C, T, H, W]
    Therefore, we must do the same here if we are not using them.
    """

    def __init__(self, modalities):
        self.modalities = modalities

    def __call__(self, d):
        new_dict = {}
        for k, v in d.items():
            new_dict[k] = torch.from_numpy(v)
            new_dict[k] = rearrange(
                new_dict[k], "time height width channels -> channels time height width"
            )
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

    Subclasses may additionally override the optional hooks
    ``_get_timestamps`` (per-timestep acquisition dates, added to the output
    as a ``timestamps`` key) and ``_get_location`` (per-chip ``[lat, lon]``,
    added as a ``location`` key). Both default to ``None``, in which case the
    corresponding key is absent and behavior is unchanged.
    """

    def __init__(
        self,
        bands: dict[str, list[str]],
        all_band_names: dict[str, list[str]],
        transform: A.Compose | None = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] | None = None,
        perturb_bands: dict[str, dict[str, float]] | None = None,
        db_scale_bands: dict[str, list[str]] | None = None,
        clip_range_bands: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:

        self.bands = bands
        self.all_band_names = all_band_names
        self.concat_bands = concat_bands
        self.repeat_bands = repeat_bands
        self.perturb_bands = perturb_bands
        self.db_scale_bands = db_scale_bands
        self.clip_range_bands = clip_range_bands

        assert set(self.bands.keys()).issubset(set(self.all_band_names.keys())), (
            f"Please choose a subset of valid sensors: {self.all_band_names.keys()}"
        )

        self.band_indices = {
            sens: [self.all_band_names[sens].index(band) for band in self.bands[sens]]
            for sens in self.bands.keys()
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

    def _get_timestamps(self, index: int) -> np.ndarray | None:
        """Optional hook: per-timestep acquisition dates for the sample at index.

        Override to return a ``(T, 3)`` integer array of canonical calendar
        dates ``[year, month, day]`` (month 1-12, day 1-31), one row per
        timestep of the primary temporal sensor (S2 for OlmoEarth). The value
        is added to the output dict as ``output["timestamps"]`` (a
        ``torch.long`` tensor) and collates to ``(B, T, 3)``; backbones
        convert to their own packing at consumption time. Returning ``None``
        (the default) means no ``timestamps`` key is added and behavior is
        unchanged for non-overriding subclasses.
        """
        return None

    def _get_location(self, index: int) -> np.ndarray | None:
        """Optional hook: chip-center location for the sample at index.

        Override to return a ``(2,)`` float array ``[lat, lon]`` in decimal
        degrees (latitude first) — per-sample, deliberately not per-timestep:
        a chip has one footprint, matching Prithvi TL's ``location_coords``
        ``(B, 2)`` consumption shape. The value is added to the output dict as
        ``output["location"]`` (a ``torch.float32`` tensor) and collates to
        ``(B, 2)``. Returning ``None`` (the default) means no ``location``
        key is added and behavior is unchanged for non-overriding subclasses.
        """
        return None

    def __getitem__(self, index: int) -> dict:
        output = {}

        for sensor in self.bands.keys():
            image = self._load_sensor_images(index, sensor)
            output[sensor] = image.astype(np.float32)

        # Convert linear-power bands to decibels right after loading, before any
        # perturbation or transforms. Matches olmoearth_pretrain's convert_to_db:
        # clip to 1e-10 to avoid log(0), then 10 * log10(x).
        if self.db_scale_bands:
            for sensor, db_band_list in self.db_scale_bands.items():
                band_indices = [self.bands[sensor].index(band) for band in db_band_list]
                for band_index in band_indices:
                    output[sensor][..., band_index] = 10 * np.log10(
                        np.clip(output[sensor][..., band_index], 1e-10, None)
                    )

        # Clip bands to a fixed value range right after loading (after any dB
        # conversion), before perturbation or transforms. Used to reproduce
        # clip-and-stretch preprocessing expected by RGB-pretrained backbones.
        if self.clip_range_bands:
            for sensor, clip_dict in self.clip_range_bands.items():
                for band, (clip_min, clip_max) in clip_dict.items():
                    band_index = self.bands[sensor].index(band)
                    output[sensor][..., band_index] = np.clip(
                        output[sensor][..., band_index], clip_min, clip_max
                    )

        if self.repeat_bands:
            for sensor, repeats in self.repeat_bands.items():
                output[sensor] = np.tile(output[sensor], (repeats, 1, 1, 1))

        if self.perturb_bands:
            for sensor, perturb_band_dict in self.perturb_bands.items():
                band_indices_dict = {
                    self.bands[sensor].index(band): alpha
                    for band, alpha in perturb_band_dict.items()
                }
                output = self._perturb_bands(output, sensor, band_indices_dict)

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

        # filename is the name of the output parquet file record, while file_id is metadata within that parquet.a
        filename, file_id = self._get_sample_id(index)
        output["filename"] = np.array(filename, dtype=str)
        output["file_id"] = file_id

        # Optional metadata hooks — added AFTER the transform/image-format
        # block so image-space transforms never touch them.
        timestamps = self._get_timestamps(index)
        if timestamps is not None:
            ts = np.asarray(timestamps)
            if ts.ndim != 2 or ts.shape[1] != 3:
                raise ValueError(
                    f"_get_timestamps must return a (T, 3) [year, month, day] "
                    f"array, got shape {ts.shape}."
                )
            output["timestamps"] = torch.as_tensor(ts, dtype=torch.long)

        location = self._get_location(index)
        if location is not None:
            loc = np.asarray(location)
            if loc.ndim != 1 or loc.shape[0] != 2:
                raise ValueError(
                    f"_get_location must return a (2,) [lat, lon] array, got shape {loc.shape}."
                )
            output["location"] = torch.as_tensor(loc, dtype=torch.float32)

        return output

    def _load_sensor_images(self, index: int, sensor: str) -> np.ndarray:
        """Load and stack sensor images into [T, H, W, C] array."""
        file_paths = self._get_file_paths(index, sensor)
        band_indices = self.band_indices[sensor]
        sensor_images = [self._load_file(path, band_indices) for path in file_paths]
        return np.stack(sensor_images, axis=0)

    def _perturb_bands(self, output, sensor, band_dict):
        for band_index, alpha in band_dict.items():
            size = output[sensor][:, :, :, band_index].shape
            original_band = output[sensor][:, :, :, band_index]
            noise = np.random.normal(
                loc=np.mean(original_band), scale=np.std(original_band), size=size
            )
            output[sensor][..., band_index] = (1 - alpha) * original_band + alpha * noise

        return output
