import importlib
from pathlib import Path
from typing import Any, List

import albumentations as A
from kornia.augmentation import AugmentationSequential
from loguru import logger
from terratorch.datamodules.generic_multimodal_data_module import (
    MultimodalNormalize,
    collate_samples,
    wrap_in_compose_is_list,
)
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule


class IdentityAug:
    """No-op batch augmentation used when ``normalize=False``.

    Returns the batch unchanged (same dict structure), for backbones that apply
    their own pretraining normalization internally (e.g. OlmoEarth).
    """

    def __call__(self, batch: dict) -> dict:
        return batch


class NoDataRemap:
    """Wrap a batch aug so on-disk nodata pixels reach the model as a chosen value.

    Detection happens BEFORE the wrapped aug runs: the on-disk sentinel (e.g.
    ``-999``) is only exactly detectable in the raw batch — after z-score
    normalization it would become an arbitrary extreme float
    (``(-999 - mean) / std``) that cannot be reliably matched. The write happens
    AFTER the wrapped aug runs: ``set_nodata`` is written into the returned
    (normalized) tensors, so the model receives exactly ``set_nodata`` at nodata
    positions, untouched by the normalization math.

    Works identically whichever aug is wrapped (``Normalize``,
    ``MultimodalNormalize``, ``IdentityAug``, or a user-supplied aug).

    Args:
        aug: The batch augmentation to wrap (called between mask computation and
            the nodata write).
        nodata_value: On-disk nodata sentinel. A scalar applies to every
            modality; a ``{modality: value}`` dict applies only to its listed
            modalities (others pass through unmasked).
        set_nodata: Value the model should receive at nodata positions. A scalar
            or a ``{modality: value}`` dict covering every masked modality.
    """

    def __init__(
        self,
        aug,
        nodata_value: float | dict[str, float],
        set_nodata: float | dict[str, float],
    ) -> None:
        self.aug = aug
        self.nodata_value = nodata_value
        self.set_nodata = set_nodata

    def __call__(self, batch: dict) -> dict:
        image = batch["image"]
        if isinstance(image, dict):
            masks = {}
            for modality, tensor in image.items():
                if isinstance(self.nodata_value, dict):
                    if modality not in self.nodata_value:
                        continue
                    nodata = self.nodata_value[modality]
                else:
                    nodata = self.nodata_value
                masks[modality] = tensor == nodata
            batch = self.aug(batch)
            for modality, mask in masks.items():
                target = (
                    self.set_nodata[modality]
                    if isinstance(self.set_nodata, dict)
                    else self.set_nodata
                )
                batch["image"][modality][mask] = target
        else:
            if isinstance(self.nodata_value, dict) or isinstance(self.set_nodata, dict):
                # ValueError (not TypeError): dicts are valid config, just not for
                # this batch layout.
                raise ValueError(
                    "Per-modality nodata_value/set_nodata dicts cannot be applied to a "
                    "single image tensor (single modality or concat_bands=True): modality "
                    "boundaries are unknown in the tensor. Use scalar values instead."
                )
            mask = image == self.nodata_value
            batch = self.aug(batch)
            batch["image"][mask] = self.set_nodata
        return batch


class GELOSDataModule(NonGeoDataModule):
    """
    This is the datamodule for Geospatial Exploration of Latent Observation Space (GELOS)
    """

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int,
        num_workers: int,
        dataset_class: type | str,
        means: dict[str, dict[str, float]] | None = None,
        stds: dict[str, dict[str, float]] | None = None,
        bands: dict[str, List[str]] | None = None,
        transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential | None = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] | None = None,
        perturb_bands: dict[str, dict[str, float]] | None = None,
        normalize: bool = True,
        db_scale_bands: dict[str, list[str]] | None = None,
        nodata_value: float | dict[str, float] | None = None,
        set_nodata: float | dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the DataModule for GELOS.

        Args:
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for data loading.
            data_root (str | Path): Root directory for dataset.
            dataset_class (type, optional): Dataset class to use.
            means: (dict[str, dict[str, float]]): Dictionary defining modalities and bands with mean values
            stds: (dict[str, dict[str, float]]): Dictionary defining modalities and bands with std values
            bands: (dict[str, List[str]], optional): Dictionary with format "modality" : List['band_a', 'band_b']
            transform (A.Compose, optional): Transforms for data, defaults to ToTensorV2.
            aug (AugmentationSequential, optional): Augmentation or normalization to apply. Defaults to normalization if not provided.
            concat_bands (bool): Whether to concat all sensors into one 'image' tensor or keep separate
            repeat_bands (dict[str, int], optional): repeat bands when loading from disc, intended to repeat single time step modalities e.g. DEM
            perturb_bands (dict[str, dict[str, float]], optional): perturb bands with additive gaussian noise. Dictionary defining modalities and bands with weights for perturbation.
            normalize (bool): When False (and no custom ``aug`` is passed), skip z-score
                normalization entirely — ``self.aug`` is an identity. Use for backbones that
                apply their own pretraining normalization internally (e.g. OlmoEarth).
            db_scale_bands (dict[str, list[str]], optional): bands to convert from linear power
                to decibels (``10 * log10(clip(x, 1e-10))``) at load time, e.g.
                ``{"S1RTC": ["VV", "VH"]}``. Passed through to the dataset class. Do NOT use for
                backbones that already convert S1 to dB internally (e.g. OlmoEarth with
                ``apply_pretraining_normalization=True``).
            nodata_value (float | dict[str, float], optional): on-disk nodata sentinel to
                detect in the raw batch, e.g. ``-999``. A scalar applies to every modality;
                a ``{modality: value}`` dict (e.g. ``{"S2L2A": -999}``) masks only its listed
                modalities. Must be provided together with ``set_nodata``. Detected pixels
                are remapped to ``set_nodata`` AFTER normalization, so with
                ``nodata_value=-999, set_nodata=0`` the model receives exactly ``0`` at
                nodata positions (not a normalized sentinel).
            set_nodata (float | dict[str, float], optional): value the model should receive
                at nodata positions, e.g. ``0``. A scalar, or a ``{modality: value}`` dict
                covering every modality masked by ``nodata_value``. Must be provided
                together with ``nodata_value``.
            **kwargs: Additional keyword arguments.
        """
        if isinstance(dataset_class, str):
            module_path, _, class_name = dataset_class.rpartition(".")
            dataset_class = getattr(importlib.import_module(module_path), class_name)

        super().__init__(dataset_class, batch_size, num_workers, **kwargs)

        self.data_root = data_root
        if bands is None:
            bands = dataset_class.all_band_names
        self.bands = bands
        self.modalities = list(self.bands.keys())
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.concat_bands = concat_bands
        self.repeat_bands = repeat_bands
        self.perturb_bands = perturb_bands
        self.normalize = normalize
        self.db_scale_bands = db_scale_bands
        self.nodata_value = nodata_value
        self.set_nodata = set_nodata

        if (nodata_value is None) != (set_nodata is None):
            raise ValueError(
                "nodata_value and set_nodata must be provided together: the on-disk "
                "sentinel (nodata_value) and the value fed to the model (set_nodata) "
                "are only meaningful as a pair."
            )
        if nodata_value is not None:
            for param_name, param in (("nodata_value", nodata_value), ("set_nodata", set_nodata)):
                if isinstance(param, dict):
                    unknown = sorted(set(param) - set(self.modalities))
                    if unknown:
                        raise ValueError(
                            f"{param_name} references unknown modalities {unknown}; "
                            f"known modalities: {self.modalities}"
                        )
            masked_modalities = (
                list(nodata_value) if isinstance(nodata_value, dict) else list(self.modalities)
            )
            if isinstance(set_nodata, dict):
                missing = sorted(set(masked_modalities) - set(set_nodata))
                if missing:
                    raise ValueError(
                        f"set_nodata is missing target values for masked modalities "
                        f"{missing}: a scalar nodata_value masks every modality, a dict "
                        "masks its keys, and set_nodata must cover all of them."
                    )
            overlapping = sorted(
                modality
                for modality in masked_modalities
                if modality in (db_scale_bands or {}) or modality in (perturb_bands or {})
            )
            if overlapping:
                logger.warning(
                    f"Modalities {overlapping} are masked by nodata_value but also listed in "
                    "db_scale_bands or perturb_bands. Those per-sample steps run BEFORE "
                    "batch-wise nodata detection and corrupt the sentinel (dB conversion "
                    "clips it, perturbation adds noise), so nodata pixels may be silently "
                    "missed in those modalities."
                )

        # Resolve per-modality/band stats, first match wins:
        # explicit means/stds args -> lowercase means/stds class attrs ->
        # uppercase MEANS/STDS class attrs -> default (mean 0.0, std 1.0).
        means = means or {}
        stds = stds or {}
        means_sources = (
            means,
            getattr(dataset_class, "means", None) or {},
            getattr(dataset_class, "MEANS", None) or {},
        )
        stds_sources = (
            stds,
            getattr(dataset_class, "stds", None) or {},
            getattr(dataset_class, "STDS", None) or {},
        )
        self.means = {}
        self.stds = {}
        for modality in self.modalities:
            self.means[modality] = [
                self._resolve_stat(modality, band, means_sources, default=0.0)
                for band in self.bands[modality]
            ]
            self.stds[modality] = [
                self._resolve_stat(modality, band, stds_sources, default=1.0)
                for band in self.bands[modality]
            ]
            if (
                normalize
                and aug is None
                and modality not in means
                and modality not in stds
                and all(mean == 0.0 for mean in self.means[modality])
                and all(std == 1.0 for std in self.stds[modality])
            ):
                logger.warning(
                    f"No normalization statistics found for modality '{modality}' on dataset "
                    f"class '{dataset_class.__name__}': every band resolved to mean 0.0 / "
                    "std 1.0, so normalization will be an identity (no-op) and the model will "
                    "receive raw pixel values. Pass explicit means/stds to GELOSDataModule or "
                    "define means/stds (or MEANS/STDS) on the dataset class."
                )

        self.transform = wrap_in_compose_is_list(transform)
        if aug is not None:
            self.aug = aug
        elif not normalize:
            self.aug = IdentityAug()
        elif len(self.bands.keys()) == 1:
            self.aug = Normalize(self.means[self.modalities[0]], self.stds[self.modalities[0]])
        else:
            self.aug = MultimodalNormalize(self.means, self.stds)
        if self.nodata_value is not None:
            self.aug = NoDataRemap(self.aug, self.nodata_value, self.set_nodata)
        self.collate_fn = collate_samples

    @staticmethod
    def _resolve_stat(
        modality: str,
        band: str,
        sources: tuple[dict[str, dict[str, float]], ...],
        default: float,
    ) -> float:
        """Return the first stat found for (modality, band) across sources, else default."""
        for source in sources:
            value = source.get(modality, {}).get(band)
            if value is not None:
                return value
        return default

    def setup(self, stage: str = "predict") -> None:
        """
        Set up GELOS dataset
        """
        if stage != "predict":
            raise ValueError("GELOS dataset is for prediction only")
        # Only forward db_scale_bands when set, so dataset subclasses that predate
        # the parameter keep working as long as the feature is unused.
        extra_kwargs = {}
        if self.db_scale_bands is not None:
            extra_kwargs["db_scale_bands"] = self.db_scale_bands
        self.dataset = self.dataset_class(
            data_root=self.data_root,
            bands=self.bands,
            transform=self.transform,
            concat_bands=self.concat_bands,
            repeat_bands=self.repeat_bands,
            perturb_bands=self.perturb_bands,
            **extra_kwargs,
        )

    def _dataloader_factory(self, stage: str = "predict"):
        if stage != "predict":
            raise ValueError("GELOS is for prediction only")
        dataset = self.dataset
        batch_size = self.batch_size
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
