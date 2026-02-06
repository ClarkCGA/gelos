from pathlib import Path
from typing import Any, List

import albumentations as A
from kornia.augmentation import AugmentationSequential
from terratorch.datamodules.generic_multimodal_data_module import (
    MultimodalNormalize,
    collate_samples,
    wrap_in_compose_is_list,
)
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
import typer
import torch

from gelos.gelosdataset import GELOSDataSet, GELOSCropDataSet

app = typer.Typer()


# instantiate GELOS datamodule class
class GELOSDataModule(NonGeoDataModule):
    """
    This is the datamodule for Geospatial Exploration of Latent Observation Space (GELOS)
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_root: str | Path,
        means: dict[str, dict[str, float]] = None,
        stds: dict[str, dict[str, float]] = None,
        bands: dict[str, List[str]] = GELOSDataSet.all_band_names,
        transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] = None,
        perturb_bands: dict[str, dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the DataModule for GELOS.

        Args:
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for data loading.
            data_root (str | Path): Root directory for dataset.
            means: (dict[str, dict[str, float]]): Dictionary defining modalities and bands with mean values
            stds: (dict[str, dict[str, float]]): Dictionary defining modalities and bands with std values
            bands: (dict[str, List[str]], optional): Dictionary with format "modality" : List['band_a', 'band_b']
            transform (A.Compose, optional): Transforms for data, defaults to ToTensorV2.
            aug (AugmentationSequential, optional): Augmentation or normalization to apply. Defaults to normalization if not provided.
            concat_bands (bool): Whether to concat all sensors into one 'image' tensor or keep separate
            repeat_bands (dict[str, int], optional): repeat bands when loading from disc, intended to repeat single time step modalities e.g. DEM
            perturb_bands (dict[str, dict[str, float]], optional): perturb bands with additive gaussian noise. Dictionary defining modalities and bands with weights for perturbation.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(GELOSDataSet, batch_size, num_workers, **kwargs)

        self.data_root = data_root
        self.bands = bands
        self.modalities = list(self.bands.keys())
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.concat_bands = concat_bands
        self.repeat_bands = repeat_bands
        self.perturb_bands = perturb_bands
        self.means = means or {}  # init empty stats dict if none
        self.stds = means or {}
        for modality in self.modalities:
            # if a statistics are not passed, default to 0 for mean and 1 for std for all unprovided bands
            self.means[modality] = [
                self.means.get(modality, {}).get(band, 0.0) for band in self.bands[modality]
            ]
            self.stds[modality] = [
                self.stds.get(modality, {}).get(band, 1.0) for band in self.bands[modality]
            ]
        self.transform = wrap_in_compose_is_list(transform)
        if len(self.bands.keys()) == 1:
            self.aug = (
                Normalize(self.means[self.modalities[0]], self.stds[self.modalities[0]])
                if aug is None
                else aug
            )
        else:
            self.aug = MultimodalNormalize(self.means, self.stds) if aug is None else aug
        self.collate_fn = collate_samples

    def setup(self, stage: str = "predict") -> None:
        """
        Set up GELOS dataset
        """
        if stage != "predict":
            raise ValueError("GELOS dataset is for prediction only")
        self.dataset = self.dataset_class(
            data_root=self.data_root,
            bands=self.bands,
            means=self.means,
            stds=self.stds,
            transform=self.transform,
            concat_bands=self.concat_bands,
            repeat_bands=self.repeat_bands,
            perturb_bands=self.perturb_bands,
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
            # drop_last=self.drop_last,
        )

############ ---- GELOS CROP DATAMODULE ---- ############

MEANS =  {
     "S2L2A": {
        "BLUE": 253.34048461914062,
        "GREEN": 387.3401794433594,
        "RED": 421.0535888671875,
        "RED_EDGE_1": 599.3675537109375,
        "RED_EDGE_2": 1093.0345458984375,
        "RED_EDGE_3": 1272.2730712890625,
        "NIR_BROAD": 1307.6336669921875,
        "NIR_NARROW": 1389.8450927734375,
        "SWIR_1": 1097.331298828125,
        "SWIR_2": 712.2874145507812
    }

}

STDS =  {
    "S2L2A": {
        "BLUE": 318.81640625,
        "GREEN": 468.80877685546875,
        "RED": 560.62548828125,
        "RED_EDGE_1": 713.4432373046875,
        "RED_EDGE_2": 1260.6995849609375,
        "RED_EDGE_3": 1485.360595703125,
        "NIR_BROAD": 1529.9556884765625,
        "NIR_NARROW": 1615.656494140625,
        "SWIR_1": 1289.3074951171875,
        "SWIR_2": 896.275146484375
    }
}

# instantiate GELOS Crop datamodule class
class GELOSCropDataModule(NonGeoDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        data_root: str | Path,
        bands: dict[str] = GELOSCropDataSet.all_band_names,
        transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        concat_bands: bool = False,
        target_size: int = None,
        pin_memory: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the DataModule.

        Args:
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for data loading.
            data_root (str | Path): Root directory for dataset.
            bands: (Dict[str, List[str]], optional): Dictionary with format "modality" : List['band_a', 'band_b']
            transform (A.Compose, optional): Transforms for data, defaults to ToTensorV2.
            aug (AugmentationSequential, optional): Augmentation or normalization to apply. Defaults to normalization if not provided.
            concat_bands (bool): Whether to concat all sensors into one 'image' tensor or keep separate
            **kwargs: Additional keyword arguments.
        """
        super().__init__(GELOSCropDataSet, batch_size, num_workers, **kwargs)

        self.data_root = data_root
        self.bands = bands
        self.modalities = list(self.bands.keys())
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.concat_bands = concat_bands
        self.target_size = target_size
        self.means = {}
        self.stds = {}
        for modality in self.modalities:
            self.means[modality] = [MEANS[modality][band] for band in self.bands[modality]]
            self.stds[modality] = [STDS[modality][band] for band in self.bands[modality]]
        self.transform = wrap_in_compose_is_list(transform)
        if len(self.bands.keys()) == 1:
            self.aug = (
                Normalize(self.means[self.modalities[0]], self.stds[self.modalities[0]])
                if aug is None
                else aug
            )
        else:
            self.aug = (MultimodalNormalize(self.means, self.stds) if aug is None else aug)
        self.collate_fn = collate_samples
        self.pin_memory = pin_memory

    def setup(self, stage: str = "predict") -> None:
        """
        Set up dataset
        """
        if stage != "predict":
            raise ValueError("dataset is for prediction only")
        self.dataset = self.dataset_class(
            data_root=self.data_root,
            bands=self.bands,
            transform=self.transform,
            concat_bands=self.concat_bands,
            target_size=self.target_size
        )

    def _dataloader_factory(self, stage: str = "predict"):
        if stage != "predict":
            raise ValueError("dataset is for prediction only")
        dataset = self.dataset
        batch_size = self.batch_size
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            # drop_last=self.drop_last,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
        """
        Ensure model-expected channel count after normalization.

        The datamodule applies `Normalize` during `on_after_batch_transfer`.
        Duplicate or trim the channel dimension here (after normalization)
        so `means`/`stds` broadcasting works correctly.
        """
        # let parent apply augmentations/normalization first
        try:
            batch = super().on_after_batch_transfer(batch, dataloader_idx)
        except Exception:
            # If parent has no hook, ignore
            pass

        if "image" not in batch:
            return batch

        img = batch["image"]
        # infer channel axis: commonly B x C x T x H x W
        if img.ndim == 5:
            channel_axis = 1
        elif img.ndim == 4:
            # either B x C x H x W or B x T x C x H
            if img.shape[1] in (6, 12):
                channel_axis = 1
            else:
                channel_axis = 2
        else:
            channel_axis = 1

        current = img.shape[channel_axis]
        target = 12
        if current != target:
            if current > target:
                slices = [slice(None)] * img.ndim
                slices[channel_axis] = slice(0, target)
                img = img[tuple(slices)]
            else:
                reps = -(-target // current)
                img = torch.cat([img] * reps, dim=channel_axis)
                slices = [slice(None)] * img.ndim
                slices[channel_axis] = slice(0, target)
                img = img[tuple(slices)]
            batch["image"] = img

        return batch