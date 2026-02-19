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


class GELOSDataModule(NonGeoDataModule):
    """
    This is the datamodule for Geospatial Exploration of Latent Observation Space (GELOS)
    """

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int,
        num_workers: int,
        dataset_class: type,
        means: dict[str, dict[str, float]] | None = None,
        stds: dict[str, dict[str, float]] | None = None,
        bands: dict[str, List[str]] | None = None,
        transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential | None = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] | None = None,
        perturb_bands: dict[str, dict[str, float]] | None = None,
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
            **kwargs: Additional keyword arguments.
        """
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
        
        # handle passing stats with missing values, check against dataset class
        # if none are found, mean defaults to 0 and std defaults to 1
        means = means or {}
        stds = stds or {}
        class_means = getattr(dataset_class, 'means', {})
        class_stds = getattr(dataset_class, 'stds', {})
        self.means = {}
        self.stds = {}
        for modality in self.modalities:
            self.means[modality] = [
                means.get(modality, class_means.get(modality, {})).get(band, 0.0)
                for band in self.bands[modality]
            ]
            self.stds[modality] = [
                stds.get(modality, class_stds.get(modality, {})).get(band, 1.0)
                for band in self.bands[modality]
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
        )
