from pathlib import Path
from typing import Any, List
import albumentations as A
from kornia.augmentation import AugmentationSequential
from terratorch.datamodules.generic_multimodal_data_module import (
    MultimodalNormalize,
    wrap_in_compose_is_list,
    collate_samples
)
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
import typer

from src.data.dataset import GELOSDataSet

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
        metadata_path: str | Path,
        means: dict[str, dict[str, float]],
        stds: dict[str, dict[str, float]],
        bands: dict[str, List[str]] | None = None,
        transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] = None,
        perturb_bands: dict[str, dict[str, float]] = None,
        fire: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the DataModule for GELOS.

        Args:
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for data loading.
            metadata_path (str | Path): Path to the metadata file.
            means: (dict[str, dict[str, float]]): Dictionary defining modalities and bands with mean values
            stds: (dict[str, dict[str, float]]): Dictionary defining modalities and bands with std values 
            bands: (dict[str, List[str]], optional): Dictionary with format "modality" : List['band_a', 'band_b']. If None, defaults are chosen based on `fire` flag.
            transform (A.Compose, optional): Transforms for data, defaults to ToTensorV2.
            aug (AugmentationSequential, optional): Augmentation or normalization to apply. Defaults to normalization if not provided.
            concat_bands (bool): Whether to concat all sensors into one 'image' tensor or keep separate
            repeat_bands (dict[str, int], optional): repeat bands when loading from disc, intended to repeat single time step modalities e.g. DEM
            perturb_bands (dict[str, dict[str, float]], optional): perturb bands with additive gaussian noise. Dictionary defining modalities and bands with weights for perturbation.
            fire (bool): If True, use fire band set as default when `bands` is None; otherwise use landcover band set.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(GELOSDataSet, batch_size, num_workers, **kwargs)

        self.metadata_path = metadata_path
        # If bands not explicitly provided, choose default based on fire flag
        if bands is None:
            self.bands = GELOSDataSet.fire_band_names if fire else GELOSDataSet.lc_band_names
        else:
            self.bands = bands
        self.fire = fire
        self.modalities = list(self.bands.keys())
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.concat_bands = concat_bands
        self.repeat_bands = repeat_bands
        self.perturb_bands = perturb_bands
        self.means = {}
        self.stds = {}
        
        print("Setting up GELOS DataModule with modalities:", self.modalities)
        for modality in self.modalities:
            if modality == "mask" or modality == "landcover":
                continue
            self.means[modality] = [means[modality][band] for band in self.bands[modality]]
            self.stds[modality] = [stds[modality][band] for band in self.bands[modality]]
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

    def setup(self, stage: str = "predict") -> None:
        """
        Set up GELOS dataset
        """
        if stage != "predict":
            raise ValueError("GELOS dataset is for prediction only")
        self.dataset = self.dataset_class(
            metadata_path=self.metadata_path,
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
