from pathlib import Path
from typing import Any

import albumentations as A
from kornia.augmentation import AugmentationSequential
from terratorch.datamodules.generic_multimodal_data_module import (
    MultimodalNormalize,
    wrap_in_compose_is_list,
)
from terratorch.datamodules.generic_pixel_wise_data_module import Normalize
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule
import typer

from gelos.gelosdataset import GELOSDataSet

app = typer.Typer()

MEANS =  {
    "sentinel_1": {
        "VV": 0.14450763165950775,
        "VH": 0.029020152986049652
    },
    "sentinel_2": {
        "COASTAL_AEROSOL": 1852.9951171875,
        "BLUE": 2046.738525390625,
        "GREEN": 2346.2802734375,
        "RED": 2593.03857421875,
        "RED_EDGE_1": 2900.828857421875,
        "RED_EDGE_2": 3365.597900390625,
        "RED_EDGE_3": 3576.141357421875,
        "NIR_BROAD": 3657.3046875,
        "NIR_NARROW": 3703.0908203125,
        "WATER_VAPOR": 3709.93359375,
        "SWIR_1": 3543.164794921875,
        "SWIR_2": 3048.239990234375
    },
    "landsat": {
        "coastal": 0.08165209740400314,
        "blue": 0.09596806019544601,
        "green": 0.1315794140100479,
        "red": 0.1531316637992859,
        "nir08": 0.2621993124485016,
        "swir16": 0.23768098652362823,
        "swir22": 0.18106447160243988
    },
    "dem": {
        "dem": 642.7003173828125
    }
}

STDS =  {
    "sentinel_1": {
        "VV": 2.600670576095581,
        "VH": 0.26772621273994446
    },
    "sentinel_2": {
        "COASTAL_AEROSOL": 1201.80078125,
        "BLUE": 1267.075927734375,
        "GREEN": 1316.0233154296875,
        "RED": 1520.836669921875,
        "RED_EDGE_1": 1518.5592041015625,
        "RED_EDGE_2": 1419.7735595703125,
        "RED_EDGE_3": 1442.878662109375,
        "NIR_BROAD": 1476.5181884765625,
        "NIR_NARROW": 1437.5333251953125,
        "WATER_VAPOR": 1440.673095703125,
        "SWIR_1": 1588.948974609375,
        "SWIR_2": 1524.4881591796875
    },
    "landsat": {
        "coastal": 0.15966829657554626,
        "blue": 0.16089804470539093,
        "green": 0.15540584921836853,
        "red": 0.1680557280778885,
        "nir08": 0.15390564501285553,
        "swir16": 0.14630644023418427,
        "swir22": 0.1311405450105667
    },
    "dem": {
        "dem": 783.0748291015625
    }
}

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
        bands: dict[str] = GELOSDataSet.all_band_names,
        transform: A.Compose | None | list[A.BasicTransform] = None,
        aug: AugmentationSequential = None,
        concat_bands: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the DataModule for GELOS.

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
        super().__init__(GELOSDataSet, batch_size, num_workers, **kwargs)

        self.data_root = data_root
        self.bands = bands
        self.modalities = list(self.bands.keys())
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.concat_bands = concat_bands
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
            MultimodalNormalize(self.means, self.stds) if aug is None else aug

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
            concat_bands=self.concat_bands
            
        )

    def _dataloader_factory(self, stage: str = "predict"):
        if stage != "predict":
            raise ValueError("GELOS dataset is for prediction only")
        dataset = self.dataset
        batch_size = self.batch_size
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=self.collate_fn,
            # drop_last=self.drop_last,
        )
