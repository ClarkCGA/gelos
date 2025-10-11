from pathlib import Path
from typing import List

import albumentations as A
import geopandas as gpd
import numpy as np
import rioxarray as rxr
from terratorch.datasets.transforms import MultimodalTransforms
from terratorch.datasets.utils import (
    default_transform,
)
import torch
from torchgeo.datasets import GeoDataset


class MultimodalToTensor:
    def __init__(self, modalities):
        self.modalities = modalities

    def __call__(self, d):
        new_dict = {}
        for k, v in d.items():
            if not isinstance(v, np.ndarray):
                new_dict[k] = v
            else:
                if (
                    k in self.modalities and len(v.shape) >= 3
                ):  # Assuming raster modalities with 3+ dimensions
                    if len(v.shape) <= 4:
                        v = np.moveaxis(v, -1, 0)  # C, H, W or C, T, H, W
                    elif len(v.shape) == 5:
                        v = np.moveaxis(v, -1, 1)  # B, C, T, H, W
                    else:
                        raise ValueError(f"Unexpected shape for {k}: {v.shape}")
                new_dict[k] = torch.from_numpy(v)
        return new_dict


class GELOSDataSet(GeoDataset):
    """
    Dataset intended for embedding extraction and exploration.
    Contains Sentinel 1 and 2 data, DEM, and Landsat 8 and 9 data.

    Dataset Format:

    .tif files for Sentinel 1, Sentinel 2, DEM, and Landsat 8 and 9 data
    .csv chip tracker with chip-level land cover classification

    Dataset Features:
    TBD Dataset Size
    4 time steps for each land cover chip
    """

    S2_BAND_NAMES = [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2",
        # "WATER_VAPOR",
        "CIRRUS",
        "THEMRAL_INFRARED_1",
    ]
    S1_BAND_NAMES = [
        "VV",
        "VH",
        "ASC_VV",
        "ASC_VH",
        "DSC_VV",
        "DSC_VH",
        "VV_VH",
    ]
    LANDSAT_BAND_NAMES = [
        "coastal",  # Coastal/Aerosol (Band 1)
        "blue",  # Blue (Band 2)
        "green",  # Green (Band 3)
        "red",  # Red (Band 4)
        "nir08",  # Near Infrared (NIR, Band 5)
        "swir16",  # Shortwave Infrared 1 (SWIR1, Band 6)
        "swir22",  # Shortwave Infrared 2 (SWIR2, Band 7)
    ]
    DEM_BAND_NAMES = ["dem"]
    all_band_names = {
        "sentinel_1": S1_BAND_NAMES,
        "sentinel_2": S2_BAND_NAMES,
        "landsat": LANDSAT_BAND_NAMES,
        "dem": DEM_BAND_NAMES,
    }

    rgb_bands = {
        "sentinel_1": [],
        "sentinel_2": ["RED", "GREEN", "BLUE"],
        "landsat": ["red", "green", "blue"],
        "dem": [],
    }

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(
        self,
        data_root: str | Path,
        bands: dict[str, List[str]] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        concat_bands: bool = False,
    ) -> None:
        """
        Initializes an instance of GELOS.

        Args:
        data_root (str | Path): root directory where the dataset can be found
        bands: (Dict[str, List[str]], optional): Dictionary with format "modality" : List['band_a', 'band_b']
        transform (A.compose, optional): transform to apply. Defaults to ToTensorV2.
        """
        self.data_root = Path(data_root)
        self.bands = bands
        self.concat_bands = concat_bands

        assert set(self.bands.keys()).issubset(set(self.all_band_names.keys())), (
            f"Please choose a subset of valid sensors: {self.all_band_names.keys()}"
        )

        self.band_indices = {
            sens: [self.all_band_names[sens].index(band) for band in self.bands[sens]]
            for sens in self.bands.keys()
        }

        self.gdf = gpd.read_file(self.data_root / "cleaned_df.geojson")
        self.gdf = self._process_metadata_df()

        # Adjust transforms based on the number of sensors
        if len(self.bands.keys()) == 1:
            self.transform = transform if transform else default_transform
        elif transform is None:
            self.transform = MultimodalToTensor(self.bands.keys())
        else:
            transform = {
                s: transform[s] if s in transform else default_transform for s in self.bands.keys()
            }
            self.transform = MultimodalTransforms(transform, shared=False)

    def __len__(self) -> int:
        return len(self.gdf)

    def __getitem__(self, index: int) -> dict:
        sample_row = self.gdf.iloc[index]

        output = {}

        for sensor in self.bands.keys():
            sensor_filepaths = sample_row[f"{sensor}_paths"]
            image = self._load_sensor_images(sensor_filepaths)
            output[sensor] = image.astype(np.float32)

        if len(output.keys()) == 1:
            # Rename the single sensor key to "image"
            sensor = list(output.keys())[0]
            output["image"] = output.pop(sensor)
        if self.transform:
            output = self.transform(output)

        if self.concat_bands:
            # Concatenate bands of all image modalities
            data = [output.pop(m) for m in self.bands.keys() if m in output]
            output["image"] = torch.cat(data, dim=1 if self.data_with_sample_dim else 0)
        else:
            # Tasks expect data to be stored in "image", moving modalities to image dict
            output["image"] = {m: output.pop(m) for m in self.bands.keys() if m in output}

        output["filename"] = sample_row["chip_id"]

        return output

    def _load_file(self, path) -> np.array:
        data = rxr.open_rasterio(path, masked=True).to_numpy()

        return data

    def _load_sensor_images(self, sensor_filepaths: List[Path]) -> np.array:
        sensor_images = [self._load_file(path) for path in sensor_filepaths]
        sensor_image = np.stack(sensor_images, axis=0)

        return sensor_image

    def _process_metadata_df(self) -> gpd.GeoDataFrame:
        # for each modality, construct file paths
        # if the modality has multiple dates, construct them from the dates column
        # otherwise, for single time step variables, construct from chip id
        def _construct_file_paths(row, modality: str, data_root: Path) -> List[Path]:
            date_list = row[f"{modality}_dates"].split(",")
            chip_id = row["chip_id"]
            path_list = [data_root / f"{modality}_{chip_id:06}_{date}.tif" for date in date_list]
            return path_list

        def _construct_dem_path(row, data_root: Path) -> List[Path]:
            chip_id = row["chip_id"]
            dem_list = [data_root / f"dem_{chip_id:06}.tif"]
            return dem_list

        for modality in self.bands.keys():
            if modality == "dem":
                self.gdf["dem_paths"] = self.gdf.apply(
                    _construct_dem_path, data_root=self.data_root, axis=1
                )
                continue

            self.gdf[f"{modality}_paths"] = self.gdf.apply(
                _construct_file_paths, modality=modality, data_root=self.data_root, axis=1
            )
        return self.gdf
