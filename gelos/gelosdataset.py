from pathlib import Path
import pdb
from typing import List

import albumentations as A
import geopandas as gpd
import numpy as np
import rioxarray as rxr
from terratorch.datasets.transforms import MultimodalTransforms
import torch.nn.functional as F

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
    Dataset intended for embedding extraction and exploration.
    Contains Sentinel 1 and 2 data, DEM, and Landsat 8 and 9 data.

    Dataset Format:

    .tif files for Sentinel 1, Sentinel 2, DEM, and Landsat 8 and 9 data
    .csv chip tracker with chip-level land cover classification

    Dataset Features:
    TBD Dataset Size
    4 time steps for each land cover chip
    """

    S2RTC_BAND_NAMES = [
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "WATER_VAPOR",
        "SWIR_1",
        "SWIR_2",
    ]
    S1RTC_BAND_NAMES = [
        "VV",
        "VH",
        # TODO 2025-10-17 GELOS v0.40 does not differentiate ASC and DSC S1 passes
        # "ASC_VV",
        # "ASC_VH",
        # "DSC_VV",
        # "DSC_VH",
        # "VV_VH",
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
    DEM_BAND_NAMES = ["DEM"]
    all_band_names = {
        "S1RTC": S1RTC_BAND_NAMES,
        "S2L2A": S2RTC_BAND_NAMES,
        "landsat": LANDSAT_BAND_NAMES,
        "DEM": DEM_BAND_NAMES,
    }

    rgb_bands = {
        "S1RTC": [],
        "S2L2A": ["RED", "GREEN", "BLUE"],
        "landsat": ["red", "green", "blue"],
        "DEM": [],
    }

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    def __init__(
        self,
        data_root: str | Path,
        bands: dict[str, List[str]] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] = None,
        target_size: int = None
    ) -> None:
        """
        Initializes an instance of GELOS.

        Args:
        data_root (str | Path): root directory where the dataset can be found
        bands: (Dict[str, List[str]], optional): Dictionary with format "modality" : List['band_a', 'band_b']
        transform (A.compose, optional): transform to apply. Defaults to ToTensorV2.
        concat_bands (bool, optional): concatenate all modalities into the channel dimension
        repeat_bands (dict[str, int], optional): repeat bands when loading from disc, intended to repeat single time step modalities e.g. DEM
        """
        self.data_root = Path(data_root)
        self.bands = bands
        self.concat_bands = concat_bands
        self.repeat_bands = repeat_bands
        self.target_size = target_size
        self.modality_rename_dict = {
            "S2L2A": "sentinel_2",
            "S1RTC": "sentinel_1",
            "DEM": "dem"
        }

        assert set(self.bands.keys()).issubset(set(self.all_band_names.keys())), (
            f"Please choose a subset of valid sensors: {self.all_band_names.keys()}"
        )

        self.band_indices = {
            sens: [self.all_band_names[sens].index(band) for band in self.bands[sens]]
            for sens in self.bands.keys()
        }

        self.gdf = gpd.read_file(self.data_root / "gelos_chip_tracker.geojson")
        self.gdf = self._process_metadata_df()

        # Adjust transforms based on the number of sensors
        if transform is None:
            self.transform = MultimodalToTensor(self.bands.keys())
        else:
            transform = {
                s: transform for s in self.bands.keys()
            }
            self.transform = MultimodalTransforms(transform, shared=False)
    def __len__(self) -> int:
        return len(self.gdf)

    def __getitem__(self, index: int) -> dict:
        sample_row = self.gdf.iloc[index]

        output = {}
        
        for sensor in self.bands.keys():
            sensor_filepaths = sample_row[f"{sensor}_paths"]
            image = self._load_sensor_images(sensor_filepaths, sensor)
            output[sensor] = image.astype(np.float32)
            
        if self.repeat_bands:
            for sensor, repeats in self.repeat_bands.items():
                output[sensor] = np.tile(output[sensor], (1, repeats, 1, 1))


        if self.target_size:
            h = self.target_size
            w = self.target_size
            for sensor in output.keys():
                # output[sensor] shape: [C, T, H, W] -> reshape to [C*T, H, W]
                original_shape = output[sensor].shape
                h_sensor, w_sensor = original_shape[-2:]
                if h == h_sensor and w == w_sensor:
                    continue
                c, t = original_shape[:2]
                reshaped = output[sensor].reshape(c * t, *original_shape[2:])
                # Interpolate
                interpolated = F.interpolate(
                    reshaped.unsqueeze(0), 
                    size=(h, w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                # Reshape back to [C, T, H, W]
                output[sensor] = interpolated.reshape(c, t, h, w)
        for k, v in output.items():
            print(k, v.shape)

        if self.transform:
            output = self.transform(output)
        for k, v in output.items():
            print(k, v.shape)
        

        if len(self.bands.keys()) == 1:
            # Rename the single sensor key to "image"
            sensor = list(output.keys())[0]
            output["image"] = output.pop(sensor)
        elif self.concat_bands:
            # Concatenate bands of all image modalities
            data = [output.pop(m) for m in self.bands.keys() if m in output]
            output["image"] = torch.cat(data, dim=1) # concatenate into channel dimension
        else:
            # Tasks expect data to be stored in "image", moving modalities to image dict
            output["image"] = {m: output.pop(m) for m in self.bands.keys() if m in output}

        # create id with timestep as output filenames
        id = str(sample_row["id"]).zfill(6)
        output["filename"] = np.array(id, dtype=str)
        output["file_id"] = sample_row["id"]


        return output

    def _load_file(self, path, band_indices: List[int]) -> np.array:
        data = rxr.open_rasterio(path, masked=True).to_numpy()
        try:
            return data[band_indices, :, :]
        except:
            return data

    def _load_sensor_images(self, sensor_filepaths: List[Path], sensor: str) -> np.array:
        band_indices = self.band_indices[sensor]
        sensor_images = [self._load_file(path, band_indices) for path in sensor_filepaths]
        sensor_image = np.stack(sensor_images, axis=0)

        return sensor_image


    def _process_metadata_df(self) -> gpd.GeoDataFrame:

        # for each modality, construct file paths
        def _construct_file_paths(row, modality: str, data_root: Path) -> List[Path]:
            modality = self.modality_rename_dict.get(modality, modality)
            date_list = row[f"{modality}_dates"].split(",")
            id = row["id"]
            path_list = [data_root / f"{modality}_{id:06}_{date}.tif" for date in date_list]
            return path_list

        def _construct_DEM_path(row, data_root: Path) -> List[Path]:
            id = row["id"]
            DEM_list = [data_root / f"dem_{id:06}.tif"]
            return DEM_list

        for modality in self.bands.keys():

            if modality == "DEM":
                self.gdf["DEM_paths"] = self.gdf.apply(
                    _construct_DEM_path, data_root=self.data_root, axis=1
                )
                continue

            self.gdf[f"{modality}_paths"] = self.gdf.apply(
                _construct_file_paths, modality=modality, data_root=self.data_root, axis=1
            )
        return self.gdf
