from pathlib import Path
import pdb
from typing import List
import matplotlib.pyplot as plt
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

def scale(array: np.array):
    """Scales a numpy array to 0-1 according to maximum value."""
    if array.max() > 1.0:
        array_scaled = array / 4000
    else:
        array_scaled = array * 5

    array_norm = np.clip(array_scaled, 0, 1)
    return array_norm


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
        means: dict[str, dict[str, float]] = None,
        stds: dict[str, dict[str, float]] = None,
        transform: A.Compose | None = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] = None,
        perturb_bands: dict[str, List[str]] = None,
        perturb_alpha: float = 1
        
    ) -> None:
        """
        Initializes an instance of GELOS.

        Args:
        data_root (str | Path): root directory where the dataset can be found
        means (dict[str, dict[str, float]]): Dataset means by sensor and band for scaling perturbations 
        stds (dict[str, dict[str, float]]): Dataset standard deviations by sensor and band for scaling perturbations 
        bands: (Dict[str, List[str]], optional): Dictionary with format "modality" : List['band_a', 'band_b']
        transform (A.compose, optional): transform to apply. Defaults to ToTensorV2.
        concat_bands (bool, optional): concatenate all modalities into the channel dimension
        repeat_bands (dict[str, int], optional): repeat bands when loading from disc, intended to repeat single time step modalities e.g. DEM
        perturb_bands (dict[str, List[str]], optional): perturb bands with additive gaussian noise. Dictionary defining modalities and bands for perturbation.
        perturb_alpha (float, optional): relative weight given to source data vs perturbation noise. 0 signifies all noise, 1 signifies equal weights
        """
        self.data_root = Path(data_root)
        self.bands = bands
        self.means = means
        self.stds = stds
        self.concat_bands = concat_bands
        self.repeat_bands = repeat_bands
        self.perturb_bands = perturb_bands
        self.perturb_alpha = perturb_alpha
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
        
        if self.perturb_bands:
            self.perturb_band_indices = {
                sens: [self.all_band_names[sens].index(band) for band in self.perturb_bands[sens]]
                for sens in self.perturb_bands.keys()
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
            # image shape: [T, H, W, C]
            
        if self.repeat_bands:
            for sensor, repeats in self.repeat_bands.items():
               output[sensor] = np.tile(output[sensor], (repeats, 1, 1, 1)) # repeat along T dimension
        
        # Add or replace individual bands with Gaussian noise scaled to each band's dataset-wide mean and std
        if self.perturb_bands:
            for sensor, perturb_band_list in self.perturb_bands.items():
                band_indices = [self.bands[sensor].index(b) for b in perturb_band_list]
                output = self._perturb_bands(output, sensor, band_indices)

        if self.transform:
            output = self.transform(output)

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

    def _perturb_bands(self, output, sensor, band_indices):
        # perturb given bands of one sensor output
        # get mean and std of given band of sensor
        for band_index in band_indices:
            loc = self.means[sensor][band_index]
            scale = self.stds[sensor][band_index]
            
            # get size of noise tensor to generate
            size = output[sensor][:, :, :, band_index].shape
            noise = np.random.normal(loc=loc, scale=scale, size=size)
            combined_noise = (noise + output[sensor][:, :, :, band_index] * self.perturb_alpha) / (1 + self.perturb_alpha)
            # combine noise back into output
            output[sensor][:, :, :, band_index] = combined_noise
        return output

    def _load_file(self, path, band_indices: List[int]) -> np.array:
        data = rxr.open_rasterio(path, masked=True).to_numpy()
        return data[band_indices, :, :].transpose(1,2,0) # [H, W, C]

    def _load_sensor_images(self, sensor_filepaths: List[Path], sensor: str) -> np.array:
        band_indices = self.band_indices[sensor]
        sensor_images = [self._load_file(path, band_indices) for path in sensor_filepaths]
        sensor_image = np.stack(sensor_images, axis=0) # stack into [T, H, W, C]

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

    def plot(
        self,
        sample: dict[str, torch.Tensor],
        vis_bands: dict[str, dict[str, int]] = rgb_bands,
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> plt.Figure:

        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            bands: bands from sensors to visualize in composites
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        # Determine if the sample contains multiple sensors or a single sensor
        if isinstance(sample["image"], dict):
            nrows = len(self.bands.keys())
        else:
            nrows = 1
        ncols=4

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5), squeeze=False)

        if not isinstance(sample["image"], dict):
            "Only one modality, setup dict for plotting"
            sens = list(self.bands.keys())[0]
            sample["image"] = {sens: sample["image"]}


        for row, sens in enumerate(self.bands.keys()):
            band_indices = [self.bands[sens].index(band) for band in vis_bands[sens]]
            print(band_indices)
            img = sample["image"][sens].numpy()
            img = scale(img)
            c, t, h, w = img.shape
            for col, t in enumerate(range(t)):
                img_t = img[band_indices,t,:,:].transpose(1,2,0)
                axs[row, col].imshow(img_t)
                axs[row, col].axis("off")

        if show_titles:
            axs[row, 0].set_title(sens)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig