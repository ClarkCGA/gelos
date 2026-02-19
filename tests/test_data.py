import gc
from pathlib import Path
from typing import Any

import albumentations as A
import geopandas as gpd
import numpy as np
import pytest
import rioxarray as rxr
import torch
from jsonargparse import ArgumentParser

from gelos.gelosdatamodule import GELOSDataModule
from gelos.gelosdataset import GELOSDataSet
from tests.utils import create_test_geojson


# ---------------------------------------------------------------------------
# Reference implementation: ExampleGELOSDataSet
# ---------------------------------------------------------------------------

class ExampleGELOSDataSet(GELOSDataSet):
    """Concrete GELOSDataSet subclass for testing and as a reference implementation.

    Demonstrates how to subclass GELOSDataSet with three sensors (S2L2A, S1RTC, DEM),
    a gelos_chip_tracker.geojson metadata file, and rioxarray-based file loading.

    An example yaml for this dataaset can be found in tests/fixtures/example_config.yaml
    """


    S2L2A_BAND_NAMES = [
        "coastal",
        "blue",
        "green",
        "red",
        "rededge1",
        "rededge2",
        "rededge3",
        "nir",
        "nir08",
        "swir16",
        "swir22",
    ]
    S1RTC_BAND_NAMES = ["VV", "VH"]
    DEM_BAND_NAMES = ["DEM"]

    all_band_names = {
        "S2L2A": S2L2A_BAND_NAMES,
        "S1RTC": S1RTC_BAND_NAMES,
        "DEM": DEM_BAND_NAMES,
    }

    BAND_SETS = {
        "all": all_band_names,
        "rgb": {"S2L2A": ["red", "green", "blue"]},
        "s2_6band": {
            "S2L2A": ["blue", "green", "red", "nir08", "swir16", "swir22"],
        },
    }

    def __init__(
        self,
        data_root: str | Path,
        bands: dict[str, list[str]] | None = None,
        means: dict[str, dict[str, float]] | None = None,
        stds: dict[str, dict[str, float]] | None = None,
        transform: A.Compose | None = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] | None = None,
        perturb_bands: dict[str, list[str]] | None = None,
        perturb_alpha: float = 1,
    ) -> None:
        if bands is None:
            bands = self.all_band_names

        super().__init__(
            bands=bands,
            all_band_names=self.all_band_names,
            means=means,
            stds=stds,
            transform=transform,
            concat_bands=concat_bands,
            repeat_bands=repeat_bands,
            perturb_bands=perturb_bands,
            perturb_alpha=perturb_alpha,
        )

        self.data_root = Path(data_root)
        self.gdf = gpd.read_file(self.data_root / "gelos_chip_tracker.geojson")
        self.zfill_length = int(self.gdf["id"].astype(str).str.len().max())

    def __len__(self) -> int:
        return len(self.gdf)

    def _get_file_paths(self, index: int, sensor: str) -> list[Path]:
        sample_row = self.gdf.iloc[index]
        return [
            self.data_root / filepath
            for filepath in sample_row[f"{sensor.lower()}_paths"].split(",")
        ]

    def _load_file(self, path: Path, band_indices: list[int]) -> np.ndarray:
        data = rxr.open_rasterio(path, masked=True).to_numpy()
        return data[band_indices, :, :].transpose(1, 2, 0)  # [H, W, C]

    def _get_sample_id(self, index: int) -> tuple[str, Any]:
        sample_row = self.gdf.iloc[index]
        filename = str(sample_row["id"]).zfill(self.zfill_length)
        return filename, sample_row["id"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 3
IMG_SIZE = 96
N_TIMESTEPS_S2 = 4
N_TIMESTEPS_S1 = 4
N_TIMESTEPS_DEM = 1


@pytest.fixture()
def data_root(tmp_path):
    """Create a temporary dataset directory with dummy tiffs and geojson."""
    sensors = {
        "S2L2A": len(ExampleGELOSDataSet.S2L2A_BAND_NAMES),
        "S1RTC": len(ExampleGELOSDataSet.S1RTC_BAND_NAMES),
        "DEM": len(ExampleGELOSDataSet.DEM_BAND_NAMES),
    }
    n_timesteps = {"S2L2A": N_TIMESTEPS_S2, "S1RTC": N_TIMESTEPS_S2, "DEM": N_TIMESTEPS_DEM}
    create_test_geojson(tmp_path, N_SAMPLES, sensors, n_timesteps, img_size=IMG_SIZE)
    return tmp_path


@pytest.fixture()
def single_sensor_dataset(data_root):
    """Dataset with only S2L2A bands (single sensor path)."""
    bands = {"S2L2A": ["blue", "green", "red"]}
    ds = ExampleGELOSDataSet(data_root=data_root, bands=bands)
    yield ds
    gc.collect()


@pytest.fixture()
def multi_sensor_dataset(data_root):
    """Dataset with S2L2A + DEM (multi-sensor, no concat)."""
    bands = {"S2L2A": ["blue", "green", "red"], "DEM": ["DEM"]}
    ds = ExampleGELOSDataSet(data_root=data_root, bands=bands)
    yield ds
    gc.collect()


# ---------------------------------------------------------------------------
# Tests: base class contract
# ---------------------------------------------------------------------------


def test_base_class_not_instantiable():
    """GELOSDataSet is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        GELOSDataSet(
            bands={"S2L2A": ["RED"]},
            all_band_names={"S2L2A": ["RED", "GREEN", "BLUE"]},
        )
    gc.collect()


def test_invalid_sensor_raises(data_root):
    """Passing a sensor key not in all_band_names raises AssertionError."""
    with pytest.raises(AssertionError):
        ExampleGELOSDataSet(data_root=data_root, bands={"INVALID": ["band"]})


def test_invalid_band_name_raises(data_root):
    """Passing a band name not in the sensor's band list raises ValueError."""
    with pytest.raises(ValueError):
        ExampleGELOSDataSet(data_root=data_root, bands={"S2L2A": ["nonexistent_band"]})


# ---------------------------------------------------------------------------
# Tests: dataset length and sample id
# ---------------------------------------------------------------------------


def test_len(single_sensor_dataset):
    """Dataset length matches the number of samples in the geojson."""
    assert len(single_sensor_dataset) == N_SAMPLES


def test_sample_id_format(single_sensor_dataset):
    """filename is a zero-padded string and file_id is an integer."""
    sample = single_sensor_dataset[0]
    assert isinstance(sample["filename"], np.ndarray)
    filename_str = str(sample["filename"])
    assert filename_str.isdigit()
    assert isinstance(sample["file_id"], (int, np.integer))


# ---------------------------------------------------------------------------
# Tests: __getitem__ single sensor
# ---------------------------------------------------------------------------


def test_getitem_single_sensor_keys(single_sensor_dataset):
    """Single-sensor output has image, filename, and file_id keys."""
    sample = single_sensor_dataset[0]
    assert "image" in sample
    assert "filename" in sample
    assert "file_id" in sample


def test_getitem_single_sensor_image_is_tensor(single_sensor_dataset):
    """Single-sensor image output is a torch.Tensor, not a dict."""
    sample = single_sensor_dataset[0]
    assert isinstance(sample["image"], torch.Tensor)


def test_getitem_single_sensor_shape(single_sensor_dataset):
    """Single-sensor image has expected [C, T, H, W] shape."""
    sample = single_sensor_dataset[0]
    image = sample["image"]
    assert image.shape == (3, N_TIMESTEPS_S2, IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# Tests: __getitem__ multi sensor
# ---------------------------------------------------------------------------


def test_getitem_multi_sensor_no_concat(multi_sensor_dataset):
    """Multi-sensor without concat returns a dict of tensors for image."""
    sample = multi_sensor_dataset[0]
    assert isinstance(sample["image"], dict)
    assert "S2L2A" in sample["image"]
    assert "DEM" in sample["image"]
    assert isinstance(sample["image"]["S2L2A"], torch.Tensor)
    assert isinstance(sample["image"]["DEM"], torch.Tensor)


def test_getitem_multi_sensor_concat(data_root):
    """Multi-sensor with concat returns a single concatenated tensor.

    Uses two single-band, single-timestep sensors (S1RTC VV + DEM) so all
    non-concat dimensions match under the default [T, H, W, C] layout.
    """
    bands = {"S1RTC": ["VV"], "DEM": ["DEM"]}
    ds = ExampleGELOSDataSet(data_root=data_root, bands=bands, concat_bands=True)
    sample = ds[0]
    assert isinstance(sample["image"], torch.Tensor)
    gc.collect()


# ---------------------------------------------------------------------------
# Tests: repeat_bands
# ---------------------------------------------------------------------------


def test_repeat_bands(data_root):
    """repeat_bands tiles the temporal dimension for the specified sensor."""
    bands = {"S2L2A": ["blue", "green", "red"], "DEM": ["DEM"]}
    repeats = 4
    ds = ExampleGELOSDataSet(
        data_root=data_root, bands=bands, repeat_bands={"DEM": repeats}
    )
    sample = ds[0]
    # DEM has 1 timestep, tiled 4 times → T=4
    dem_image = sample["image"]["DEM"]
    assert dem_image.shape[1] == repeats
    gc.collect()


# ---------------------------------------------------------------------------
# Tests: GELOSDataModule integration
# ---------------------------------------------------------------------------


def test_datamodule_rejects_non_predict(data_root):
    """GELOSDataModule.setup() only accepts stage='predict'."""
    dm = GELOSDataModule(
        data_root=data_root,
        batch_size=1,
        num_workers=0,
        dataset_class=ExampleGELOSDataSet,
        bands={"S2L2A": ["blue", "green", "red"]},
    )
    with pytest.raises(ValueError):
        dm.setup(stage="fit")
    gc.collect()


def test_datamodule_setup_and_iterate(data_root):
    """DataModule creates dataset and produces batches with expected keys."""
    dm = GELOSDataModule(
        data_root=data_root,
        batch_size=2,
        num_workers=0,
        dataset_class=ExampleGELOSDataSet,
        bands={"S2L2A": ["blue", "green", "red"]},
    )
    dm.setup(stage="predict")
    assert len(dm.dataset) == N_SAMPLES

    dl = dm.predict_dataloader()
    batch = next(iter(dl))
    assert "image" in batch
    assert "filename" in batch
    assert "file_id" in batch
    gc.collect()


def test_example_config_instantiates(data_root):
    """tests/fixtures/example_config.yaml can instantiate GELOSDataModule and produce batches.

    Validates that the documented YAML config stays in sync with the code:
    class paths are importable, band names are valid, and the DataModule can
    set up and iterate.
    """
    import importlib

    import yaml

    yaml_path = Path(__file__).parent / "fixtures" / "example_config.yaml"

    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    parser = ArgumentParser()
    parser.add_class_arguments(GELOSDataModule, "data")

    data_init_args = yaml_config['data']['init_args']
    data_init_args['data_root'] = str(data_root)

    cfg = parser.parse_object({"data": data_init_args})
    init = parser.instantiate_classes(cfg)
    gelos_datamodule = init.data

    gelos_datamodule.setup(stage="predict")
    assert len(gelos_datamodule.dataset) == N_SAMPLES

    batch = next(iter(gelos_datamodule.predict_dataloader()))
    assert "image" in batch
    gc.collect()
