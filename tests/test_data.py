import gc
from pathlib import Path
from typing import Any

import albumentations as A
import geopandas as gpd
import numpy as np
import pytest
import rioxarray as rxr
import torch
from gelos.gelosdatamodule import GELOSDataModule
from gelos.generation import instantiate_recursive
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

    means = {
        "S2L2A": {
            "coastal": 0.0,
            "blue": 0.0,
            "green": 0.0,
            "red": 0.0,
            "rededge1": 0.0,
            "rededge2": 0.0,
            "rededge3": 0.0,
            "nir": 0.0,
            "nir08": 0.0,
            "swir16": 0.0,
            "swir22": 0.0,
        },
        "S1RTC": {"VV": 0.0, "VH": 0.0},
        "DEM": {"DEM": 0.0},
    }

    stds = {
        "S2L2A": {
            "coastal": 1.0,
            "blue": 1.0,
            "green": 1.0,
            "red": 1.0,
            "rededge1": 1.0,
            "rededge2": 1.0,
            "rededge3": 1.0,
            "nir": 1.0,
            "nir08": 1.0,
            "swir16": 1.0,
            "swir22": 1.0,
        },
        "S1RTC": {"VV": 1.0, "VH": 1.0},
        "DEM": {"DEM": 1.0},
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
        transform: A.Compose | None = None,
        concat_bands: bool = False,
        repeat_bands: dict[str, int] | None = None,
        perturb_bands: dict[str, dict[str, float]] | None = None,
        db_scale_bands: dict[str, list[str]] | None = None,
    ) -> None:
        if bands is None:
            bands = self.all_band_names

        super().__init__(
            bands=bands,
            all_band_names=self.all_band_names,
            transform=transform,
            concat_bands=concat_bands,
            repeat_bands=repeat_bands,
            perturb_bands=perturb_bands,
            db_scale_bands=db_scale_bands,
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
# Tests: perturb_bands
# ---------------------------------------------------------------------------


def test_perturb_bands(data_root):
    """perturb bands adds gaussian noise based on the band distrubtion of a single chip"""
    bands = {"S2L2A": ["blue", "green", "red"], "DEM": ["DEM"]}
    perturb_bands = {"S2L2A": {"blue": 0.1}} 
    perturb_ds = ExampleGELOSDataSet(
        data_root=data_root, bands=bands, perturb_bands=perturb_bands
    )
    non_perturb_ds = ExampleGELOSDataSet(
        data_root=data_root, bands=bands,
    )
    
    non_perturb_sample = non_perturb_ds[0]
    perturb_sample = perturb_ds[0]
    s2l2a_equal = non_perturb_sample["image"]["S2L2A"] == perturb_sample["image"]["S2L2A"]
    dem_equal = non_perturb_sample["image"]["DEM"] == perturb_sample["image"]["DEM"]
    assert dem_equal.all() and not s2l2a_equal.all() 
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


# ---------------------------------------------------------------------------
# Tests: normalization stats resolution (explicit args -> means/stds ->
# MEANS/STDS -> defaults) and the identity-normalization warning
# ---------------------------------------------------------------------------


class _UppercaseStatsDataSet:
    """Minimal dataset-class stand-in defining only uppercase MEANS/STDS."""

    all_band_names = {"S2L2A": ["blue", "green"]}
    MEANS = {"S2L2A": {"blue": 10.0, "green": 20.0}}
    STDS = {"S2L2A": {"blue": 2.0, "green": 4.0}}


class _BothCaseStatsDataSet:
    """Defines both cases; lowercase means/stds must win over MEANS/STDS."""

    all_band_names = {"S2L2A": ["blue"]}
    means = {"S2L2A": {"blue": 1.0}}
    stds = {"S2L2A": {"blue": 3.0}}
    MEANS = {"S2L2A": {"blue": 100.0}}
    STDS = {"S2L2A": {"blue": 300.0}}


class _NoStatsDataSet:
    """No stats attributes at all: every band resolves to the 0.0/1.0 defaults."""

    all_band_names = {"S2L2A": ["blue", "green"]}


def _capture_loguru_warnings():
    """Return (messages, sink_id): a list capturing loguru WARNING+ output."""
    from loguru import logger

    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    return messages, sink_id


def test_stats_resolution_uppercase_class_attrs():
    """Uppercase MEANS/STDS class attributes are found (the gelos-lc convention)."""
    dm = GELOSDataModule(
        data_root="unused",
        batch_size=1,
        num_workers=0,
        dataset_class=_UppercaseStatsDataSet,
        bands={"S2L2A": ["blue", "green"]},
    )
    assert dm.means["S2L2A"] == [10.0, 20.0]
    assert dm.stds["S2L2A"] == [2.0, 4.0]


def test_stats_resolution_lowercase_wins_over_uppercase():
    """Lowercase means/stds class attributes take precedence over MEANS/STDS."""
    dm = GELOSDataModule(
        data_root="unused",
        batch_size=1,
        num_workers=0,
        dataset_class=_BothCaseStatsDataSet,
        bands={"S2L2A": ["blue"]},
    )
    assert dm.means["S2L2A"] == [1.0]
    assert dm.stds["S2L2A"] == [3.0]


def test_stats_resolution_explicit_args_win():
    """Explicit means/stds arguments take precedence over any class attributes."""
    dm = GELOSDataModule(
        data_root="unused",
        batch_size=1,
        num_workers=0,
        dataset_class=_BothCaseStatsDataSet,
        bands={"S2L2A": ["blue"]},
        means={"S2L2A": {"blue": 7.0}},
        stds={"S2L2A": {"blue": 9.0}},
    )
    assert dm.means["S2L2A"] == [7.0]
    assert dm.stds["S2L2A"] == [9.0]


def test_stats_resolution_all_defaults_warns():
    """A modality resolving entirely to defaults logs a loud identity warning."""
    messages, sink_id = _capture_loguru_warnings()
    from loguru import logger

    try:
        dm = GELOSDataModule(
            data_root="unused",
            batch_size=1,
            num_workers=0,
            dataset_class=_NoStatsDataSet,
            bands={"S2L2A": ["blue", "green"]},
        )
    finally:
        logger.remove(sink_id)
    assert dm.means["S2L2A"] == [0.0, 0.0]
    assert dm.stds["S2L2A"] == [1.0, 1.0]
    assert any(
        "identity" in message and "_NoStatsDataSet" in message and "S2L2A" in message
        for message in messages
    )


def test_stats_resolution_real_stats_do_not_warn():
    """No identity warning when real statistics are resolved."""
    messages, sink_id = _capture_loguru_warnings()
    from loguru import logger

    try:
        GELOSDataModule(
            data_root="unused",
            batch_size=1,
            num_workers=0,
            dataset_class=_UppercaseStatsDataSet,
            bands={"S2L2A": ["blue", "green"]},
        )
    finally:
        logger.remove(sink_id)
    assert not any("identity" in message for message in messages)


# ---------------------------------------------------------------------------
# Tests: normalize=False (identity aug)
# ---------------------------------------------------------------------------


def test_normalize_false_aug_is_identity(data_root):
    """With normalize=False the aug leaves a batch completely unchanged."""
    import copy

    dm = GELOSDataModule(
        data_root=data_root,
        batch_size=2,
        num_workers=0,
        dataset_class=ExampleGELOSDataSet,
        bands={"S2L2A": ["blue", "green", "red"], "DEM": ["DEM"]},
        means={"S2L2A": {"blue": 5.0, "green": 5.0, "red": 5.0}, "DEM": {"DEM": 5.0}},
        stds={"S2L2A": {"blue": 2.0, "green": 2.0, "red": 2.0}, "DEM": {"DEM": 2.0}},
        normalize=False,
    )
    dm.setup(stage="predict")
    batch = next(iter(dm.predict_dataloader()))
    original = copy.deepcopy(batch)
    out = dm.aug(batch)
    for sensor in ("S2L2A", "DEM"):
        torch.testing.assert_close(out["image"][sensor], original["image"][sensor])
    gc.collect()


def test_normalize_true_aug_changes_batch(data_root):
    """Control: with normalize=True and real stats, the aug does modify the batch."""
    import copy

    dm = GELOSDataModule(
        data_root=data_root,
        batch_size=2,
        num_workers=0,
        dataset_class=ExampleGELOSDataSet,
        bands={"S2L2A": ["blue", "green", "red"], "DEM": ["DEM"]},
        means={"S2L2A": {"blue": 5.0, "green": 5.0, "red": 5.0}, "DEM": {"DEM": 5.0}},
        stds={"S2L2A": {"blue": 2.0, "green": 2.0, "red": 2.0}, "DEM": {"DEM": 2.0}},
    )
    dm.setup(stage="predict")
    batch = next(iter(dm.predict_dataloader()))
    original = copy.deepcopy(batch)
    out = dm.aug(batch)
    assert not torch.equal(out["image"]["S2L2A"], original["image"]["S2L2A"])
    gc.collect()


# ---------------------------------------------------------------------------
# Tests: db_scale_bands (linear power -> decibel conversion)
# ---------------------------------------------------------------------------


def test_db_scale_bands_converts_to_decibels(data_root):
    """db_scale_bands output equals 10*log10(clip(raw, 1e-10)) of the raw output."""
    bands = {"S1RTC": ["VV", "VH"]}
    raw_ds = ExampleGELOSDataSet(data_root=data_root, bands=bands)
    db_ds = ExampleGELOSDataSet(
        data_root=data_root, bands=bands, db_scale_bands={"S1RTC": ["VV", "VH"]}
    )
    raw = raw_ds[0]["image"]
    db = db_ds[0]["image"]
    expected = 10 * torch.log10(torch.clamp(raw, min=1e-10))
    torch.testing.assert_close(db, expected)
    gc.collect()


def test_db_scale_bands_converts_only_listed_bands(data_root):
    """Bands not listed in db_scale_bands are untouched."""
    bands = {"S1RTC": ["VV", "VH"]}
    raw_ds = ExampleGELOSDataSet(data_root=data_root, bands=bands)
    db_ds = ExampleGELOSDataSet(
        data_root=data_root, bands=bands, db_scale_bands={"S1RTC": ["VV"]}
    )
    raw = raw_ds[0]["image"]
    db = db_ds[0]["image"]
    # Band layout is [C, T, H, W]: VV is channel 0, VH channel 1.
    torch.testing.assert_close(db[0], 10 * torch.log10(torch.clamp(raw[0], min=1e-10)))
    torch.testing.assert_close(db[1], raw[1])
    gc.collect()


def test_db_scale_bands_invalid_band_raises(data_root):
    """Band names are validated against self.bands like perturb_bands."""
    ds = ExampleGELOSDataSet(
        data_root=data_root,
        bands={"S1RTC": ["VV", "VH"]},
        db_scale_bands={"S1RTC": ["nonexistent_band"]},
    )
    with pytest.raises(ValueError):
        ds[0]
    gc.collect()


def test_datamodule_passes_db_scale_bands_to_dataset(data_root):
    """GELOSDataModule forwards db_scale_bands to the dataset in setup()."""
    db_scale_bands = {"S1RTC": ["VV", "VH"]}
    dm = GELOSDataModule(
        data_root=data_root,
        batch_size=1,
        num_workers=0,
        dataset_class=ExampleGELOSDataSet,
        bands={"S1RTC": ["VV", "VH"]},
        db_scale_bands=db_scale_bands,
    )
    dm.setup(stage="predict")
    assert dm.dataset.db_scale_bands == db_scale_bands
    gc.collect()


def test_example_config_instantiates(data_root):
    """tests/fixtures/example_config.yaml can instantiate GELOSDataModule and produce batches.

    Validates that the documented YAML config stays in sync with the code:
    class paths are importable, band names are valid, and the DataModule can
    set up and iterate.
    """
    import yaml

    yaml_path = Path(__file__).parent / "fixtures" / "example_config.yaml"

    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    yaml_config["data"]["init_args"]["data_root"] = str(data_root)
    gelos_datamodule = instantiate_recursive(yaml_config["data"])

    gelos_datamodule.setup(stage="predict")
    assert len(gelos_datamodule.dataset) == N_SAMPLES

    batch = next(iter(gelos_datamodule.predict_dataloader()))
    assert "image" in batch
    gc.collect()
