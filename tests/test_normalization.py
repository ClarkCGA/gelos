import pytest

from gelos.normalization import (
    MODEL_NORMALIZATION,
    inject_model_normalization,
    resolve_model_normalization,
)

PRITHVI_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
TERRAMIND_S2_BANDS = [
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


def test_prithvi_resolves_v2_stats():
    resolved = resolve_model_normalization("prithvi_eo_v2_300", {"S2L2A": PRITHVI_BANDS})
    assert set(resolved) == {"means", "stds", "set_nodata"}
    assert resolved["set_nodata"] == 0
    assert resolved["means"]["S2L2A"]["BLUE"] == 1087.0
    assert resolved["means"]["S2L2A"]["NIR_NARROW"] == 2734.0
    assert resolved["stds"]["S2L2A"]["SWIR_2"] == 1049.0


def test_prithvi_tl_coords_name_resolves_v2_stats():
    # The gelos-registered TL wrapper names share the prithvi_eo_v2 prefix, so
    # TL configs get Prithvi's pretraining normalization automatically.
    tl = resolve_model_normalization("prithvi_eo_v2_300_tl_coords", {"S2L2A": PRITHVI_BANDS})
    plain = resolve_model_normalization("prithvi_eo_v2_300", {"S2L2A": PRITHVI_BANDS})
    assert tl == plain
    assert tl["means"]["S2L2A"]["BLUE"] == 1087.0


def test_prithvi_band_subset_and_order_respected():
    resolved = resolve_model_normalization("prithvi_eo_v2_600", {"S2L2A": ["RED", "BLUE"]})
    assert list(resolved["means"]["S2L2A"]) == ["RED", "BLUE"]
    assert resolved["means"]["S2L2A"] == {"RED": 1433.0, "BLUE": 1087.0}


def test_prithvi_unknown_band_raises():
    with pytest.raises(ValueError, match="COASTAL_AEROSOL"):
        resolve_model_normalization("prithvi_eo_v2_300", {"S2L2A": ["BLUE", "COASTAL_AEROSOL"]})


def test_prithvi_unknown_modality_raises():
    with pytest.raises(ValueError, match="S1RTC"):
        resolve_model_normalization(
            "prithvi_eo_v2_300", {"S2L2A": PRITHVI_BANDS, "S1RTC": ["VV", "VH"]}
        )


def test_terramind_resolves_multimodal_stats_and_db_scale():
    resolved = resolve_model_normalization(
        "terramind_v1_base",
        {"S2L2A": TERRAMIND_S2_BANDS, "S1RTC": ["VV", "VH"], "DEM": ["DEM"]},
    )
    assert resolved["means"]["S1RTC"] == {"VV": -10.93, "VH": -17.329}
    assert resolved["stds"]["S1RTC"] == {"VV": 4.391, "VH": 4.459}
    assert resolved["means"]["S2L2A"]["COASTAL_AEROSOL"] == 1390.458
    assert resolved["stds"]["S2L2A"]["SWIR_2"] == 1334.311
    assert resolved["means"]["DEM"] == {"DEM": 670.665}
    # dB conversion applies to S1 only
    assert resolved["db_scale_bands"] == {"S1RTC": ["VV", "VH"]}
    assert resolved["set_nodata"] == 0


def test_terramind_s2_only_has_no_db_scale():
    resolved = resolve_model_normalization("terramind_v1_large", {"S2L2A": TERRAMIND_S2_BANDS})
    assert "db_scale_bands" not in resolved


def test_olmoearth_disables_datamodule_normalization():
    resolved = resolve_model_normalization("olmoearth_v1_base_s1s2", {"S2L2A": ["BLUE"]})
    assert resolved == {"normalize": False, "set_nodata": 0}
    resolved = resolve_model_normalization("olmoearth_v1_2_base", {"S2L2A": ["BLUE"]})
    assert resolved == {"normalize": False, "set_nodata": 0}


def test_dinov3_resolves_clip_and_stretch_imagenet_stats():
    """DINOv3 gets clip-to-2500 plus ImageNet stats folded onto the DN scale."""
    resolved = resolve_model_normalization(
        "dinov3_vitb16_pretrained", {"S2L2A": ["RED", "GREEN", "BLUE"]}
    )
    assert set(resolved) == {"means", "stds", "clip_range_bands", "set_nodata"}
    assert resolved["means"]["S2L2A"] == {
        "RED": 0.485 * 2500,
        "GREEN": 0.456 * 2500,
        "BLUE": 0.406 * 2500,
    }
    assert resolved["stds"]["S2L2A"] == {
        "RED": 0.229 * 2500,
        "GREEN": 0.224 * 2500,
        "BLUE": 0.225 * 2500,
    }
    assert resolved["clip_range_bands"] == {
        "S2L2A": {
            "RED": [0.0, 2500.0],
            "GREEN": [0.0, 2500.0],
            "BLUE": [0.0, 2500.0],
        }
    }


def test_dinov3_non_rgb_band_raises():
    with pytest.raises(ValueError, match="NIR_NARROW"):
        resolve_model_normalization("dinov3_vitb16", {"S2L2A": ["RED", "NIR_NARROW"]})


def test_dinov3_sat_resolves_clip_and_stretch_sat_stats():
    """The satellite variant gets SAT-493M stats folded onto the DN scale, not ImageNet's."""
    resolved = resolve_model_normalization(
        "dinov3_vitl16_sat_pretrained", {"S2L2A": ["RED", "GREEN", "BLUE"]}
    )
    assert set(resolved) == {"means", "stds", "clip_range_bands", "set_nodata"}
    assert resolved["means"]["S2L2A"] == {
        "RED": 0.430 * 2500,
        "GREEN": 0.411 * 2500,
        "BLUE": 0.296 * 2500,
    }
    assert resolved["stds"]["S2L2A"] == {
        "RED": 0.213 * 2500,
        "GREEN": 0.156 * 2500,
        "BLUE": 0.143 * 2500,
    }
    assert resolved["clip_range_bands"] == {
        "S2L2A": {
            "RED": [0.0, 2500.0],
            "GREEN": [0.0, 2500.0],
            "BLUE": [0.0, 2500.0],
        }
    }


def test_dinov3_sat_prefix_precedes_generic_dinov3():
    # Resolution takes the first startswith match, so the sat entry must sit
    # before the generic "dinov3" entry or it would silently get ImageNet stats.
    keys = list(MODEL_NORMALIZATION)
    assert keys.index("dinov3_vitl16_sat") < keys.index("dinov3")
    # And the generic entry still serves the web variant (ImageNet RED mean).
    resolved = resolve_model_normalization("dinov3_vitb16_pretrained", {"S2L2A": ["RED"]})
    assert resolved["means"]["S2L2A"]["RED"] == 0.485 * 2500


def test_dinov3_sat_non_rgb_band_raises():
    with pytest.raises(ValueError, match="NIR_NARROW"):
        resolve_model_normalization("dinov3_vitl16_sat", {"S2L2A": ["RED", "NIR_NARROW"]})


def test_unknown_model_returns_none():
    assert resolve_model_normalization("some_future_model", {"S2L2A": ["BLUE"]}) is None


def test_inject_fills_missing_keys_only():
    data_init = {
        "bands": {"S1RTC": ["VV", "VH"]},
        "means": {"S1RTC": {"VV": -1.0, "VH": -2.0}},
    }
    injected = inject_model_normalization(data_init, "terramind_v1_base")
    assert sorted(injected) == ["db_scale_bands", "stds"]
    # explicit config means untouched
    assert data_init["means"] == {"S1RTC": {"VV": -1.0, "VH": -2.0}}
    assert data_init["stds"]["S1RTC"] == {"VV": 4.391, "VH": 4.459}


def test_inject_unknown_model_or_missing_bands_is_noop():
    data_init = {"bands": {"S2L2A": ["BLUE"]}}
    assert inject_model_normalization(data_init, "some_future_model") == []
    assert set(data_init) == {"bands"}
    assert inject_model_normalization({}, "prithvi_eo_v2_300") == []


def test_registry_covers_expected_models():
    assert set(MODEL_NORMALIZATION) == {
        "prithvi_eo_v2",
        "terramind_v1",
        "olmoearth_v1",
        "dinov3_vitl16_sat",
        "dinov3",
    }
    assert all(spec["set_nodata"] == 0 for spec in MODEL_NORMALIZATION.values())


# ---------------------------------------------------------------------------
# set_nodata injection: only alongside an explicit nodata_value.
# ---------------------------------------------------------------------------


def _example_dataset():
    from tests.test_data import ExampleGELOSDataSet

    return ExampleGELOSDataSet


@pytest.mark.parametrize("model_name", ["olmoearth_v1_2_base", "prithvi_eo_v2_300"])
def test_inject_set_nodata_when_nodata_value_present(model_name):
    from gelos.gelosdatamodule import GELOSDataModule

    bands = {"S2L2A": ["BLUE"]}
    data_init = {"bands": bands, "nodata_value": -999}
    injected = inject_model_normalization(data_init, model_name)
    assert "set_nodata" in injected
    assert data_init["set_nodata"] == 0
    assert data_init["nodata_value"] == -999
    # The injected pair constructs a datamodule (nodata_value/set_nodata together).
    dm = GELOSDataModule(
        data_root="unused",
        batch_size=1,
        num_workers=0,
        dataset_class=_example_dataset(),
        **data_init,
    )
    assert dm.set_nodata == 0


@pytest.mark.parametrize("model_name", ["olmoearth_v1_2_base", "prithvi_eo_v2_300"])
def test_inject_skips_set_nodata_without_nodata_value(model_name):
    from gelos.gelosdatamodule import GELOSDataModule

    data_init = {"bands": {"S2L2A": ["BLUE"]}}
    injected = inject_model_normalization(data_init, model_name)
    assert "set_nodata" not in injected
    assert "set_nodata" not in data_init
    assert "nodata_value" not in data_init
    # No "must be provided together" ValueError from the datamodule.
    GELOSDataModule(
        data_root="unused",
        batch_size=1,
        num_workers=0,
        dataset_class=_example_dataset(),
        **data_init,
    )


def test_inject_explicit_set_nodata_wins():
    data_init = {"bands": {"S2L2A": ["BLUE"]}, "nodata_value": -999, "set_nodata": -1}
    injected = inject_model_normalization(data_init, "olmoearth_v1_2_base")
    assert injected == ["normalize"]
    assert data_init["set_nodata"] == -1
