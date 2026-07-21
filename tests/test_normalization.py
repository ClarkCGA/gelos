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
    assert set(resolved) == {"means", "stds"}
    assert resolved["means"]["S2L2A"]["BLUE"] == 1087.0
    assert resolved["means"]["S2L2A"]["NIR_NARROW"] == 2734.0
    assert resolved["stds"]["S2L2A"]["SWIR_2"] == 1049.0


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


def test_terramind_s2_only_has_no_db_scale():
    resolved = resolve_model_normalization("terramind_v1_large", {"S2L2A": TERRAMIND_S2_BANDS})
    assert "db_scale_bands" not in resolved


def test_olmoearth_disables_datamodule_normalization():
    resolved = resolve_model_normalization("olmoearth_v1_base_s1s2", {"S2L2A": ["BLUE"]})
    assert resolved == {"normalize": False}


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
    assert set(MODEL_NORMALIZATION) == {"prithvi_eo_v2", "terramind_v1", "olmoearth_v1"}
