"""Model-matched normalization defaults for embedding generation.

Frozen encoders should receive inputs normalized the way the model saw data
during pretraining, not with dataset statistics: the full preprocessing chain
(raw sensor values -> scaled tensor) is effectively the model's first layer.
This module maps backbone names to their pretraining statistics (keyed by
GELOS band names) plus any required scale conversions (e.g. Sentinel-1 linear
power -> dB), and ``gelos.generation`` injects them into the datamodule config
for recognized models. Keys set explicitly in the config always win, and
unrecognized models fall back to the dataset-statistics resolution in
``GELOSDataModule``.

Statistics are imported from terratorch where available so they stay in sync;
hard-coded fallbacks (copied from the same terratorch modules) cover older
terratorch layouts.
"""

from loguru import logger

try:
    from terratorch.models.backbones.prithvi_vit import PRITHVI_V2_MEAN, PRITHVI_V2_STD
except ImportError:  # terratorch layout changed; values from prithvi_vit.py
    PRITHVI_V2_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
    PRITHVI_V2_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

try:
    from terratorch.models.backbones.terramind.model.terramind_register import (
        v1_pretraining_mean as _tm_v1_mean,
    )
    from terratorch.models.backbones.terramind.model.terramind_register import (
        v1_pretraining_std as _tm_v1_std,
    )

    _TM_S2L2A_MEAN = _tm_v1_mean["untok_sen2l2a@224"]
    _TM_S2L2A_STD = _tm_v1_std["untok_sen2l2a@224"]
    _TM_S1RTC_MEAN = _tm_v1_mean["untok_sen1rtc@224"]
    _TM_S1RTC_STD = _tm_v1_std["untok_sen1rtc@224"]
    _TM_DEM_MEAN = _tm_v1_mean["untok_dem@224"]
    _TM_DEM_STD = _tm_v1_std["untok_dem@224"]
except ImportError:  # terratorch layout changed; values from terramind_register.py
    _TM_S2L2A_MEAN = [
        1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975,
        2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648,
    ]  # fmt: skip
    _TM_S2L2A_STD = [
        2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
        1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311,
    ]  # fmt: skip
    _TM_S1RTC_MEAN = [-10.93, -17.329]
    _TM_S1RTC_STD = [4.391, 4.459]
    _TM_DEM_MEAN = [670.665]
    _TM_DEM_STD = [951.272]

# Band orders below mirror terratorch's PRETRAINED_BANDS for each backbone.
_PRITHVI_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
_TERRAMIND_S2L2A_BANDS = [
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


def _band_stats(bands: list[str], values: list[float]) -> dict[str, float]:
    if len(bands) != len(values):
        raise ValueError(f"band/stat length mismatch: {bands} vs {values}")
    return dict(zip(bands, values))


# Backbone-name prefix -> normalization spec. A spec is either
# {"normalize": False} (the backbone normalizes internally, e.g. OlmoEarth) or
# per-modality pretraining stats with optional dB conversion requirements.
MODEL_NORMALIZATION = {
    "prithvi_eo_v2": {
        "means": {"S2L2A": _band_stats(_PRITHVI_BANDS, PRITHVI_V2_MEAN)},
        "stds": {"S2L2A": _band_stats(_PRITHVI_BANDS, PRITHVI_V2_STD)},
    },
    "terramind_v1": {
        "means": {
            "S2L2A": _band_stats(_TERRAMIND_S2L2A_BANDS, _TM_S2L2A_MEAN),
            "S1RTC": _band_stats(["VV", "VH"], _TM_S1RTC_MEAN),
            "DEM": _band_stats(["DEM"], _TM_DEM_MEAN),
        },
        "stds": {
            "S2L2A": _band_stats(_TERRAMIND_S2L2A_BANDS, _TM_S2L2A_STD),
            "S1RTC": _band_stats(["VV", "VH"], _TM_S1RTC_STD),
            "DEM": _band_stats(["DEM"], _TM_DEM_STD),
        },
        # TerraMind's S1 pretraining stats are in dB; the on-disk chips are
        # linear power, so the datamodule must convert before normalizing.
        "db_scale_bands": {"S1RTC": ["VV", "VH"]},
    },
    "olmoearth_v1": {"normalize": False},
}


def resolve_model_normalization(
    model_name: str, bands: dict[str, list[str]]
) -> dict[str, object] | None:
    """Resolve datamodule kwargs matching a backbone's pretraining normalization.

    Args:
        model_name: terratorch/gelos backbone name (e.g. ``prithvi_eo_v2_300``).
        bands: modality -> band names, as configured on the datamodule.

    Returns:
        Dict of ``GELOSDataModule`` init kwargs (``means``/``stds`` and
        optionally ``db_scale_bands``, or ``normalize: False`` for backbones
        that normalize internally), or ``None`` if the model is not registered.

    Raises:
        ValueError: the model is registered but a configured modality or band
            has no pretraining statistic. Set ``means``/``stds`` explicitly in
            the config to override.
    """
    spec = next(
        (s for prefix, s in MODEL_NORMALIZATION.items() if model_name.startswith(prefix)),
        None,
    )
    if spec is None:
        return None
    if "normalize" in spec:
        return {"normalize": spec["normalize"]}

    means: dict[str, dict[str, float]] = {}
    stds: dict[str, dict[str, float]] = {}
    for modality, band_list in bands.items():
        model_means = spec["means"].get(modality)
        if model_means is None:
            raise ValueError(
                f"{model_name} has no pretraining statistics for modality '{modality}'; "
                "set means/stds for it explicitly in the config to override"
            )
        missing = [band for band in band_list if band not in model_means]
        if missing:
            raise ValueError(
                f"{model_name} has no pretraining statistics for {modality} bands {missing}; "
                "set means/stds for them explicitly in the config to override"
            )
        means[modality] = {band: model_means[band] for band in band_list}
        stds[modality] = {band: spec["stds"][modality][band] for band in band_list}

    resolved: dict[str, object] = {"means": means, "stds": stds}
    db_scale = {
        modality: [b for b in band_list if b in spec.get("db_scale_bands", {}).get(modality, [])]
        for modality, band_list in bands.items()
    }
    db_scale = {modality: band_list for modality, band_list in db_scale.items() if band_list}
    if db_scale:
        resolved["db_scale_bands"] = db_scale
    return resolved


def inject_model_normalization(data_init_args: dict, model_name: str) -> list[str]:
    """Inject model-matched normalization defaults into a datamodule config dict.

    Mutates ``data_init_args`` in place. Keys already present in the config are
    never overwritten, so explicit config values always win.

    Returns:
        The list of injected keys (empty if the model is unregistered, the
        config left no gaps to fill, or no ``bands`` are configured).
    """
    bands = data_init_args.get("bands")
    if not bands:
        logger.warning(
            f"no explicit 'bands' in the data config; cannot resolve model-matched "
            f"normalization for '{model_name}' — falling back to dataset statistics"
        )
        return []
    resolved = resolve_model_normalization(model_name, bands)
    if resolved is None:
        logger.info(
            f"no model-matched normalization registered for '{model_name}'; "
            "falling back to dataset statistics"
        )
        return []
    injected = [key for key in resolved if key not in data_init_args]
    for key in injected:
        data_init_args[key] = resolved[key]
    if injected:
        logger.info(f"model-matched normalization for '{model_name}': injected {injected}")
    else:
        logger.info(
            f"model-matched normalization for '{model_name}' fully overridden by the config"
        )
    return injected
