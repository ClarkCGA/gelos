"""Terratorch-compatible backbone wrapper for Ai2's OlmoEarth foundation model.

GELOS selects geospatial foundation models purely via YAML (``model:`` block),
resolved through terratorch's ``BACKBONE_REGISTRY``. OlmoEarth is not shipped in
the pinned terratorch, so this module adapts an ``olmoearth-pretrain`` encoder to
the terratorch backbone contract (``forward_features`` returning a list of layer
tensors). Registration lives in ``gelos/backbones/olmoearth_backbone.py``
(imported from ``gelos/generation.py``).

The ``olmoearth-pretrain`` package is an optional, heavy dependency (extra
``gelos[olmoearth]``); it is imported lazily inside ``__init__``/``forward`` so the
base install and unimported test collection never require it.

API NOTE (see plan risk R2): the exact ``MaskedOlmoEarthSample`` field names
(``sentinel2_l2a``, ``timestamps``) and the encoder signature
(``encoder(sample, fast_pass=True, patch_size=...)`` returning
``out["tokens_and_masks"].sentinel2_l2a``) come from the public OlmoEarth API docs
and were NOT verifiable at implementation time (the package is not installed here).
These touch-points are isolated in this module and must be verified against the
installed ``olmoearth_pretrain`` package. The band-reorder helper below is pure
index logic and is fully correct/tested regardless.

S1 extension (API NOTE R2):
  sentinel1 field name: "sentinel1" (NOT "sentinel1_rtc").
  S1 band order: ["vv", "vh"] (lowercase, 2 bands, 1 band set).
  S1 mask shape: (B, H, W, T, 1) — 1 band set.
  S1 output key: out["tokens_and_masks"].sentinel1, shape (B, H', W', T, 1, D).
  S2 mask shape: (B, H, W, T, S) — S band sets, read from the loaded encoder's
    ``tokenization_config.get_num_bandsets("sentinel2_l2a")``: 1 for v1.2
    checkpoints (one 12-band set), 3 for v1 (10m/20m/60m).
  Source: verified against allenai/olmoearth_pretrain datatypes.py and constants.py.

Checkpoint generations (issue #81):
  OlmoEarth v1.2 is the supported generation. The ``olmoearth_v1_*`` factories
  (v1 checkpoints) are DEPRECATED: they still work for the full 12-band S2L2A
  input but emit a ``DeprecationWarning`` and will be removed in a future
  release. ``OlmoEarthBackbone``'s default ``model_id`` is v1.2 Base.

Band subsets (issue #81, v1.2 only):
  ``bands`` may be a SUBSET of the 12 S2L2A bands. Bands not listed are
  zero-filled AFTER pretraining normalization, per band for the whole sample,
  which is exactly what v1.2's pretraining band dropout fed the encoder
  (``MultiModalPatchEmbeddings._apply_band_dropout`` multiplies a whole
  normalized band channel by 0; paper arXiv 2605.20804 §2.2). One warning is
  emitted at construction naming the zero-filled bands; a stronger warning
  fires when more than ``_MAX_ABSENT_BANDS_IN_DISTRIBUTION`` bands are absent
  (pretraining dropped about 10% of bands on average). v1 checkpoints have no
  band dropout, so a band subset with a v1 checkpoint raises ``ValueError``.

Pretraining normalization (``apply_pretraining_normalization=True``, the default):
  OlmoEarth was pretrained on data normalized by its own data loader (the encoder
  itself performs no normalization), replicating
  ``olmoearth_pretrain.data.normalize.Normalizer._normalize_computed``
  (std_multiplier=2) and ``olmoearth_pretrain.data.utils.convert_to_db``:

  - S1 (RAW LINEAR POWER in): ``clip(x, 1e-10)`` -> ``10*log10(x)`` -> per-band
    min-max over mean±2σ: ``(x_dB - (mean - 2σ)) / (4σ)``.
  - S2 L2A (RAW DN 0–10000 in): per-band ``(x - (mean - 2σ)) / (4σ)``, no log.

  Inputs to this backbone must therefore be RAW sensor scale: S1 linear-power
  gamma0 (e.g. Planetary Computer sentinel-1-rtc) and S2 L2A digital numbers.
  Configure the datamodule with ``normalize: false`` and do NOT apply
  ``db_scale_bands`` to S1, or values get double-transformed. Per-band stats are
  read from the installed ``olmoearth_pretrain`` package's
  ``data/norm_configs/computed.json`` when importable, else from a hard-coded
  copy of those constants.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import json
import warnings

import torch
from torch import nn

# OlmoEarth's expected Sentinel-2 L2A band order, expressed in GELOS-LC band names.
# Spectral mapping (OlmoEarth band id -> GELOS-LC name):
#   B02=BLUE, B03=GREEN, B04=RED, B08=NIR_BROAD, B05=RED_EDGE_1, B06=RED_EDGE_2,
#   B07=RED_EDGE_3, B8A=NIR_NARROW, B11=SWIR_1, B12=SWIR_2, B01=COASTAL_AEROSOL,
#   B09=WATER_VAPOR
# (See plan: [B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09].)
OLMOEARTH_S2_BAND_ORDER: list[str] = [
    "BLUE",  # B02
    "GREEN",  # B03
    "RED",  # B04
    "NIR_BROAD",  # B08
    "RED_EDGE_1",  # B05
    "RED_EDGE_2",  # B06
    "RED_EDGE_3",  # B07
    "NIR_NARROW",  # B8A
    "SWIR_1",  # B11
    "SWIR_2",  # B12
    "COASTAL_AEROSOL",  # B01
    "WATER_VAPOR",  # B09
]

# OlmoEarth's expected Sentinel-1 band order (lowercase per olmoearth_pretrain API).
# GELOS-LC names these "VV" and "VH" (uppercase); mapping is case-insensitive.
# S1 has 1 band set (S=1), so sentinel1_mask shape is (B, H, W, T, 1).
OLMOEARTH_S1_BAND_ORDER: list[str] = ["vv", "vh"]

# GELOS band name -> olmoearth_pretrain computed.json sentinel2_l2a band key.
GELOS_TO_OLMOEARTH_S2_KEY: dict[str, str] = {
    "COASTAL_AEROSOL": "B01",
    "BLUE": "B02",
    "GREEN": "B03",
    "RED": "B04",
    "RED_EDGE_1": "B05",
    "RED_EDGE_2": "B06",
    "RED_EDGE_3": "B07",
    "NIR_BROAD": "B08",
    "NIR_NARROW": "B8A",
    "WATER_VAPOR": "B09",
    "SWIR_1": "B11",
    "SWIR_2": "B12",
}

# Hard-coded copy of the per-band pretraining stats from
# olmoearth_pretrain/data/norm_configs/computed.json (keys "sentinel1" and
# "sentinel2_l2a"), used as a fallback when the installed package cannot be read.
_FALLBACK_COMPUTED_STATS: dict[str, dict[str, dict[str, float]]] = {
    "sentinel1": {
        "vv": {"mean": -11.648990747328444, "std": 10.840350299936597},
        "vh": {"mean": -17.745436133270044, "std": 10.216274681392647},
    },
    "sentinel2_l2a": {
        "B01": {"mean": 1115.8494388252218, "std": 1955.6991794280914},
        "B02": {"mean": 1188.9412572078477, "std": 1859.1923971769581},
        "B03": {"mean": 1407.7739105458452, "std": 1727.7387413631088},
        "B04": {"mean": 1513.0573882432757, "std": 1740.7757298757895},
        "B05": {"mean": 1890.9893042634167, "std": 1754.7320388511766},
        "B06": {"mean": 2483.7812315422448, "std": 1622.1172720635755},
        "B07": {"mean": 2722.728248756412, "std": 1621.8226170190276},
        "B08": {"mean": 2755.481305028308, "std": 1612.2565699990187},
        "B09": {"mean": 3269.812554170875, "std": 2651.088425085525},
        "B11": {"mean": 2562.852607796968, "std": 1441.5471830655151},
        "B12": {"mean": 1914.1383249044113, "std": 1328.8914382756407},
        "B8A": {"mean": 2885.570263047681, "std": 1611.3587521042082},
    },
}


def _load_computed_stats() -> dict[str, dict[str, dict[str, float]]]:
    """Load OlmoEarth's computed normalization stats.

    Prefers the ``computed.json`` shipped inside the installed
    ``olmoearth_pretrain`` package (the authoritative source); falls back to the
    hard-coded copy above when the package is not importable.
    """
    try:
        from importlib.resources import files

        with (files("olmoearth_pretrain.data.norm_configs") / "computed.json").open() as f:
            return json.load(f)
    except Exception:
        return _FALLBACK_COMPUTED_STATS


OLMOEARTH_COMPUTED_STATS: dict[str, dict[str, dict[str, float]]] = _load_computed_stats()


def convert_to_db(x: torch.Tensor) -> torch.Tensor:
    """Convert linear-power SAR backscatter to decibels.

    Replicates ``olmoearth_pretrain.data.utils.convert_to_db``: clip to 1e-10 to
    avoid log(0), then ``10 * log10(x)``.
    """
    return 10.0 * torch.log10(torch.clamp(x, min=1e-10))


def minmax_normalize(
    x: torch.Tensor,
    means: Sequence[float],
    stds: Sequence[float],
    std_multiplier: float = 2.0,
) -> torch.Tensor:
    """Per-band mean±kσ min-max normalization over the last (channel) axis.

    Replicates ``olmoearth_pretrain.data.normalize.Normalizer._normalize_computed``
    with its default ``std_multiplier=2``:
    ``(x - (mean - k*std)) / ((mean + k*std) - (mean - k*std))``.
    """
    mean_vals = torch.as_tensor(means, dtype=x.dtype, device=x.device)
    std_vals = torch.as_tensor(stds, dtype=x.dtype, device=x.device)
    min_vals = mean_vals - std_multiplier * std_vals
    max_vals = mean_vals + std_multiplier * std_vals
    return (x - min_vals) / (max_vals - min_vals)


def resolve_s2_band_stats(bands: Sequence[str]) -> tuple[list[float], list[float]]:
    """Per-band (means, stds) for GELOS-named S2 bands from computed.json.

    Args:
        bands: GELOS band names (e.g. ``"BLUE"``) in tensor channel order.

    Raises:
        ValueError: if a band has no computed.json mapping.
    """
    s2_stats = OLMOEARTH_COMPUTED_STATS["sentinel2_l2a"]
    missing = [b for b in bands if b not in GELOS_TO_OLMOEARTH_S2_KEY]
    if missing:
        raise ValueError(
            f"No OlmoEarth normalization stats mapping for S2 band(s) {missing}. "
            f"Known bands: {sorted(GELOS_TO_OLMOEARTH_S2_KEY)}."
        )
    keys = [GELOS_TO_OLMOEARTH_S2_KEY[b] for b in bands]
    return [s2_stats[k]["mean"] for k in keys], [s2_stats[k]["std"] for k in keys]


def resolve_s1_band_stats(bands: Sequence[str]) -> tuple[list[float], list[float]]:
    """Per-band (means, stds) for S1 bands (case-insensitive VV/VH) from computed.json.

    Note: the computed.json sentinel1 stats are in DECIBELS — apply them to
    dB-converted data (see :func:`convert_to_db`).

    Raises:
        ValueError: if a band is not vv/vh.
    """
    s1_stats = OLMOEARTH_COMPUTED_STATS["sentinel1"]
    keys = [b.lower() for b in bands]
    missing = [b for b, k in zip(bands, keys) if k not in s1_stats]
    if missing:
        raise ValueError(
            f"No OlmoEarth normalization stats for S1 band(s) {missing}. "
            f"Known bands: {sorted(s1_stats)}."
        )
    return [s1_stats[k]["mean"] for k in keys], [s1_stats[k]["std"] for k in keys]


def build_s1_band_reorder_index(bands_s1: list[str]) -> list[int]:
    """Build channel-index permutation for GELOS S1 bands -> OlmoEarth ["vv","vh"] order.

    Case-insensitive: accepts GELOS-LC uppercase names ("VV", "VH").

    Raises:
        ValueError: if either "vv" or "vh" is missing from bands_s1.
    """
    band_to_pos: dict[str, int] = {}
    for pos, name in enumerate(bands_s1):
        band_to_pos.setdefault(name.lower(), pos)

    missing = [b for b in OLMOEARTH_S1_BAND_ORDER if b not in band_to_pos]
    if missing:
        raise ValueError(
            "OlmoEarth S1 requires VV and VH; missing: "
            f"{missing}. Configured S1 bands: {bands_s1}."
        )
    return [band_to_pos[b] for b in OLMOEARTH_S1_BAND_ORDER]


# Default OlmoEarth hidden size for the Base checkpoint (D=768). Used as a fallback
# for ``out_channels`` when the encoder does not expose an introspectable dim.
_DEFAULT_HIDDEN_DIM = 768


def calendar_to_olmoearth_timestamps(timestamps: torch.Tensor) -> torch.Tensor:
    """Convert canonical calendar dates to OlmoEarth's timestamp packing.

    Pure reindex logic (no model dependency). Maps a ``(..., 3)`` tensor of
    canonical ``[year, month, day]`` dates (month 1-12) to OlmoEarth's
    ``[day, month_index, year]`` packing (month zero-indexed): a column
    permutation plus ``month - 1``.

    Emits a ``UserWarning`` when the input looks implausible as canonical
    dates — any month outside 1-12 or any value in column 0 (year) below
    1900 — which catches callers still producing the legacy
    ``[day, month_index, year]`` packing (whose column 0 is a 1-31 day and
    whose month can be 0).

    Args:
        timestamps: ``(..., 3)`` integer tensor of ``[year, month, day]``.

    Returns:
        Tensor of the same shape packed as ``[day, month - 1, year]``.
    """
    year = timestamps[..., 0]
    month = timestamps[..., 1]
    day = timestamps[..., 2]
    if ((month < 1) | (month > 12)).any() or (year < 1900).any():
        warnings.warn(
            "calendar_to_olmoearth_timestamps: input has month outside 1-12 "
            "or year < 1900; expected canonical [year, month, day] dates "
            "(month 1-12). Legacy [day, month_index, year] packing must be "
            "converted to the canonical format.",
            UserWarning,
            stacklevel=2,
        )
    return torch.stack([day, month - 1, year], dim=-1)


def build_band_reorder_index(bands: list[str]) -> list[int | None]:
    """Build the channel-index map from ``bands`` -> OlmoEarth order.

    Pure index logic (no model dependency). Given the configured input ``bands``
    (the channel order of the incoming GELOS tensor), returns, for each band in
    :data:`OLMOEARTH_S2_BAND_ORDER`, the position of that band in ``bands`` —
    or ``None`` when the band is absent (a subset configuration: the backbone
    zero-fills that channel after pretraining normalization, see
    :class:`OlmoEarthBackbone`). With the full 12-band set the result is a plain
    permutation usable directly with ``index_select``.

    Args:
        bands: Channel names in the order they appear in the input tensor.

    Returns:
        A list of length ``len(OLMOEARTH_S2_BAND_ORDER)`` where element ``i`` is
        the position in ``bands`` of the ``i``-th OlmoEarth band, or ``None``
        if that band is not configured.

    Raises:
        ValueError: If ``bands`` is empty or names a band that is not one of the
            12 OlmoEarth S2L2A bands (unknown names are never silently
            zero-filled: a typo such as ``nir09`` must not drop ``WATER_VAPOR``).
    """
    unknown = [b for b in bands if b not in OLMOEARTH_S2_BAND_ORDER]
    if unknown:
        raise ValueError(
            f"Unknown OlmoEarth S2L2A band name(s): {unknown}. Configured bands: "
            f"{list(bands)}. Known bands: {OLMOEARTH_S2_BAND_ORDER}."
        )
    if not bands:
        raise ValueError("OlmoEarth requires at least one S2L2A band; got an empty band list.")

    band_to_pos: dict[str, int] = {}
    for pos, name in enumerate(bands):
        # First occurrence wins; duplicates are ignored deterministically.
        band_to_pos.setdefault(name, pos)

    return [band_to_pos.get(b) for b in OLMOEARTH_S2_BAND_ORDER]


def absent_s2_bands(reorder_index: Sequence[int | None]) -> list[str]:
    """Names (in OlmoEarth order) of the bands :func:`build_band_reorder_index` left ``None``."""
    return [b for b, src in zip(OLMOEARTH_S2_BAND_ORDER, reorder_index) if src is None]


# Largest number of absent S2 bands still considered in distribution with v1.2's
# pretraining band dropout (rate ~U(0, 0.2), i.e. ~10% of 12 bands on average).
# Above this the constructor emits a stronger warning (it never raises for it).
_MAX_ABSENT_BANDS_IN_DISTRIBUTION = 3


# olmoearth_pretrain.datatypes.MaskValue integer codes, duplicated here so the
# pure mask helpers stay usable (and testable) without the package installed.
# ``patch_mask_to_olmoearth_mask`` prefers the package's values when importable.
_MASK_ONLINE_ENCODER = 0
_MASK_MISSING = 3


def build_patch_nodata_mask(
    nodata: torch.Tensor, patch_size: int, threshold: float
) -> torch.Tensor:
    """Reduce a pixel-level nodata mask to a per-patch MISSING decision.

    Pure tensor logic (no model dependency). OlmoEarth's patch embedding reads
    only the top-left pixel of each ``patch_size x patch_size`` patch from the
    mask and requires masks to be patch-constant, so GELOS must decide per
    patch. With ``threshold=0`` a patch is MISSING when ANY of its pixels is
    nodata in ANY band at that timestep; with ``threshold>0`` it is MISSING
    when at least that fraction of its pixels is nodata.

    Args:
        nodata: Bool tensor ``(B, C, T, H, W)`` in GELOS layout, ``True`` = nodata.
        patch_size: Encoder patch size in pixels.
        threshold: Fraction in ``[0, 1]``; ``0`` masks any patch touching
            nodata, ``1.0`` masks only fully-nodata patches.

    Returns:
        Bool tensor ``(B, H//patch_size, W//patch_size, T)``, ``True`` = MISSING.

    Raises:
        ValueError: if ``nodata`` is not 5-D or H/W are not divisible by
            ``patch_size``.
    """
    if nodata.dim() != 5:
        raise ValueError(f"nodata mask must be (B, C, T, H, W), got shape {tuple(nodata.shape)}.")
    b, _c, t, h, w = nodata.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"Spatial dims (H={h}, W={w}) must be divisible by patch_size={patch_size}."
        )
    any_band = nodata.any(dim=1)  # (B, T, H, W)
    fraction = (
        any_band.reshape(b, t, h // patch_size, patch_size, w // patch_size, patch_size)
        .to(torch.float32)
        .mean(dim=(3, 5))
    )  # (B, T, H', W')
    missing = fraction > 0 if threshold == 0 else fraction >= threshold
    return missing.permute(0, 2, 3, 1).contiguous()  # (B, H', W', T)


def patch_mask_to_olmoearth_mask(
    patch_missing: torch.Tensor,
    patch_size: int,
    num_band_sets: int,
    missing_value: int | None = None,
    online_value: int | None = None,
) -> torch.Tensor:
    """Broadcast a per-patch MISSING decision to OlmoEarth's pixel-level mask.

    Args:
        patch_missing: Bool tensor ``(B, H', W', T)``, ``True`` = MISSING.
        patch_size: Encoder patch size; each patch decision is repeated over a
            ``patch_size x patch_size`` pixel block (so the mask is patch-constant
            and the top-left pixel the encoder reads carries the patch value).
        num_band_sets: ``S`` in the returned ``(B, H, W, T, S)`` mask (the
            encoder's S2 band-set count — 1 for v1.2, 3 for v1 — or 1 for S1);
            every band set shares the patch decision.
        missing_value / online_value: Integer codes. Default to
            ``MaskValue.MISSING`` / ``MaskValue.ONLINE_ENCODER`` from
            ``olmoearth_pretrain`` when importable, else the hard-coded copies.

    Returns:
        Int32 tensor ``(B, H'*patch_size, W'*patch_size, T, num_band_sets)``.
    """
    if missing_value is None or online_value is None:
        try:
            from olmoearth_pretrain.datatypes import MaskValue  # type: ignore[import-not-found]

            pkg_missing, pkg_online = MaskValue.MISSING.value, MaskValue.ONLINE_ENCODER.value
        except ImportError:
            pkg_missing, pkg_online = _MASK_MISSING, _MASK_ONLINE_ENCODER
        missing_value = pkg_missing if missing_value is None else missing_value
        online_value = pkg_online if online_value is None else online_value
    pixel_missing = patch_missing.repeat_interleave(patch_size, dim=1).repeat_interleave(
        patch_size, dim=2
    )  # (B, H, W, T)
    pixel_missing = pixel_missing.unsqueeze(-1).expand(*pixel_missing.shape, num_band_sets)
    return torch.where(
        pixel_missing,
        torch.tensor(missing_value, dtype=torch.int32, device=patch_missing.device),
        torch.tensor(online_value, dtype=torch.int32, device=patch_missing.device),
    )


def masked_mean(
    x: torch.Tensor, valid: torch.Tensor, dims: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean of ``x`` over ``dims`` counting only positions where ``valid`` is True.

    Args:
        x: Float tensor ``(..., D)``.
        valid: Bool tensor broadcastable to ``x.shape[:-1]`` (no trailing ``D``).
        dims: Non-negative axes of ``x`` to reduce (must not include the last).

    Returns:
        ``(mean, still_valid)``: ``mean`` has ``x``'s shape with ``dims``
        removed; positions with zero valid contributors are zero vectors.
        ``still_valid`` (bool, ``mean.shape[:-1]``) is ``True`` where at least
        one valid contributor existed, so callers can chain validity through
        successive pooling stages.
    """
    weight = valid.to(x.dtype)
    count = weight.expand(x.shape[:-1]).sum(dim=dims)
    total = (x * weight.unsqueeze(-1)).sum(dim=dims)
    return total / count.clamp(min=1).unsqueeze(-1), count > 0


def _resolve_s2_num_band_sets(inner_encoder: nn.Module, model_id: str) -> int:
    """Number of Sentinel-2 L2A band sets the loaded OlmoEarth encoder tokenizes.

    Prefers the encoder's own ``tokenization_config.get_num_bandsets("sentinel2_l2a")``
    (olmoearth_pretrain 0.1.1: 1 for v1.2 checkpoints, 3 for v1). Falls back to
    parsing ``model_id`` only when the encoder exposes no tokenization config.
    """
    tokenization_config = getattr(inner_encoder, "tokenization_config", None)
    get_num_bandsets = getattr(tokenization_config, "get_num_bandsets", None)
    if get_num_bandsets is not None:
        return int(get_num_bandsets("sentinel2_l2a"))
    # v1.2 (and v1.1) checkpoints use a single 12-band set; v1 uses three.
    return 1 if ("v1_2" in model_id or "v1_1" in model_id) else 3


@contextmanager
def _attention_mask_enabled(encoder: nn.Module):
    """Make an eval-mode OlmoEarth ``Encoder`` honor its token mask under attention.

    With ``fast_pass=False`` the encoder removes MISSING tokens, sorts the rest
    valid-first and zero-pads every sample to the batch's longest sequence
    (``flexi_vit.py: remove_masked_tokens``). The attention mask that would hide
    those pads is only built when ``self.training`` is True
    (``_maybe_get_attn_mask`` returns ``None`` otherwise), so in eval mode the
    zero pads still act as attention keys and a sample's output depends on its
    batch-mates' mask counts (measured: 0.11 abs diff vs. a B=1 run; 2.5e-6 with
    the flag set). We therefore flip the ``training`` attribute on the
    ``Encoder`` object ONLY — not ``.train()``, which would recurse into
    submodules and re-enable DropPath/dropout/band-dropout. ``Encoder`` reads
    ``self.training`` nowhere else (olmoearth_pretrain 0.1.1).
    """
    previous = encoder.training
    encoder.training = True
    try:
        yield
    finally:
        encoder.training = previous


class OlmoEarthBackbone(nn.Module):
    """Adapts an OlmoEarth encoder to terratorch's backbone ``forward_features``.

    Accepts GELOS's channels-first ``(B, C, T, H, W)`` Sentinel-2 L2A tensor,
    reorders bands to OlmoEarth's expected 12-band order, transposes to
    channels-last ``(B, H, W, T, C)``, applies OlmoEarth's pretraining
    normalization (see below), runs the OlmoEarth encoder, mean-pools the
    token tensor over the spectral-group axis, and returns a single-element list
    ``[tokens]`` (terratorch necks expect a list of layer tensors).

    Band subsets (v1.2 checkpoints only): ``bands`` may list only some of the
    12 S2L2A bands (``C == len(bands)``; it must match ``data.bands.S2L2A``).
    Each absent band's channel is set to exactly 0 *after* pretraining
    normalization, for the whole sample (every pixel and timestep), whether or
    not ``apply_pretraining_normalization`` is on — the same input v1.2's
    pretraining band dropout produced, so the encoder infers the missing bands
    from the present ones. Nodata masking is unaffected (the mask keeps the
    dataset's channel count). Construction warns once, naming the zero-filled
    bands, and more strongly when more than three bands are absent (far outside
    the ~10% pretraining dropout rate). v1 checkpoints (deprecated) have no band
    dropout, so they raise ``ValueError`` on a subset; the generation is read
    from the loaded encoder's S2 band-set count (1 = v1.2, 3 = v1).

    With ``apply_pretraining_normalization=True`` (default), inputs must be RAW
    sensor scale — S2 L2A digital numbers (0–10000) and S1 linear-power gamma0 —
    and the wrapper replicates the normalization OlmoEarth's pretraining data
    loader applied (the encoder itself has none): S1 is converted to dB
    (``10*log10(clip(x, 1e-10))``) and both modalities are min-max scaled per
    band over mean±2σ using ``olmoearth_pretrain``'s computed.json stats. The
    datamodule must be configured with ``normalize: false`` and must NOT apply
    ``db_scale_bands`` to S1, or values get double-transformed. Set
    ``apply_pretraining_normalization=False`` only if you pre-normalize inputs
    to OlmoEarth's pretraining scale yourself.

    The temporal axis is handled per ``temporal_pooling``:

    - ``"mean"`` (default): mean over T, tokens shape ``(B, H'*W', D)``.
    - ``"keep"``: per-timestep tokens preserved, flattened time-major to
      ``(B, T*H'*W', D)`` — token ``t*H'*W' + i`` is spatial patch ``i`` at
      timestep ``t``, matching the layout Prithvi produces so the same strided
      ``slice_args`` extraction strategies apply (e.g. center patch across all
      timesteps: ``start=center_idx, step=H'*W'``). Note these tokens come from
      one joint space-time attention pass, so a single timestep's tokens still
      carry cross-time context (unlike running the encoder with T=1).

    ``spatial_pooling`` (default ``None`` = off) aggregates the ``H'xW'`` token
    grid after encoding:

    - int factor ``s``: average-pools non-overlapping ``sxs`` neighborhoods, so
      each output token covers ``(s*patch_size)^2`` input pixels. Use it to
      match the spatial footprint of larger-patch models: with ``patch_size=4,
      spatial_pooling=4`` each token covers 16x16 pixels and the grid (and token
      indices) line up exactly with Prithvi/TerraMind's 16-pixel patches.
    - ``"mean"``: masked mean over the whole valid token grid per timestep, so
      every sample yields the same-size vector whatever its nodata footprint:
      ``(B, T, D)`` with ``temporal_pooling="keep"``, ``(B, 1, D)`` with
      ``"mean"``.

    Encoding always happens at the fine ``patch_size``; only outputs are pooled.

    Nodata masking (``mask_nodata=True``, default): when the task stashed a
    ``nodata_mask`` for the batch (``GELOSDataModule`` with ``nodata_value``
    set), each ``patch_size x patch_size`` patch whose nodata fraction (any
    band, per timestep) is nodata (default ``nodata_patch_threshold=0``; a value
    in ``(0, 1]`` instead requires at least that fraction) is flagged
    ``MaskValue.MISSING``. Missing tokens are removed before attention (so they
    neither attend nor are attended to) and excluded from every pooling mean
    (band-set, S1/S2 fusion, spatial, temporal). In un-pooled grids masked
    positions come back as zero vectors (the encoder's convention); a pooled
    position with zero valid contributors is also a zero vector and triggers a
    warning. A sample with no valid patch at all is encoded unmasked (with a
    warning) because the encoder cannot process an empty token sequence.
    """

    def __init__(
        self,
        pretrained: bool = True,
        model_id: str = "allenai/OlmoEarth-v1_2-Base",
        bands: list[str] | None = None,
        patch_size: int = 4,
        hidden_dim: int | None = None,
        bands_s1: list[str] | None = None,
        warn_missing_s1: bool = True,
        temporal_pooling: str = "mean",
        spatial_pooling: int | str | None = None,
        apply_pretraining_normalization: bool = True,
        mask_nodata: bool = True,
        nodata_patch_threshold: float = 0.0,
        **kwargs,  # tolerate terratorch-injected args
    ) -> None:
        super().__init__()
        if temporal_pooling not in ("mean", "keep"):
            raise ValueError(
                f"temporal_pooling must be 'mean' or 'keep', got {temporal_pooling!r}."
            )
        if (
            spatial_pooling is not None
            and spatial_pooling != "mean"
            and (not isinstance(spatial_pooling, int) or spatial_pooling < 1)
        ):
            raise ValueError(
                "spatial_pooling must be None, a positive int, or 'mean', "
                f"got {spatial_pooling!r}."
            )
        if not (0 <= nodata_patch_threshold <= 1):
            raise ValueError(
                "nodata_patch_threshold must be in [0, 1] (fraction of nodata pixels "
                f"that marks a patch MISSING), got {nodata_patch_threshold!r}."
            )
        self.model_id = model_id
        self.pretrained = pretrained
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.temporal_pooling = temporal_pooling
        self.spatial_pooling = spatial_pooling
        self.mask_nodata = mask_nodata
        self.nodata_patch_threshold = nodata_patch_threshold
        self.bands = list(bands) if bands else list(OLMOEARTH_S2_BAND_ORDER)

        # Transient per-batch acquisition dates, stashed by the task's
        # predict_step (see gelos.generation.LenientEmbeddingGenerationTask).
        # Plain attribute on purpose: NOT a buffer/parameter, so it is never saved
        # in the state_dict nor moved by ``.to()`` — it is cleared after each batch.
        self._batch_timestamps: torch.Tensor | None = None
        # Transient per-batch pixel nodata mask from GELOSDataModule's NoDataRemap
        # ({modality: BoolTensor(B, C, T, H, W)} or a single BoolTensor), stashed
        # the same way. Plain attribute for the same reason: never in state_dict.
        self._batch_nodata_mask: dict[str, torch.Tensor] | torch.Tensor | None = None

        # Precompute and validate the band-reorder map eagerly so misconfigured
        # bands (unknown names) fail at construction time, not mid-forward.
        # Absent bands are ``None`` here; they are zero-filled in forward_features
        # (v1.2 only — the v1 check below needs the loaded encoder).
        self.reorder_index = build_band_reorder_index(self.bands)
        self.absent_bands: list[str] = absent_s2_bands(self.reorder_index)
        # Gather/scatter indices for the subset path: present source channels
        # of the input tensor and their target slots in the 12-band layout.
        self._present_source_index = [s for s in self.reorder_index if s is not None]
        self._present_target_index = [i for i, s in enumerate(self.reorder_index) if s is not None]
        self._absent_target_index = [i for i, s in enumerate(self.reorder_index) if s is None]

        self.bands_s1 = list(bands_s1) if bands_s1 else None
        self.warn_missing_s1 = warn_missing_s1
        self.reorder_index_s1 = (
            build_s1_band_reorder_index(self.bands_s1) if self.bands_s1 is not None else None
        )

        # Pretraining normalization stats, resolved for the POST-reorder channel
        # order (reordering maps the configured band subset/order onto the
        # canonical OlmoEarth orders, so stats are always indexed consistently).
        self.apply_pretraining_normalization = apply_pretraining_normalization
        if apply_pretraining_normalization:
            self._s2_norm_means, self._s2_norm_stds = resolve_s2_band_stats(
                OLMOEARTH_S2_BAND_ORDER
            )
            self._s1_norm_means, self._s1_norm_stds = resolve_s1_band_stats(
                OLMOEARTH_S1_BAND_ORDER
            )

        try:
            from olmoearth_pretrain.model_loader import (  # type: ignore[import-not-found]
                ModelID,
                load_model_from_id,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "OlmoEarthBackbone requires the 'olmoearth-pretrain' package. "
                "Install it with `pip install gelos` (olmoearth-pretrain is a "
                "core gelos dependency)."
            ) from exc

        # ModelID values are bare checkpoint names ("OlmoEarth-v1-Base"); strip
        # the optional "allenai/" org prefix that the factory defaults include.
        model_id_clean = model_id.split("/")[-1]
        self.encoder = load_model_from_id(ModelID(model_id_clean), load_weights=pretrained)

        # Expose the embedding dim so terratorch necks can introspect. The
        # LatentMIM wrapper exposes its inner encoder as .encoder; prefer that
        # for dim introspection, then fall back to the size-variant hidden_dim
        # hint (set by the factory functions), then the Base default.
        fallback_dim = self.hidden_dim if self.hidden_dim is not None else _DEFAULT_HIDDEN_DIM
        inner_enc = getattr(self.encoder, "encoder", self.encoder)
        self.out_channels = getattr(
            inner_enc,
            "embedding_dim",
            getattr(inner_enc, "embed_dim", fallback_dim),
        )

        # S2 band-set count of the loaded checkpoint: 1 for v1.2 (one 12-band
        # set, pretrained with band dropout), 3 for v1 (10m/20m/60m, no band
        # dropout). Drives the S2 mask's last dim and the band-subset guard.
        self.s2_num_band_sets = _resolve_s2_num_band_sets(inner_enc, model_id_clean)

        if self.absent_bands:
            if self.s2_num_band_sets != 1:
                raise ValueError(
                    f"OlmoEarth checkpoint {model_id!r} tokenizes Sentinel-2 L2A as "
                    f"{self.s2_num_band_sets} band sets (a deprecated v1 checkpoint) and "
                    "was not pretrained with band dropout, so a band subset is not "
                    f"supported: absent band(s) {self.absent_bands}. Supply all 12 "
                    f"bands {OLMOEARTH_S2_BAND_ORDER} or use a v1.2 checkpoint "
                    "(e.g. 'allenai/OlmoEarth-v1_2-Base'), which zero-fills absent bands."
                )
            n_absent, n_total = len(self.absent_bands), len(OLMOEARTH_S2_BAND_ORDER)
            if n_absent > _MAX_ABSENT_BANDS_IN_DISTRIBUTION:
                warnings.warn(
                    f"OlmoEarthBackbone: {n_absent} of {n_total} Sentinel-2 L2A bands are "
                    f"absent from model_args.bands and will be zero-filled after "
                    f"pretraining normalization: {self.absent_bands}. This is FAR "
                    "OUTSIDE OlmoEarth v1.2's pretraining band-dropout regime (about "
                    f"10% of bands, at most {_MAX_ABSENT_BANDS_IN_DISTRIBUTION} absent "
                    "is in distribution); expect degraded embeddings. Configured "
                    f"bands: {self.bands}.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"OlmoEarthBackbone: Sentinel-2 L2A band(s) {self.absent_bands} are "
                    "absent from model_args.bands and will be zero-filled after "
                    "pretraining normalization (matching OlmoEarth v1.2's pretraining "
                    f"band dropout; {n_total - n_absent} of {n_total} bands present). "
                    f"Configured bands: {self.bands}.",
                    UserWarning,
                    stacklevel=2,
                )

    def set_batch_timestamps(self, timestamps: torch.Tensor | None) -> None:
        """Stash the current batch's per-timestep timestamps for ``forward_features``.

        The stash is canonical calendar dates ``(B, T, 3)`` ``[year, month, day]``
        (month 1-12); conversion to OlmoEarth's ``[day, month_index, year]``
        packing happens at consumption time in ``forward_features`` via
        :func:`calendar_to_olmoearth_timestamps`.

        Called by the task's ``predict_step`` because terratorch's
        ``get_embeddings``/``self.model(input)`` call site forwards only ``input``
        to the backbone — there is no kwargs channel for extra batch keys.
        """
        self._batch_timestamps = timestamps

    def clear_batch_timestamps(self) -> None:
        """Clear the stashed timestamps (called in the task's ``finally``)."""
        self._batch_timestamps = None

    def set_batch_nodata_mask(self, mask: dict[str, torch.Tensor] | torch.Tensor | None) -> None:
        """Stash the current batch's pixel nodata mask for ``forward_features``.

        Same side-channel as :meth:`set_batch_timestamps`: terratorch forwards
        only ``batch["image"]`` to the backbone, so the task pops
        ``batch["nodata_mask"]`` and parks it here for one batch. Accepts the
        ``{modality: BoolTensor(B, C, T, H, W)}`` dict produced for dict batches
        or the single ``BoolTensor`` of the tensor path (treated as the S2 mask).
        """
        self._batch_nodata_mask = mask

    def clear_batch_nodata_mask(self) -> None:
        """Clear the stashed nodata mask (called in the task's ``finally``)."""
        self._batch_nodata_mask = None

    def _resolve_patch_missing(
        self,
        nodata: torch.Tensor | None,
        expected: tuple[int, int, int, int],
        device: torch.device,
        modality: str,
    ) -> torch.Tensor | None:
        """Turn a stashed pixel nodata mask into a ``(B, H', W', T)`` MISSING mask.

        Returns ``None`` (= nothing masked) when no mask was stashed or its
        ``(B, T, H, W)`` disagrees with the input (warns, mirroring the
        timestamp shape fallback).
        """
        if nodata is None:
            return None
        got = (
            (nodata.shape[0], nodata.shape[2], nodata.shape[3], nodata.shape[4])
            if nodata.dim() == 5
            else tuple(nodata.shape)
        )
        if got != expected:
            warnings.warn(
                f"OlmoEarthBackbone: stashed {modality} nodata mask has (B, T, H, W) "
                f"{got}, expected {expected}; ignoring the mask for this batch.",
                UserWarning,
                stacklevel=3,
            )
            return None
        return build_patch_nodata_mask(
            nodata.to(device=device, dtype=torch.bool),
            self.patch_size,
            self.nodata_patch_threshold,
        )

    def forward_features(self, x, **kwargs) -> list[torch.Tensor]:
        """Run the OlmoEarth encoder and return ``[tokens]``.

        When ``set_batch_timestamps`` stashed a canonical ``(B, T, 3)``
        ``[year, month, day]`` tensor matching the input's batch/timestep
        dims, it is converted here to OlmoEarth's ``[day, month_index, year]``
        packing via :func:`calendar_to_olmoearth_timestamps`; otherwise the
        constant dummy date ``[15, 0, 2020]`` is used (with a warning on
        shape mismatch).

        Args:
            x: Either the S2L2A tensor ``(B, C, T, H, W)`` or a dict of modalities
                keyed by sensor name. When a dict, ``"S2L2A"`` is required;
                ``"S1RTC"`` is used when ``bands_s1`` was configured.

        Returns:
            A single-element list whose tensor has shape ``(B, H'*W', D)`` when
            ``temporal_pooling="mean"``, or ``(B, T*H'*W', D)`` (time-major) when
            ``temporal_pooling="keep"``. With ``spatial_pooling=s``, ``H'`` and
            ``W'`` above are the pooled grid dims (``H/patch_size/s``). With
            ``spatial_pooling="mean"`` the grid collapses entirely: ``(B, T, D)``
            for ``"keep"``, ``(B, 1, D)`` for ``"mean"``. When a nodata mask is
            stashed, masked patches are excluded from attention and from every
            mean; un-pooled masked grid positions are zero vectors.
        """
        from olmoearth_pretrain.datatypes import (  # type: ignore[import-not-found]
            MaskedOlmoEarthSample,
            MaskValue,
        )

        # --- Unpack modalities ---
        x_s1 = None
        if isinstance(x, dict):
            x_s2 = x["S2L2A"]
            if self.reorder_index_s1 is not None:
                if "S1RTC" in x:
                    x_s1 = x["S1RTC"]
                elif self.warn_missing_s1:
                    warnings.warn(
                        "OlmoEarthBackbone: S1 bands configured but 'S1RTC' not found "
                        "in batch dict; running S2-only forward.",
                        UserWarning,
                        stacklevel=2,
                    )
        else:
            x_s2 = x

        if x_s2.dim() != 5:
            raise ValueError(
                f"OlmoEarthBackbone expects a (B, C, T, H, W) tensor, got shape {tuple(x_s2.shape)}."
            )

        b, c, t, h, w = x_s2.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Spatial dims (H={h}, W={w}) must be divisible by patch_size={self.patch_size}."
            )

        if c != len(self.bands):
            raise ValueError(
                f"OlmoEarthBackbone: input has {c} S2L2A channels but model_args.bands "
                f"lists {len(self.bands)} ({self.bands}); data.bands.S2L2A and "
                "model_args.bands must match."
            )

        # 1. Reorder S2 channels to OlmoEarth band order. With the full 12-band
        # set this is a plain gather (unchanged path); with a subset, present
        # channels are scattered into a zero 12-channel tensor and the absent
        # slots are re-zeroed after normalization below.
        n_oe = len(OLMOEARTH_S2_BAND_ORDER)
        if self.absent_bands:
            x_full = x_s2.new_zeros((b, n_oe, t, h, w))
            x_full[:, self._present_target_index] = x_s2[:, self._present_source_index]
            x_s2 = x_full  # (B, 12, T, H, W)
        else:
            idx_s2 = torch.as_tensor(self.reorder_index, device=x_s2.device, dtype=torch.long)
            x_s2 = x_s2.index_select(dim=1, index=idx_s2)  # (B, 12, T, H, W)

        # 2. channels-first -> channels-last: (B, C, T, H, W) -> (B, H, W, T, C)
        x_s2 = x_s2.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, T, 12)

        # 2b. Pretraining normalization (expects RAW DN 0-10000): per-band
        # min-max over mean±2σ, replicating olmoearth_pretrain's Normalizer.
        if self.apply_pretraining_normalization:
            x_s2 = minmax_normalize(x_s2, self._s2_norm_means, self._s2_norm_stds)

        # 2c. Zero-fill absent bands AFTER normalization (raw 0 would normalize
        # to a non-zero value): exactly what v1.2's pretraining band dropout fed
        # the encoder — the whole band channel, every pixel and timestep, is 0.
        if self.absent_bands:
            x_s2[..., self._absent_target_index] = 0.0

        # 3. Per-timestep timestamps (real or dummy fallback). The stash is
        # canonical [year, month, day]; convert to OlmoEarth's
        # [day, month_index, year] packing at consumption.
        timestamps = None
        if self._batch_timestamps is not None:
            candidate = self._batch_timestamps.to(device=x_s2.device, dtype=torch.long)
            if tuple(candidate.shape) == (b, t, 3):
                timestamps = calendar_to_olmoearth_timestamps(candidate)
            else:
                warnings.warn(
                    "OlmoEarthBackbone: stashed timestamps have shape "
                    f"{tuple(candidate.shape)}, expected {(b, t, 3)}; "
                    "falling back to the constant date."
                )
        if timestamps is None:
            timestamps = (
                torch.tensor([15, 0, 2020], dtype=torch.long, device=x_s2.device)
                .view(1, 1, 3)
                .expand(b, t, 3)
                .contiguous()
            )

        # 4. S2 patch-level MISSING decision from the stashed pixel nodata mask
        # (dict batch: per-modality entry; tensor batch: the tensor is the S2
        # mask). None / mask_nodata=False -> nothing masked.
        stash = self._batch_nodata_mask if self.mask_nodata else None
        nodata_s2 = nodata_s1 = None
        if isinstance(stash, dict):
            nodata_s2, nodata_s1 = stash.get("S2L2A"), stash.get("S1RTC")
        elif stash is not None:
            nodata_s2 = stash
        p = self.patch_size
        hp, wp = h // p, w // p
        missing_s2 = self._resolve_patch_missing(nodata_s2, (b, t, h, w), x_s2.device, "S2L2A")
        if missing_s2 is None:
            missing_s2 = torch.zeros(b, hp, wp, t, dtype=torch.bool, device=x_s2.device)

        # 5. S1 path (optional).
        sentinel1_tensor = None
        sentinel1_mask = None
        missing_s1 = None
        if x_s1 is not None:
            if x_s1.dim() != 5:
                raise ValueError(
                    f"OlmoEarthBackbone S1 expects (B, C, T, H, W), got {tuple(x_s1.shape)}."
                )
            idx_s1 = torch.as_tensor(self.reorder_index_s1, device=x_s1.device, dtype=torch.long)
            x_s1 = x_s1.index_select(dim=1, index=idx_s1)  # (B, 2, T, H, W)
            x_s1 = x_s1.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, T, 2)
            # Pretraining normalization (expects RAW LINEAR POWER): clip 1e-10 ->
            # 10*log10 -> per-band min-max over mean±2σ (dB-scale stats).
            if self.apply_pretraining_normalization:
                x_s1 = minmax_normalize(
                    convert_to_db(x_s1), self._s1_norm_means, self._s1_norm_stds
                )
            sentinel1_tensor = x_s1
            missing_s1 = self._resolve_patch_missing(
                nodata_s1, tuple(x_s1.shape[i] for i in (0, 3, 1, 2)), x_s1.device, "S1RTC"
            )
            if missing_s1 is None:
                missing_s1 = torch.zeros(
                    b,
                    x_s1.shape[1] // p,
                    x_s1.shape[2] // p,
                    t,
                    dtype=torch.bool,
                    device=x_s1.device,
                )

        # 5b. Fully-masked guard: the encoder asserts on an empty token sequence
        # (add_removed_tokens) and its mean pooling raises on zero valid tokens,
        # so a sample with no valid patch in any modality is encoded unmasked.
        valid_any = ~missing_s2
        if missing_s1 is not None:
            valid_any = valid_any | ~missing_s1
        no_valid = ~valid_any.flatten(1).any(dim=1)  # (B,)
        if bool(no_valid.any()):
            warnings.warn(
                "OlmoEarthBackbone: batch samples "
                f"{no_valid.nonzero().flatten().tolist()} have no valid patch after nodata "
                "masking; encoding them unmasked (their embeddings are nodata-dominated).",
                UserWarning,
                stacklevel=2,
            )
            missing_s2[no_valid] = False
            if missing_s1 is not None:
                missing_s1[no_valid] = False

        # 5c. Broadcast patch decisions to OlmoEarth's pixel-level int masks:
        # S2 (B, H, W, T, S) with S = the encoder's S2 band-set count (1 for
        # v1.2, 3 for v1: 10m/20m/60m); S1 (B, H, W, T, 1).
        missing_code, online_code = MaskValue.MISSING.value, MaskValue.ONLINE_ENCODER.value
        sentinel2_mask = patch_mask_to_olmoearth_mask(
            missing_s2, p, self.s2_num_band_sets, missing_code, online_code
        )
        if sentinel1_tensor is not None:
            sentinel1_mask = patch_mask_to_olmoearth_mask(
                missing_s1, p, 1, missing_code, online_code
            )

        # 6. Build OlmoEarth sample.
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=x_s2,
            sentinel2_l2a_mask=sentinel2_mask,
            sentinel1=sentinel1_tensor,
            sentinel1_mask=sentinel1_mask,
            timestamps=timestamps,
        )

        # 7. Encode — call the inner encoder directly to skip the decoder.
        # fast_pass=True ignores the mask entirely (masked tokens are neither
        # removed nor hidden from attention), so it is only used when nothing is
        # masked, keeping that path bit-identical to the unmasked behaviour.
        any_masked = bool(missing_s2.any()) or (missing_s1 is not None and bool(missing_s1.any()))
        inner_encoder = self.encoder.encoder
        if not any_masked:
            output_dict = inner_encoder(sample, fast_pass=True, patch_size=p)
        else:
            # fast_pass=False removes MISSING tokens before attention; see
            # _attention_mask_enabled for why the training flag must be set so
            # zero-padded batch-mates cannot leak into attention. The extra
            # "project_aggregated" output key is ignored.
            with _attention_mask_enabled(inner_encoder):
                output_dict = inner_encoder(sample, fast_pass=False, patch_size=p)
        tokens_and_masks = output_dict["tokens_and_masks"]

        # 8. Pool S2 tokens over band-sets: (B, H', W', T, S, D) -> (B, H', W', T, D)
        # (S = 1 for v1.2, so the mean is the identity; 3 for v1). All band sets
        # share the patch decision, so a plain mean is exact.
        s2_tokens = tokens_and_masks.sentinel2_l2a  # (B, H', W', T, S, D)
        pooled = s2_tokens.mean(dim=4)  # (B, H', W', T, D)
        valid = ~missing_s2  # (B, H', W', T)

        # 9. Fuse S1 tokens when present: average over the modalities valid at
        # each patch (equal-weight /2 when both are valid).
        if sentinel1_tensor is not None:
            s1_tokens = tokens_and_masks.sentinel1  # (B, H', W', T, 1, D)
            s1_pooled = s1_tokens.mean(dim=4)
            valid_s1 = ~missing_s1
            w_s2 = valid.to(pooled.dtype).unsqueeze(-1)
            w_s1 = valid_s1.to(pooled.dtype).unsqueeze(-1)
            pooled = (pooled * w_s2 + s1_pooled * w_s1) / (w_s2 + w_s1).clamp(min=1)
            valid = valid | valid_s1

        # 9b. Optional spatial pooling (masked means; validity chained through).
        pooled_any = False
        if isinstance(self.spatial_pooling, int) and self.spatial_pooling > 1:
            # Average sxs token neighborhoods so each output token covers
            # (s*patch_size)^2 pixels.
            s = self.spatial_pooling
            bb, hp, wp, tt, d = pooled.shape
            if hp % s != 0 or wp % s != 0:
                raise ValueError(
                    f"Token grid ({hp}x{wp}) must be divisible by spatial_pooling={s}."
                )
            pooled, valid = masked_mean(
                pooled.reshape(bb, hp // s, s, wp // s, s, tt, d),
                valid.reshape(bb, hp // s, s, wp // s, s, tt),
                dims=(2, 4),
            )  # (B, H'/s, W'/s, T, D), (B, H'/s, W'/s, T)
            pooled_any = True
        elif self.spatial_pooling == "mean":
            # Whole-grid masked mean per timestep: same-size vector per sample.
            pooled, valid = masked_mean(pooled, valid, dims=(1, 2))  # (B, T, D), (B, T)
            pooled_any = True

        # 10. Handle time, flatten to a token sequence.
        if self.spatial_pooling == "mean":
            if self.temporal_pooling == "mean":
                pooled, valid = masked_mean(pooled, valid, dims=(1,))  # (B, D), (B,)
                tokens = pooled.unsqueeze(1)  # (B, 1, D)
            else:  # "keep"
                tokens = pooled  # (B, T, D)
        elif self.temporal_pooling == "mean":
            pooled, valid = masked_mean(pooled, valid, dims=(3,))  # (B, H', W', D)
            pooled_any = True
            bb, hp, wp, d = pooled.shape
            tokens = pooled.reshape(bb, hp * wp, d)
        else:  # "keep": time-major (B, T*H'*W', D)
            bb, hp, wp, tt, d = pooled.shape
            tokens = pooled.permute(0, 3, 1, 2, 4).reshape(bb, tt * hp * wp, d)

        if pooled_any and any_masked and not bool(valid.all()):
            warnings.warn(
                "OlmoEarthBackbone: some pooled output positions had zero valid (non-"
                f"nodata) tokens ({int((~valid).sum())} of {valid.numel()}); they are "
                "zero vectors.",
                UserWarning,
                stacklevel=2,
            )

        return [tokens]

    def forward(self, x, **kwargs) -> list[torch.Tensor]:
        return self.forward_features(x, **kwargs)


def _warn_v1_deprecated(factory_name: str) -> None:
    """Emit the OlmoEarth v1 factory deprecation warning (issue #81)."""
    replacement = factory_name.replace("olmoearth_v1_", "olmoearth_v1_2_").replace(
        "_large", "_base"
    )
    warnings.warn(
        f"{factory_name} (OlmoEarth v1) is deprecated and will be removed in a future "
        f"GELOS release; use {replacement} (OlmoEarth v1.2) instead. v1 checkpoints "
        "still require the full 12-band S2L2A input (band subsets are zero-filled "
        "only for v1.2).",
        DeprecationWarning,
        stacklevel=3,
    )


def olmoearth_v1_nano(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Nano",
    bands: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 128,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth Nano checkpoint (D=128). DEPRECATED.

    .. deprecated::
        OlmoEarth v1 checkpoints are deprecated in GELOS; use the v1.2 factory
        ``olmoearth_v1_2_nano`` instead. Emits a ``DeprecationWarning`` and still
        requires the full 12-band S2L2A input (no band-subset zero-fill).

    Registered under its own name (``olmoearth_v1_nano``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_nano",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.

    NOTE: inputs must be RAW S2 L2A digital numbers (0-10000). With
    ``apply_pretraining_normalization=True`` (default) the wrapper min-max
    normalizes each band over mean±2σ, exactly as OlmoEarth's pretraining data
    loader did. Configure the datamodule with ``normalize: false``.
    """
    _warn_v1_deprecated("olmoearth_v1_nano")
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_tiny(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Tiny",
    bands: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 192,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth Tiny checkpoint (D=192). DEPRECATED.

    .. deprecated::
        OlmoEarth v1 checkpoints are deprecated in GELOS; use the v1.2 factory
        ``olmoearth_v1_2_tiny`` instead. Emits a ``DeprecationWarning`` and still
        requires the full 12-band S2L2A input (no band-subset zero-fill).

    Registered under its own name (``olmoearth_v1_tiny``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_tiny",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.

    NOTE: inputs must be RAW S2 L2A digital numbers (0-10000). With
    ``apply_pretraining_normalization=True`` (default) the wrapper min-max
    normalizes each band over mean±2σ, exactly as OlmoEarth's pretraining data
    loader did. Configure the datamodule with ``normalize: false``.
    """
    _warn_v1_deprecated("olmoearth_v1_tiny")
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_base(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Base",
    bands: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 768,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth Base checkpoint (D=768). DEPRECATED.

    .. deprecated::
        OlmoEarth v1 checkpoints are deprecated in GELOS; use the v1.2 factory
        ``olmoearth_v1_2_base`` instead. Emits a ``DeprecationWarning`` and still
        requires the full 12-band S2L2A input (no band-subset zero-fill).

    Registered under its own name (``olmoearth_v1_base``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_base",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.

    NOTE: inputs must be RAW S2 L2A digital numbers (0-10000). With
    ``apply_pretraining_normalization=True`` (default) the wrapper min-max
    normalizes each band over mean±2σ, exactly as OlmoEarth's pretraining data
    loader did. Configure the datamodule with ``normalize: false``.
    """
    _warn_v1_deprecated("olmoearth_v1_base")
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_large(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Large",
    bands: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 1024,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth Large checkpoint (D=1024). DEPRECATED.

    .. deprecated::
        OlmoEarth v1 checkpoints are deprecated in GELOS; use the v1.2 factory
        ``olmoearth_v1_2_base`` instead. Emits a ``DeprecationWarning`` and still
        requires the full 12-band S2L2A input (no band-subset zero-fill).

    Registered under its own name (``olmoearth_v1_large``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_large",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.

    NOTE: inputs must be RAW S2 L2A digital numbers (0-10000). With
    ``apply_pretraining_normalization=True`` (default) the wrapper min-max
    normalizes each band over mean±2σ, exactly as OlmoEarth's pretraining data
    loader did. Configure the datamodule with ``normalize: false``.
    """
    _warn_v1_deprecated("olmoearth_v1_large")
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_nano_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Nano",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 128,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth Nano with S2+S1 combined input (D=128). DEPRECATED.

    .. deprecated::
        OlmoEarth v1 checkpoints are deprecated in GELOS; use the v1.2 factory
        ``olmoearth_v1_2_nano_s1s2`` instead. Emits a ``DeprecationWarning`` and still
        requires the full 12-band S2L2A input (no band-subset zero-fill).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_nano``.

    NOTE: inputs must be RAW sensor scale — S2 L2A digital numbers (0-10000) and
    S1 linear-power gamma0 (e.g. Planetary Computer sentinel-1-rtc). With
    ``apply_pretraining_normalization=True`` (default) the wrapper converts S1 to
    dB and min-max normalizes both modalities per band over mean±2σ, exactly as
    OlmoEarth's pretraining data loader did. Configure the datamodule with
    ``normalize: false`` and do NOT apply ``db_scale_bands`` to S1, or values get
    double-transformed.
    """
    _warn_v1_deprecated("olmoearth_v1_nano_s1s2")
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_tiny_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Tiny",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 192,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth Tiny with S2+S1 combined input (D=192). DEPRECATED.

    .. deprecated::
        OlmoEarth v1 checkpoints are deprecated in GELOS; use the v1.2 factory
        ``olmoearth_v1_2_tiny_s1s2`` instead. Emits a ``DeprecationWarning`` and still
        requires the full 12-band S2L2A input (no band-subset zero-fill).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_tiny``.

    NOTE: inputs must be RAW sensor scale — S2 L2A digital numbers (0-10000) and
    S1 linear-power gamma0 (e.g. Planetary Computer sentinel-1-rtc). With
    ``apply_pretraining_normalization=True`` (default) the wrapper converts S1 to
    dB and min-max normalizes both modalities per band over mean±2σ, exactly as
    OlmoEarth's pretraining data loader did. Configure the datamodule with
    ``normalize: false`` and do NOT apply ``db_scale_bands`` to S1, or values get
    double-transformed.
    """
    _warn_v1_deprecated("olmoearth_v1_tiny_s1s2")
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_base_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Base",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 768,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth Base with S2+S1 combined input (D=768). DEPRECATED.

    .. deprecated::
        OlmoEarth v1 checkpoints are deprecated in GELOS; use the v1.2 factory
        ``olmoearth_v1_2_base_s1s2`` instead. Emits a ``DeprecationWarning`` and still
        requires the full 12-band S2L2A input (no band-subset zero-fill).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_base``.

    NOTE: inputs must be RAW sensor scale — S2 L2A digital numbers (0-10000) and
    S1 linear-power gamma0 (e.g. Planetary Computer sentinel-1-rtc). With
    ``apply_pretraining_normalization=True`` (default) the wrapper converts S1 to
    dB and min-max normalizes both modalities per band over mean±2σ, exactly as
    OlmoEarth's pretraining data loader did. Configure the datamodule with
    ``normalize: false`` and do NOT apply ``db_scale_bands`` to S1, or values get
    double-transformed.
    """
    _warn_v1_deprecated("olmoearth_v1_base_s1s2")
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_large_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Large",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 1024,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth Large with S2+S1 combined input (D=1024). DEPRECATED.

    .. deprecated::
        OlmoEarth v1 checkpoints are deprecated in GELOS; use the v1.2 factory
        ``olmoearth_v1_2_base_s1s2`` instead. Emits a ``DeprecationWarning`` and still
        requires the full 12-band S2L2A input (no band-subset zero-fill).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_large``.

    NOTE: inputs must be RAW sensor scale — S2 L2A digital numbers (0-10000) and
    S1 linear-power gamma0 (e.g. Planetary Computer sentinel-1-rtc). With
    ``apply_pretraining_normalization=True`` (default) the wrapper converts S1 to
    dB and min-max normalizes both modalities per band over mean±2σ, exactly as
    OlmoEarth's pretraining data loader did. Configure the datamodule with
    ``normalize: false`` and do NOT apply ``db_scale_bands`` to S1, or values get
    double-transformed.
    """
    _warn_v1_deprecated("olmoearth_v1_large_s1s2")
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_2_nano(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1_2-Nano",
    bands: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 128,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth v1.2 Nano checkpoint (D=128).

    Registered under its own name (``olmoearth_v1_2_nano``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_2_nano",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.

    NOTE: inputs must be RAW S2 L2A digital numbers (0-10000). With
    ``apply_pretraining_normalization=True`` (default) the wrapper min-max
    normalizes each band over mean±2σ, exactly as OlmoEarth's pretraining data
    loader did. Configure the datamodule with ``normalize: false``.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_2_tiny(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1_2-Tiny",
    bands: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 192,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth v1.2 Tiny checkpoint (D=192).

    Registered under its own name (``olmoearth_v1_2_tiny``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_2_tiny",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.

    NOTE: inputs must be RAW S2 L2A digital numbers (0-10000). With
    ``apply_pretraining_normalization=True`` (default) the wrapper min-max
    normalizes each band over mean±2σ, exactly as OlmoEarth's pretraining data
    loader did. Configure the datamodule with ``normalize: false``.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_2_small(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1_2-Small",
    bands: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 384,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth v1.2 Small checkpoint (D=384).

    Registered under its own name (``olmoearth_v1_2_small``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_2_small",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.

    NOTE: inputs must be RAW S2 L2A digital numbers (0-10000). With
    ``apply_pretraining_normalization=True`` (default) the wrapper min-max
    normalizes each band over mean±2σ, exactly as OlmoEarth's pretraining data
    loader did. Configure the datamodule with ``normalize: false``.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_2_base(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1_2-Base",
    bands: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 768,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth v1.2 Base checkpoint (D=768).

    Registered under its own name (``olmoearth_v1_2_base``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_2_base",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.

    NOTE: inputs must be RAW S2 L2A digital numbers (0-10000). With
    ``apply_pretraining_normalization=True`` (default) the wrapper min-max
    normalizes each band over mean±2σ, exactly as OlmoEarth's pretraining data
    loader did. Configure the datamodule with ``normalize: false``.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_2_nano_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1_2-Nano",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 128,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth v1.2 Nano with S2+S1 combined input (D=128).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_2_nano``.

    NOTE: inputs must be RAW sensor scale — S2 L2A digital numbers (0-10000) and
    S1 linear-power gamma0 (e.g. Planetary Computer sentinel-1-rtc). With
    ``apply_pretraining_normalization=True`` (default) the wrapper converts S1 to
    dB and min-max normalizes both modalities per band over mean±2σ, exactly as
    OlmoEarth's pretraining data loader did. Configure the datamodule with
    ``normalize: false`` and do NOT apply ``db_scale_bands`` to S1, or values get
    double-transformed.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_2_tiny_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1_2-Tiny",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 192,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth v1.2 Tiny with S2+S1 combined input (D=192).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_2_tiny``.

    NOTE: inputs must be RAW sensor scale — S2 L2A digital numbers (0-10000) and
    S1 linear-power gamma0 (e.g. Planetary Computer sentinel-1-rtc). With
    ``apply_pretraining_normalization=True`` (default) the wrapper converts S1 to
    dB and min-max normalizes both modalities per band over mean±2σ, exactly as
    OlmoEarth's pretraining data loader did. Configure the datamodule with
    ``normalize: false`` and do NOT apply ``db_scale_bands`` to S1, or values get
    double-transformed.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_2_small_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1_2-Small",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 384,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth v1.2 Small with S2+S1 combined input (D=384).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_2_small``.

    NOTE: inputs must be RAW sensor scale — S2 L2A digital numbers (0-10000) and
    S1 linear-power gamma0 (e.g. Planetary Computer sentinel-1-rtc). With
    ``apply_pretraining_normalization=True`` (default) the wrapper converts S1 to
    dB and min-max normalizes both modalities per band over mean±2σ, exactly as
    OlmoEarth's pretraining data loader did. Configure the datamodule with
    ``normalize: false`` and do NOT apply ``db_scale_bands`` to S1, or values get
    double-transformed.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_2_base_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1_2-Base",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    apply_pretraining_normalization: bool = True,
    hidden_dim: int | None = 768,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth v1.2 Base with S2+S1 combined input (D=768).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_2_base``.

    NOTE: inputs must be RAW sensor scale — S2 L2A digital numbers (0-10000) and
    S1 linear-power gamma0 (e.g. Planetary Computer sentinel-1-rtc). With
    ``apply_pretraining_normalization=True`` (default) the wrapper converts S1 to
    dB and min-max normalizes both modalities per band over mean±2σ, exactly as
    OlmoEarth's pretraining data loader did. Configure the datamodule with
    ``normalize: false`` and do NOT apply ``db_scale_bands`` to S1, or values get
    double-transformed.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        apply_pretraining_normalization=apply_pretraining_normalization,
        hidden_dim=hidden_dim,
        **kwargs,
    )


import logging as _logging

_logger = _logging.getLogger("terratorch")
try:
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY as _REG

    for _f in (
        olmoearth_v1_nano,
        olmoearth_v1_tiny,
        olmoearth_v1_base,
        olmoearth_v1_large,
        olmoearth_v1_nano_s1s2,
        olmoearth_v1_tiny_s1s2,
        olmoearth_v1_base_s1s2,
        olmoearth_v1_large_s1s2,
        olmoearth_v1_2_nano,
        olmoearth_v1_2_tiny,
        olmoearth_v1_2_small,
        olmoearth_v1_2_base,
        olmoearth_v1_2_nano_s1s2,
        olmoearth_v1_2_tiny_s1s2,
        olmoearth_v1_2_small_s1s2,
        olmoearth_v1_2_base_s1s2,
    ):
        _REG.register(_f)
    _logger.info(
        "Registered OlmoEarth backbones: 'olmoearth_v1_2_nano', 'olmoearth_v1_2_tiny', "
        "'olmoearth_v1_2_small', 'olmoearth_v1_2_base', "
        "'olmoearth_v1_2_nano_s1s2', 'olmoearth_v1_2_tiny_s1s2', "
        "'olmoearth_v1_2_small_s1s2', 'olmoearth_v1_2_base_s1s2'; "
        "DEPRECATED (v1, still registered): 'olmoearth_v1_nano', "
        "'olmoearth_v1_tiny', 'olmoearth_v1_base', 'olmoearth_v1_large', "
        "'olmoearth_v1_nano_s1s2', 'olmoearth_v1_tiny_s1s2', "
        "'olmoearth_v1_base_s1s2', 'olmoearth_v1_large_s1s2'."
    )
except Exception as _exc:
    import traceback as _tb

    _logger.warning(
        "Skipping OlmoEarth backbone registration: %s.\n%s\n"
        "Install with `pip install gelos` to enable it.",
        _exc,
        _tb.format_exc(),
    )
