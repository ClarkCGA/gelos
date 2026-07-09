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
  S2 mask shape: (B, H, W, T, 3) — 3 band sets (10m/20m/60m).
  S1 data must be in decibel scale (VV mean≈-11.6 dB, VH mean≈-17.7 dB).
  Source: verified against allenai/olmoearth_pretrain datatypes.py and constants.py.
"""

from __future__ import annotations

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


def build_band_reorder_index(bands: list[str]) -> list[int]:
    """Build the channel-index permutation mapping ``bands`` -> OlmoEarth order.

    Pure index logic (no model dependency). Given the configured input ``bands``
    (the channel order of the incoming GELOS tensor), returns a list of indices
    into that tensor that, when used to gather the channel axis, reorders the
    channels to :data:`OLMOEARTH_S2_BAND_ORDER`.

    Args:
        bands: Channel names in the order they appear in the input tensor.

    Returns:
        A list of length ``len(OLMOEARTH_S2_BAND_ORDER)`` where element ``i`` is
        the position in ``bands`` of the ``i``-th OlmoEarth band.

    Raises:
        ValueError: If any band required by OlmoEarth is missing from ``bands``.
    """
    band_to_pos: dict[str, int] = {}
    for pos, name in enumerate(bands):
        # First occurrence wins; duplicates are ignored deterministically.
        band_to_pos.setdefault(name, pos)

    missing = [b for b in OLMOEARTH_S2_BAND_ORDER if b not in band_to_pos]
    if missing:
        raise ValueError(
            "OlmoEarth requires the full 12-band Sentinel-2 L2A set; missing "
            f"band(s): {missing}. Configured bands: {bands}. Expected order: "
            f"{OLMOEARTH_S2_BAND_ORDER}."
        )

    return [band_to_pos[b] for b in OLMOEARTH_S2_BAND_ORDER]


class OlmoEarthBackbone(nn.Module):
    """Adapts an OlmoEarth encoder to terratorch's backbone ``forward_features``.

    Accepts GELOS's channels-first ``(B, C, T, H, W)`` Sentinel-2 L2A tensor,
    reorders bands to OlmoEarth's expected 12-band order, transposes to
    channels-last ``(B, H, W, T, C)``, runs the OlmoEarth encoder, mean-pools the
    token tensor over the spectral-group axis, and returns a single-element list
    ``[tokens]`` (terratorch necks expect a list of layer tensors).

    The temporal axis is handled per ``temporal_pooling``:

    - ``"mean"`` (default): mean over T, tokens shape ``(B, H'*W', D)``.
    - ``"keep"``: per-timestep tokens preserved, flattened time-major to
      ``(B, T*H'*W', D)`` — token ``t*H'*W' + i`` is spatial patch ``i`` at
      timestep ``t``, matching the layout Prithvi produces so the same strided
      ``slice_args`` extraction strategies apply (e.g. center patch across all
      timesteps: ``start=center_idx, step=H'*W'``). Note these tokens come from
      one joint space-time attention pass, so a single timestep's tokens still
      carry cross-time context (unlike running the encoder with T=1).

    ``spatial_pooling`` (int factor ``s``, default ``None`` = off) average-pools
    the ``H'xW'`` token grid over non-overlapping ``sxs`` neighborhoods after
    encoding, so each output token covers ``(s*patch_size)^2`` input pixels. Use
    it to match the spatial footprint of larger-patch models: with
    ``patch_size=4, spatial_pooling=4`` each token covers 16x16 pixels and the
    grid (and token indices) line up exactly with Prithvi/TerraMind's 16-pixel
    patches. Encoding still happens at the fine ``patch_size``; only the output
    tokens are aggregated.
    """

    def __init__(
        self,
        pretrained: bool = True,
        model_id: str = "allenai/OlmoEarth-v1-Base",
        bands: list[str] | None = None,
        patch_size: int = 4,
        hidden_dim: int | None = None,
        bands_s1: list[str] | None = None,
        warn_missing_s1: bool = True,
        temporal_pooling: str = "mean",
        spatial_pooling: int | None = None,
        **kwargs,  # tolerate terratorch-injected args
    ) -> None:
        super().__init__()
        if temporal_pooling not in ("mean", "keep"):
            raise ValueError(
                f"temporal_pooling must be 'mean' or 'keep', got {temporal_pooling!r}."
            )
        if spatial_pooling is not None and (
            not isinstance(spatial_pooling, int) or spatial_pooling < 1
        ):
            raise ValueError(
                f"spatial_pooling must be a positive int or None, got {spatial_pooling!r}."
            )
        self.model_id = model_id
        self.pretrained = pretrained
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.temporal_pooling = temporal_pooling
        self.spatial_pooling = spatial_pooling
        self.bands = list(bands) if bands else list(OLMOEARTH_S2_BAND_ORDER)

        # Transient per-batch acquisition dates, stashed by the task's
        # predict_step (see gelos.generation.LenientEmbeddingGenerationTask).
        # Plain attribute on purpose: NOT a buffer/parameter, so it is never saved
        # in the state_dict nor moved by ``.to()`` — it is cleared after each batch.
        self._batch_timestamps: torch.Tensor | None = None

        # Precompute and validate the band-reorder map eagerly so misconfigured
        # bands fail at construction time, not mid-forward.
        self.reorder_index = build_band_reorder_index(self.bands)

        self.bands_s1 = list(bands_s1) if bands_s1 else None
        self.warn_missing_s1 = warn_missing_s1
        self.reorder_index_s1 = (
            build_s1_band_reorder_index(self.bands_s1) if self.bands_s1 is not None else None
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

    def set_batch_timestamps(self, timestamps: torch.Tensor | None) -> None:
        """Stash the current batch's per-timestep timestamps for ``forward_features``.

        Called by the task's ``predict_step`` because terratorch's
        ``get_embeddings``/``self.model(input)`` call site forwards only ``input``
        to the backbone — there is no kwargs channel for extra batch keys.
        """
        self._batch_timestamps = timestamps

    def clear_batch_timestamps(self) -> None:
        """Clear the stashed timestamps (called in the task's ``finally``)."""
        self._batch_timestamps = None

    def forward_features(self, x, **kwargs) -> list[torch.Tensor]:
        """Run the OlmoEarth encoder and return ``[tokens]``.

        Args:
            x: Either the S2L2A tensor ``(B, C, T, H, W)`` or a dict of modalities
                keyed by sensor name. When a dict, ``"S2L2A"`` is required;
                ``"S1RTC"`` is used when ``bands_s1`` was configured.

        Returns:
            A single-element list whose tensor has shape ``(B, H'*W', D)`` when
            ``temporal_pooling="mean"``, or ``(B, T*H'*W', D)`` (time-major) when
            ``temporal_pooling="keep"``. With ``spatial_pooling=s``, ``H'`` and
            ``W'`` above are the pooled grid dims (``H/patch_size/s``).
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

        # 1. Reorder S2 channels to OlmoEarth band order.
        idx_s2 = torch.as_tensor(self.reorder_index, device=x_s2.device, dtype=torch.long)
        x_s2 = x_s2.index_select(dim=1, index=idx_s2)  # (B, 12, T, H, W)

        # 2. channels-first -> channels-last: (B, C, T, H, W) -> (B, H, W, T, C)
        x_s2 = x_s2.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, T, 12)

        # 3. Per-timestep timestamps (real or dummy fallback).
        timestamps = None
        if self._batch_timestamps is not None:
            candidate = self._batch_timestamps.to(device=x_s2.device, dtype=torch.long)
            if tuple(candidate.shape) == (b, t, 3):
                timestamps = candidate
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

        # 4. S2 mask: (B, H, W, T, 3) — 3 band sets (10m/20m/60m).
        sentinel2_mask = (
            torch.ones(b, h, w, t, 3, dtype=torch.int32, device=x_s2.device)
            * MaskValue.ONLINE_ENCODER.value
        )

        # 5. S1 path (optional).
        sentinel1_tensor = None
        sentinel1_mask = None
        if x_s1 is not None:
            if x_s1.dim() != 5:
                raise ValueError(
                    f"OlmoEarthBackbone S1 expects (B, C, T, H, W), got {tuple(x_s1.shape)}."
                )
            idx_s1 = torch.as_tensor(self.reorder_index_s1, device=x_s1.device, dtype=torch.long)
            x_s1 = x_s1.index_select(dim=1, index=idx_s1)  # (B, 2, T, H, W)
            x_s1 = x_s1.permute(0, 3, 4, 2, 1).contiguous()  # (B, H, W, T, 2)
            sentinel1_tensor = x_s1
            # S1 has 1 band set -> mask shape (B, H, W, T, 1)
            sentinel1_mask = (
                torch.ones(b, h, w, t, 1, dtype=torch.int32, device=x_s1.device)
                * MaskValue.ONLINE_ENCODER.value
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
        output_dict = self.encoder.encoder(sample, fast_pass=True, patch_size=self.patch_size)
        tokens_and_masks = output_dict["tokens_and_masks"]

        # 8. Pool S2 tokens over band-sets: (B, H', W', T, 3, D) -> (B, H', W', T, D)
        s2_tokens = tokens_and_masks.sentinel2_l2a  # (B, H', W', T, 3, D)
        pooled = s2_tokens.mean(dim=4)  # (B, H', W', T, D)

        # 9. Fuse S1 tokens when present (equal-weight average, per timestep).
        if sentinel1_tensor is not None:
            s1_tokens = tokens_and_masks.sentinel1  # (B, H', W', T, 1, D)
            pooled = (pooled + s1_tokens.mean(dim=4)) / 2.0

        # 9b. Optional spatial pooling: average sxs token neighborhoods so each
        # output token covers (s*patch_size)^2 pixels.
        if self.spatial_pooling is not None and self.spatial_pooling > 1:
            s = self.spatial_pooling
            bb, hp, wp, tt, d = pooled.shape
            if hp % s != 0 or wp % s != 0:
                raise ValueError(
                    f"Token grid ({hp}x{wp}) must be divisible by spatial_pooling={s}."
                )
            pooled = pooled.reshape(bb, hp // s, s, wp // s, s, tt, d).mean(dim=(2, 4))

        # 10. Handle time, flatten to a token sequence.
        if self.temporal_pooling == "mean":
            pooled = pooled.mean(dim=3)  # (B, H', W', D)
            bb, hp, wp, d = pooled.shape
            tokens = pooled.reshape(bb, hp * wp, d)
        else:  # "keep": time-major (B, T*H'*W', D)
            bb, hp, wp, tt, d = pooled.shape
            tokens = pooled.permute(0, 3, 1, 2, 4).reshape(bb, tt * hp * wp, d)

        return [tokens]

    def forward(self, x, **kwargs) -> list[torch.Tensor]:
        return self.forward_features(x, **kwargs)


def olmoearth_v1_nano(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Nano",
    bands: list[str] | None = None,
    patch_size: int = 4,
    hidden_dim: int | None = 128,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth Nano checkpoint (D=128).

    Registered under its own name (``olmoearth_v1_nano``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_nano",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_tiny(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Tiny",
    bands: list[str] | None = None,
    patch_size: int = 4,
    hidden_dim: int | None = 192,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth Tiny checkpoint (D=192).

    Registered under its own name (``olmoearth_v1_tiny``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_tiny",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_base(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Base",
    bands: list[str] | None = None,
    patch_size: int = 4,
    hidden_dim: int | None = 768,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth Base checkpoint (D=768).

    Registered under its own name (``olmoearth_v1_base``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_base",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_large(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Large",
    bands: list[str] | None = None,
    patch_size: int = 4,
    hidden_dim: int | None = 1024,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for the OlmoEarth Large checkpoint (D=1024).

    Registered under its own name (``olmoearth_v1_large``) in
    ``gelos.backbones.olmoearth_backbone``; ``BACKBONE_REGISTRY.build("olmoearth_v1_large",
    **model_args)`` returns an :class:`OlmoEarthBackbone`.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_nano_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Nano",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    hidden_dim: int | None = 128,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth Nano with S2+S1 combined input (D=128).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_nano``.

    NOTE: S1 data must be in decibel scale. OlmoEarth pretraining used S1 dB values
    (VV mean≈-11.6 dB, VH mean≈-17.7 dB). Linear-power S1 inputs produce incorrect
    embeddings without raising an error.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_tiny_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Tiny",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    hidden_dim: int | None = 192,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth Tiny with S2+S1 combined input (D=192).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_tiny``.

    NOTE: S1 data must be in decibel scale. OlmoEarth pretraining used S1 dB values
    (VV mean≈-11.6 dB, VH mean≈-17.7 dB). Linear-power S1 inputs produce incorrect
    embeddings without raising an error.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_base_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Base",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    hidden_dim: int | None = 768,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth Base with S2+S1 combined input (D=768).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_base``.

    NOTE: S1 data must be in decibel scale. OlmoEarth pretraining used S1 dB values
    (VV mean≈-11.6 dB, VH mean≈-17.7 dB). Linear-power S1 inputs produce incorrect
    embeddings without raising an error.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def olmoearth_v1_large_s1s2(
    pretrained: bool = True,
    model_id: str = "allenai/OlmoEarth-v1-Large",
    bands: list[str] | None = None,
    bands_s1: list[str] | None = None,
    patch_size: int = 4,
    hidden_dim: int | None = 1024,
    **kwargs,
) -> OlmoEarthBackbone:
    """Terratorch backbone factory for OlmoEarth Large with S2+S1 combined input (D=1024).

    Pass ``bands_s1=["VV", "VH"]`` (or via YAML ``model_args.bands_s1``) to enable S1.
    Omitting ``bands_s1`` falls back to S2-only, identical to ``olmoearth_v1_large``.

    NOTE: S1 data must be in decibel scale. OlmoEarth pretraining used S1 dB values
    (VV mean≈-11.6 dB, VH mean≈-17.7 dB). Linear-power S1 inputs produce incorrect
    embeddings without raising an error.
    """
    return OlmoEarthBackbone(
        pretrained=pretrained,
        model_id=model_id,
        bands=bands,
        bands_s1=bands_s1,
        patch_size=patch_size,
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
    ):
        _REG.register(_f)
    _logger.info(
        "Registered OlmoEarth backbones: 'olmoearth_v1_nano', "
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
