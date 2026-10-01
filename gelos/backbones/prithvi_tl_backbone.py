"""Prithvi EO v2 "TL" (time + location) backbone wrapper for gelos.

Prithvi EO v2 ships TL checkpoints whose encoder adds sinusoidal temporal
(``[year, day_of_year]``) and location (``[lat, lon]``) embeddings to every
patch token. terratorch registers them as ``prithvi_eo_v2_{tiny,100,300,600}_tl``,
but its ``PrithviViT.forward_features`` only applies those embeddings when
``temporal_coords`` / ``location_coords`` are passed as keyword arguments —
and terratorch's ``EmbeddingGenerationTask`` (as well as ``TemporalWrapper``)
calls the backbone with the image tensor alone. Running a stock TL name
through gelos therefore **silently** produces embeddings with no time or
location signal at all.

This module closes that gap with :class:`PrithviTLBackbone`, a thin wrapper
that consumes the per-batch side-channel already used by OlmoEarth:
``GELOSDataSet._get_timestamps`` / ``_get_location`` populate
``batch["timestamps"]`` ``(B, T, 3)`` ``[year, month, day]`` and
``batch["location"]`` ``(B, 2)`` ``[lat, lon]``;
``LenientEmbeddingGenerationTask.predict_step`` pops them and stashes them on
the backbone via ``set_batch_timestamps`` / ``set_batch_location``. The wrapper
converts the stash to Prithvi's coordinate layout and forwards it explicitly to
the terratorch encoder. If either stash is missing it raises — TL embeddings
are never silently skipped.

The wrapper is registered under its own names, suffixed ``_coords``
(``prithvi_eo_v2_300_tl_coords`` etc.), because terratorch's registry
``register`` is a bare ``__name__`` dict write: re-registering the stock
``prithvi_eo_v2_300_tl`` name would globally overwrite terratorch's entry
whenever ``gelos.generation`` is imported. The names still start with
``prithvi_eo_v2`` so :mod:`gelos.normalization` injects Prithvi's pretraining
means/stds unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

PRITHVI_TL_VARIANTS = (
    "prithvi_eo_v2_tiny_tl",
    "prithvi_eo_v2_100_tl",
    "prithvi_eo_v2_300_tl",
    "prithvi_eo_v2_600_tl",
)

# Cumulative days before each month in a common (non-leap) year.
_CUM_DAYS = torch.tensor([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def calendar_to_prithvi_temporal_coords(timestamps: torch.Tensor) -> torch.Tensor:
    """Convert canonical calendar dates to Prithvi TL's ``[year, day_of_year]``.

    Pure conversion logic (no model dependency), mirroring
    :func:`gelos.backbones.olmoearth_backbone.calendar_to_olmoearth_timestamps`.
    Maps a ``(..., 3)`` integer tensor of canonical ``[year, month, day]`` dates
    (month 1-12, day 1-31) to a ``(..., 2)`` float32 tensor of
    ``[year, day_of_year]`` with a 1-based day of year (Jan 1 = 1) and
    Gregorian leap years (divisible by 4, except centuries unless divisible by
    400). terratorch's ``TemporalEncoder`` asserts a floating dtype, hence the
    float32 output.

    Args:
        timestamps: ``(..., 3)`` integer tensor of ``[year, month, day]``.

    Returns:
        ``(..., 2)`` float32 tensor of ``[year, day_of_year]`` preserving the
        leading dims.

    Raises:
        ValueError: last dim is not 3, or any month/day is out of range.
    """
    if timestamps.dim() < 1 or timestamps.shape[-1] != 3:
        raise ValueError(
            "calendar_to_prithvi_temporal_coords expects a (..., 3) [year, month, day] "
            f"tensor, got shape {tuple(timestamps.shape)}."
        )
    ts = timestamps.to(torch.long)
    year = ts[..., 0]
    month = ts[..., 1]
    day = ts[..., 2]
    if ((month < 1) | (month > 12)).any():
        raise ValueError(
            "calendar_to_prithvi_temporal_coords: month must be in 1..12 (canonical "
            "[year, month, day] dates); got values outside that range."
        )
    if ((day < 1) | (day > 31)).any():
        raise ValueError(
            "calendar_to_prithvi_temporal_coords: day must be in 1..31 (canonical "
            "[year, month, day] dates); got values outside that range."
        )
    leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    cum_days = _CUM_DAYS.to(ts.device)
    doy = cum_days[month - 1] + day + (leap & (month > 2)).to(torch.long)
    return torch.stack([year, doy], dim=-1).to(torch.float32)


class PrithviTLBackbone(nn.Module):
    """Prithvi EO v2 TL encoder that consumes gelos's timestamps/location stash.

    Builds the terratorch TL encoder (``variant`` must be one of
    :data:`PRITHVI_TL_VARIANTS`) and holds it as ``self.encoder``. At forward
    time the stashed canonical ``(B, T, 3)`` ``[year, month, day]`` timestamps
    are converted to ``[year, day_of_year]`` via
    :func:`calendar_to_prithvi_temporal_coords` and, together with the ``(B, 2)``
    ``[lat, lon]`` location, forwarded explicitly as ``temporal_coords`` /
    ``location_coords`` to the encoder.

    Two input layouts are accepted:

    - **4D** ``(B*T, C, H, W)`` — the ``TemporalWrapper`` path
      (``temporal_cfg.temporal_wrapper: true``, the same path plain Prithvi
      uses in gelos). ``TemporalWrapper`` flattens ``(B, C, T, H, W)`` in
      b-major order, so ``(B, T)`` is derived from the stashed timestamps and
      coords are flattened the same way: temporal ``(B*T, 1, 2)``, location
      ``(B*T, 2)`` via ``repeat_interleave``. Each timestep is encoded on its
      own (CLS at index 0), keeping the token layout identical to the non-TL
      Prithvi baseline.
    - **5D** ``(B, C, T, H, W)`` — joint space-time encoding
      (``temporal_wrapper: false``). Coords pass through as ``(B, T, 2)`` /
      ``(B, 2)``, but ``T`` must equal the encoder's configured ``num_frames``
      (terratorch derives ``tokens_per_frame`` from it, so a mismatch would
      silently misalign the temporal embedding).

    Fail-loud contract: a missing stash (dataset does not override
    ``_get_timestamps`` / ``_get_location``) or a shape mismatch raises
    ``ValueError`` instead of falling back to the un-encoded path.
    """

    def __init__(
        self,
        variant: str,
        pretrained: bool = True,
        bands: list[str] | None = None,
        **model_kwargs,
    ) -> None:
        super().__init__()
        if variant not in PRITHVI_TL_VARIANTS:
            raise ValueError(
                f"PrithviTLBackbone supports only the Prithvi TL variants "
                f"{PRITHVI_TL_VARIANTS}, got {variant!r}."
            )
        from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

        self.variant = variant
        self.encoder = TERRATORCH_BACKBONE_REGISTRY.build(
            variant, pretrained=pretrained, bands=bands, **model_kwargs
        )
        if not (
            getattr(self.encoder, "temporal_encoding", False)
            and getattr(self.encoder, "location_encoding", False)
        ):
            raise ValueError(
                f"terratorch built {variant!r} without both temporal and location "
                "encoding enabled; PrithviTLBackbone requires a TL encoder "
                "(coords_encoding=['time', 'location'])."
            )
        self.out_channels = self.encoder.out_channels
        self.num_frames = self.encoder.num_frames

        # Transient per-batch stashes, set by the task's predict_step (see
        # gelos.generation.LenientEmbeddingGenerationTask). Plain attributes on
        # purpose: NOT buffers/parameters, so they are never saved in the
        # state_dict nor moved by ``.to()`` — they are cleared after each batch.
        self._batch_timestamps: torch.Tensor | None = None
        self._batch_location: torch.Tensor | None = None

    def set_batch_timestamps(self, timestamps: torch.Tensor | None) -> None:
        """Stash the current batch's per-timestep timestamps for ``forward``.

        The stash is canonical calendar dates ``(B, T, 3)`` ``[year, month, day]``
        (month 1-12); conversion to Prithvi's ``[year, day_of_year]`` happens at
        consumption time via :func:`calendar_to_prithvi_temporal_coords`.

        Called by the task's ``predict_step`` because terratorch's
        ``get_embeddings``/``self.model(input)`` call site forwards only ``input``
        to the backbone — there is no kwargs channel for extra batch keys.
        """
        self._batch_timestamps = timestamps

    def clear_batch_timestamps(self) -> None:
        """Clear the stashed timestamps (called in the task's ``finally``)."""
        self._batch_timestamps = None

    def set_batch_location(self, location: torch.Tensor | None) -> None:
        """Stash the current batch's chip locations ``(B, 2)`` ``[lat, lon]``.

        Same side-channel as :meth:`set_batch_timestamps`; consumed as
        ``location_coords`` at forward time.
        """
        self._batch_location = location

    def clear_batch_location(self) -> None:
        """Clear the stashed location (called in the task's ``finally``)."""
        self._batch_location = None

    def _resolve_coords(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate the stashes against ``x`` and return ``(temporal, location)`` coords."""
        if self._batch_timestamps is None:
            raise ValueError(
                "PrithviTLBackbone requires batch['timestamps'] (B, T, 3) [year, month, day] "
                "but none was stashed for this batch. Override "
                "GELOSDataSet._get_timestamps(index) to return a (T, 3) array "
                f"(Prithvi TL was requested via model {self.variant!r}; TL embeddings "
                "are never silently skipped)."
            )
        if self._batch_location is None:
            raise ValueError(
                "PrithviTLBackbone requires batch['location'] (B, 2) [lat, lon] but none "
                "was stashed for this batch. Override GELOSDataSet._get_location(index) "
                f"to return a (2,) array (Prithvi TL was requested via model "
                f"{self.variant!r}; TL embeddings are never silently skipped)."
            )
        ts = self._batch_timestamps.to(x.device)
        if ts.dim() != 3 or ts.shape[-1] != 3:
            raise ValueError(
                "PrithviTLBackbone: batch['timestamps'] must be (B, T, 3) [year, month, day], "
                f"got shape {tuple(ts.shape)}."
            )
        B, T = ts.shape[0], ts.shape[1]
        loc = self._batch_location.to(x.device, torch.float32)
        if loc.shape != (B, 2):
            raise ValueError(
                f"PrithviTLBackbone: batch['location'] must be (B, 2) [lat, lon] with B={B} "
                f"(from timestamps), got shape {tuple(loc.shape)}."
            )
        tc = calendar_to_prithvi_temporal_coords(ts)  # (B, T, 2)

        if x.dim() == 4:
            # TemporalWrapper path: (B, C, T, H, W) was flattened b-major to
            # (B*T, C, H, W), so every timestep is its own single-frame sample.
            if x.shape[0] != B * T:
                raise ValueError(
                    f"PrithviTLBackbone: got N={x.shape[0]} input samples but timestamps "
                    f"imply B*T={B * T} (B={B}, T={T}). Under TemporalWrapper the batch "
                    "is flattened to (B*T, C, H, W); check the dataset's _get_timestamps."
                )
            return tc.reshape(B * T, 1, 2), loc.repeat_interleave(T, dim=0)
        if x.dim() == 5:
            if x.shape[0] != B or x.shape[2] != T:
                raise ValueError(
                    f"PrithviTLBackbone: input (B, C, T, H, W)={tuple(x.shape)} does not "
                    f"match timestamps (B, T)=({B}, {T})."
                )
            if T != self.num_frames:
                raise ValueError(
                    f"PrithviTLBackbone: 5D input has T={T} timesteps but the encoder was "
                    f"built with num_frames={self.num_frames}; terratorch derives "
                    "tokens_per_frame from num_frames, so a mismatch misaligns the "
                    f"temporal embedding. Set model_args.num_frames: {T} or use "
                    "temporal_cfg.temporal_wrapper: true."
                )
            return tc, loc
        raise ValueError(
            "PrithviTLBackbone expects a 4D (B*T, C, H, W) or 5D (B, C, T, H, W) input, "
            f"got {x.dim()}D with shape {tuple(x.shape)}."
        )

    def forward_features(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        tc, lc = self._resolve_coords(x)
        return self.encoder.forward_features(x, temporal_coords=tc, location_coords=lc)

    def forward(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        # Goes through terratorch's ``forward_filter_indices`` (honors out_indices).
        tc, lc = self._resolve_coords(x)
        return self.encoder(x, temporal_coords=tc, location_coords=lc)


def prithvi_eo_v2_tiny_tl_coords(
    pretrained: bool = True, bands: list[str] | None = None, **kwargs
) -> PrithviTLBackbone:
    """Prithvi EO v2 tiny TL (HF ``ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL``).

    Requires the dataset to override ``GELOSDataSet._get_timestamps`` and
    ``_get_location``; generation raises otherwise.
    """
    return PrithviTLBackbone("prithvi_eo_v2_tiny_tl", pretrained=pretrained, bands=bands, **kwargs)


def prithvi_eo_v2_100_tl_coords(
    pretrained: bool = True, bands: list[str] | None = None, **kwargs
) -> PrithviTLBackbone:
    """Prithvi EO v2 100M TL (HF ``ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL``).

    Requires the dataset to override ``GELOSDataSet._get_timestamps`` and
    ``_get_location``; generation raises otherwise.
    """
    return PrithviTLBackbone("prithvi_eo_v2_100_tl", pretrained=pretrained, bands=bands, **kwargs)


def prithvi_eo_v2_300_tl_coords(
    pretrained: bool = True, bands: list[str] | None = None, **kwargs
) -> PrithviTLBackbone:
    """Prithvi EO v2 300M TL (HF ``ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL``).

    Requires the dataset to override ``GELOSDataSet._get_timestamps`` and
    ``_get_location``; generation raises otherwise.
    """
    return PrithviTLBackbone("prithvi_eo_v2_300_tl", pretrained=pretrained, bands=bands, **kwargs)


def prithvi_eo_v2_600_tl_coords(
    pretrained: bool = True, bands: list[str] | None = None, **kwargs
) -> PrithviTLBackbone:
    """Prithvi EO v2 600M TL (HF ``ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL``).

    Requires the dataset to override ``GELOSDataSet._get_timestamps`` and
    ``_get_location``; generation raises otherwise.
    """
    return PrithviTLBackbone("prithvi_eo_v2_600_tl", pretrained=pretrained, bands=bands, **kwargs)


import logging as _logging

_logger = _logging.getLogger("terratorch")
try:
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY as _REG

    for _f in (
        prithvi_eo_v2_tiny_tl_coords,
        prithvi_eo_v2_100_tl_coords,
        prithvi_eo_v2_300_tl_coords,
        prithvi_eo_v2_600_tl_coords,
    ):
        _REG.register(_f)
    _logger.info(
        "Registered Prithvi TL backbones: 'prithvi_eo_v2_tiny_tl_coords', "
        "'prithvi_eo_v2_100_tl_coords', 'prithvi_eo_v2_300_tl_coords', "
        "'prithvi_eo_v2_600_tl_coords'."
    )
except Exception as _exc:
    import traceback as _tb

    _logger.warning(
        "Skipping Prithvi TL backbone registration: %s.\n%s",
        _exc,
        _tb.format_exc(),
    )
