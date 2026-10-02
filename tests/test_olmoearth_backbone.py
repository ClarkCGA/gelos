"""Tests for the OlmoEarth terratorch backbone wrapper.

The band-reorder helper is pure index logic with no model dependency, so those
tests run unconditionally and are the most valuable to keep green in CI without the
heavy optional ``olmoearth-pretrain`` extra installed. Tests that need the actual
model are gated with ``pytest.importorskip("olmoearth_pretrain")``.
"""

import warnings

import pytest
import torch

from gelos.backbones.olmoearth_backbone import (
    OLMOEARTH_S2_BAND_ORDER,
    build_band_reorder_index,
    calendar_to_olmoearth_timestamps,
)

# The 12 GELOS-LC band names OlmoEarth requires, in the natural dataset (non-
# OlmoEarth) input order, to exercise the reorder logic.
ALL_12_BANDS = [
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


# ---------------------------------------------------------------------------
# Pure helper tests (no model dependency) — run unconditionally.
# ---------------------------------------------------------------------------


def test_band_reorder_index_maps_to_olmoearth_order():
    index = build_band_reorder_index(ALL_12_BANDS)
    # Applying the index to ALL_12_BANDS must yield OlmoEarth's expected order.
    reordered = [ALL_12_BANDS[i] for i in index]
    assert reordered == OLMOEARTH_S2_BAND_ORDER


def test_band_reorder_index_when_input_already_in_target_order_is_identity():
    index = build_band_reorder_index(OLMOEARTH_S2_BAND_ORDER)
    assert index == list(range(len(OLMOEARTH_S2_BAND_ORDER)))


def test_band_reorder_index_handles_shuffled_input():
    shuffled = list(reversed(ALL_12_BANDS))
    index = build_band_reorder_index(shuffled)
    reordered = [shuffled[i] for i in index]
    assert reordered == OLMOEARTH_S2_BAND_ORDER


def test_band_reorder_index_marks_missing_band_as_none():
    # Drop WATER_VAPOR (B09) — the band ExampleGELOSDataSet also lacks. A subset
    # no longer raises: the absent band's slot is None (zero-filled in forward).
    from gelos.backbones.olmoearth_backbone import absent_s2_bands

    missing = [b for b in ALL_12_BANDS if b != "WATER_VAPOR"]
    index = build_band_reorder_index(missing)
    assert len(index) == 12
    assert index[OLMOEARTH_S2_BAND_ORDER.index("WATER_VAPOR")] is None
    present = [(i, s) for i, s in enumerate(index) if s is not None]
    assert len(present) == 11
    assert all(missing[s] == OLMOEARTH_S2_BAND_ORDER[i] for i, s in present)
    assert absent_s2_bands(index) == ["WATER_VAPOR"]


def test_band_reorder_index_reports_all_missing_bands():
    from gelos.backbones.olmoearth_backbone import absent_s2_bands

    index = build_band_reorder_index(["BLUE", "GREEN", "RED"])
    absent = absent_s2_bands(index)
    assert len(absent) == 9
    assert "COASTAL_AEROSOL" in absent and "WATER_VAPOR" in absent
    assert set(absent).isdisjoint({"BLUE", "GREEN", "RED"})
    # Absent names come back in OlmoEarth order.
    assert absent == [b for b in OLMOEARTH_S2_BAND_ORDER if b in absent]


def test_band_reorder_index_unknown_band_raises():
    # A typo (nir09) must raise, not silently zero-fill WATER_VAPOR.
    bands = [b if b != "WATER_VAPOR" else "nir09" for b in ALL_12_BANDS]
    with pytest.raises(ValueError, match="nir09"):
        build_band_reorder_index(bands)


def test_band_reorder_index_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        build_band_reorder_index([])


# ---------------------------------------------------------------------------
# Canonical -> OlmoEarth timestamp conversion tests (pure logic, no model
# dependency) — run unconditionally.
# ---------------------------------------------------------------------------


def test_calendar_conversion_basic():
    # Canonical [year, month, day] -> OlmoEarth [day, month_index, year].
    ts = torch.tensor([[2021, 6, 15]], dtype=torch.long)
    out = calendar_to_olmoearth_timestamps(ts)
    assert torch.equal(out, torch.tensor([[15, 5, 2021]], dtype=torch.long))


def test_calendar_conversion_preserves_batched_shape():
    ts = torch.stack(
        [
            torch.tensor([[2020, 1, 1], [2020, 12, 31], [2021, 7, 4]]),
            torch.tensor([[2019, 3, 10], [2019, 6, 20], [2019, 9, 30]]),
        ]
    )  # (B=2, T=3, 3)
    out = calendar_to_olmoearth_timestamps(ts)
    assert out.shape == (2, 3, 3)
    # Spot-check one entry: [2019, 6, 20] -> [20, 5, 2019].
    assert out[1, 1].tolist() == [20, 5, 2019]


def test_calendar_conversion_month_boundaries():
    ts = torch.tensor([[2022, 1, 5], [2022, 12, 25]], dtype=torch.long)
    out = calendar_to_olmoearth_timestamps(ts)
    assert out[0].tolist() == [5, 0, 2022]  # January -> month_index 0
    assert out[1].tolist() == [25, 11, 2022]  # December -> month_index 11


@pytest.mark.parametrize("month", [0, 13])
def test_calendar_conversion_warns_on_out_of_range_month(month):
    ts = torch.tensor([[2021, month, 15]], dtype=torch.long)
    with pytest.warns(UserWarning, match="year, month, day"):
        calendar_to_olmoearth_timestamps(ts)


def test_calendar_conversion_warns_on_implausible_year_column():
    # Legacy [day, month_index, year] packing: column 0 is a 1-31 day (< 1900).
    ts = torch.tensor([[15, 5, 2021]], dtype=torch.long)
    with pytest.warns(UserWarning, match="year, month, day"):
        calendar_to_olmoearth_timestamps(ts)


# ---------------------------------------------------------------------------
# Model-dependent tests — skipped cleanly when the extra is absent.
# ---------------------------------------------------------------------------


def test_forward_features_output_shape():
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    backbone = OlmoEarthBackbone(
        pretrained=False,
        model_id="allenai/OlmoEarth-v1-Base",
        bands=ALL_12_BANDS,
        patch_size=4,
    )
    x = torch.randn(2, 12, 1, 32, 32)  # (B, C, T, H, W)
    out = backbone.forward_features(x)

    assert isinstance(out, list)
    assert len(out) == 1
    tokens = out[0]
    assert tokens.dim() == 3
    assert tokens.shape[0] == 2  # batch dim preserved


def test_forward_features_shape_mismatch_falls_back_to_constant(monkeypatch):
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    backbone = OlmoEarthBackbone(pretrained=False, bands=ALL_12_BANDS, patch_size=4)
    # Wrong-shaped timestamps for a (B=1, T=1) input: must warn and not crash.
    backbone.set_batch_timestamps(torch.zeros(5, 9, 3, dtype=torch.int64))
    x = torch.randn(1, 12, 1, 32, 32)
    with pytest.warns(UserWarning, match="timestamps"):
        out = backbone.forward_features(x)
    assert isinstance(out, list) and len(out) == 1


def test_forward_features_rejects_non_divisible_spatial():
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    backbone = OlmoEarthBackbone(pretrained=False, bands=ALL_12_BANDS, patch_size=4)
    x = torch.randn(1, 12, 1, 30, 30)  # 30 not divisible by 4
    with pytest.raises(ValueError, match="divisible"):
        backbone.forward_features(x)


def test_forward_features_temporal_keep_output_shape():
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    backbone = OlmoEarthBackbone(
        pretrained=False,
        model_id="allenai/OlmoEarth-v1-Base",
        bands=ALL_12_BANDS,
        patch_size=4,
        temporal_pooling="keep",
    )
    x = torch.randn(2, 12, 3, 32, 32)  # (B, C, T=3, H, W)
    out = backbone.forward_features(x)

    tokens = out[0]
    # time-major: T * (H/p) * (W/p) = 3 * 8 * 8 tokens
    assert tokens.shape[:2] == (2, 3 * 8 * 8)


def test_forward_features_mean_matches_keep_averaged():
    # With temporal_pooling="mean", the center token must equal the mean of the
    # per-timestep center tokens from a "keep" run (same weights, same input).
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    torch.manual_seed(0)
    x = torch.randn(1, 12, 2, 32, 32)
    kwargs = dict(pretrained=False, bands=ALL_12_BANDS, patch_size=4)

    torch.manual_seed(42)
    mean_bb = OlmoEarthBackbone(temporal_pooling="mean", **kwargs)
    torch.manual_seed(42)
    keep_bb = OlmoEarthBackbone(temporal_pooling="keep", **kwargs)
    keep_bb.load_state_dict(mean_bb.state_dict())
    # eval() disables the encoder's DropPath layers; in train mode the two
    # forward passes diverge stochastically.
    mean_bb.eval()
    keep_bb.eval()

    with torch.no_grad():
        mean_tokens = mean_bb.forward_features(x)[0]  # (1, 64, D)
        keep_tokens = keep_bb.forward_features(x)[0]  # (1, 128, D)

    n_spatial = mean_tokens.shape[1]
    per_step = keep_tokens.reshape(1, 2, n_spatial, -1)
    torch.testing.assert_close(per_step.mean(dim=1), mean_tokens, rtol=1e-4, atol=1e-5)


def test_constructor_rejects_invalid_temporal_pooling():
    # Validation happens before the lazy olmoearth_pretrain import, so this
    # runs without the extra installed.
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    with pytest.raises(ValueError, match="temporal_pooling"):
        OlmoEarthBackbone(pretrained=False, bands=ALL_12_BANDS, temporal_pooling="max")


def test_constructor_rejects_invalid_spatial_pooling():
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    with pytest.raises(ValueError, match="spatial_pooling"):
        OlmoEarthBackbone(pretrained=False, bands=ALL_12_BANDS, spatial_pooling=0)


def test_forward_features_spatial_pooling_output_shape():
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    backbone = OlmoEarthBackbone(
        pretrained=False,
        bands=ALL_12_BANDS,
        patch_size=4,
        temporal_pooling="keep",
        spatial_pooling=4,
    )
    x = torch.randn(1, 12, 2, 96, 96)  # 24x24 token grid -> pooled to 6x6
    tokens = backbone.forward_features(x)[0]
    assert tokens.shape[:2] == (1, 2 * 6 * 6)


def test_forward_features_spatial_pooling_rejects_non_divisible_grid():
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    backbone = OlmoEarthBackbone(
        pretrained=False, bands=ALL_12_BANDS, patch_size=4, spatial_pooling=5
    )
    x = torch.randn(1, 12, 1, 96, 96)  # 24x24 grid not divisible by 5
    with pytest.raises(ValueError, match="spatial_pooling"):
        backbone.forward_features(x)


def test_constructor_raises_on_unknown_band_without_model():
    # Construction validates band names before touching the model, so an
    # unknown band raises ValueError regardless of whether the extra is installed.
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    bands = [b if b != "WATER_VAPOR" else "nir09" for b in ALL_12_BANDS]
    with pytest.raises(ValueError, match="nir09"):
        OlmoEarthBackbone(pretrained=False, bands=bands)


def test_default_model_id_is_v1_2_base():
    import inspect

    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    params = inspect.signature(OlmoEarthBackbone.__init__).parameters
    assert params["model_id"].default == "allenai/OlmoEarth-v1_2-Base"


# ---------------------------------------------------------------------------
# Band-subset zero-fill (issue #81) — needs the model package for construction
# (the band-set count is read from the loaded encoder); the encoder forward is
# replaced by a stub that captures the sample the wrapper built.
# ---------------------------------------------------------------------------

_V1_2_NANO = "allenai/OlmoEarth-v1_2-Nano"
_V1_NANO = "allenai/OlmoEarth-v1-Nano"
# S2-Agri-Patch style subset: no COASTAL_AEROSOL (B01) / WATER_VAPOR (B09).
TEN_BANDS = [b for b in ALL_12_BANDS if b not in ("COASTAL_AEROSOL", "WATER_VAPOR")]


class _CapturingEncoder(torch.nn.Module):
    """Stands in for ``backbone.encoder.encoder``; records the sample it is given."""

    def __init__(self, hidden_dim: int, num_band_sets: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_band_sets = num_band_sets
        self.samples = []

    def forward(self, sample, fast_pass=True, patch_size=4):
        from types import SimpleNamespace

        self.samples.append(sample)
        b, h, w, t, _ = sample.sentinel2_l2a.shape
        tokens = torch.zeros(
            b, h // patch_size, w // patch_size, t, self.num_band_sets, self.hidden_dim
        )
        return {"tokens_and_masks": SimpleNamespace(sentinel2_l2a=tokens, sentinel1=None)}


def _capturing_backbone(bands, model_id=_V1_2_NANO, **kwargs):
    """Backbone with the inner encoder swapped for ``_CapturingEncoder`` (after init)."""
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    backbone = OlmoEarthBackbone(pretrained=False, model_id=model_id, bands=bands, **kwargs)
    stub = _CapturingEncoder(backbone.out_channels, backbone.s2_num_band_sets)
    backbone.encoder.encoder = stub
    return backbone.eval(), stub


def _captured_s2(backbone, stub, x):
    with torch.no_grad():
        backbone.forward_features(x)
    return stub.samples[-1].sentinel2_l2a  # (B, H, W, T, 12)


def _raw_s2(c, b=1, t=2, h=8, w=8):
    torch.manual_seed(0)
    return 1500 + 500 * torch.randn(b, c, t, h, w)


def test_subset_warns_once_naming_zero_filled_bands():
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    with pytest.warns(UserWarning, match="zero-filled") as record:
        backbone = OlmoEarthBackbone(pretrained=False, model_id=_V1_2_NANO, bands=TEN_BANDS)
    fill_warnings = [w for w in record if "zero-filled" in str(w.message)]
    assert len(fill_warnings) == 1
    msg = str(fill_warnings[0].message)
    assert "COASTAL_AEROSOL" in msg and "WATER_VAPOR" in msg
    assert "FAR OUTSIDE" not in msg  # 2 of 12 absent is in distribution
    assert backbone.absent_bands == ["COASTAL_AEROSOL", "WATER_VAPOR"]


def test_many_absent_bands_warns_loudly_but_does_not_raise():
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    six = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]
    with pytest.warns(UserWarning, match="FAR OUTSIDE") as record:
        backbone = OlmoEarthBackbone(pretrained=False, model_id=_V1_2_NANO, bands=six)
    msg = str(next(w for w in record if "FAR OUTSIDE" in str(w.message)).message)
    assert "6 of 12" in msg and "RED_EDGE_1" in msg
    assert len(backbone.absent_bands) == 6


def test_full_band_set_emits_no_zero_fill_warning():
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        backbone = OlmoEarthBackbone(pretrained=False, model_id=_V1_2_NANO, bands=ALL_12_BANDS)
    assert backbone.absent_bands == []


def test_subset_with_v1_checkpoint_raises():
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    with pytest.raises(ValueError, match="band subset is not supported"):
        OlmoEarthBackbone(pretrained=False, model_id=_V1_NANO, bands=TEN_BANDS)


def test_full_band_set_with_v1_checkpoint_still_works():
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    backbone = OlmoEarthBackbone(pretrained=False, model_id=_V1_NANO, bands=ALL_12_BANDS)
    assert backbone.s2_num_band_sets == 3
    assert backbone.absent_bands == []


def test_subset_absent_channels_zero_and_present_channels_unchanged():
    from gelos.backbones.olmoearth_backbone import minmax_normalize

    x12 = _raw_s2(12)
    full_bb, full_stub = _capturing_backbone(ALL_12_BANDS)
    with pytest.warns(UserWarning, match="zero-filled"):
        sub_bb, sub_stub = _capturing_backbone(TEN_BANDS)

    full = _captured_s2(full_bb, full_stub, x12)
    keep = [i for i, b in enumerate(ALL_12_BANDS) if b in TEN_BANDS]
    sub = _captured_s2(sub_bb, sub_stub, x12[:, keep])
    assert full.shape == sub.shape == (1, 8, 8, 2, 12)

    absent_idx = [OLMOEARTH_S2_BAND_ORDER.index(b) for b in ("COASTAL_AEROSOL", "WATER_VAPOR")]
    present_idx = [i for i in range(12) if i not in absent_idx]
    # Absent channels are exactly 0 after normalization ...
    assert torch.equal(sub[..., absent_idx], torch.zeros_like(sub[..., absent_idx]))
    # ... which is NOT what a raw-0 input would normalize to.
    assert not torch.equal(full[..., absent_idx], torch.zeros_like(full[..., absent_idx]))
    # Present channels are bit-identical to the full-band path.
    assert torch.equal(sub[..., present_idx], full[..., present_idx])
    # Full-band path is bit-identical to the hand-applied normalization.
    expected = minmax_normalize(
        x12.index_select(1, torch.tensor(full_bb.reorder_index)).permute(0, 3, 4, 2, 1),
        full_bb._s2_norm_means,
        full_bb._s2_norm_stds,
    )
    assert torch.equal(full, expected)


def test_subset_zero_fill_without_pretraining_normalization():
    x10 = _raw_s2(10)
    with pytest.warns(UserWarning, match="zero-filled"):
        bb, stub = _capturing_backbone(TEN_BANDS, apply_pretraining_normalization=False)
    out = _captured_s2(bb, stub, x10)
    absent_idx = [OLMOEARTH_S2_BAND_ORDER.index(b) for b in ("COASTAL_AEROSOL", "WATER_VAPOR")]
    assert torch.equal(out[..., absent_idx], torch.zeros_like(out[..., absent_idx]))
    # Present channels pass through raw (no normalization), reordered.
    for i, name in enumerate(OLMOEARTH_S2_BAND_ORDER):
        if name in TEN_BANDS:
            assert torch.equal(out[..., i], x10[:, TEN_BANDS.index(name)].permute(0, 2, 3, 1))


def test_subset_channel_count_mismatch_raises():
    with pytest.warns(UserWarning, match="zero-filled"):
        bb, _stub = _capturing_backbone(TEN_BANDS)
    with pytest.raises(ValueError, match="must match"):
        bb.forward_features(_raw_s2(12))


def test_subset_nodata_mask_keeps_dataset_channel_count():
    # The stashed nodata mask has the dataset's 10 channels, not 12; masking
    # still works and the S2 mask's last dim follows the encoder's band sets (1).
    with pytest.warns(UserWarning, match="zero-filled"):
        bb, stub = _capturing_backbone(TEN_BANDS, temporal_pooling="keep")
    x10 = _raw_s2(10)
    mask = torch.zeros(1, 10, 2, 8, 8, dtype=torch.bool)
    mask[0, :, :, 0:4, 0:4] = True  # patch (0, 0) nodata at both timesteps
    bb.set_batch_nodata_mask(mask)
    try:
        with torch.no_grad():
            bb.forward_features(x10)
    finally:
        bb.clear_batch_nodata_mask()
    s2_mask = stub.samples[-1].sentinel2_l2a_mask
    assert s2_mask.shape == (1, 8, 8, 2, 1)
    assert (s2_mask[0, 0:4, 0:4] == 3).all()
    assert (s2_mask[0, 4:, 4:] == 0).all()


@pytest.mark.parametrize("model_id, expected", [(_V1_2_NANO, 1), (_V1_NANO, 3)])
def test_s2_mask_band_sets_follow_encoder(model_id, expected):
    bb, stub = _capturing_backbone(ALL_12_BANDS, model_id=model_id)
    assert bb.s2_num_band_sets == expected
    _captured_s2(bb, stub, _raw_s2(12))
    assert stub.samples[-1].sentinel2_l2a_mask.shape[-1] == expected


# ---------------------------------------------------------------------------
# OlmoEarth v1 factory deprecation (issue #81).
# ---------------------------------------------------------------------------

V1_FACTORIES = [
    "olmoearth_v1_nano",
    "olmoearth_v1_tiny",
    "olmoearth_v1_base",
    "olmoearth_v1_large",
    "olmoearth_v1_nano_s1s2",
    "olmoearth_v1_tiny_s1s2",
    "olmoearth_v1_base_s1s2",
    "olmoearth_v1_large_s1s2",
]


@pytest.mark.parametrize("name", V1_FACTORIES)
def test_v1_factories_emit_deprecation_warning(name):
    pytest.importorskip("olmoearth_pretrain")
    import gelos.backbones.olmoearth_backbone as oe

    # Nano checkpoint for every factory: the warning is what is under test.
    with pytest.warns(DeprecationWarning, match=name):
        backbone = getattr(oe, name)(pretrained=False, model_id=_V1_NANO, bands=ALL_12_BANDS)
    assert backbone.s2_num_band_sets == 3


@pytest.mark.parametrize("name", V1_FACTORIES)
def test_v1_factory_defaults_still_point_at_v1_checkpoints(name):
    import inspect

    import gelos.backbones.olmoearth_backbone as oe

    params = inspect.signature(getattr(oe, name)).parameters
    assert params["model_id"].default.startswith("allenai/OlmoEarth-v1-")


def test_v1_2_factories_do_not_warn():
    pytest.importorskip("olmoearth_pretrain")
    import gelos.backbones.olmoearth_backbone as oe

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        oe.olmoearth_v1_2_nano(pretrained=False, bands=ALL_12_BANDS)
    ours = [
        w
        for w in record
        if issubclass(w.category, DeprecationWarning) and "olmoearth_v1" in str(w.message)
    ]
    assert not ours


# ---------------------------------------------------------------------------
# Timestamp stash tests — no model dependency (operate on a bare instance).
# ---------------------------------------------------------------------------


def test_set_and_clear_batch_timestamps():
    # Exercise the setter/clearer without constructing the (model-dependent)
    # backbone: __init__ always lazy-imports olmoearth_pretrain, so build a bare
    # instance via __new__ and seed the attribute __init__ would set.
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    backbone = OlmoEarthBackbone.__new__(OlmoEarthBackbone)
    backbone._batch_timestamps = None

    ts = torch.tensor([[[15, 2, 2020]]], dtype=torch.int64)
    backbone.set_batch_timestamps(ts)
    assert backbone._batch_timestamps is ts

    backbone.clear_batch_timestamps()
    assert backbone._batch_timestamps is None


# ---------------------------------------------------------------------------
# Task predict_step tests — verify the timestamp side-channel plumbing without
# the real terratorch predict pipeline.
# ---------------------------------------------------------------------------


class _TimestampsOnlyBackbone:
    """Minimal backbone exposing only the timestamp setter/clearer."""

    def __init__(self):
        self.set_calls = []
        self.clear_calls = 0
        self._batch_timestamps = None

    def set_batch_timestamps(self, ts):
        self.set_calls.append(ts)
        self._batch_timestamps = ts

    def clear_batch_timestamps(self):
        self.clear_calls += 1
        self._batch_timestamps = None


class _DummyBackbone(_TimestampsOnlyBackbone):
    """Backbone exposing both the timestamp and the nodata-mask side-channels."""

    def __init__(self):
        super().__init__()
        self.mask_set_calls = []
        self.mask_clear_calls = 0
        self._batch_nodata_mask = None

    def set_batch_nodata_mask(self, mask):
        self.mask_set_calls.append(mask)
        self._batch_nodata_mask = mask

    def clear_batch_nodata_mask(self):
        self.mask_clear_calls += 1
        self._batch_nodata_mask = None


class _DummyLocationBackbone(_TimestampsOnlyBackbone):
    """Backbone exposing BOTH the timestamps and location setter/clearer."""

    def __init__(self):
        super().__init__()
        self.loc_set_calls = []
        self.loc_clear_calls = 0
        self._batch_location = None

    def set_batch_location(self, loc):
        self.loc_set_calls.append(loc)
        self._batch_location = loc

    def clear_batch_location(self):
        self.loc_clear_calls += 1
        self._batch_location = None


def _make_task():
    from gelos.generation import LenientEmbeddingGenerationTask

    return LenientEmbeddingGenerationTask.__new__(LenientEmbeddingGenerationTask)


def test_predict_step_sets_then_clears_timestamps(monkeypatch):
    from gelos.generation import LenientEmbeddingGenerationTask

    task = _make_task()
    backbone = _DummyBackbone()
    task.model = backbone

    captured = {}

    def fake_super_predict_step(self, batch):
        # super() should see neither popped side-channel key.
        captured["batch_keys"] = set(batch.keys())
        captured["stashed"] = backbone._batch_timestamps
        captured["stashed_mask"] = backbone._batch_nodata_mask
        return "result"

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        fake_super_predict_step,
        raising=False,
    )

    ts = torch.tensor([[[18, 1, 2023]]], dtype=torch.int64)
    mask = {"S2L2A": torch.zeros(1, 12, 1, 8, 8, dtype=torch.bool)}
    out = task.predict_step({"image": "stub", "timestamps": ts, "nodata_mask": mask})

    assert out == "result"
    assert captured["batch_keys"] == {"image"}
    assert captured["stashed"] is ts  # set before super ran
    assert captured["stashed_mask"] is mask
    assert backbone.set_calls == [ts]
    assert backbone.clear_calls == 1  # cleared in finally
    assert backbone.mask_set_calls == [mask]
    assert backbone.mask_clear_calls == 1
    assert backbone._batch_nodata_mask is None


def test_predict_step_clears_on_exception(monkeypatch):
    from gelos.generation import LenientEmbeddingGenerationTask

    task = _make_task()
    backbone = _DummyBackbone()
    task.model = backbone

    def boom(self, batch):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        boom,
        raising=False,
    )

    ts = torch.tensor([[[18, 1, 2023]]], dtype=torch.int64)
    mask = torch.zeros(1, 12, 1, 8, 8, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="kaboom"):
        task.predict_step({"image": "stub", "timestamps": ts, "nodata_mask": mask})
    assert backbone.clear_calls == 1  # finally still ran
    assert backbone.mask_clear_calls == 1
    assert backbone._batch_nodata_mask is None


def test_predict_step_mask_is_noop_for_timestamps_only_backbone(monkeypatch):
    from gelos.generation import LenientEmbeddingGenerationTask

    task = _make_task()
    backbone = _TimestampsOnlyBackbone()
    task.model = backbone

    captured = {}

    def fake_super_predict_step(self, batch):
        captured["batch_keys"] = set(batch.keys())
        return "ok"

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        fake_super_predict_step,
        raising=False,
    )

    ts = torch.tensor([[[18, 1, 2023]]], dtype=torch.int64)
    mask = torch.zeros(1, 12, 1, 8, 8, dtype=torch.bool)
    out = task.predict_step({"image": "stub", "timestamps": ts, "nodata_mask": mask})
    assert out == "ok"
    # The mask key is still popped (terratorch never sees it); timestamps still flow.
    assert captured["batch_keys"] == {"image"}
    assert backbone.set_calls == [ts]
    assert backbone.clear_calls == 1
    assert not hasattr(backbone, "_batch_nodata_mask")


def test_predict_step_finds_setters_on_wrapped_encoder(monkeypatch):
    from gelos.generation import LenientEmbeddingGenerationTask

    class _Wrapper:
        def __init__(self, encoder):
            self.encoder = encoder

    task = _make_task()
    backbone = _DummyBackbone()
    task.model = _Wrapper(backbone)  # e.g. terratorch TemporalWrapper

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        lambda self, batch: "ok",
        raising=False,
    )

    ts = torch.tensor([[[18, 1, 2023]]], dtype=torch.int64)
    mask = torch.zeros(1, 12, 1, 8, 8, dtype=torch.bool)
    assert task.predict_step({"image": "stub", "timestamps": ts, "nodata_mask": mask}) == "ok"
    assert backbone.set_calls == [ts] and backbone.clear_calls == 1
    assert backbone.mask_set_calls == [mask] and backbone.mask_clear_calls == 1


def test_predict_step_noop_for_backbone_without_setter(monkeypatch):
    from gelos.generation import LenientEmbeddingGenerationTask

    task = _make_task()
    task.model = object()  # no set_batch_timestamps / encoder

    def fake_super_predict_step(self, batch):
        return "ok"

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        fake_super_predict_step,
        raising=False,
    )

    # Must not raise even though the backbone lacks both setters.
    out = task.predict_step(
        {
            "image": "stub",
            "timestamps": torch.zeros(1, 1, 3),
            "nodata_mask": torch.zeros(1, 1, 1, 4, 4, dtype=torch.bool),
        }
    )
    assert out == "ok"


def test_write_parquet_stores_float32_and_roundtrips(tmp_path):
    # The override must store the embedding as float32 (stock terratorch lands
    # float64 via .tolist()) while preserving values, nesting, and metadata.
    import numpy as np
    import pyarrow.dataset as ds

    task = _make_task()
    emb = torch.randn(2, 3, 4)  # [timesteps, tokens, dim]
    task.write_parquet(emb, "chip_0001.tif", {"file_id": torch.tensor(7)}, tmp_path)

    dataset = ds.dataset([str(tmp_path / "chip_0001_embedding.parquet")], format="parquet")
    table = dataset.to_table(columns=["embedding", "file_id"])
    import pyarrow as pa

    # three levels of list nesting with a float32 leaf, not the float64 the
    # stock .tolist() path would produce
    emb_type = table.schema.field("embedding").type
    for _ in range(3):
        assert pa.types.is_list(emb_type)
        emb_type = emb_type.value_type
    assert emb_type == pa.float32()
    read = np.array(table.column("embedding").to_pylist()[0], dtype=np.float32)
    np.testing.assert_array_equal(read, emb.numpy().astype(np.float32))
    assert table.column("file_id").to_pylist() == [7]
def test_predict_step_sets_then_clears_location(monkeypatch):
    from gelos.generation import LenientEmbeddingGenerationTask

    task = _make_task()
    backbone = _DummyLocationBackbone()
    task.model = backbone

    captured = {}

    def fake_super_predict_step(self, batch):
        # super() should see neither popped key.
        captured["batch_keys"] = set(batch.keys())
        captured["stashed_loc"] = backbone._batch_location
        captured["stashed_ts"] = backbone._batch_timestamps
        return "result"

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        fake_super_predict_step,
        raising=False,
    )

    ts = torch.tensor([[[2023, 2, 18]]], dtype=torch.int64)
    loc = torch.tensor([[42.25, -71.82]])
    out = task.predict_step({"image": "stub", "timestamps": ts, "location": loc})

    assert out == "result"
    assert "location" not in captured["batch_keys"]
    assert "timestamps" not in captured["batch_keys"]
    assert captured["stashed_loc"] is loc  # set before super ran
    assert captured["stashed_ts"] is ts
    assert backbone.loc_set_calls == [loc]
    assert backbone.set_calls == [ts]
    assert backbone.loc_clear_calls == 1  # cleared in finally
    assert backbone.clear_calls == 1


def test_predict_step_clears_location_on_exception(monkeypatch):
    from gelos.generation import LenientEmbeddingGenerationTask

    task = _make_task()
    backbone = _DummyLocationBackbone()
    task.model = backbone

    def boom(self, batch):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        boom,
        raising=False,
    )

    loc = torch.tensor([[42.25, -71.82]])
    with pytest.raises(RuntimeError, match="kaboom"):
        task.predict_step({"image": "stub", "location": loc})
    assert backbone.loc_clear_calls == 1  # finally still ran


def test_predict_step_location_noop_when_backbone_has_only_timestamps_setter(monkeypatch):
    # Independence of the two probes: a backbone with set_batch_timestamps but
    # NOT set_batch_location must not raise, and timestamps still dispatch.
    from gelos.generation import LenientEmbeddingGenerationTask

    task = _make_task()
    backbone = _DummyBackbone()  # timestamps setter only
    task.model = backbone

    def fake_super_predict_step(self, batch):
        return "ok"

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        fake_super_predict_step,
        raising=False,
    )

    ts = torch.tensor([[[2023, 2, 18]]], dtype=torch.int64)
    loc = torch.tensor([[42.25, -71.82]])
    out = task.predict_step({"image": "stub", "timestamps": ts, "location": loc})
    assert out == "ok"
    assert backbone.set_calls == [ts]  # timestamps still dispatched
    assert backbone.clear_calls == 1


def test_predict_step_location_noop_for_backbone_without_any_setter(monkeypatch):
    from gelos.generation import LenientEmbeddingGenerationTask

    task = _make_task()
    task.model = object()  # neither setter, no encoder

    def fake_super_predict_step(self, batch):
        return "ok"

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        fake_super_predict_step,
        raising=False,
    )

    out = task.predict_step({"image": "stub", "location": torch.zeros(1, 2)})
    assert out == "ok"


def test_predict_step_location_only_batch_skips_timestamps_machinery(monkeypatch):
    # A batch carrying only "location" dispatches it and never calls the
    # timestamps setter (clear still runs — it is resolved independently).
    from gelos.generation import LenientEmbeddingGenerationTask

    task = _make_task()
    backbone = _DummyLocationBackbone()
    task.model = backbone

    def fake_super_predict_step(self, batch):
        return "ok"

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        fake_super_predict_step,
        raising=False,
    )

    loc = torch.tensor([[42.25, -71.82]])
    out = task.predict_step({"image": "stub", "location": loc})
    assert out == "ok"
    assert backbone.loc_set_calls == [loc]
    assert backbone.set_calls == []  # timestamps setter never invoked
    assert backbone.loc_clear_calls == 1


# ---------------------------------------------------------------------------
# S1 band-reorder helper tests — pure logic, no model dependency.
# ---------------------------------------------------------------------------


def test_s1_band_reorder_index_identity():
    from gelos.backbones.olmoearth_backbone import build_s1_band_reorder_index

    assert build_s1_band_reorder_index(["VV", "VH"]) == [0, 1]


def test_s1_band_reorder_index_reversed():
    from gelos.backbones.olmoearth_backbone import build_s1_band_reorder_index

    assert build_s1_band_reorder_index(["VH", "VV"]) == [1, 0]


def test_s1_band_reorder_index_case_insensitive():
    from gelos.backbones.olmoearth_backbone import build_s1_band_reorder_index

    assert build_s1_band_reorder_index(["vv", "vh"]) == build_s1_band_reorder_index(["VV", "VH"])


def test_s1_band_reorder_index_missing_band_raises():
    from gelos.backbones.olmoearth_backbone import build_s1_band_reorder_index

    with pytest.raises(ValueError, match="vh"):
        build_s1_band_reorder_index(["VV"])


# ---------------------------------------------------------------------------
# Pretraining normalization helper tests — pure math, no model dependency.
# Expected values are hand-computed as (x - (mean - 2σ)) / (4σ) using the
# constants from olmoearth_pretrain/data/norm_configs/computed.json.
# ---------------------------------------------------------------------------

# computed.json sentinel1 stats (dB scale).
_VV_MEAN, _VV_STD = -11.648990747328444, 10.840350299936597
_VH_MEAN, _VH_STD = -17.745436133270044, 10.216274681392647
# computed.json sentinel2_l2a stats for B02 (BLUE) and B08 (NIR_BROAD).
_B02_MEAN, _B02_STD = 1188.9412572078477, 1859.1923971769581
_B08_MEAN, _B08_STD = 2755.481305028308, 1612.2565699990187


def _expected_minmax(x: float, mean: float, std: float) -> float:
    return (x - (mean - 2 * std)) / (4 * std)


def test_convert_to_db_matches_10_log10():
    from gelos.backbones.olmoearth_backbone import convert_to_db

    x = torch.tensor([1.0, 0.1, 0.01])
    torch.testing.assert_close(convert_to_db(x), torch.tensor([0.0, -10.0, -20.0]))


def test_convert_to_db_clips_at_1e_minus_10():
    from gelos.backbones.olmoearth_backbone import convert_to_db

    # Zero (and negative) linear power is clipped to 1e-10 -> -100 dB, not -inf.
    x = torch.tensor([0.0, -5.0, 1e-12])
    torch.testing.assert_close(convert_to_db(x), torch.tensor([-100.0, -100.0, -100.0]))


def test_s1_normalization_matches_hand_computed():
    from gelos.backbones.olmoearth_backbone import (
        convert_to_db,
        minmax_normalize,
        resolve_s1_band_stats,
    )

    means, stds = resolve_s1_band_stats(["VV", "VH"])
    assert means == [_VV_MEAN, _VH_MEAN]
    assert stds == [_VV_STD, _VH_STD]

    # Raw linear power (VV, VH), last axis = bands.
    x = torch.tensor([[0.1, 0.01]], dtype=torch.float64)  # -> -10 dB, -20 dB
    out = minmax_normalize(convert_to_db(x), means, stds)
    expected = torch.tensor(
        [
            [
                _expected_minmax(-10.0, _VV_MEAN, _VV_STD),
                _expected_minmax(-20.0, _VH_MEAN, _VH_STD),
            ]
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(out, expected)


def test_s2_normalization_matches_hand_computed():
    from gelos.backbones.olmoearth_backbone import minmax_normalize, resolve_s2_band_stats

    means, stds = resolve_s2_band_stats(["BLUE", "NIR_BROAD"])
    assert means == [_B02_MEAN, _B08_MEAN]
    assert stds == [_B02_STD, _B08_STD]

    # Raw DN values, no log for S2.
    x = torch.tensor([[1500.0, 3000.0]], dtype=torch.float64)
    out = minmax_normalize(x, means, stds)
    expected = torch.tensor(
        [
            [
                _expected_minmax(1500.0, _B02_MEAN, _B02_STD),
                _expected_minmax(3000.0, _B08_MEAN, _B08_STD),
            ]
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(out, expected)


def test_s2_normalization_band_mean_maps_to_half():
    # By construction, x == mean must normalize to exactly 0.5.
    from gelos.backbones.olmoearth_backbone import minmax_normalize, resolve_s2_band_stats

    means, stds = resolve_s2_band_stats(OLMOEARTH_S2_BAND_ORDER)
    x = torch.tensor(means, dtype=torch.float64)
    out = minmax_normalize(x, means, stds)
    torch.testing.assert_close(out, torch.full_like(out, 0.5))


def test_resolve_s2_band_stats_full_order_covers_all_12_bands():
    from gelos.backbones.olmoearth_backbone import (
        GELOS_TO_OLMOEARTH_S2_KEY,
        OLMOEARTH_COMPUTED_STATS,
        resolve_s2_band_stats,
    )

    means, stds = resolve_s2_band_stats(OLMOEARTH_S2_BAND_ORDER)
    s2_stats = OLMOEARTH_COMPUTED_STATS["sentinel2_l2a"]
    expected_keys = [GELOS_TO_OLMOEARTH_S2_KEY[b] for b in OLMOEARTH_S2_BAND_ORDER]
    assert means == [s2_stats[k]["mean"] for k in expected_keys]
    assert stds == [s2_stats[k]["std"] for k in expected_keys]


def test_resolve_s2_band_stats_unknown_band_raises():
    from gelos.backbones.olmoearth_backbone import resolve_s2_band_stats

    with pytest.raises(ValueError, match="NOT_A_BAND"):
        resolve_s2_band_stats(["BLUE", "NOT_A_BAND"])


def test_resolve_s1_band_stats_unknown_band_raises():
    from gelos.backbones.olmoearth_backbone import resolve_s1_band_stats

    with pytest.raises(ValueError, match="HH"):
        resolve_s1_band_stats(["HH"])


def test_minmax_normalize_broadcasts_over_leading_dims():
    # Same per-band math must apply at every (B, H, W, T) position.
    from gelos.backbones.olmoearth_backbone import minmax_normalize

    means, stds = [10.0, 20.0], [2.0, 4.0]
    x = torch.full((2, 3, 3, 4, 2), 10.0)
    x[..., 1] = 20.0  # each band sits exactly at its mean
    out = minmax_normalize(x, means, stds)
    torch.testing.assert_close(out, torch.full_like(out, 0.5))


def test_example_s1s2_fixture_yaml_valid():
    from pathlib import Path

    import yaml

    path = Path(__file__).parent / "fixtures" / "example_olmoearth_s1s2_config.yaml"
    config = yaml.safe_load(path.read_text())
    # The backbone normalizes internally; the datamodule must not double-normalize.
    assert config["data"]["init_args"]["normalize"] is False
    bands = config["data"]["init_args"]["bands"]
    assert "S1RTC" in bands
    assert set(bands["S1RTC"]) == {"VV", "VH"}
    model_args = config["model"]["init_args"]["model_args"]
    assert "bands_s1" in model_args
    assert set(model_args["bands_s1"]) == {"VV", "VH"}
    # Fixtures track the supported generation (v1 is deprecated).
    assert config["model"]["init_args"]["model"] == "olmoearth_v1_2_base_s1s2"
    assert model_args["model_id"] == "allenai/OlmoEarth-v1_2-Base"


def test_example_olmoearth_fixture_yaml_valid():
    from pathlib import Path

    import yaml

    path = Path(__file__).parent / "fixtures" / "example_olmoearth_config.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["data"]["init_args"]["normalize"] is False
    assert config["model"]["init_args"]["model"] == "olmoearth_v1_2_base"
    model_args = config["model"]["init_args"]["model_args"]
    assert model_args["model_id"] == "allenai/OlmoEarth-v1_2-Base"
    assert model_args["bands"] == config["data"]["init_args"]["bands"]["S2L2A"]


# ---------------------------------------------------------------------------
# OlmoEarth v1.2 factory tests (model-free: signature + registry only).
# ---------------------------------------------------------------------------

# (factory name, hidden_dim default, model_id default) for all 8 v1.2 factories.
V1_2_FACTORIES = [
    ("olmoearth_v1_2_nano", 128, "allenai/OlmoEarth-v1_2-Nano"),
    ("olmoearth_v1_2_tiny", 192, "allenai/OlmoEarth-v1_2-Tiny"),
    ("olmoearth_v1_2_small", 384, "allenai/OlmoEarth-v1_2-Small"),
    ("olmoearth_v1_2_base", 768, "allenai/OlmoEarth-v1_2-Base"),
    ("olmoearth_v1_2_nano_s1s2", 128, "allenai/OlmoEarth-v1_2-Nano"),
    ("olmoearth_v1_2_tiny_s1s2", 192, "allenai/OlmoEarth-v1_2-Tiny"),
    ("olmoearth_v1_2_small_s1s2", 384, "allenai/OlmoEarth-v1_2-Small"),
    ("olmoearth_v1_2_base_s1s2", 768, "allenai/OlmoEarth-v1_2-Base"),
]


@pytest.mark.parametrize("name, hidden_dim, model_id", V1_2_FACTORIES)
def test_v1_2_factory_defaults(name, hidden_dim, model_id):
    import inspect

    import gelos.backbones.olmoearth_backbone as oe

    fn = getattr(oe, name)
    params = inspect.signature(fn).parameters
    assert params["hidden_dim"].default == hidden_dim
    assert params["model_id"].default == model_id
    # The _s1s2 variants must expose the extra bands_s1 pass-through param.
    if name.endswith("_s1s2"):
        assert "bands_s1" in params


def test_v1_2_factories_registered():
    try:
        from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
    except ImportError:
        pytest.skip("terratorch registry not importable in this environment")
    # Importing the module triggers self-registration of the factories.
    import gelos.backbones.olmoearth_backbone  # noqa: F401

    for name, _hidden_dim, _model_id in V1_2_FACTORIES:
        assert name in TERRATORCH_BACKBONE_REGISTRY


# ---------------------------------------------------------------------------
# Nodata patch-mask helper tests — pure tensor logic, no model dependency.
# ---------------------------------------------------------------------------


def _pixel_mask(b=1, c=12, t=2, h=32, w=32):
    return torch.zeros(b, c, t, h, w, dtype=torch.bool)


def test_build_patch_nodata_mask_any_band_reduction():
    from gelos.backbones.olmoearth_backbone import build_patch_nodata_mask

    mask = _pixel_mask()
    # One band only, one patch (rows 0-3, cols 4-7), timestep 1 only.
    mask[0, 7, 1, 0:4, 4:8] = True
    out = build_patch_nodata_mask(mask, patch_size=4, threshold=0.5)
    assert out.shape == (1, 8, 8, 2)
    assert out.dtype == torch.bool
    assert out[0, 0, 1, 1].item() is True
    assert out[0, 0, 1, 0].item() is False  # other timestep untouched
    assert out.sum().item() == 1


def test_build_patch_nodata_mask_threshold_is_inclusive():
    from gelos.backbones.olmoearth_backbone import build_patch_nodata_mask

    mask = _pixel_mask()
    mask[0, :, :, 0:2, 0:4] = True  # exactly half of patch (0, 0)
    assert build_patch_nodata_mask(mask, 4, 0.5)[0, 0, 0].all()
    assert not build_patch_nodata_mask(mask, 4, 0.51)[0, 0, 0].any()


def test_build_patch_nodata_mask_threshold_one_keeps_partial_patches():
    from gelos.backbones.olmoearth_backbone import build_patch_nodata_mask

    mask = _pixel_mask()
    mask[0, :, :, 0:2, 0:4] = True  # half-nodata patch (0, 0)
    mask[0, :, :, 4:8, 4:8] = True  # fully-nodata patch (1, 1)
    out = build_patch_nodata_mask(mask, 4, 1.0)
    assert not out[0, 0, 0].any()
    assert out[0, 1, 1].all()
    assert out.sum().item() == 2  # (1, 1) at both timesteps


def test_build_patch_nodata_mask_zero_threshold_masks_any_touched_patch():
    from gelos.backbones.olmoearth_backbone import build_patch_nodata_mask

    mask = _pixel_mask()
    mask[0, 0, 0, 5, 5] = True  # a single pixel inside patch (1, 1)
    out = build_patch_nodata_mask(mask, 4, 0.0)
    assert out[0, 1, 1, 0].item() is True
    assert out.sum().item() == 1


def test_build_patch_nodata_mask_rejects_non_divisible():
    from gelos.backbones.olmoearth_backbone import build_patch_nodata_mask

    with pytest.raises(ValueError, match="divisible"):
        build_patch_nodata_mask(_pixel_mask(h=30, w=32), 4, 0.5)


def test_patch_mask_to_olmoearth_mask_is_patch_constant():
    from gelos.backbones.olmoearth_backbone import patch_mask_to_olmoearth_mask

    patch_missing = torch.zeros(1, 2, 3, 2, dtype=torch.bool)  # (B, H', W', T)
    patch_missing[0, 1, 2, 0] = True
    out = patch_mask_to_olmoearth_mask(patch_missing, patch_size=4, num_band_sets=3)
    assert out.shape == (1, 8, 12, 2, 3)
    assert out.dtype == torch.int32
    assert set(out.unique().tolist()) == {0, 3}
    # The whole 4x4 block of patch (1, 2) at t=0 is MISSING for every band set;
    # the top-left pixel (what the encoder reads) carries the patch value.
    block = out[0, 4:8, 8:12, 0, :]
    assert (block == 3).all()
    assert (out[0, 4, 8, 0, :] == 3).all()
    # Everything else is ONLINE_ENCODER.
    assert (out == 3).sum().item() == 4 * 4 * 3
    assert (out[0, :, :, 1, :] == 0).all()


def test_patch_mask_to_olmoearth_mask_explicit_codes():
    from gelos.backbones.olmoearth_backbone import patch_mask_to_olmoearth_mask

    patch_missing = torch.tensor([[[[True]]]])  # (1, 1, 1, 1)
    out = patch_mask_to_olmoearth_mask(
        patch_missing, patch_size=2, num_band_sets=1, missing_value=7, online_value=1
    )
    assert out.shape == (1, 2, 2, 1, 1)
    assert (out == 7).all()


def test_masked_mean_zero_count_yields_zeros_and_false():
    from gelos.backbones.olmoearth_backbone import masked_mean

    x = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2) + 1.0
    valid = torch.tensor([[True, False, True], [False, False, False]])
    mean, still_valid = masked_mean(x, valid, dims=(1,))
    assert mean.shape == (2, 2) and still_valid.shape == (2,)
    torch.testing.assert_close(mean[0], (x[0, 0] + x[0, 2]) / 2)
    assert torch.equal(mean[1], torch.zeros(2))
    assert still_valid.tolist() == [True, False]


def test_masked_mean_all_valid_matches_plain_mean():
    from gelos.backbones.olmoearth_backbone import masked_mean

    x = torch.randn(2, 4, 5, 3)
    mean, still_valid = masked_mean(x, torch.ones(2, 4, 5, dtype=torch.bool), dims=(1, 2))
    torch.testing.assert_close(mean, x.mean(dim=(1, 2)))
    assert still_valid.all()


def test_constructor_rejects_invalid_nodata_patch_threshold():
    # Validated before the lazy olmoearth_pretrain import.
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    for bad in (1.5, -0.1):
        with pytest.raises(ValueError, match="nodata_patch_threshold"):
            OlmoEarthBackbone(pretrained=False, bands=ALL_12_BANDS, nodata_patch_threshold=bad)


def test_constructor_rejects_unknown_spatial_pooling_string():
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    with pytest.raises(ValueError, match="spatial_pooling"):
        OlmoEarthBackbone(pretrained=False, bands=ALL_12_BANDS, spatial_pooling="max")


def test_set_and_clear_batch_nodata_mask():
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    backbone = OlmoEarthBackbone.__new__(OlmoEarthBackbone)
    backbone._batch_nodata_mask = None

    mask = {"S2L2A": torch.zeros(1, 12, 1, 8, 8, dtype=torch.bool)}
    backbone.set_batch_nodata_mask(mask)
    assert backbone._batch_nodata_mask is mask

    backbone.clear_batch_nodata_mask()
    assert backbone._batch_nodata_mask is None


# ---------------------------------------------------------------------------
# Nodata masking with the real encoder — random weights, eval mode.
# ---------------------------------------------------------------------------

# Grid for a (32, 32) chip at patch_size=4 is 8x8 = 64 tokens per timestep.
_GRID = 8
_MASKED_PATCHES = [(0, 0), (3, 5)]  # (row, col) of nodata patches
_VALID_PATCH = (4, 4)


def _masked_backbone(reference=None, **kwargs):
    """Eval-mode backbone (random weights); shares weights with ``reference`` if given."""
    pytest.importorskip("olmoearth_pretrain")
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    torch.manual_seed(42)
    backbone = OlmoEarthBackbone(pretrained=False, bands=ALL_12_BANDS, patch_size=4, **kwargs)
    if reference is not None:
        backbone.load_state_dict(reference.state_dict())
    return backbone.eval()


def _s2_input(b=1, t=2):
    torch.manual_seed(0)
    return 1500 + 500 * torch.randn(b, 12, t, 32, 32)


def _s2_nodata_mask(b=1, t=2, sample_indices=None):
    """Pixel mask with ``_MASKED_PATCHES`` fully nodata (all bands, all timesteps)."""
    mask = torch.zeros(b, 12, t, 32, 32, dtype=torch.bool)
    for i in range(b) if sample_indices is None else sample_indices:
        for row, col in _MASKED_PATCHES:
            mask[i, :, :, row * 4 : (row + 1) * 4, col * 4 : (col + 1) * 4] = True
    return mask


def _token_index(t, row, col):
    """Time-major token index for temporal_pooling='keep' on the 8x8 grid."""
    return t * _GRID * _GRID + row * _GRID + col


def _run(backbone, x, mask=None):
    backbone.set_batch_nodata_mask(mask)
    try:
        with torch.no_grad():
            return backbone.forward_features(x)[0]
    finally:
        backbone.clear_batch_nodata_mask()


def test_nodata_mask_zeroes_masked_tokens_and_changes_valid_ones():
    backbone = _masked_backbone(temporal_pooling="keep")
    x = _s2_input()
    unmasked = _run(backbone, x)
    masked = _run(backbone, x, _s2_nodata_mask())
    assert masked.shape == unmasked.shape == (1, 2 * 64, backbone.out_channels)

    for t in range(2):
        for row, col in _MASKED_PATCHES:
            assert torch.equal(
                masked[0, _token_index(t, row, col)], torch.zeros(backbone.out_channels)
            )
        assert not torch.equal(unmasked[0, _token_index(t, 0, 0)], torch.zeros_like(unmasked[0, 0]))
        # Valid tokens see a different attention context once nodata tokens are removed.
        assert not torch.allclose(
            masked[0, _token_index(t, *_VALID_PATCH)], unmasked[0, _token_index(t, *_VALID_PATCH)]
        )


def test_nodata_mask_dict_stash_matches_tensor_stash():
    backbone = _masked_backbone(temporal_pooling="keep")
    x = _s2_input()
    mask = _s2_nodata_mask()
    from_tensor = _run(backbone, x, mask)
    from_dict = _run(backbone, {"S2L2A": x}, {"S2L2A": mask})
    assert torch.equal(from_tensor, from_dict)


def test_all_false_nodata_mask_is_bit_identical_to_no_mask():
    backbone = _masked_backbone(temporal_pooling="keep")
    x = _s2_input()
    unmasked = _run(backbone, x)
    all_false = _run(backbone, x, torch.zeros(1, 12, 2, 32, 32, dtype=torch.bool))
    assert torch.equal(unmasked, all_false)


def test_nodata_mask_no_batch_leakage():
    # Guards the encoder.training toggle: in eval mode OlmoEarth builds no
    # attention mask, so the zero pads of a batch-mate with fewer valid tokens
    # would leak into attention and the B=2 result would differ from B=1.
    backbone = _masked_backbone(temporal_pooling="keep")
    x = _s2_input(b=2)
    mask = _s2_nodata_mask(b=2, sample_indices=[1])  # sample 0 unmasked
    batched = _run(backbone, x, mask)
    solo_masked = _run(backbone, x[1:2], mask[1:2])
    solo_unmasked = _run(backbone, x[0:1])
    torch.testing.assert_close(batched[1:2], solo_masked, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(batched[0:1], solo_unmasked, rtol=1e-4, atol=1e-4)
    # The toggle must be undone after the forward.
    assert backbone.encoder.encoder.training is False


def test_spatial_pooling_mean_matches_masked_mean_of_grid():
    grid_bb = _masked_backbone(temporal_pooling="keep")
    mean_bb = _masked_backbone(reference=grid_bb, temporal_pooling="keep", spatial_pooling="mean")
    from gelos.backbones.olmoearth_backbone import build_patch_nodata_mask

    x = _s2_input()
    mask = _s2_nodata_mask()
    grid = _run(grid_bb, x, mask).reshape(1, 2, 64, -1)  # (B, T, H'*W', D)
    pooled = _run(mean_bb, x, mask)
    assert pooled.shape == (1, 2, grid_bb.out_channels)

    valid = ~build_patch_nodata_mask(mask, 4, 0.5)  # (B, H', W', T)
    valid = valid.permute(0, 3, 1, 2).reshape(1, 2, 64)
    expected = torch.stack(
        [grid[0, t][valid[0, t]].mean(dim=0) for t in range(2)]
    ).unsqueeze(0)
    torch.testing.assert_close(pooled, expected, rtol=1e-4, atol=1e-5)

    both_bb = _masked_backbone(reference=grid_bb, temporal_pooling="mean", spatial_pooling="mean")
    both = _run(both_bb, x, mask)
    assert both.shape == (1, 1, grid_bb.out_channels)
    torch.testing.assert_close(both[:, 0], expected.mean(dim=1), rtol=1e-4, atol=1e-5)


def test_spatial_pooling_mean_unmasked_shapes():
    backbone = _masked_backbone(temporal_pooling="keep", spatial_pooling="mean")
    assert _run(backbone, _s2_input(b=2, t=3)).shape == (2, 3, backbone.out_channels)
    backbone = _masked_backbone(temporal_pooling="mean", spatial_pooling="mean")
    assert _run(backbone, _s2_input(b=2, t=3)).shape == (2, 1, backbone.out_channels)


def test_fully_masked_sample_warns_and_encodes_unmasked():
    backbone = _masked_backbone(temporal_pooling="keep")
    x = _s2_input()
    unmasked = _run(backbone, x)
    with pytest.warns(UserWarning, match="no valid patch"):
        out = _run(backbone, x, torch.ones(1, 12, 2, 32, 32, dtype=torch.bool))
    assert torch.equal(out, unmasked)


def test_nodata_mask_shape_mismatch_warns_and_falls_back():
    backbone = _masked_backbone(temporal_pooling="keep")
    x = _s2_input()
    unmasked = _run(backbone, x)
    with pytest.warns(UserWarning, match="nodata mask"):
        out = _run(backbone, x, torch.ones(1, 12, 2, 16, 16, dtype=torch.bool))
    assert torch.equal(out, unmasked)


def test_mask_nodata_false_ignores_stash():
    backbone = _masked_backbone(temporal_pooling="keep", mask_nodata=False)
    x = _s2_input()
    unmasked = _run(backbone, x)
    assert torch.equal(_run(backbone, x, _s2_nodata_mask()), unmasked)


def test_spatial_pooling_int_warns_when_pooled_window_has_no_valid_token():
    # 8x8 grid pooled by 8 -> a single output token per timestep; masking the
    # whole grid at t=0 only (t=1 stays valid so the sample is not fully masked)
    # leaves that pooled position with zero contributors -> zero vector + warning.
    backbone = _masked_backbone(temporal_pooling="keep", spatial_pooling=8)
    x = _s2_input()
    mask = torch.zeros(1, 12, 2, 32, 32, dtype=torch.bool)
    mask[:, :, 0] = True
    with pytest.warns(UserWarning, match="zero valid"):
        out = _run(backbone, x, mask)
    assert out.shape == (1, 2, backbone.out_channels)
    assert torch.equal(out[0, 0], torch.zeros(backbone.out_channels))
    assert not torch.equal(out[0, 1], torch.zeros(backbone.out_channels))


def test_s1_only_nodata_keeps_s2_token_at_that_patch():
    # Nodata only in S1RTC: the S2 mask stays all-ONLINE, so the fused token at
    # the S1-masked patch is the S2 token alone (non-zero), while attention
    # context (and hence every token) differs from the unmasked run.
    backbone = _masked_backbone(temporal_pooling="keep", bands_s1=["VV", "VH"])
    x_s2 = _s2_input()
    torch.manual_seed(1)
    x_s1 = 0.05 + 0.02 * torch.rand(1, 2, 2, 32, 32)
    batch = {"S2L2A": x_s2, "S1RTC": x_s1}
    unmasked = _run(backbone, batch)
    s1_mask = torch.zeros(1, 2, 2, 32, 32, dtype=torch.bool)
    s1_mask[:, :, :, 0:4, 0:4] = True  # patch (0, 0), all timesteps
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no fully-masked / zero-valid warnings
        masked = _run(backbone, batch, {"S1RTC": s1_mask})
    zero = torch.zeros(backbone.out_channels)
    for t in range(2):
        assert not torch.equal(masked[0, _token_index(t, 0, 0)], zero)
        assert not torch.allclose(masked[0, _token_index(t, 0, 0)], unmasked[0, _token_index(t, 0, 0)])
        assert not torch.allclose(
            masked[0, _token_index(t, *_VALID_PATCH)], unmasked[0, _token_index(t, *_VALID_PATCH)]
        )

