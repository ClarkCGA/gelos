"""Tests for the Prithvi EO v2 TL (time + location) backbone wrapper.

Model-dependent tests build a *random-init tiny* TL encoder (no weights
download): ``prithvi_eo_v2_tiny_tl`` with ``embed_dim=32, depth=2, num_heads=2``
on 32x32 inputs (16-pixel patches -> 4 patches + CLS).
"""

import inspect

from gelos.backbones.prithvi_tl_backbone import (
    PRITHVI_TL_VARIANTS,
    PrithviTLBackbone,
    calendar_to_prithvi_temporal_coords,
)
import pytest
import torch

TL_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
TINY_KW = dict(pretrained=False, bands=TL_BANDS, embed_dim=32, depth=2, num_heads=2)
EMBED = 32
N_PATCHES = 4  # 32x32 input, 16x16 patches


def _tiny(**overrides) -> PrithviTLBackbone:
    kw = {**TINY_KW, **overrides}
    torch.manual_seed(0)
    model = PrithviTLBackbone("prithvi_eo_v2_tiny_tl", **kw)
    return model.eval()


@pytest.fixture(scope="module")
def tiny_tl() -> PrithviTLBackbone:
    return _tiny()


def _stash(model: PrithviTLBackbone, ts: torch.Tensor, loc: torch.Tensor) -> None:
    model.set_batch_timestamps(ts)
    model.set_batch_location(loc)


def _clear(model: PrithviTLBackbone) -> None:
    model.clear_batch_timestamps()
    model.clear_batch_location()


# ---------------------------------------------------------------------------
# calendar_to_prithvi_temporal_coords — pure conversion, no model dependency.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ymd, expected_doy",
    [
        ([2021, 1, 1], 1),
        ([2021, 12, 31], 365),
        ([2020, 12, 31], 366),
        ([2020, 3, 1], 61),
        ([2021, 3, 1], 60),
        ([2000, 3, 1], 61),  # divisible by 400 -> leap
        ([1900, 3, 1], 60),  # divisible by 100 but not 400 -> not leap
    ],
)
def test_calendar_to_prithvi_temporal_coords_day_of_year(ymd, expected_doy):
    out = calendar_to_prithvi_temporal_coords(torch.tensor([ymd], dtype=torch.long))
    assert out.dtype == torch.float32
    assert out.shape == (1, 2)
    assert out[0, 0].item() == ymd[0]
    assert out[0, 1].item() == expected_doy


def test_calendar_to_prithvi_temporal_coords_preserves_leading_dims():
    ts = torch.tensor(
        [[[2021, 1, 1], [2021, 6, 15], [2021, 12, 31]], [[2020, 2, 29], [2020, 3, 1], [2020, 7, 4]]],
        dtype=torch.long,
    )
    out = calendar_to_prithvi_temporal_coords(ts)
    assert out.shape == (2, 3, 2)
    assert out.dtype == torch.float32
    assert out[0, :, 1].tolist() == [1.0, 166.0, 365.0]
    assert out[1, :, 1].tolist() == [60.0, 61.0, 186.0]


@pytest.mark.parametrize("month", [0, 13])
def test_calendar_to_prithvi_temporal_coords_bad_month_raises(month):
    with pytest.raises(ValueError, match="month"):
        calendar_to_prithvi_temporal_coords(torch.tensor([[2021, month, 1]]))


def test_calendar_to_prithvi_temporal_coords_bad_shape_raises():
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        calendar_to_prithvi_temporal_coords(torch.zeros(3, 2, dtype=torch.long))


# ---------------------------------------------------------------------------
# Stash tests on a bare instance (no model build).
# ---------------------------------------------------------------------------


def test_set_and_clear_batch_timestamps_and_location():
    backbone = PrithviTLBackbone.__new__(PrithviTLBackbone)
    backbone._batch_timestamps = None
    backbone._batch_location = None

    ts = torch.tensor([[[2020, 2, 15]]], dtype=torch.long)
    loc = torch.tensor([[42.0, -71.0]])
    backbone.set_batch_timestamps(ts)
    backbone.set_batch_location(loc)
    assert backbone._batch_timestamps is ts
    assert backbone._batch_location is loc

    backbone.clear_batch_timestamps()
    backbone.clear_batch_location()
    assert backbone._batch_timestamps is None
    assert backbone._batch_location is None


# ---------------------------------------------------------------------------
# Fail-loud branches.
# ---------------------------------------------------------------------------


def _ts(B: int, T: int) -> torch.Tensor:
    return torch.tensor(
        [[[2021, 1 + (b + t) % 12, 1 + 3 * b + 5 * t] for t in range(T)] for b in range(B)],
        dtype=torch.long,
    )


def _loc(B: int) -> torch.Tensor:
    return torch.tensor([[42.25 + b, -71.82 - b] for b in range(B)], dtype=torch.float32)


def test_forward_without_stash_raises(tiny_tl):
    _clear(tiny_tl)
    with pytest.raises(ValueError, match=r"timestamps.*_get_timestamps") as exc:
        tiny_tl(torch.randn(2, 6, 32, 32))
    assert "prithvi_eo_v2_tiny_tl" in str(exc.value)


def test_forward_with_timestamps_only_raises(tiny_tl):
    _clear(tiny_tl)
    tiny_tl.set_batch_timestamps(_ts(2, 1))
    try:
        with pytest.raises(ValueError, match=r"location.*_get_location"):
            tiny_tl(torch.randn(2, 6, 32, 32))
    finally:
        _clear(tiny_tl)


def test_location_wrong_shape_raises(tiny_tl):
    _stash(tiny_tl, _ts(2, 1), torch.zeros(2, 3))
    try:
        with pytest.raises(ValueError, match=r"location.*\(B, 2\)"):
            tiny_tl(torch.randn(2, 6, 32, 32))
    finally:
        _clear(tiny_tl)


def test_4d_batch_mismatch_raises(tiny_tl):
    _stash(tiny_tl, _ts(2, 3), _loc(2))  # implies N = 6
    try:
        with pytest.raises(ValueError, match=r"N=5.*B\*T=6"):
            tiny_tl(torch.randn(5, 6, 32, 32))
    finally:
        _clear(tiny_tl)


def test_5d_num_frames_mismatch_raises(tiny_tl):
    assert tiny_tl.num_frames == 1
    _stash(tiny_tl, _ts(2, 3), _loc(2))
    try:
        with pytest.raises(ValueError, match="num_frames"):
            tiny_tl(torch.randn(2, 6, 3, 32, 32))
    finally:
        _clear(tiny_tl)


def test_3d_input_raises(tiny_tl):
    _stash(tiny_tl, _ts(2, 1), _loc(2))
    try:
        with pytest.raises(ValueError, match="4D.*5D"):
            tiny_tl(torch.randn(2, 6, 32))
    finally:
        _clear(tiny_tl)


# ---------------------------------------------------------------------------
# Plumbing correctness against a direct encoder call.
# ---------------------------------------------------------------------------


def test_4d_path_matches_direct_encoder_call_and_coords_matter(tiny_tl):
    B, T = 2, 3
    ts, loc = _ts(B, T), _loc(B)
    torch.manual_seed(1)
    x = torch.randn(B * T, 6, 32, 32)
    _stash(tiny_tl, ts, loc)
    try:
        out = tiny_tl(x)
    finally:
        _clear(tiny_tl)
    assert isinstance(out, list)
    assert out[-1].shape == (B * T, 1 + N_PATCHES, EMBED)

    tc = calendar_to_prithvi_temporal_coords(ts).reshape(B * T, 1, 2)
    lc = loc.repeat_interleave(T, dim=0)
    expected = tiny_tl.encoder(x, temporal_coords=tc, location_coords=lc)
    torch.testing.assert_close(out[-1], expected[-1])
    # Coords actually change the output — the stock silent-no-op path did not run.
    assert not torch.allclose(out[-1], tiny_tl.encoder(x)[-1])


def test_5d_path_matches_direct_encoder_call():
    model = _tiny(num_frames=2)
    B, T = 2, 2
    ts, loc = _ts(B, T), _loc(B)
    torch.manual_seed(2)
    x = torch.randn(B, 6, T, 32, 32)
    _stash(model, ts, loc)
    try:
        out = model(x)
    finally:
        _clear(model)
    assert out[-1].shape == (B, 1 + T * N_PATCHES, EMBED)
    tc = calendar_to_prithvi_temporal_coords(ts)
    expected = model.encoder(x, temporal_coords=tc, location_coords=loc)
    torch.testing.assert_close(out[-1], expected[-1])


def test_no_leakage_after_clear(tiny_tl):
    x = torch.randn(2, 6, 32, 32)
    _stash(tiny_tl, _ts(2, 1), _loc(2))
    tiny_tl(x)
    _clear(tiny_tl)
    with pytest.raises(ValueError, match="timestamps"):
        tiny_tl(x)


# ---------------------------------------------------------------------------
# Task integration under TemporalWrapper.
# ---------------------------------------------------------------------------


def test_task_predict_step_under_temporal_wrapper(tiny_tl, monkeypatch):
    from terratorch.models.utils import TemporalWrapper

    from gelos.generation import LenientEmbeddingGenerationTask

    task = LenientEmbeddingGenerationTask.__new__(LenientEmbeddingGenerationTask)
    # The bare task skipped Module.__init__, so assigning a Module attribute via
    # the normal setattr path raises; store it directly on the instance dict.
    object.__setattr__(task, "model", TemporalWrapper(tiny_tl, pooling="keep"))

    def fake_super_predict_step(self, batch):
        return self.model(batch["image"])

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        fake_super_predict_step,
        raising=False,
    )

    B, T = 2, 3
    ts, loc = _ts(B, T), _loc(B)
    x5d = torch.randn(B, 6, T, 32, 32)
    result = task.predict_step({"image": x5d, "timestamps": ts, "location": loc})
    assert result[0].shape == (B, T, 1 + N_PATCHES, EMBED)
    assert tiny_tl._batch_timestamps is None
    assert tiny_tl._batch_location is None

    with pytest.raises(ValueError, match="location"):
        task.predict_step({"image": x5d, "timestamps": ts})
    assert tiny_tl._batch_timestamps is None
    assert tiny_tl._batch_location is None


# ---------------------------------------------------------------------------
# Registry / factory tests.
# ---------------------------------------------------------------------------

COORDS_NAMES = [f"{v}_coords" for v in PRITHVI_TL_VARIANTS]


def test_factories_registered():
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

    import gelos.backbones.prithvi_tl_backbone  # noqa: F401

    assert COORDS_NAMES == [
        "prithvi_eo_v2_tiny_tl_coords",
        "prithvi_eo_v2_100_tl_coords",
        "prithvi_eo_v2_300_tl_coords",
        "prithvi_eo_v2_600_tl_coords",
    ]
    for name in COORDS_NAMES:
        assert name in TERRATORCH_BACKBONE_REGISTRY


def test_registry_build_returns_wrapper():
    from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

    model = TERRATORCH_BACKBONE_REGISTRY.build("prithvi_eo_v2_tiny_tl_coords", **TINY_KW)
    assert isinstance(model, PrithviTLBackbone)
    assert model.encoder.temporal_encoding and model.encoder.location_encoding


@pytest.mark.parametrize("name", COORDS_NAMES)
def test_factory_pretrained_default_true(name):
    import gelos.backbones.prithvi_tl_backbone as m

    assert inspect.signature(getattr(m, name)).parameters["pretrained"].default is True


def test_non_tl_variant_raises():
    with pytest.raises(ValueError, match="prithvi_eo_v2_300"):
        PrithviTLBackbone("prithvi_eo_v2_300", pretrained=False)


def test_example_prithvi_tl_fixture_yaml_valid():
    from pathlib import Path

    import yaml

    path = Path(__file__).parent / "fixtures" / "example_prithvi_tl_config.yaml"
    config = yaml.safe_load(path.read_text())
    init = config["model"]["init_args"]
    assert init["model"].endswith("_tl_coords")
    assert config["data"]["init_args"]["dataset_class"] == (
        "tests.test_data.TimestampedExampleGELOSDataSet"
    )
    assert init["has_cls"] is True
    assert init["temporal_cfg"]["temporal_wrapper"] is True
