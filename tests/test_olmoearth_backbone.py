"""Tests for the OlmoEarth terratorch backbone wrapper.

The band-reorder helper is pure index logic with no model dependency, so those
tests run unconditionally and are the most valuable to keep green in CI without the
heavy optional ``olmoearth-pretrain`` extra installed. Tests that need the actual
model are gated with ``pytest.importorskip("olmoearth_pretrain")``.
"""

from gelos.backbones.olmoearth_backbone import (
    OLMOEARTH_S2_BAND_ORDER,
    build_band_reorder_index,
)
import pytest
import torch

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


def test_band_reorder_index_raises_on_missing_band():
    # Drop WATER_VAPOR (B09) — the band ExampleGELOSDataSet also lacks.
    missing = [b for b in ALL_12_BANDS if b != "WATER_VAPOR"]
    with pytest.raises(ValueError, match="WATER_VAPOR"):
        build_band_reorder_index(missing)


def test_band_reorder_index_reports_all_missing_bands():
    with pytest.raises(ValueError) as exc:
        build_band_reorder_index(["BLUE", "GREEN", "RED"])
    msg = str(exc.value)
    # Several required bands should be named in the error.
    assert "COASTAL_AEROSOL" in msg and "WATER_VAPOR" in msg


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


def test_constructor_raises_on_missing_band_without_model():
    # Construction validates bands before touching the model, so a missing band
    # raises ValueError regardless of whether the extra is installed.
    from gelos.backbones.olmoearth_backbone import OlmoEarthBackbone

    missing = [b for b in ALL_12_BANDS if b != "WATER_VAPOR"]
    with pytest.raises(ValueError, match="WATER_VAPOR"):
        OlmoEarthBackbone(pretrained=False, bands=missing)


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


class _DummyBackbone:
    """Minimal backbone exposing the timestamp setter/clearer."""

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
        # super() should not see the popped timestamps key.
        captured["batch_keys"] = set(batch.keys())
        captured["stashed"] = backbone._batch_timestamps
        return "result"

    monkeypatch.setattr(
        LenientEmbeddingGenerationTask.__mro__[1],
        "predict_step",
        fake_super_predict_step,
        raising=False,
    )

    ts = torch.tensor([[[18, 1, 2023]]], dtype=torch.int64)
    out = task.predict_step({"image": "stub", "timestamps": ts})

    assert out == "result"
    assert "timestamps" not in captured["batch_keys"]
    assert captured["stashed"] is ts  # set before super ran
    assert backbone.set_calls == [ts]
    assert backbone.clear_calls == 1  # cleared in finally


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
    with pytest.raises(RuntimeError, match="kaboom"):
        task.predict_step({"image": "stub", "timestamps": ts})
    assert backbone.clear_calls == 1  # finally still ran


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

    # Must not raise even though the backbone lacks the setter.
    out = task.predict_step({"image": "stub", "timestamps": torch.zeros(1, 1, 3)})
    assert out == "ok"


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


def test_example_s1s2_fixture_yaml_valid():
    import yaml
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "example_olmoearth_s1s2_config.yaml"
    config = yaml.safe_load(path.read_text())
    bands = config["data"]["init_args"]["bands"]
    assert "S1RTC" in bands
    assert set(bands["S1RTC"]) == {"VV", "VH"}
    model_args = config["model"]["init_args"]["model_args"]
    assert "bands_s1" in model_args
    assert set(model_args["bands_s1"]) == {"VV", "VH"}
