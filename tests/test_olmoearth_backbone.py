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
    import yaml
    from pathlib import Path

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
