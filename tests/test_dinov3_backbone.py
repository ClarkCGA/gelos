"""Tests for the DINOv3 pretrained backbone variants.

The variant table and checkpoint-path handling are pure logic with no weight
download, so those tests run unconditionally (module import needs terratorch +
torch, like the olmoearth backbone tests). Nothing here loads real weights —
the first real load happens in the pipeline run.
"""

import pytest

from gelos.backbones.dinov3_backbone import _VARIANTS, _load_hub_model


def test_web_variant_weight_sources():
    variant = _VARIANTS["dinov3_vitb16"]
    assert variant.hub_model == "dinov3_vitb16"
    assert variant.hf_repo == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert variant.ckpt_env == "DINOV3_VITB16_CKPT"
    # Hub default weights ARE the web LVD-1689M checkpoint.
    assert variant.try_hub_pretrained is True


def test_sat_variant_weight_sources():
    variant = _VARIANTS["dinov3_vitl16_sat"]
    # Same hub architecture as the web ViT-L, different weights.
    assert variant.hub_model == "dinov3_vitl16"
    assert variant.hf_repo == "facebook/dinov3-vitl16-pretrain-sat493m"
    assert variant.ckpt_env == "DINOV3_VITL16_SAT_CKPT"


def test_sat_variant_never_uses_hub_pretrained_default():
    # torch.hub's default weights for dinov3_vitl16 are the WEB checkpoint;
    # loading them for the sat variant would be a silent wrong-weights bug.
    assert _VARIANTS["dinov3_vitl16_sat"].try_hub_pretrained is False


def test_both_variants_registered():
    try:
        from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
    except ImportError:
        pytest.skip("terratorch registry not importable in this environment")

    assert "dinov3_vitb16_pretrained" in TERRATORCH_BACKBONE_REGISTRY
    assert "dinov3_vitl16_sat_pretrained" in TERRATORCH_BACKBONE_REGISTRY


@pytest.mark.parametrize("variant_name", list(_VARIANTS))
def test_missing_ckpt_path_raises_instead_of_falling_through(variant_name, monkeypatch, tmp_path):
    # A set-but-invalid ckpt env var must fail loudly, not fall through to a
    # download that could resolve to the wrong weights.
    missing = tmp_path / "no_such_checkpoint.pth"
    monkeypatch.setenv(_VARIANTS[variant_name].ckpt_env, str(missing))
    with pytest.raises(FileNotFoundError, match=_VARIANTS[variant_name].ckpt_env):
        _load_hub_model(variant_name)
