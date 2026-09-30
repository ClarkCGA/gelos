"""Tests for the DINOv3 pretrained backbone variants.

The variant table and checkpoint-path handling are pure logic with no weight
download, so those tests run unconditionally (module import needs terratorch +
torch, like the olmoearth backbone tests). Nothing here loads real weights —
the first real load happens in the pipeline run.
"""

import pytest
import torch

from gelos.backbones.dinov3_backbone import (
    _VARIANTS,
    _convert_hf_transformers_state_dict,
    _load_hub_model,
)


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


def _fake_hf_state_dict(dim: int = 4, n_layers: int = 2) -> dict:
    """Tiny HF-transformers-format DINOv3 state dict for conversion tests."""
    sd = {
        "embeddings.cls_token": torch.randn(1, 1, dim),
        "embeddings.mask_token": torch.randn(1, 1, dim),
        "embeddings.register_tokens": torch.randn(1, 4, dim),
        "embeddings.patch_embeddings.weight": torch.randn(dim, 3, 16, 16),
        "embeddings.patch_embeddings.bias": torch.randn(dim),
        "norm.weight": torch.randn(dim),
        "norm.bias": torch.randn(dim),
    }
    for i in range(n_layers):
        p = f"layer.{i}."
        sd |= {
            f"{p}attention.q_proj.weight": torch.randn(dim, dim),
            f"{p}attention.q_proj.bias": torch.randn(dim),
            f"{p}attention.k_proj.weight": torch.randn(dim, dim),
            f"{p}attention.v_proj.weight": torch.randn(dim, dim),
            f"{p}attention.v_proj.bias": torch.randn(dim),
            f"{p}attention.o_proj.weight": torch.randn(dim, dim),
            f"{p}attention.o_proj.bias": torch.randn(dim),
            f"{p}layer_scale1.lambda1": torch.randn(dim),
            f"{p}layer_scale2.lambda1": torch.randn(dim),
            f"{p}mlp.up_proj.weight": torch.randn(4 * dim, dim),
            f"{p}mlp.up_proj.bias": torch.randn(4 * dim),
            f"{p}mlp.down_proj.weight": torch.randn(dim, 4 * dim),
            f"{p}mlp.down_proj.bias": torch.randn(dim),
            f"{p}norm1.weight": torch.randn(dim),
            f"{p}norm1.bias": torch.randn(dim),
            f"{p}norm2.weight": torch.randn(dim),
            f"{p}norm2.bias": torch.randn(dim),
        }
    return sd


def test_convert_hf_state_dict_fuses_qkv_and_masks_k_bias():
    dim = 4
    hf_sd = _fake_hf_state_dict(dim=dim, n_layers=2)
    out = _convert_hf_transformers_state_dict(hf_sd)

    qkv_w = out["blocks.0.attn.qkv.weight"]
    assert qkv_w.shape == (3 * dim, dim)
    torch.testing.assert_close(qkv_w[:dim], hf_sd["layer.0.attention.q_proj.weight"])
    torch.testing.assert_close(qkv_w[dim : 2 * dim], hf_sd["layer.0.attention.k_proj.weight"])
    torch.testing.assert_close(qkv_w[2 * dim :], hf_sd["layer.0.attention.v_proj.weight"])

    # HF ships no k bias (the hub arch zero-masks it): the k segment of the
    # fused bias must be zero and bias_mask must zero exactly that segment.
    qkv_b = out["blocks.0.attn.qkv.bias"]
    torch.testing.assert_close(qkv_b[:dim], hf_sd["layer.0.attention.q_proj.bias"])
    assert (qkv_b[dim : 2 * dim] == 0).all()
    torch.testing.assert_close(qkv_b[2 * dim :], hf_sd["layer.0.attention.v_proj.bias"])
    mask = out["blocks.0.attn.qkv.bias_mask"]
    assert (mask[:dim] == 1).all() and (mask[dim : 2 * dim] == 0).all()
    assert (mask[2 * dim :] == 1).all()


def test_convert_hf_state_dict_renames_and_reshapes():
    hf_sd = _fake_hf_state_dict()
    out = _convert_hf_transformers_state_dict(hf_sd)
    assert out["mask_token"].shape == (1, 4)  # (1, 1, D) -> (1, D)
    torch.testing.assert_close(out["storage_tokens"], hf_sd["embeddings.register_tokens"])
    torch.testing.assert_close(out["blocks.1.ls1.gamma"], hf_sd["layer.1.layer_scale1.lambda1"])
    torch.testing.assert_close(
        out["blocks.1.mlp.fc2.weight"], hf_sd["layer.1.mlp.down_proj.weight"]
    )
    # every hub key belongs to the expected namespace
    prefixes = ("cls_token", "storage_tokens", "mask_token", "patch_embed.", "norm.", "blocks.")
    assert all(k.startswith(prefixes) for k in out)


def test_convert_hf_state_dict_rejects_unknown_keys():
    hf_sd = _fake_hf_state_dict()
    hf_sd["embeddings.some_new_thing"] = torch.randn(1)
    with pytest.raises(RuntimeError, match="some_new_thing"):
        _convert_hf_transformers_state_dict(hf_sd)


def test_sat_variant_builds_arch_from_hash_suffixed_weights_name():
    # The hub ViT-L entrypoint derives untie_global_and_local_cls_norm from the
    # -eadcf0ff hash in the weights filename; the sat variant must pass it.
    kwargs = _VARIANTS["dinov3_vitl16_sat"].hub_build_kwargs
    assert kwargs["weights"].endswith("-eadcf0ff.pth")
    assert "sat493m" in kwargs["weights"]


@pytest.mark.parametrize("variant_name", list(_VARIANTS))
def test_missing_ckpt_path_raises_instead_of_falling_through(variant_name, monkeypatch, tmp_path):
    # A set-but-invalid ckpt env var must fail loudly, not fall through to a
    # download that could resolve to the wrong weights.
    missing = tmp_path / "no_such_checkpoint.pth"
    monkeypatch.setenv(_VARIANTS[variant_name].ckpt_env, str(missing))
    with pytest.raises(FileNotFoundError, match=_VARIANTS[variant_name].ckpt_env):
        _load_hub_model(variant_name)
