"""DINOv3 backbones with auto-downloaded pretrained weights.

TerraTorch's built-in ``dinov3_*`` backbones require a local ``ckpt_path`` and
load ``pretrained=False`` when none is given. This module registers
``dinov3_*_pretrained`` backbones that resolve pretrained weights at first run,
trying in order:

1. A local checkpoint pointed at by the variant's env var (see ``_VARIANTS``),
   for weights downloaded manually from Meta's page
2. ``torch.hub.load(..., pretrained=True)`` — only for variants whose weights
   ARE the hub default (the sat variants share an architecture with the web
   checkpoints, so the hub default would silently load the wrong weights).
   Meta's CDN currently answers 403 without a signed URL, so in practice this
   step falls through and step 3 does the work.
3. HuggingFace Hub — requires ``HF_TOKEN`` in the environment with approved
   access to the gated ``facebook/dinov3-*`` repos. Those repos ship
   transformers-format ``model.safetensors`` (split q/k/v, ``layer.N.*``
   naming), not hub ``.pth`` checkpoints, so the download is converted to the
   hub layout by ``_convert_hf_transformers_state_dict``.
4. A clear error naming the variant's env var and the Meta download page.

Registration is triggered by ``import gelos.backbones.dinov3_backbone`` in
``gelos/generation.py``, which runs before any YAML config is parsed.

Configs must list S2 bands in RED, GREEN, BLUE order — the pretrained patch
embed expects RGB channel order. ``gelos.normalization`` injects the matching
clip-and-stretch normalization per model name (ImageNet stats for the web
LVD-1689M variants, SAT-493M stats for the satellite variants).
"""

from dataclasses import dataclass, field
import os

import numpy as np
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
import torch
from torch import Tensor, nn

_HUB_REPO = "facebookresearch/dinov3"


@dataclass(frozen=True)
class _Variant:
    """Weight sources for one registered DINOv3 variant."""

    hub_model: str  # architecture name in the facebookresearch/dinov3 hub repo
    hf_repo: str  # gated HuggingFace repo carrying this variant's weights
    ckpt_env: str  # env var pointing at a manually downloaded checkpoint
    try_hub_pretrained: bool  # hub default weights ARE this variant's weights
    # Extra kwargs when building the arch WITHOUT downloading weights. The hub
    # entrypoints derive arch flags from the weights filename hash (e.g.
    # -eadcf0ff flips untie_global_and_local_cls_norm for the sat ViT-L), so
    # passing the canonical filename with pretrained=False builds the right
    # arch without touching the network.
    hub_build_kwargs: dict = field(default_factory=dict)


# The sat/web variants share hub architectures and differ only in weights, so
# each registered name pins its own weight sources; ``try_hub_pretrained`` must
# stay False wherever the hub default resolves to a different checkpoint.
_VARIANTS: dict[str, _Variant] = {
    "dinov3_vitb16": _Variant(
        hub_model="dinov3_vitb16",
        hf_repo="facebook/dinov3-vitb16-pretrain-lvd1689m",
        ckpt_env="DINOV3_VITB16_CKPT",
        try_hub_pretrained=True,
    ),
    "dinov3_vitl16_sat": _Variant(
        hub_model="dinov3_vitl16",
        hf_repo="facebook/dinov3-vitl16-pretrain-sat493m",
        ckpt_env="DINOV3_VITL16_SAT_CKPT",
        try_hub_pretrained=False,  # hub default = web LVD-1689M weights
        hub_build_kwargs={"weights": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"},
    ),
}

# Buffers/params legitimately absent from the HF transformers export:
# rope_embed.periods is a deterministic buffer recomputed at arch init, and
# local_cls_norm.* (sat ViT-L) is only used for local crops during training.
_ALLOWED_MISSING_KEYS = {"rope_embed.periods", "local_cls_norm.weight", "local_cls_norm.bias"}


def _convert_hf_transformers_state_dict(hf_sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Convert an HF transformers DINOv3 state dict to the torch.hub layout.

    The HF repos ship split q/k/v projections under ``layer.N.attention.*``;
    the hub arch fuses them into ``blocks.N.attn.qkv``. The k bias is absent
    from the HF export because the hub arch zero-masks it (LinearKMaskedBias),
    so the zero k-bias segment and the matching ``bias_mask`` buffer are
    synthesized here. Raises if any source key goes unconsumed, so a layout
    change in the HF export fails loudly instead of loading garbage.
    """
    consumed = set()

    def take(key: str) -> Tensor:
        consumed.add(key)
        return hf_sd[key]

    out = {
        "cls_token": take("embeddings.cls_token"),
        "storage_tokens": take("embeddings.register_tokens"),
        # HF stores the mask token as (1, 1, D); the hub arch wants (1, D).
        "mask_token": take("embeddings.mask_token").reshape(1, -1),
        "patch_embed.proj.weight": take("embeddings.patch_embeddings.weight"),
        "patch_embed.proj.bias": take("embeddings.patch_embeddings.bias"),
        "norm.weight": take("norm.weight"),
        "norm.bias": take("norm.bias"),
    }
    n_layers = 1 + max(int(k.split(".")[1]) for k in hf_sd if k.startswith("layer."))
    for i in range(n_layers):
        hf, hub = f"layer.{i}.", f"blocks.{i}."
        q_bias = take(f"{hf}attention.q_proj.bias")
        zeros, ones = torch.zeros_like(q_bias), torch.ones_like(q_bias)
        out[f"{hub}attn.qkv.weight"] = torch.cat(
            [take(f"{hf}attention.{p}_proj.weight") for p in ("q", "k", "v")], dim=0
        )
        out[f"{hub}attn.qkv.bias"] = torch.cat(
            [q_bias, zeros, take(f"{hf}attention.v_proj.bias")], dim=0
        )
        out[f"{hub}attn.qkv.bias_mask"] = torch.cat([ones, zeros, ones], dim=0)
        out[f"{hub}attn.proj.weight"] = take(f"{hf}attention.o_proj.weight")
        out[f"{hub}attn.proj.bias"] = take(f"{hf}attention.o_proj.bias")
        out[f"{hub}ls1.gamma"] = take(f"{hf}layer_scale1.lambda1")
        out[f"{hub}ls2.gamma"] = take(f"{hf}layer_scale2.lambda1")
        out[f"{hub}mlp.fc1.weight"] = take(f"{hf}mlp.up_proj.weight")
        out[f"{hub}mlp.fc1.bias"] = take(f"{hf}mlp.up_proj.bias")
        out[f"{hub}mlp.fc2.weight"] = take(f"{hf}mlp.down_proj.weight")
        out[f"{hub}mlp.fc2.bias"] = take(f"{hf}mlp.down_proj.bias")
        for norm in ("norm1", "norm2"):
            for kind in ("weight", "bias"):
                out[f"{hub}{norm}.{kind}"] = take(f"{hf}{norm}.{kind}")
    leftover = set(hf_sd) - consumed
    if leftover:
        raise RuntimeError(
            f"unrecognized keys in HF DINOv3 state dict (layout changed?): {sorted(leftover)}"
        )
    return out


def _load_from_hf(variant: _Variant) -> nn.Module:
    """Download a variant's HF safetensors and load them into the hub arch."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    local_path = hf_hub_download(repo_id=variant.hf_repo, filename="model.safetensors")
    converted = _convert_hf_transformers_state_dict(load_file(local_path))
    model = torch.hub.load(
        _HUB_REPO, variant.hub_model, pretrained=False, **variant.hub_build_kwargs
    )
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if unexpected or not set(missing) <= _ALLOWED_MISSING_KEYS:
        raise RuntimeError(
            f"converted HF state dict does not match the hub arch for "
            f"'{variant.hub_model}': missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return model


def _load_hub_model(variant_name: str) -> nn.Module:
    """Load pretrained weights for a registered DINOv3 variant."""
    variant = _VARIANTS[variant_name]

    # 1. Explicit local checkpoint. A set-but-invalid path raises instead of
    # falling through, which could silently land on the wrong weights. The
    # file must keep its original hash-suffixed name (…-eadcf0ff.pth): the hub
    # ViT-L entrypoint derives arch flags from that hash.
    ckpt_path = os.environ.get(variant.ckpt_env)
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"{variant.ckpt_env}={ckpt_path} does not exist (note: inside a "
                "container this must be the in-container path). Fix the path or "
                "unset the variable to fall back to auto-download."
            )
        return torch.hub.load(_HUB_REPO, variant.hub_model, weights=ckpt_path)

    # 2. Try torch.hub with pretrained=True, only where the hub default IS this
    # variant (works only if Meta's CDN ever serves the weights publicly)
    if variant.try_hub_pretrained:
        try:
            return torch.hub.load(_HUB_REPO, variant.hub_model, pretrained=True)
        except Exception:
            pass

    # 3. Try HuggingFace Hub (requires HF_TOKEN env var for the gated repos).
    try:
        return _load_from_hf(variant)
    except Exception as hf_error:
        raise RuntimeError(
            f"Could not auto-download pretrained DINOv3 weights for '{variant_name}' "
            f"(HuggingFace repo {variant.hf_repo}). Options:\n"
            "  1. Set HF_TOKEN in your environment and ensure your account has "
            "approved access to the gated HuggingFace repo.\n"
            "  2. Download manually from "
            "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ "
            f"and set {variant.ckpt_env} to the checkpoint path."
        ) from hf_error


class DinoV3PretrainedWrapper(nn.Module):
    """Mirror of TerraTorch's ``DinoV3Wrapper`` with auto-download of pretrained weights.

    ``forward`` returns a list of embeddings for all intermediate layer blocks.
    When ``return_cls_token=True`` the CLS token is appended as the LAST token
    in each sequence (matching TerraTorch's wrapper code, whose docstring
    incorrectly says "in front").
    """

    def __init__(self, model: str, return_cls_token: bool = True):
        super().__init__()
        self.dinov3 = _load_hub_model(model)
        self.out_channels = (
            [self.dinov3.embed_dim] * len(self.dinov3.blocks)
            if hasattr(self.dinov3, "blocks")
            else self.dinov3.embed_dims
        )
        self.output_indexes = (
            list(np.arange(len(self.dinov3.blocks))) if hasattr(self.dinov3, "blocks") else 4
        )
        self.return_cls_token = return_cls_token

    def forward(self, x: Tensor):
        feats = self.dinov3.get_intermediate_layers(
            x, n=self.output_indexes, return_class_token=self.return_cls_token
        )
        if self.return_cls_token:
            feats = [torch.cat([f[0], f[1].unsqueeze(1)], dim=1) for f in feats]
        return list(feats)


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vitb16_pretrained(return_cls_token: bool = True, **kwargs) -> DinoV3PretrainedWrapper:
    """DINOv3 ViT-B/16 (web LVD-1689M) with auto-downloaded pretrained weights."""
    return DinoV3PretrainedWrapper(model="dinov3_vitb16", return_cls_token=return_cls_token)


@TERRATORCH_BACKBONE_REGISTRY.register
def dinov3_vitl16_sat_pretrained(
    return_cls_token: bool = True, **kwargs
) -> DinoV3PretrainedWrapper:
    """DINOv3 ViT-L/16 (satellite SAT-493M) with auto-downloaded pretrained weights."""
    return DinoV3PretrainedWrapper(model="dinov3_vitl16_sat", return_cls_token=return_cls_token)
