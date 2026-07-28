"""DINOv3 backbones with auto-downloaded pretrained weights.

TerraTorch's built-in ``dinov3_*`` backbones require a local ``ckpt_path`` and
load ``pretrained=False`` when none is given. This module registers
``dinov3_*_pretrained`` backbones that resolve pretrained weights at first run,
trying in order:

1. A local checkpoint pointed at by the variant's env var (see ``_VARIANTS``),
   for weights downloaded manually from Meta's page
2. ``torch.hub.load(..., pretrained=True)`` — only for variants whose weights
   ARE the hub default (the sat variants share an architecture with the web
   checkpoints, so the hub default would silently load the wrong weights)
3. HuggingFace Hub (``huggingface_hub.hf_hub_download``) — requires ``HF_TOKEN``
   in the environment for gated repos
4. A clear error naming the variant's env var and the Meta download page.

Registration is triggered by ``import gelos.backbones.dinov3_backbone`` in
``gelos/generation.py``, which runs before any YAML config is parsed.

Configs must list S2 bands in RED, GREEN, BLUE order — the pretrained patch
embed expects RGB channel order. ``gelos.normalization`` injects the matching
clip-and-stretch normalization per model name (ImageNet stats for the web
LVD-1689M variants, SAT-493M stats for the satellite variants).
"""

from dataclasses import dataclass
import os

import numpy as np
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class _Variant:
    """Weight sources for one registered DINOv3 variant."""

    hub_model: str  # architecture name in the facebookresearch/dinov3 hub repo
    hf_repo: str  # gated HuggingFace repo carrying this variant's weights
    ckpt_env: str  # env var pointing at a manually downloaded checkpoint
    try_hub_pretrained: bool  # hub default weights ARE this variant's weights


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
    ),
}


def _load_hub_model(variant_name: str) -> nn.Module:
    """Load pretrained weights for a registered DINOv3 variant."""
    variant = _VARIANTS[variant_name]

    # 1. Explicit local checkpoint. A set-but-invalid path raises instead of
    # falling through, which could silently land on the wrong weights.
    ckpt_path = os.environ.get(variant.ckpt_env)
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"{variant.ckpt_env}={ckpt_path} does not exist (note: inside a "
                "container this must be the in-container path). Fix the path or "
                "unset the variable to fall back to auto-download."
            )
        return torch.hub.load("facebookresearch/dinov3", variant.hub_model, weights=ckpt_path)

    # 2. Try torch.hub with pretrained=True, only where the hub default IS this
    # variant (works if the hub repo serves weights publicly)
    if variant.try_hub_pretrained:
        try:
            return torch.hub.load("facebookresearch/dinov3", variant.hub_model, pretrained=True)
        except Exception:
            pass

    # 3. Try HuggingFace Hub (requires HF_TOKEN env var for gated repos). The
    # checkpoint filename carries a hash suffix, so we pick the first .pth file
    # in the repo rather than hardcoding the full name.
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        filename = next(f for f in list_repo_files(variant.hf_repo) if f.endswith(".pth"))
        local_path = hf_hub_download(repo_id=variant.hf_repo, filename=filename)
        return torch.hub.load("facebookresearch/dinov3", variant.hub_model, weights=local_path)
    except Exception:
        pass

    raise RuntimeError(
        f"Could not auto-download pretrained DINOv3 weights for '{variant_name}' "
        f"(HuggingFace repo {variant.hf_repo}). Options:\n"
        "  1. Set HF_TOKEN in your environment and ensure your account has "
        "approved access to the gated HuggingFace repo.\n"
        "  2. Download manually from "
        "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ "
        f"and set {variant.ckpt_env} to the checkpoint path."
    )


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
