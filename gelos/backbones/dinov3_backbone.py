"""DINOv3 backbone with auto-downloaded pretrained weights.

TerraTorch's built-in ``dinov3_*`` backbones require a local ``ckpt_path`` and
load ``pretrained=False`` when none is given. This module registers a
``dinov3_vitb16_pretrained`` backbone that auto-downloads pretrained weights at
first run, trying in order:

1. ``torch.hub.load(..., pretrained=True)`` (works if the hub repo serves weights)
2. HuggingFace Hub (``huggingface_hub.hf_hub_download``) — requires ``HF_TOKEN``
   in the environment for gated repos
3. A clear error pointing at the Meta download page and the standard
   ``dinov3_vitb16`` backbone with an explicit ``ckpt_path``.

Registration is triggered by ``import gelos.backbones.dinov3_backbone`` in
``gelos/generation.py``, which runs before any YAML config is parsed.
"""

import numpy as np
import torch
from torch import Tensor, nn

from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

# HuggingFace Hub repo IDs for gated DINOv3 checkpoints.
# The checkpoint filename carries a hash suffix, so we pick the first .pth file
# in the repo rather than hardcoding the full name.
_HF_REPOS: dict[str, str] = {
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}


def _load_hub_model(model_name: str) -> nn.Module:
    """Try to load pretrained DINOv3 weights, falling back gracefully."""
    # 1. Try torch.hub with pretrained=True (works if hub repo serves weights publicly)
    try:
        return torch.hub.load("facebookresearch/dinov3", model_name, pretrained=True)
    except Exception:
        pass

    # 2. Try HuggingFace Hub (requires HF_TOKEN env var for gated repos)
    if model_name in _HF_REPOS:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            repo_id = _HF_REPOS[model_name]
            filename = next(f for f in list_repo_files(repo_id) if f.endswith(".pth"))
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
            return torch.hub.load(
                "facebookresearch/dinov3", model_name, weights=local_path
            )
        except Exception:
            pass

    raise RuntimeError(
        f"Could not auto-download pretrained DINOv3 weights for '{model_name}'. "
        "Options:\n"
        "  1. Set HF_TOKEN in your environment and ensure your account has "
        "approved access to the gated HuggingFace repo.\n"
        "  2. Download manually from "
        "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ "
        "and use the standard 'dinov3_vitb16' backbone with ckpt_path= in model_args."
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
            list(np.arange(len(self.dinov3.blocks)))
            if hasattr(self.dinov3, "blocks")
            else 4
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
    """DINOv3 ViT-B/16 with auto-downloaded pretrained weights."""
    return DinoV3PretrainedWrapper(model="dinov3_vitb16", return_cls_token=return_cls_token)
