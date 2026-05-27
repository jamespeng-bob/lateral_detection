"""U-Net architectures for lateral_detection.

Two options:

- :class:`SMPUnet` — thin wrapper around ``segmentation_models_pytorch.Unet``
  with a 1-channel output. The baseline.
- :class:`DeepUnet` — two-stream variant: a low-resolution global branch
  whose coarse prediction is concatenated as a fourth channel back onto the
  full-resolution input, then refined by a second UNet. Reserved for v2
  experiments where per-tile context proves insufficient.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class SMPUnet(nn.Module):
    """Single-stream UNet from ``segmentation_models_pytorch``."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepUnet(nn.Module):
    """Two-stream UNet with a low-resolution global branch."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        self.ds_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        )
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=4,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        x_ds = F.interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=False)
        coarse = self.ds_unet(x_ds)
        coarse = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
        return self.model(torch.cat([x, coarse], dim=1))


def build_model(cfg: dict) -> nn.Module:
    """Construct a model from the ``model:`` config block."""
    name = cfg.get("name", "smp_unet")
    encoder = cfg.get("encoder", "resnet34")
    weights = cfg.get("encoder_weights", "imagenet")
    if weights in ("null", "none", None, ""):
        weights = None
    if name == "smp_unet":
        return SMPUnet(encoder_name=encoder, encoder_weights=weights)
    if name == "deep_unet":
        return DeepUnet(encoder_name=encoder, encoder_weights=weights)
    raise ValueError(f"Unknown model name: {name!r}")
