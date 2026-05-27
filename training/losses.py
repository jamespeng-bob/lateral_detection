"""Binary segmentation losses for lateral_detection.

The lateral foreground is roughly 0.1–0.8% of pixels per full-image, and
1–5% per positive-centered training tile. Plain BCE collapses to "predict
zero everywhere"; we always pair it with Dice, which is class-imbalance
robust by construction.

Two recipes are exposed:

- :class:`BCEDiceLoss`  — ``BCE(pos_weight) + dice_weight * Dice``.
- :class:`FocalDiceLoss` — ``Focal(α, γ) + dice_weight * Dice``.

Both return a ``dict`` so trainers can log component losses separately.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss on ``sigmoid(logits)`` vs ``target ∈ [0, 1]``."""
    prob = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    inter = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = target * prob + (1.0 - target) * (1.0 - prob)
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
    return (alpha_t * (1.0 - pt) ** gamma * bce).mean()


class BCEDiceLoss(nn.Module):
    """``BCE(pos_weight) + dice_weight * Dice``. Returns dict for logging."""

    def __init__(self, pos_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.dice_weight = float(dice_weight)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        dl = dice_loss(logits, target)
        total = bce + self.dice_weight * dl
        return {"total": total, "bce": bce, "dice": dl}


class FocalDiceLoss(nn.Module):
    """``Focal(α, γ) + dice_weight * Dice``. Returns dict for logging."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        dice_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.dice_weight = float(dice_weight)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fl = focal_loss(logits, target, alpha=self.alpha, gamma=self.gamma)
        dl = dice_loss(logits, target)
        total = fl + self.dice_weight * dl
        return {"total": total, "focal": fl, "dice": dl}


def build_loss(cfg: dict) -> nn.Module:
    """Construct a loss from the ``loss:`` config block."""
    name = cfg.get("name", "bce_dice")
    if name == "bce_dice":
        return BCEDiceLoss(
            pos_weight=cfg.get("bce_pos_weight", 1.0),
            dice_weight=cfg.get("dice_weight", 1.0),
        )
    if name == "focal_dice":
        return FocalDiceLoss(
            alpha=cfg.get("focal_alpha", 0.25),
            gamma=cfg.get("focal_gamma", 2.0),
            dice_weight=cfg.get("dice_weight", 1.0),
        )
    raise ValueError(f"Unknown loss name: {name!r}")
