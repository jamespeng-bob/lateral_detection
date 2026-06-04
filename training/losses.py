"""Binary segmentation losses for lateral_detection.

The lateral foreground is roughly 0.1–0.8% of pixels per full-image, and
1–5% per positive-centered training tile. Plain BCE collapses to "predict
zero everywhere"; we always pair it with Dice, which is class-imbalance
robust by construction.

Recipes exposed via ``build_loss``:

- ``bce_dice``    →  :class:`BCEDiceLoss`     — ``BCE(pos_weight) + w*Dice``
- ``focal_dice``  →  :class:`FocalDiceLoss`   — ``Focal(α, γ) + w*Dice``
- ``composite``   →  :class:`CompositeLoss`   — ``BCEDice [+ w_cl*clDice] [+ w_lv*Lovász]``
                                                with optional per-component linear warmup.

All return a ``dict`` containing at least ``{"total": ..., "dice": ...}``
so trainers can log component losses and the trainer's existing
``losses["dice"]`` aggregation keeps working unchanged.

Auxiliary losses (clDice, Lovász) are intended to be USED VIA CompositeLoss,
not standalone — they don't carry the BCE/Dice supervision the segmenter
needs to converge from scratch.
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


# ---------------------------------------------------------------------------
# clDice — soft, differentiable centerline Dice
# ---------------------------------------------------------------------------
#
# Reference: Shit et al., "clDice — a Novel Topology-Preserving Loss for
# Tubular Structures", CVPR 2021. https://github.com/jocpae/clDice
#
# The soft skeleton is obtained by iterated morphological opening: each
# iteration peels off one layer of border pixels (soft_erode), then re-
# opens to remove non-skeleton parts, and accumulates the pixels that
# would be removed by re-opening (those are the skeleton pixels at that
# erosion depth). 3 iterations is enough for our 4 px-wide lines.


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Min-pool 3x3 (3x1 then 1x3 = approximate erosion). Differentiable."""
    p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Max-pool 3x3. Differentiable."""
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int = 3) -> torch.Tensor:
    """Differentiable approximation of the morphological skeleton."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        # Union of skel and delta, but kept differentiable. relu(delta - skel*delta)
        # = delta where skel=0, ≈0 where skel=1.
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cl_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    iter_: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """1 - clDice over the batch. Inputs: ``logits`` raw, ``target ∈ [0,1]``."""
    prob = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    skel_pred = soft_skel(prob,   iter_=iter_)
    skel_targ = soft_skel(target, iter_=iter_)
    tprec = (torch.sum(skel_pred * target) + smooth) / (torch.sum(skel_pred) + smooth)
    tsens = (torch.sum(skel_targ * prob)   + smooth) / (torch.sum(skel_targ) + smooth)
    cl_dice = 2.0 * tprec * tsens / (tprec + tsens + smooth)
    return 1.0 - cl_dice


class ClDiceLoss(nn.Module):
    def __init__(self, iter_: int = 3, smooth: float = 1.0) -> None:
        super().__init__()
        self.iter = int(iter_)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cl_dice_loss(logits, target, self.iter, self.smooth)


# ---------------------------------------------------------------------------
# Lovász hinge (binary)
# ---------------------------------------------------------------------------
#
# Reference: Berman et al., "The Lovász-Softmax loss", CVPR 2018.
# Canonical impl: https://github.com/bermanmaxim/LovaszSoftmax
#
# Lovász hinge is a direct (smooth) surrogate for 1 - IoU. We use the
# per-image variant: compute one Lovász loss per sample, then average,
# which works better than pooling all pixels into one big sort for
# class-imbalanced data like ours.


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Discrete gradient of the Lovász extension of the Jaccard loss."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    p = len(gt_sorted)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[:-1].clone()
    return jaccard


def _lovasz_hinge_flat(logits_flat: torch.Tensor, labels_flat: torch.Tensor) -> torch.Tensor:
    if labels_flat.numel() == 0:
        return logits_flat.sum() * 0.0
    signs  = 2.0 * labels_flat.float() - 1.0
    errors = 1.0 - logits_flat * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels_flat[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(
    logits: torch.Tensor,
    target: torch.Tensor,
    per_image: bool = True,
) -> torch.Tensor:
    """Binary Lovász hinge. ``logits``/``target`` shape: ``(B, 1, H, W)``."""
    if per_image:
        losses = [
            _lovasz_hinge_flat(log.reshape(-1), lab.reshape(-1).long())
            for log, lab in zip(logits, target)
        ]
        return torch.stack(losses).mean()
    return _lovasz_hinge_flat(logits.reshape(-1), target.reshape(-1).long())


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image: bool = True) -> None:
        super().__init__()
        self.per_image = bool(per_image)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return lovasz_hinge(logits, target, per_image=self.per_image)


# ---------------------------------------------------------------------------
# Composite (BCEDice + optional clDice + optional Lovász, with per-aux warmup)
# ---------------------------------------------------------------------------


class CompositeLoss(nn.Module):
    """Sum of BCEDice + optionally weighted clDice + optionally weighted Lovász.

    Auxiliary losses can be linearly ramped from 0 to their target weight
    over ``warmup_epochs`` to avoid early-training instability (when the
    model's predictions are nearly uniform, skeletonization and Lovász
    sorts give noisy gradients that can derail optimization).

    Set ``cldice_weight=0`` or ``lovasz_weight=0`` to disable a component.
    A loss with all aux weights = 0 is functionally equivalent to BCEDice.

    Expects the trainer to call ``set_epoch(epoch)`` once per epoch before
    train_epoch — otherwise weights stay at full strength (epoch=1, warmup
    behavior is to ramp UP TO and INCLUDING ``warmup_epochs``, so by epoch
    ``warmup_epochs`` the aux is at its target weight).
    """

    def __init__(
        self,
        bce_pos_weight: float = 1.0,
        dice_weight:    float = 1.0,
        cldice_weight:  float = 0.0,
        cldice_warmup:  int   = 0,
        cldice_iter:    int   = 3,
        lovasz_weight:  float = 0.0,
        lovasz_warmup:  int   = 0,
    ) -> None:
        super().__init__()
        self.bce_dice = BCEDiceLoss(pos_weight=bce_pos_weight, dice_weight=dice_weight)
        self.cldice         = ClDiceLoss(iter_=cldice_iter) if cldice_weight > 0 else None
        self.cldice_weight  = float(cldice_weight)
        self.cldice_warmup  = int(cldice_warmup)
        self.lovasz         = LovaszHingeLoss() if lovasz_weight > 0 else None
        self.lovasz_weight  = float(lovasz_weight)
        self.lovasz_warmup  = int(lovasz_warmup)
        self.current_epoch  = 1

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _eff(self, weight: float, warmup: int) -> float:
        """Linear ramp 0 → ``weight`` over ``warmup`` epochs (1-indexed)."""
        if warmup <= 0 or self.current_epoch >= warmup:
            return weight
        return weight * (self.current_epoch / warmup)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bd = self.bce_dice(logits, target)
        out: dict[str, torch.Tensor] = {
            "bce":   bd["bce"],
            "dice":  bd["dice"],
            "total": bd["total"],
        }
        if self.cldice is not None:
            cd = self.cldice(logits, target)
            out["cldice"] = cd
            out["total"]  = out["total"] + self._eff(self.cldice_weight, self.cldice_warmup) * cd
        if self.lovasz is not None:
            lv = self.lovasz(logits, target)
            out["lovasz"] = lv
            out["total"]  = out["total"] + self._eff(self.lovasz_weight, self.lovasz_warmup) * lv
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


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
    if name == "composite":
        return CompositeLoss(
            bce_pos_weight=cfg.get("bce_pos_weight", 1.0),
            dice_weight=cfg.get("dice_weight",       1.0),
            cldice_weight=cfg.get("cldice_weight",   0.0),
            cldice_warmup=cfg.get("cldice_warmup",   0),
            cldice_iter=cfg.get("cldice_iter",       3),
            lovasz_weight=cfg.get("lovasz_weight",   0.0),
            lovasz_warmup=cfg.get("lovasz_warmup",   0),
        )
    raise ValueError(f"Unknown loss name: {name!r}")
