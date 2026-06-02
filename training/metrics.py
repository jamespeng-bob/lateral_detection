"""Validation-time metric accumulators for lateral_detection.

All metrics here follow a ``reset / update / compute`` lifecycle so the
trainer can drive them uniformly. Every accumulator's ``compute()`` is
DDP-safe — when ``torch.distributed`` is initialized, it ``all_reduce``s
its raw counts (or ``all_gather``s its per-sample lists) across ranks
before returning, so the reported number is the *global* metric across
the whole val set, not just the local rank's shard.

What's in this module:

- :class:`BinaryDiceAccumulator` — sum-based Dice over the whole val set.
  Equivalent to the legacy per-batch ``_dice_metric`` in trainer.py but
  more correct numerically (one sum / one divide vs many averages).

- :class:`BinaryIoUAccumulator` — sum-based IoU over the whole val set.
  Same numeric improvement as Dice.

- :class:`BinaryClDiceAccumulator` — centerline Dice (clDice). Width-
  invariant: a 4 px and 12 px prediction at the same centerline give
  the same clDice. Tells us "is the line in the right *place*"
  decoupled from "is the line the right *width*".

- :class:`LengthRatioAccumulator` — per-sample ``pred_fg / gt_fg`` ratios
  in two flavours (raw pixel count, width-invariant skeleton count).
  Reports mean / median / p25 / p75 of the distribution across val tiles.

Notes on DDP semantics:

- BinaryDice / BinaryIoU / BinaryClDice keep four scalar float counters
  each; we ``all_reduce(SUM)`` them at compute time.
- LengthRatio keeps a list of per-sample ratios; we ``all_gather_object``
  the lists at compute time so the distribution stats are over the
  whole val set.
- When ``torch.distributed`` is not initialized (single-GPU runs), all
  reductions short-circuit and return local values — backward-compatible.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.distributed as dist
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum(value: float) -> float:
    """Sum ``value`` across all ranks (no-op in single-GPU mode)."""
    if not _ddp_active():
        return float(value)
    # Place the scalar on the current CUDA device so NCCL can reduce it.
    device = torch.device("cuda", torch.cuda.current_device())
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _all_gather_list(local_list: list) -> list:
    """Concatenate per-rank Python lists (no-op in single-GPU mode)."""
    if not _ddp_active():
        return list(local_list)
    world = dist.get_world_size()
    gathered: List[Any] = [None] * world
    dist.all_gather_object(gathered, list(local_list))
    out: list = []
    for sub in gathered:
        if sub:
            out.extend(sub)
    return out


# ---------------------------------------------------------------------------
# Binary Dice (sum-based, global over val set)
# ---------------------------------------------------------------------------


class BinaryDiceAccumulator:
    """``Dice = 2 * sum_inter / (sum_pred + sum_target)`` over the val set.

    Predictions are thresholded at ``threshold`` (default 0.5) before the
    intersection / sum are computed. DDP-safe via SUM all-reduce.
    """

    higher_is_better = True

    def __init__(self, threshold: float = 0.5, eps: float = 1e-6) -> None:
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.inter = 0.0
        self.denom = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = (torch.sigmoid(logits) >= self.threshold).float()
        tgt  = (target >= 0.5).float()
        self.inter += float((prob * tgt).sum().item())
        self.denom += float((prob.sum() + tgt.sum()).item())

    def compute(self) -> float:
        inter = _all_reduce_sum(self.inter)
        denom = _all_reduce_sum(self.denom)
        return float((2.0 * inter + self.eps) / (denom + self.eps))


# ---------------------------------------------------------------------------
# Binary IoU (sum-based, global over val set)
# ---------------------------------------------------------------------------


class BinaryIoUAccumulator:
    """``IoU = sum_inter / (sum_pred + sum_target - sum_inter)`` over val set."""

    higher_is_better = True

    def __init__(self, threshold: float = 0.5, eps: float = 1e-6) -> None:
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.inter = 0.0
        self.sum_pred = 0.0
        self.sum_target = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = (torch.sigmoid(logits) >= self.threshold).float()
        tgt  = (target >= 0.5).float()
        self.inter      += float((prob * tgt).sum().item())
        self.sum_pred   += float(prob.sum().item())
        self.sum_target += float(tgt.sum().item())

    def compute(self) -> float:
        inter      = _all_reduce_sum(self.inter)
        sum_pred   = _all_reduce_sum(self.sum_pred)
        sum_target = _all_reduce_sum(self.sum_target)
        union = sum_pred + sum_target - inter
        return float((inter + self.eps) / (union + self.eps))


# ---------------------------------------------------------------------------
# Centerline Dice (clDice)
# ---------------------------------------------------------------------------


class BinaryClDiceAccumulator:
    """clDice = 2 * tprec * tsens / (tprec + tsens), aggregated globally.

    tprec = |skel(pred)  ∩ target| / |skel(pred)|       (how much of the
                                                        predicted skeleton
                                                        lands on real GT)
    tsens = |skel(target) ∩ pred|  / |skel(target)|    (how much of the GT
                                                        skeleton is covered
                                                        by the prediction)

    Skeletonization happens per-sample on CPU with ``skimage`` (~50 ms per
    1024² tile). Acceptable at validation cadence (≪ training step time).
    Raw counts are all-reduced under DDP so the reported value is the
    global metric over the whole val set.
    """

    higher_is_better = True

    def __init__(self, threshold: float = 0.5, eps: float = 1e-6) -> None:
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.tprec_num = 0.0
        self.tprec_den = 0.0
        self.tsens_num = 0.0
        self.tsens_den = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = (torch.sigmoid(logits) >= self.threshold).detach().cpu().numpy()
        targ = (target >= 0.5).detach().cpu().numpy()
        if prob.ndim == 4:
            prob = prob.squeeze(1)
        if targ.ndim == 4:
            targ = targ.squeeze(1)

        for p, t in zip(prob.astype(bool), targ.astype(bool)):
            sp = skeletonize(p)
            st = skeletonize(t)
            self.tprec_num += float((sp & t).sum())
            self.tprec_den += float(sp.sum())
            self.tsens_num += float((st & p).sum())
            self.tsens_den += float(st.sum())

    def compute(self) -> float:
        tprec_num = _all_reduce_sum(self.tprec_num)
        tprec_den = _all_reduce_sum(self.tprec_den)
        tsens_num = _all_reduce_sum(self.tsens_num)
        tsens_den = _all_reduce_sum(self.tsens_den)
        tprec = tprec_num / (tprec_den + self.eps)
        tsens = tsens_num / (tsens_den + self.eps)
        return float(2.0 * tprec * tsens / (tprec + tsens + self.eps))


# ---------------------------------------------------------------------------
# Length ratio
# ---------------------------------------------------------------------------


class LengthRatioAccumulator:
    """Per-sample ratio of predicted foreground count to GT foreground count.

    Tracks two ratios in parallel:

    * ``pixel``    = #pred_fg_pixels / #gt_fg_pixels                — width-sensitive
    * ``skeleton`` = #skel(pred)_pixels / #skel(gt)_pixels         — width-invariant

    Each ratio is collected per sample (not pooled across the batch), so
    the final report exposes the *distribution* across val tiles (mean,
    median, p25, p75). A median far from 1.0 indicates systematic bias;
    a wide p25–p75 spread indicates the model is right on average but
    not reliably per-tile.

    Under DDP, per-rank lists are gathered to rank 0 (and everywhere via
    all_gather_object) before computing the distribution stats.
    """

    higher_is_better = None  # not unidirectional — "closer to 1.0" is good

    def __init__(self, threshold: float = 0.5, eps: float = 1e-6) -> None:
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.pixel_ratios: list[float] = []
        self.skel_ratios:  list[float] = []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = (torch.sigmoid(logits) >= self.threshold).detach().cpu().numpy()
        targ = (target >= 0.5).detach().cpu().numpy()
        if prob.ndim == 4:
            prob = prob.squeeze(1)
        if targ.ndim == 4:
            targ = targ.squeeze(1)

        for p, t in zip(prob.astype(bool), targ.astype(bool)):
            gt_pix = float(t.sum())
            if gt_pix < 1.0:
                continue
            pred_pix = float(p.sum())
            self.pixel_ratios.append(pred_pix / (gt_pix + self.eps))

            gt_skel = float(skeletonize(t).sum())
            if gt_skel < 1.0:
                continue
            pred_skel = float(skeletonize(p).sum())
            self.skel_ratios.append(pred_skel / (gt_skel + self.eps))

    def compute(self) -> dict[str, float]:
        # Concatenate per-rank lists into a single global distribution.
        pixel = _all_gather_list(self.pixel_ratios)
        skel  = _all_gather_list(self.skel_ratios)

        out: dict[str, float] = {}
        for name, ratios in (("pixel", pixel), ("skel", skel)):
            if not ratios:
                out[f"length_ratio_{name}_mean"]   = float("nan")
                out[f"length_ratio_{name}_median"] = float("nan")
                out[f"length_ratio_{name}_p25"]    = float("nan")
                out[f"length_ratio_{name}_p75"]    = float("nan")
                continue
            arr = np.asarray(ratios, dtype=np.float64)
            out[f"length_ratio_{name}_mean"]   = float(arr.mean())
            out[f"length_ratio_{name}_median"] = float(np.median(arr))
            out[f"length_ratio_{name}_p25"]    = float(np.percentile(arr, 25))
            out[f"length_ratio_{name}_p75"]    = float(np.percentile(arr, 75))
        return out
