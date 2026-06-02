"""Validation-time metric accumulators for lateral_detection.

Two metrics here, both deployment-aligned for the "estimate total lateral
length" task:

- :class:`BinaryClDiceAccumulator` — centerline Dice (clDice). Dice computed
  on the *skeletons* of prediction and target, not on the masks. This is
  width-invariant: a 4 px prediction and a 12 px prediction at the same
  centerline position give the same clDice. So clDice tells us "is the
  prediction in the right *place*" decoupled from "is the prediction the
  right *width*".

- :class:`LengthRatioAccumulator` — per-sample ratio of predicted
  foreground pixels to GT foreground pixels, in two flavors:

  * ``pixel``    — based on raw binary mask counts; sensitive to width.
  * ``skeleton`` — based on skeletonized counts; width-invariant, so a more
                   direct proxy for "did we recover the true line length?"

  Tracking both lets us decompose length error into a *width* component
  (pixel ratio drifts but skeleton ratio is fine) and a *coverage*
  component (both ratios drift together).

Both accumulators follow a ``reset / update / compute`` pattern that the
trainer drives directly (no torchmetrics dependency to keep deps light).
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# Centerline Dice (clDice)
# ---------------------------------------------------------------------------


class BinaryClDiceAccumulator:
    """clDice = 2 * tprec * tsens / (tprec + tsens), accumulated over a split.

    tprec = |skel(pred) ∩ target| / |skel(pred)|       (how much of the
                                                        predicted skeleton
                                                        lands on real GT)
    tsens = |skel(target) ∩ pred|  / |skel(target)|    (how much of the GT
                                                        skeleton is covered
                                                        by the prediction)

    We accumulate raw counts across the whole validation split, then form
    a single global clDice at ``compute`` time (rather than per-batch then
    averaging). Both formulations are common; the global form is more
    stable when batch foreground varies a lot.

    Skeletonization is done per-sample on CPU with ``skimage`` — that's
    the slow part, ~50 ms per 1024² sample. Acceptable at validation
    cadence (≪ training step time).
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
        """Accumulate counts from one batch.

        ``logits`` shape ``(B, 1, H, W)`` or ``(B, H, W)``.
        ``target`` is the GT mask in {0, 1} or {0, 255}, same shape.
        """
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
        tprec = self.tprec_num / (self.tprec_den + self.eps)
        tsens = self.tsens_num / (self.tsens_den + self.eps)
        return float(2.0 * tprec * tsens / (tprec + tsens + self.eps))


# ---------------------------------------------------------------------------
# Length ratio
# ---------------------------------------------------------------------------


class LengthRatioAccumulator:
    """Per-sample ratio of predicted foreground count to GT foreground count.

    Tracks two ratios in parallel:

    * ``pixel``    = #pred_fg_pixels / #gt_fg_pixels                — width-sensitive
    * ``skeleton`` = #skel(pred)_pixels / #skel(gt)_pixels         — width-invariant

    Each ratio is collected per sample (not pooled across the batch), so the
    final report exposes the *distribution* across val tiles (mean, median,
    p25, p75). A median far from 1.0 indicates systematic bias; a wide
    p25–p75 spread indicates that the model is right *on average* but not
    reliably per-tile.

    Samples with no GT foreground are skipped (division by zero); in
    ``pos_only_grid`` validation mode this never fires, but we defend.
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
        out = {}
        for name, ratios in (("pixel", self.pixel_ratios), ("skel", self.skel_ratios)):
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
