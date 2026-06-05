"""Symbol-based filter for predicted lateral-line masks.

Given a model's predicted lateral mask + detected symbols (with their API
embeddings and bbox), this module categorizes each symbol as "irrigation",
"callout", or "neutral" via a pre-trained per-symbol classifier, then drops
predicted line components whose endpoints touch ONLY callout symbols and
no irrigation symbol — the conservative rule the user picked earlier.

Two classes:

- :class:`SymbolClassifier` wraps a joblib-pickled sklearn pipeline that
  predicts ``P(near_gt | embedding + bbox_features)`` per symbol. Built by
  ``scripts/train_symbol_classifier.py``.

- :class:`SymbolFilter` applies that classifier to a per-image batch of
  symbols, then walks the connected components of a dilated pred mask
  to make per-component keep/drop decisions.

This is purely inference-side — no model retraining required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# Feature engineering — must match what train_symbol_classifier.py uses
# ---------------------------------------------------------------------------


def _bbox_features(bboxes: np.ndarray) -> np.ndarray:
    """Convert ``(N, 4)`` ``(x1,y1,x2,y2)`` bboxes to ``(N, 5)`` features.

    Features (in order): width, height, log(aspect_ratio), log(area),
    log(width*height/max_image_dim). The log transforms tame the scale
    range (widths span 5-2000 px across our data).
    """
    bboxes = np.asarray(bboxes, dtype=np.float64)
    w = (bboxes[:, 2] - bboxes[:, 0]).clip(min=1.0)
    h = (bboxes[:, 3] - bboxes[:, 1]).clip(min=1.0)
    log_aspect = np.log(w / h)
    log_area   = np.log(w * h)
    # A coarse "absolute size" feature, normalized to ~[0,1] for typical sizes.
    log_w = np.log(w)
    log_h = np.log(h)
    return np.stack([w, h, log_aspect, log_area, log_w + log_h], axis=1).astype(np.float32)


def build_features(
    embeddings: np.ndarray,   # (N, 128) raw embeddings
    bboxes:     np.ndarray,   # (N, 4)   x1,y1,x2,y2
) -> np.ndarray:
    """Return (N, 128+5=133) feature matrix used for training + inference."""
    emb = np.asarray(embeddings, dtype=np.float32)
    # L2-normalize embeddings so cosine geometry is preserved through downstream
    # standard scaling. Matches what mine_symbol_categories.py does for clustering.
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    bbox_feats = _bbox_features(bboxes)
    return np.concatenate([emb, bbox_feats], axis=1)


# ---------------------------------------------------------------------------
# Classifier wrapper
# ---------------------------------------------------------------------------


class SymbolClassifier:
    """Wraps the trained sklearn Pipeline + records its training metadata.

    The pickled artifact at ``model_path`` is a dict::

        {
            "pipeline":   sklearn.pipeline.Pipeline,
            "n_features": 133,
            "auc_train":  float,
            "auc_val":    float,
            "trained_on": str,
            "feature_layout": "L2(emb_128) + [w, h, log_aspect, log_area, log_w+log_h]",
        }
    """

    def __init__(self, model_path: str | Path) -> None:
        bundle = joblib.load(str(model_path))
        self.pipeline      = bundle["pipeline"]
        self.n_features    = int(bundle["n_features"])
        self.auc_train     = float(bundle.get("auc_train", float("nan")))
        self.auc_val       = float(bundle.get("auc_val",   float("nan")))
        self.trained_on    = str(bundle.get("trained_on", ""))
        self.feature_layout = str(bundle.get("feature_layout", ""))

    @classmethod
    def predict_proba_batch(
        cls,
        pipeline,
        embeddings: np.ndarray,
        bboxes:     np.ndarray,
    ) -> np.ndarray:
        if len(embeddings) == 0:
            return np.zeros(0, dtype=np.float32)
        X = build_features(embeddings, bboxes)
        return pipeline.predict_proba(X)[:, 1].astype(np.float32)

    def predict_proba(self, embeddings: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        return self.predict_proba_batch(self.pipeline, embeddings, bboxes)


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


@dataclass
class ComponentDecision:
    component_id:        int
    n_pixels:            int
    n_endpoints:         int
    nearest_p_per_endpoint: list[float] = field(default_factory=list)
    fate:                str = "keep"   # 'keep' | 'drop'
    reason:              str = ""


@dataclass
class FilterReport:
    n_components_total:   int
    n_components_dropped: int
    pixels_dropped:       int
    decisions:            list[ComponentDecision]


class SymbolFilter:
    """Inference-time filter that drops callout-style FP line components.

    Decision rule (conservative, matching the user's earlier choice):

      For each connected component of (dilated) pred_mask:

        - Get its skeleton; find endpoints (degree-1 skeleton pixels).
          A component with 0 endpoints (closed loop) is always kept.
        - For each endpoint, find symbols whose bbox is within
          ``endpoint_radius`` px of it, and take the MAX classifier
          P(irrigation) over those nearby symbols. ``nan`` if no
          symbols within reach.
        - KEEP the component if any endpoint's max-P >= ``p_irr_thresh``.
        - Otherwise DROP if any endpoint's max-P <= ``p_call_thresh``
          (and no endpoint was irrigation).
        - Otherwise KEEP (default conservative; we err toward keeping).
    """

    def __init__(
        self,
        classifier: SymbolClassifier,
        *,
        p_irr_thresh:           float = 0.85,
        p_call_thresh:          float = 0.15,
        dilation_px:            int   = 3,
        endpoint_radius:        int   = 25,
        min_component_px:       int   = 0,
        min_drop_skel_px:       int   = 0,
    ) -> None:
        """
        Parameters
        ----------
        p_irr_thresh : float
            A symbol's classifier P >= this → "irrigation". An endpoint touching
            any such symbol KEEPS the component unconditionally.
        p_call_thresh : float
            A symbol's classifier P <= this → "non-irrigation" (callout etc.).
            By itself doesn't drop; the full drop rule also requires no
            irrigation endpoint AND skeleton length below ``min_drop_skel_px``.
        dilation_px : int
            Morphological dilation of the pred mask before component analysis,
            to close hairline gaps.
        endpoint_radius : int
            How close a symbol's bbox center must be to a skeleton endpoint
            to count as "touching" that endpoint.
        min_component_px : int
            Components smaller than this many AREA pixels are skipped entirely
            (no decision recorded). Useful for noise.
        min_drop_skel_px : int
            Components are PROTECTED from being dropped if their SKELETON
            (1-px-wide centerline) has at least this many pixels — i.e., the
            line is long enough to be a real lateral, not a callout fragment.
            Set to 0 to disable. Default 0 = no length-based protection.
        """
        self.classifier        = classifier
        self.p_irr_thresh      = float(p_irr_thresh)
        self.p_call_thresh     = float(p_call_thresh)
        self.dilation_px       = int(dilation_px)
        self.endpoint_radius   = int(endpoint_radius)
        self.min_component_px  = int(min_component_px)
        self.min_drop_skel_px  = int(min_drop_skel_px)

    # ------------------------------------------------------------------

    def _endpoints_from_skeleton(self, skel: np.ndarray) -> list[tuple[int, int]]:
        """Return list of (y, x) for skeleton pixels with degree == 1.

        Degree is the count of 8-neighbours that are also skeleton pixels.
        For very short components (no degree-1 pixel), returns the two
        extreme points of the skeleton along its principal axis.
        """
        if not skel.any():
            return []
        # Count 8-connected neighbors via 3x3 sum (subtract self).
        kernel = np.ones((3, 3), dtype=np.uint8)
        nb = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        nb = nb - skel.astype(np.uint8)  # subtract self
        endpoints_yx = list(zip(*np.where((skel > 0) & (nb == 1))))
        if endpoints_yx:
            return [(int(y), int(x)) for y, x in endpoints_yx]

        # Fallback for short / closed-loop-ish components: extreme points.
        ys, xs = np.where(skel > 0)
        if len(ys) == 0:
            return []
        if len(ys) <= 2:
            return [(int(y), int(x)) for y, x in zip(ys, xs)]
        # Use the two pixels farthest from the centroid as proxy endpoints.
        cy, cx = float(ys.mean()), float(xs.mean())
        d = (ys - cy) ** 2 + (xs - cx) ** 2
        i1 = int(np.argmax(d))
        # Second-farthest = farthest from the first.
        d2 = (ys - ys[i1]) ** 2 + (xs - xs[i1]) ** 2
        i2 = int(np.argmax(d2))
        return [(int(ys[i1]), int(xs[i1])), (int(ys[i2]), int(xs[i2]))]

    # ------------------------------------------------------------------

    def apply(
        self,
        pred_mask: np.ndarray,            # H x W, uint8 {0, 255} (or bool)
        symbols:   list[dict],            # each: x1,y1,x2,y2,embedding (list of 128)
    ) -> tuple[np.ndarray, FilterReport]:
        """Return (filtered_mask, report)."""
        binary = (pred_mask > 0).astype(np.uint8)
        H, W = binary.shape

        # Categorize all symbols once for this image.
        if symbols:
            embs  = np.asarray([s["embedding"] for s in symbols], dtype=np.float32)
            bboxes = np.asarray(
                [[s["x1"], s["y1"], s["x2"], s["y2"]] for s in symbols],
                dtype=np.float32,
            )
            sym_p   = self.classifier.predict_proba(embs, bboxes)
            sym_cx  = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            sym_cy  = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
        else:
            sym_p, sym_cx, sym_cy = (np.zeros(0, dtype=np.float32),) * 3

        # Dilate to close 1-2 px gaps before component analysis.
        if self.dilation_px > 0:
            k = 2 * self.dilation_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            dilated = cv2.dilate(binary, kernel)
        else:
            dilated = binary

        # Connected components on the DILATED mask so 1-px gaps don't fragment.
        n_lab, lab_img, stats, _ = cv2.connectedComponentsWithStats(
            dilated, connectivity=8, ltype=cv2.CV_32S,
        )

        decisions: list[ComponentDecision] = []
        drop_set: set[int] = set()

        # Skip label 0 (background).
        for cid in range(1, n_lab):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < max(1, self.min_component_px):
                continue
            # Use the ORIGINAL (undilated) skeleton of this component, so we
            # don't get phantom endpoints from the dilation.
            comp_mask = ((lab_img == cid) & (binary > 0)).astype(np.uint8)
            if not comp_mask.any():
                # Component is entirely from dilation infill — skip
                continue
            skel = skeletonize(comp_mask.astype(bool))
            endpoints = self._endpoints_from_skeleton(skel)

            decision = ComponentDecision(
                component_id=cid, n_pixels=area, n_endpoints=len(endpoints),
            )

            if not endpoints:
                decision.fate = "keep"; decision.reason = "no_endpoints"
                decisions.append(decision)
                continue

            # For each endpoint, find max P(irr) among nearby symbols.
            max_p_per_endpoint: list[float] = []
            for (ey, ex) in endpoints:
                if len(sym_p) == 0:
                    max_p_per_endpoint.append(float("nan"))
                    continue
                dx = sym_cx - ex; dy = sym_cy - ey
                near = (dx * dx + dy * dy) <= (self.endpoint_radius ** 2)
                if not near.any():
                    max_p_per_endpoint.append(float("nan"))
                else:
                    max_p_per_endpoint.append(float(sym_p[near].max()))
            decision.nearest_p_per_endpoint = max_p_per_endpoint

            # Decision logic
            anys = [p for p in max_p_per_endpoint if not np.isnan(p)]
            any_irr  = any(p >= self.p_irr_thresh  for p in anys)
            any_call = any(p <= self.p_call_thresh for p in anys)

            if any_irr:
                decision.fate = "keep"; decision.reason = "irrigation_endpoint"
            elif any_call:
                # Provisional drop — but check if the component is long enough
                # to be a real lateral. Real laterals span hundreds of pixels;
                # callout fragments are usually < ~100 px. Skeleton length is
                # the right ruler because it's width-invariant.
                if self.min_drop_skel_px > 0:
                    skel_len = int(skel.sum())
                else:
                    skel_len = -1  # signal "not measured"
                if self.min_drop_skel_px > 0 and skel_len >= self.min_drop_skel_px:
                    decision.fate = "keep"
                    decision.reason = (
                        f"callout_endpoint_but_long_skel_{skel_len}>={self.min_drop_skel_px}"
                    )
                else:
                    decision.fate = "drop"
                    decision.reason = (
                        "callout_endpoint_only"
                        if skel_len < 0
                        else f"callout_endpoint_short_skel_{skel_len}"
                    )
                    drop_set.add(cid)
            else:
                decision.fate = "keep"; decision.reason = "no_strong_signal"
            decisions.append(decision)

        # Build filtered mask: zero out dropped components.
        if not drop_set:
            filtered = binary * 255
        else:
            keep_mask = np.ones(n_lab, dtype=np.uint8)
            for cid in drop_set:
                keep_mask[cid] = 0
            filtered = (keep_mask[lab_img] * binary * 255).astype(np.uint8)

        pixels_dropped = int(((binary > 0) & (filtered == 0)).sum())
        report = FilterReport(
            n_components_total=n_lab - 1,
            n_components_dropped=len(drop_set),
            pixels_dropped=pixels_dropped,
            decisions=decisions,
        )
        return filtered, report


# ---------------------------------------------------------------------------
# Helpers for the eval / viz scripts
# ---------------------------------------------------------------------------


def load_cached_symbols(cache_json: str | Path) -> list[dict]:
    """Load the per-image symbol cache. Returns ``[]`` if missing."""
    p = Path(cache_json)
    if not p.is_file():
        return []
    return json.loads(p.read_text()).get("symbols", [])
