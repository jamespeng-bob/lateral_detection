"""Recompute per-symbol ``near_gt`` in the existing symbol cache.

What "near_gt" should mean (sword-and-apple criterion)
------------------------------------------------------
A symbol is irrigation-related iff the GT lateral line either PASSES
THROUGH the symbol (sword pierces apple through the middle) or ENDS
inside the symbol (sword thrusts toward the center but doesn't pierce
through). A line that GRAZES the symbol's edge (sword glances off the
apple's surface) is NOT a real intersection.

The previous heuristics didn't capture this:

  v1 (bbox CENTER distance < d_near px):
     Biased low for LARGE symbols. A 50×50 valve sitting on a lateral has
     its center 20-25 px from the line even though the bbox contains the
     line; with d_near=30 most large valves got mislabeled as not-near-GT.

  v2 (any bbox pixel within d_near of GT — bbox overlap):
     Captured large valves correctly, but: (a) used an ABSOLUTE pixel
     threshold (unfair to symbols of different sizes), and (b) couldn't
     distinguish "line pierces the symbol" from "line grazes the symbol
     edge" — both fire near_gt=True even though only the first means
     "the symbol is on the line".

  v3 (THIS file — inner-region check):
     A symbol is near_gt iff any GT line pixel lies within the bbox's
     INNER REGION (the bbox shrunk by `center_margin` on each side).
     Inner region = bbox shrunk to (1 - 2*center_margin) × (1 - 2*center_margin)
     of original size, centered.

     - center_margin=0.30 → inner is the central 40% × 40% of the bbox.
     - Line piercing the center: hits inner region → near_gt=True ✓
     - Line ending well inside the bbox: hits inner region → near_gt=True ✓
     - Line grazing one bbox edge: all line pixels are in the OUTER ring,
       none in the inner region → near_gt=False ✓
     - Line ending exactly at bbox edge: borderline; if the line extends
       into the bbox at all, may or may not reach the inner region
       depending on geometry (this is the only case where we err
       slightly conservative).

     Threshold is RELATIVE to bbox size — same fraction works for tiny
     callout boxes and large valves alike.

What this script doesn't do
---------------------------
Doesn't change the EMBEDDINGS or bbox values in the cache. Only recomputes
the boolean `near_gt` per symbol. No new API calls.

Usage (re-run end-to-end after this):

    python -m scripts.recompute_near_gt --center-margin 0.30
    python -m scripts.train_symbol_classifier ...
    python -m scripts.eval_symbol_filter ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.coco_loader import load_split
from data.polyline_builder import build_polylines
from data.rasterize import rasterize_polylines
from train import merge_configs


def _bbox_near_gt_inner(
    gt_mask: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    center_margin: float,
) -> bool:
    """Sword-and-apple criterion: True iff any GT line pixel lies in the
    bbox's INNER region (the bbox shrunk by ``center_margin`` on each side).

    ``center_margin=0.30`` → inner region = central 40% × 40% of bbox.

    A line that pierces the bbox (sword through apple's middle) has pixels
    in the inner region; a line that just grazes an edge does not.
    """
    H, W = gt_mask.shape
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return False
    # Shrink bbox by `center_margin` of each side, from each side.
    inset_x = w * center_margin
    inset_y = h * center_margin
    ix1 = int(round(x1 + inset_x))
    iy1 = int(round(y1 + inset_y))
    ix2 = int(round(x2 - inset_x))
    iy2 = int(round(y2 - inset_y))
    # Clip to image
    ix1 = max(0, min(W, ix1))
    iy1 = max(0, min(H, iy1))
    ix2 = max(0, min(W, ix2))
    iy2 = max(0, min(H, iy2))
    if ix2 <= ix1 or iy2 <= iy1:
        # Inner region collapsed (e.g., a tiny bbox with large margin) —
        # fall back to the FULL bbox check so we don't silently return False
        # for legitimate small symbols on the line.
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(W, int(x2) + 1)
        iy2 = min(H, int(y2) + 1)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
    patch = gt_mask[iy1:iy2, ix1:ix2]
    return bool(patch.any())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config",      default="configs/train.yaml")
    parser.add_argument("--overlay",     default="configs/train_v2b_6k.yaml")
    parser.add_argument("--cache-dir",   default="results/symbols_cache")
    parser.add_argument("--splits",      nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--center-margin", type=float, default=0.30,
                        help="Fraction of each bbox side to crop OFF when "
                             "checking GT overlap. 0.30 → inner region is the "
                             "central 40%% × 40%% of the bbox. Same fraction "
                             "works for symbols of any size — that's the point "
                             "of using a relative threshold.")
    args = parser.parse_args()

    cfg = merge_configs(yaml.safe_load(open(args.base_config)),
                        yaml.safe_load(open(args.config)))
    cfg = merge_configs(cfg, yaml.safe_load(open(args.overlay)))

    cache_root   = Path(args.cache_dir)
    dataset_root = (Path.cwd() / cfg["data"]["dataset_root"]).resolve()
    lateral_cid  = int(cfg["lateral_category_id"])
    merge_radius = float(cfg["polyline"]["merge_radius"])
    thickness    = int(cfg["rasterize"]["thickness"])
    split_dir_keys = {"train": "train_dir", "valid": "valid_dir", "test": "test_dir"}

    grand_total = 0
    grand_flips = 0

    for split in args.splits:
        if split not in split_dir_keys:
            print(f"  [warn] unknown split {split!r}, skipping")
            continue
        split_dir = dataset_root / cfg["data"][split_dir_keys[split]]
        if not split_dir.is_dir():
            print(f"  [warn] split dir missing: {split_dir}")
            continue

        images, chords_by_image = load_split(split_dir, category_id=lateral_cid)
        # Build filename → record + chords map so we can look up by cache stem
        by_stem = {Path(rec.file_name).stem: (rec, chords_by_image.get(rec.id, []))
                   for rec in images.values()}

        cache_split = cache_root / split
        if not cache_split.is_dir():
            print(f"  [warn] no cache for split {split!r} at {cache_split}")
            continue

        n_imgs   = 0
        n_syms   = 0
        n_flip   = 0
        n_t_true = 0
        n_t_false = 0

        for jp in tqdm(sorted(cache_split.glob("*.json")), desc=f"recompute {split}"):
            data = json.loads(jp.read_text())
            stem = jp.stem
            if stem not in by_stem:
                continue
            record, chords = by_stem[stem]
            polylines = build_polylines(chords, merge_radius=merge_radius)
            if not polylines:
                # No GT laterals on this image — keep symbols' near_gt as None.
                for s in data["symbols"]:
                    s["near_gt"] = None
                jp.write_text(json.dumps(data))
                continue
            gt = rasterize_polylines(
                polylines=polylines, height=record.height, width=record.width,
                thickness=thickness,
            )
            if not gt.any():
                for s in data["symbols"]:
                    s["near_gt"] = None
                jp.write_text(json.dumps(data))
                continue

            for s in data["symbols"]:
                old = s.get("near_gt")
                new = _bbox_near_gt_inner(
                    gt, s["x1"], s["y1"], s["x2"], s["y2"], args.center_margin,
                )
                s["near_gt"] = bool(new)
                if old is None or bool(old) != bool(new):
                    n_flip += 1
                if new:
                    n_t_true += 1
                else:
                    n_t_false += 1
                n_syms += 1
            jp.write_text(json.dumps(data))
            n_imgs += 1

        print(f"  [{split}] {n_imgs} images, {n_syms} symbols  "
              f"(near_gt: {n_t_true} true / {n_t_false} false, {n_flip} flipped)")
        grand_total += n_syms
        grand_flips += n_flip

    print()
    print(f"=== done: {grand_total} symbols, {grand_flips} flipped "
          f"({100*grand_flips/max(1,grand_total):.1f}%) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
