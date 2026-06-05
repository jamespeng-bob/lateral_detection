"""Recompute per-symbol ``near_gt`` in the existing symbol cache.

Why this exists
---------------
The original ``mine_symbol_categories.py`` set ``near_gt`` using the bbox
CENTER distance to the nearest GT lateral pixel. For LARGE symbols (valves,
sprinklers — the irrigation-relevant ones) this is biased low: a 50×50 px
valve sitting ON a lateral has its center ~20-25 px from the line even
though the bbox literally contains the line. With ``d_near=30 px`` they
ended up under-labeled, polluting the "ambiguous" cluster with real
valves and teaching the classifier to predict low P for them. Downstream,
the symbol filter then dropped real laterals whose endpoints touched
these mis-classified valves.

This script fixes that **without re-running the API** (which dominates
the cost of mining). It walks the existing cache, re-computes ``near_gt``
using bbox-OVERLAP distance (``min distance over pixels inside the bbox
< d_near``), and writes the updated JSONs in place. Embeddings + bboxes
+ ids are untouched.

After this, you re-train the classifier and re-evaluate the filter:

    python -m scripts.recompute_near_gt --d-near 10
    python -m scripts.train_symbol_classifier ...
    python -m scripts.eval_symbol_filter ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
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


def _bbox_near_gt(
    dist_lookup: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    d_near: int,
) -> bool:
    """True if any pixel inside the bbox is within ``d_near`` of a GT pixel.

    ``dist_lookup[y, x]`` is the L2 distance from pixel (y, x) to the
    nearest GT-line pixel (computed once per image).
    """
    H, W = dist_lookup.shape
    x1i = max(0, min(W - 1, int(x1)))
    y1i = max(0, min(H - 1, int(y1)))
    x2i = max(0, min(W - 1, int(x2)))
    y2i = max(0, min(H - 1, int(y2)))
    if x2i < x1i or y2i < y1i:
        return False
    patch = dist_lookup[y1i : y2i + 1, x1i : x2i + 1]
    return bool(patch.size > 0 and patch.min() < d_near)


def _near_gt_lookup(gt_mask: np.ndarray) -> np.ndarray:
    """L2 distance transform of ``1 - gt_mask`` (0 on GT pixels, growing outward)."""
    inv = (gt_mask == 0).astype(np.uint8)
    return cv2.distanceTransform(inv, cv2.DIST_L2, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config",      default="configs/train.yaml")
    parser.add_argument("--overlay",     default="configs/train_v2b_6k.yaml")
    parser.add_argument("--cache-dir",   default="results/symbols_cache")
    parser.add_argument("--splits",      nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--d-near",      type=int, default=10,
                        help="Pixel distance for bbox-overlap with GT. Default 10 "
                             "is meaningfully tighter than the old 30 because we "
                             "no longer need to compensate for center bias.")
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
            dist = _near_gt_lookup(gt)

            for s in data["symbols"]:
                old = s.get("near_gt")
                new = _bbox_near_gt(
                    dist, s["x1"], s["y1"], s["x2"], s["y2"], args.d_near,
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
