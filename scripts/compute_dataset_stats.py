"""Compute per-channel pixel mean and std over a sample of training plans.

These statistics are intended for the ``normalization:`` block in
``configs/train.yaml``. We sample whole images rather than positive-centered
tiles so the result is dataset-level (matches what the *full plan* looks
like to the model at inference time). The encoder's BatchNorm absorbs the
remaining tile-vs-image distribution gap during training.

Reproducible: same RNG seed yields the same sample.

Usage:
    python -m scripts.compute_dataset_stats
    python -m scripts.compute_dataset_stats --split train --n-images 50
    python -m scripts.compute_dataset_stats --all-images
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

# Make repo modules importable when running the file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.coco_loader import load_split

Image.MAX_IMAGE_PIXELS = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/base.yaml",
                        help="Used to locate dataset_root + split dirs.")
    parser.add_argument("--split", default="train",
                        choices=("train", "valid", "test"),
                        help="Which split to compute stats over (almost always 'train').")
    parser.add_argument("--n-images", type=int, default=50,
                        help="Number of random images to sample. Use --all-images to override.")
    parser.add_argument("--all-images", action="store_true",
                        help="Sample every image in the split.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible sampling.")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(open(args.base_config))
    dataset_root = (Path.cwd() / base_cfg["data"]["dataset_root"]).resolve()
    split_dir = dataset_root / base_cfg["data"][f"{args.split}_dir"]
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    images, _ = load_split(split_dir)
    records = list(images.values())

    rng = random.Random(args.seed)
    if args.all_images or args.n_images >= len(records):
        sampled = records
    else:
        sampled = rng.sample(records, args.n_images)

    print(f"split:        {args.split}")
    print(f"split_dir:    {split_dir}")
    print(f"sampling:     {len(sampled)} / {len(records)} images (seed={args.seed})")
    print()

    n_pixels     = 0
    sum_per_ch   = np.zeros(3, dtype=np.float64)
    sumsq_per_ch = np.zeros(3, dtype=np.float64)

    for rec in tqdm(sampled, desc="streaming"):
        with Image.open(rec.path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        n_pixels     += flat.shape[0]
        sum_per_ch   += flat.sum(axis=0)
        sumsq_per_ch += (flat ** 2).sum(axis=0)

    mean = sum_per_ch / n_pixels
    var  = sumsq_per_ch / n_pixels - mean ** 2
    # Floating-point precision can produce tiny negative variances on
    # near-constant channels; clamp before sqrt.
    std  = np.sqrt(np.maximum(var, 0.0))

    print()
    print(f"total pixels sampled: {n_pixels:,}")
    print(f"per-channel (R, G, B):")
    print(f"  mean = [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  std  = [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print()
    print("YAML block to paste into configs/train.yaml -> normalization:")
    print(f"  mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
