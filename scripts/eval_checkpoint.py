"""eval_checkpoint.py — run a single val pass with the accumulator metrics.

Useful for:

- Re-evaluating older checkpoints (v1, v2*) under the new accumulator-style
  metrics so they're directly comparable to v3+ training runs (which use
  the new metrics natively).
- Quick "what does this best.pth actually score on val?" without spinning
  up a fresh training session.

Single-GPU only (no DDP). The accumulator metrics short-circuit their
all-reduce calls when ``torch.distributed`` is not initialized, so they
just produce single-process numbers — identical to what they'd produce
inside a single-GPU training loop.

Normalization caveat
--------------------
The checkpoint must be evaluated under the SAME normalization stats it
was trained with — feeding a different distribution at eval time degrades
the model artificially. Two ways to pin the right stats:

- ``--imagenet-norm``     pins to [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
                          → use for v1, v2a, v2b, v2c (all trained pre dataset-stats change)
- (default)               uses whatever ``normalization:`` is in the merged config
                          → use for v3+ trained after the dataset-stats default landed

For v3+ we'll also embed the stats inside the checkpoint dict so this
flag becomes optional.

Usage
-----
    # v2b — needs --imagenet-norm because it predates the dataset-stats change:
    python -m scripts.eval_checkpoint \\
        --checkpoint runs/v2b_hgnetv2b4_bcedice/best.pth \\
        --overlay    configs/train_v2b.yaml \\
        --imagenet-norm \\
        --device     cuda:0

    # A future v3 checkpoint trained with dataset-stat normalization:
    python -m scripts.eval_checkpoint \\
        --checkpoint runs/v3a_*/best.pth \\
        --overlay    configs/train_v3a.yaml \\
        --device     cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make repo modules importable when running the file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import TileDataset, collate_tile_samples, worker_init_fn
from models.unet import build_model
from training.losses import build_loss
from training.metrics import (
    BinaryClDiceAccumulator,
    BinaryDiceAccumulator,
    BinaryIoUAccumulator,
    LengthRatioAccumulator,
)
from train import _resolve_device, merge_configs


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a model_state_dict checkpoint (best.pth / last.pth).")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config",      default="configs/train.yaml")
    parser.add_argument("--overlay",     default=None,
                        help="Optional per-experiment overlay (e.g. configs/train_v2b.yaml).")
    parser.add_argument("--device",      default="cuda:0",
                        help="Inference device (cuda:0 / cuda:1 / cpu / mps).")
    parser.add_argument("--batch-size",  type=int, default=None,
                        help="Override training.batch_size for eval (default: config value).")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override training.num_workers (default: config value).")
    parser.add_argument("--imagenet-norm", action="store_true",
                        help="Force ImageNet normalization (mean/std). Use for v1/v2* "
                             "checkpoints trained before the dataset-stats default landed.")
    args = parser.parse_args()

    # ── Merge configs ───────────────────────────────────────────────────
    base_cfg  = yaml.safe_load(open(args.base_config))
    train_cfg = yaml.safe_load(open(args.config))
    cfg = merge_configs(base_cfg, train_cfg)
    if args.overlay:
        cfg = merge_configs(cfg, yaml.safe_load(open(args.overlay)))

    if args.imagenet_norm:
        cfg["normalization"] = {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}

    device = _resolve_device(args.device)
    bs = args.batch_size if args.batch_size is not None else int(cfg["training"]["batch_size"])
    nw = args.num_workers if args.num_workers is not None else int(cfg["training"]["num_workers"])

    # ── Build val dataset + loader ──────────────────────────────────────
    dataset_root = (Path.cwd() / cfg["data"]["dataset_root"]).resolve()
    val_dir = dataset_root / cfg["data"]["valid_dir"]
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    norm = cfg["normalization"]
    val_ds = TileDataset(
        split_dir=val_dir,
        tile_size=int(cfg["data"]["tile_size"]),
        stride=int(cfg["data"]["stride"]),
        mode=cfg["data"]["val_mode"],
        merge_radius=float(cfg["polyline"]["merge_radius"]),
        thickness=int(cfg["rasterize"]["thickness"]),
        augmenter=None,
        mean=tuple(norm["mean"]),
        std=tuple(norm["std"]),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=(device != "cpu"),
        collate_fn=collate_tile_samples,
        worker_init_fn=worker_init_fn,
        persistent_workers=(nw > 0),
    )

    print(f"[eval] checkpoint:   {args.checkpoint}")
    print(f"[eval] device:       {device}")
    print(f"[eval] encoder:      {cfg['model']['name']}/{cfg['model']['encoder']}")
    print(f"[eval] normalization: mean={norm['mean']}  std={norm['std']}")
    print(f"[eval] val tiles:    {len(val_ds)}  (bs={bs}, num_workers={nw})")

    # ── Build model + load checkpoint ───────────────────────────────────
    model = build_model(cfg["model"]).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"[eval] loaded:       epoch={ckpt.get('epoch', '?')}, "
          f"stored val_dice={float(ckpt.get('val_dice', float('nan'))):.4f}")

    criterion = build_loss(cfg["loss"]).to(device)

    # ── Validate ────────────────────────────────────────────────────────
    dice_acc   = BinaryDiceAccumulator(threshold=0.5)
    iou_acc    = BinaryIoUAccumulator(threshold=0.5)
    cldice_acc = BinaryClDiceAccumulator(threshold=0.5)
    length_acc = LengthRatioAccumulator(threshold=0.5)

    total_loss      = 0.0
    total_dice_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="val", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device,  non_blocking=True)
            logits = model(images)
            losses = criterion(logits, masks)
            total_loss      += float(losses["total"].item())
            total_dice_loss += float(losses["dice"].item())
            n += 1
            dice_acc.update(logits, masks)
            iou_acc.update(logits, masks)
            cldice_acc.update(logits, masks)
            length_acc.update(logits, masks)

    n = max(n, 1)
    length = length_acc.compute()

    print()
    print("=" * 60)
    print(" eval results (accumulator-style, global over val set)")
    print("=" * 60)
    print(f"  val_total_loss   = {total_loss      / n:.4f}")
    print(f"  val_dice_loss    = {total_dice_loss / n:.4f}")
    print(f"  val_dice         = {dice_acc.compute():.4f}")
    print(f"  val_iou          = {iou_acc.compute():.4f}")
    print(f"  val_cldice       = {cldice_acc.compute():.4f}")
    print(f"  val_length_pixel: mean={length['length_ratio_pixel_mean']:.3f}  "
          f"median={length['length_ratio_pixel_median']:.3f}  "
          f"p25={length['length_ratio_pixel_p25']:.3f}  "
          f"p75={length['length_ratio_pixel_p75']:.3f}")
    print(f"  val_length_skel : mean={length['length_ratio_skel_mean']:.3f}  "
          f"median={length['length_ratio_skel_median']:.3f}  "
          f"p25={length['length_ratio_skel_p25']:.3f}  "
          f"p75={length['length_ratio_skel_p75']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
