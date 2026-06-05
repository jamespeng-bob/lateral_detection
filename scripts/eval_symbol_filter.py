"""Apply the symbol filter to v2b's val predictions and report before/after.

For each val image:
  1. Tile-infer with v2b to get the predicted probability map.
  2. Threshold → binary pred mask.
  3. Load the cached symbol detections + embeddings for this image.
  4. Apply :class:`SymbolFilter` to get a filtered mask.
  5. Compute dice + len_skel_ratio + pixel counts for BOTH masks.
  6. Render a 4-color composite per image:
       white   = TP (kept and correct)
       cyan    = FN (unchanged — filter cannot recover misses)
       magenta = FP_kept (model wrong AND filter agrees)
       orange  = FP_dropped (model wrong, filter caught it)
     Plus colored symbol boxes:
       green   = irrigation (P >= p_irr_thresh)
       red     = callout    (P <= p_call_thresh)
       yellow  = neutral    (in between)

Per-image stats land in summary.csv with paired before/after columns.

Usage
-----
    python -m scripts.eval_symbol_filter \\
        --checkpoint runs/v2b_hgnetv2b4_bcedice_6k/best.pth \\
        --classifier results/symbol_categories/symbol_classifier.joblib \\
        --cache-dir  results/symbols_cache \\
        --out-dir    results/v2b_with_symbol_filter \\
        --splits     valid \\
        --device     cuda:0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from skimage.morphology import skeletonize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.coco_loader import load_split
from data.polyline_builder import build_polylines
from data.rasterize import rasterize_polylines
from inference.symbol_filter import SymbolClassifier, SymbolFilter, load_cached_symbols
from inference.tiled_predict import predict_full_image
from models.unet import build_model
from train import _resolve_device, merge_configs

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------


def _image_metrics(gt_bin: np.ndarray, pred_bin: np.ndarray) -> dict:
    gt_b   = gt_bin   > 0
    pred_b = pred_bin > 0
    tp = int(np.logical_and(gt_b, pred_b).sum())
    fp = int(np.logical_and(~gt_b, pred_b).sum())
    fn = int(np.logical_and(gt_b, ~pred_b).sum())
    gt_count   = int(gt_b.sum())
    pred_count = int(pred_b.sum())
    dice = (2 * tp + 1e-6) / (gt_count + pred_count + 1e-6)
    gt_skel = int(skeletonize(gt_b).sum()) if gt_count else 0
    pred_skel = int(skeletonize(pred_b).sum()) if pred_count else 0
    len_skel_ratio = (pred_skel / gt_skel) if gt_skel else float("nan")
    return dict(dice=dice, len_skel_ratio=len_skel_ratio,
                tp=tp, fp=fp, fn=fn, gt_skel=gt_skel, pred_skel=pred_skel)


# ---------------------------------------------------------------------------
# Composite visualization with before/after coloring
# ---------------------------------------------------------------------------


def _downsample(img: np.ndarray, max_long_side: int = 1800) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_long_side:
        return img
    scale = max_long_side / max(h, w)
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _build_composite(
    image_rgb:    np.ndarray,
    gt_mask:      np.ndarray,
    pred_mask:    np.ndarray,
    filtered_mask: np.ndarray,
    symbols:      list[dict],
    sym_p:        np.ndarray,
    *,
    p_irr_thresh:  float,
    p_call_thresh: float,
    fade_alpha:    float = 0.4,
    max_long_side: int   = 1800,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    canvas = (image_rgb.astype(np.float32) * fade_alpha).clip(0, 255)

    gt   = (gt_mask      >= 128).astype(bool)
    pred = (pred_mask    >= 128).astype(bool)
    filt = (filtered_mask >= 128).astype(bool)

    # Categories:
    #   TP_kept       : gt   ∧ filt              → white
    #   FN            : gt   ∧ ¬pred             → cyan
    #   FP_kept       : ¬gt  ∧ filt              → magenta
    #   FP_dropped    : ¬gt  ∧ pred ∧ ¬filt      → orange
    #   TP_dropped    : gt   ∧ pred ∧ ¬filt      → red (danger! filter ate real lateral)
    tp_kept   = gt   & filt
    fn        = gt   & ~pred
    fp_kept   = ~gt  & filt
    fp_drop   = ~gt  & pred & ~filt
    tp_drop   = gt   & pred & ~filt

    canvas[tp_kept] = (255, 255, 255)   # white
    canvas[fn]      = (0,   255, 255)   # cyan
    canvas[fp_kept] = (255, 0,   255)   # magenta
    canvas[fp_drop] = (255, 165, 0)     # orange
    canvas[tp_drop] = (255, 0,   0)     # red — bad! filter dropped real lateral

    canvas = canvas.astype(np.uint8)

    # Symbol bboxes color-coded by classifier P
    for s, p in zip(symbols, sym_p):
        x1, y1, x2, y2 = int(s["x1"]), int(s["y1"]), int(s["x2"]), int(s["y2"])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        if   p >= p_irr_thresh:   col = (0,   255, 0)    # green = irrigation
        elif p <= p_call_thresh:  col = (255, 0,   0)    # red   = callout
        else:                     col = (255, 255, 0)    # yellow = neutral
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, thickness=4)

    canvas = _downsample(canvas, max_long_side=max_long_side)

    # Legend strip
    legend_h = 40
    legend = np.zeros((legend_h, canvas.shape[1], 3), dtype=np.uint8)
    items = [
        ("TP",          (255, 255, 255)),
        ("FN",          (0,   255, 255)),
        ("FP_kept",     (255, 0,   255)),
        ("FP_dropped",  (255, 165, 0)),
        ("TP_dropped!", (255, 0,   0)),
        ("sym:irr",     (0,   255, 0)),
        ("sym:call",    (255, 0,   0)),
    ]
    x = 10
    for label, color in items:
        cv2.rectangle(legend, (x, 8), (x + 22, 32), color, -1)
        cv2.putText(legend, label, (x + 28, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        x += 8 + 22 + len(label) * 10 + 12
    return np.concatenate([legend, canvas], axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--classifier",  required=True,
                        help="Path to symbol_classifier.joblib")
    parser.add_argument("--cache-dir",   default="results/symbols_cache")
    parser.add_argument("--out-dir",     required=True)
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config",      default="configs/train.yaml")
    parser.add_argument("--overlay",     default="configs/train_v2b_6k.yaml")
    parser.add_argument("--splits",      nargs="+", default=["valid"])
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--prob-thresh", type=float, default=0.5)
    parser.add_argument("--p-irr-thresh",     type=float, default=0.85)
    parser.add_argument("--p-call-thresh",    type=float, default=0.05,
                        help="Drop only on very-confident non-irrigation (was 0.15).")
    parser.add_argument("--dilation-px",      type=int,   default=3)
    parser.add_argument("--endpoint-radius",  type=int,   default=25)
    parser.add_argument("--min-drop-skel-px", type=int,   default=150,
                        help="Components with skeleton-length >= this are PROTECTED "
                             "from being dropped (they're long enough to be real "
                             "laterals, not callout fragments). Default 150.")
    parser.add_argument("--max-long-side",   type=int, default=1800)
    args = parser.parse_args()

    cfg = merge_configs(yaml.safe_load(open(args.base_config)),
                        yaml.safe_load(open(args.config)))
    cfg = merge_configs(cfg, yaml.safe_load(open(args.overlay)))

    device = _resolve_device(args.device)
    print(f"[eval_filter] device: {device}")

    model = build_model(cfg["model"]).to(device).eval()
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print(f"[eval_filter] checkpoint loaded, epoch={ckpt.get('epoch', '?')}")

    classifier = SymbolClassifier(args.classifier)
    print(f"[eval_filter] classifier loaded (train AUC={classifier.auc_train:.3f}, val AUC={classifier.auc_val:.3f})")
    sfilter = SymbolFilter(
        classifier=classifier,
        p_irr_thresh=args.p_irr_thresh,
        p_call_thresh=args.p_call_thresh,
        dilation_px=args.dilation_px,
        endpoint_radius=args.endpoint_radius,
        min_drop_skel_px=args.min_drop_skel_px,
    )
    print(f"[eval_filter] filter: P_irr>={args.p_irr_thresh}, "
          f"P_call<={args.p_call_thresh}, dilation={args.dilation_px}, "
          f"endpoint_radius={args.endpoint_radius}, "
          f"min_drop_skel_px={args.min_drop_skel_px}")

    norm = cfg["normalization"]
    mean = tuple(norm["mean"]); std = tuple(norm["std"])

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    fp_sum = open(summary_path, "w", newline="")
    writer = csv.DictWriter(fp_sum, fieldnames=[
        "split", "image_id", "image_file",
        "dice_before", "dice_after", "dice_delta",
        "len_skel_ratio_before", "len_skel_ratio_after",
        "fp_before", "fp_after", "fp_dropped",
        "fn_before", "fn_after", "fn_added",
        "n_components", "n_dropped", "pixels_dropped",
        "n_symbols", "n_sym_irr", "n_sym_call",
    ])
    writer.writeheader()

    dataset_root = (Path.cwd() / cfg["data"]["dataset_root"]).resolve()
    lateral_cid  = int(cfg["lateral_category_id"])
    merge_radius = float(cfg["polyline"]["merge_radius"])
    thickness    = int(cfg["rasterize"]["thickness"])

    split_dir_keys = {"train": "train_dir", "valid": "valid_dir", "test": "test_dir"}
    for split in args.splits:
        if split not in split_dir_keys:
            continue
        split_dir = dataset_root / cfg["data"][split_dir_keys[split]]
        if not split_dir.is_dir():
            continue
        out_split = out_dir / split; out_split.mkdir(parents=True, exist_ok=True)
        images, chords_by_image = load_split(split_dir, category_id=lateral_cid)
        print(f"\n=== split: {split} ({len(images)} images) ===")
        for image_id, record in tqdm(sorted(images.items()), desc=split):
            img_path = record.path
            if not img_path.is_file():
                continue

            # GT
            polylines = build_polylines(chords_by_image.get(image_id, []),
                                        merge_radius=merge_radius)
            gt_mask = rasterize_polylines(
                polylines=polylines, height=record.height, width=record.width,
                thickness=thickness,
            )

            # Pred (tile inference)
            prob = predict_full_image(
                model, img_path,
                tile_size=int(cfg["data"]["tile_size"]),
                stride=int(cfg["data"]["stride"]),
                device=device, mean=mean, std=std,
                batch_size=4, show_progress=False,
            )
            pred_mask = (prob >= args.prob_thresh).astype(np.uint8) * 255

            # Symbols from cache
            cache_path = Path(args.cache_dir) / split / f"{Path(record.file_name).stem}.json"
            symbols = load_cached_symbols(cache_path)

            # Apply filter
            filtered_mask, report = sfilter.apply(pred_mask, symbols)

            # Metrics
            m_before = _image_metrics(gt_mask, pred_mask)
            m_after  = _image_metrics(gt_mask, filtered_mask)

            # Per-symbol categorization for viz
            if symbols:
                embs   = np.asarray([s["embedding"] for s in symbols], dtype=np.float32)
                bboxes = np.asarray(
                    [[s["x1"], s["y1"], s["x2"], s["y2"]] for s in symbols],
                    dtype=np.float32,
                )
                sym_p = classifier.predict_proba(embs, bboxes)
            else:
                sym_p = np.zeros(0, dtype=np.float32)
            n_sym_irr  = int((sym_p >= args.p_irr_thresh).sum())
            n_sym_call = int((sym_p <= args.p_call_thresh).sum())

            # Viz
            image_rgb = np.array(Image.open(img_path).convert("RGB"))
            viz = _build_composite(
                image_rgb=image_rgb,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                filtered_mask=filtered_mask,
                symbols=symbols, sym_p=sym_p,
                p_irr_thresh=args.p_irr_thresh,
                p_call_thresh=args.p_call_thresh,
                max_long_side=args.max_long_side,
            )
            out_jpg = out_split / f"{Path(record.file_name).stem}.jpg"
            cv2.imwrite(str(out_jpg), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])

            writer.writerow({
                "split":       split,
                "image_id":    image_id,
                "image_file":  record.file_name,
                "dice_before": f"{m_before['dice']:.4f}",
                "dice_after":  f"{m_after['dice']:.4f}",
                "dice_delta":  f"{m_after['dice'] - m_before['dice']:+.4f}",
                "len_skel_ratio_before": f"{m_before['len_skel_ratio']:.4f}",
                "len_skel_ratio_after":  f"{m_after['len_skel_ratio']:.4f}",
                "fp_before":   m_before["fp"],
                "fp_after":    m_after["fp"],
                "fp_dropped":  m_before["fp"] - m_after["fp"],
                "fn_before":   m_before["fn"],
                "fn_after":    m_after["fn"],
                "fn_added":    m_after["fn"] - m_before["fn"],
                "n_components": report.n_components_total,
                "n_dropped":    report.n_components_dropped,
                "pixels_dropped": report.pixels_dropped,
                "n_symbols":    len(symbols),
                "n_sym_irr":    n_sym_irr,
                "n_sym_call":   n_sym_call,
            })
            fp_sum.flush()

    fp_sum.close()
    print(f"\n[eval_filter] wrote {summary_path}")
    print(f"[eval_filter] per-image visualizations in {out_dir}/<split>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
