"""For each image in a split: run a checkpoint, overlay GT vs predicted
lateral lines vs detected symbols, save a composite visualization, and
write a per-image summary CSV.

The goal is human-eyeballing of model failure modes before committing to
a downstream filtering strategy (valve-based component filter, valve-as-
input-channel retrain, etc.). Three things become visible per drawing:

  - Where the model is correct (white)
  - Where the model misses real GT (cyan)        → false negatives
  - Where the model hallucinates lines (magenta) → false positives
  - Where valve symbols are detected (yellow boxes) — when --with-symbols

If FPs cluster in regions far from any valve, a post-processing valve
filter will help. If FPs cluster in regions with detected valves but
between irrigation and non-irrigation lines, a valve-channel retrain is
the right tool. If FPs are uniform / random, neither approach is enough.

Output layout (per chosen split):
    <out-dir>/<split>/<stem>.jpg                 composite viz
    <out-dir>/summary.csv                        one row per image

Usage
-----
    python -m scripts.failure_analysis \
        --checkpoint runs/v2b_hgnetv2b4_bcedice_6k/best.pth \
        --out-dir    results/v2b_failure_analysis \
        --splits     valid \
        --device     cuda:0   # or mps / cpu on macOS

Add ``--no-symbols`` to skip the GCP symbol API call (useful when
credentials aren't available on the host).
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
from inference.tiled_predict import predict_full_image
from models.unet import build_model
from train import _resolve_device, merge_configs

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _downsample_for_viz(img: np.ndarray, max_long_side: int = 1800) -> np.ndarray:
    """Resize so the long side is at most ``max_long_side`` (keeps aspect)."""
    h, w = img.shape[:2]
    long = max(h, w)
    if long <= max_long_side:
        return img
    scale = max_long_side / long
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    interp = cv2.INTER_AREA  # area-resample preserves thin lines well
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _build_composite(
    image_rgb: np.ndarray,
    gt_mask:   np.ndarray,   # H x W, uint8 in {0, 255}
    pred_mask: np.ndarray,   # H x W, uint8 in {0, 255}
    symbol_boxes: list[tuple[int, int, int, int, float, str]],
    *,
    fade_alpha: float = 0.4,
    max_long_side: int = 1800,
) -> np.ndarray:
    """Return a composite RGB visualization of original + GT/pred overlays.

    Color key:
      - TP  (gt ∧ pred): white
      - FN  (gt only)  : cyan
      - FP  (pred only): magenta
      - symbol box     : yellow rectangle (with label)
    """
    h, w = image_rgb.shape[:2]
    canvas = (image_rgb.astype(np.float32) * fade_alpha).clip(0, 255)

    gt_bin   = (gt_mask   >= 128).astype(np.uint8)
    pred_bin = (pred_mask >= 128).astype(np.uint8)
    tp = (gt_bin & pred_bin).astype(bool)
    fn = (gt_bin & ~pred_bin).astype(bool)
    fp = (~gt_bin & pred_bin).astype(bool)

    # White for TP, cyan for FN, magenta for FP.  Channels are R, G, B.
    canvas[tp] = (255, 255, 255)
    canvas[fn] = (0,   255, 255)
    canvas[fp] = (255, 0,   255)

    canvas = canvas.astype(np.uint8)

    # Yellow rectangles + labels for symbols. Drawn BEFORE downsample so
    # labels stay readable at full font size relative to the original image.
    for x1, y1, x2, y2, conf, label in symbol_boxes:
        x1c = max(0, int(x1)); y1c = max(0, int(y1))
        x2c = min(w - 1, int(x2)); y2c = min(h - 1, int(y2))
        cv2.rectangle(canvas, (x1c, y1c), (x2c, y2c), (255, 255, 0), thickness=4)

    # Downsample (anti-aliased) for a viewable JPG.
    canvas = _downsample_for_viz(canvas, max_long_side=max_long_side)

    # Legend strip across the top
    legend_h = 36
    legend = np.zeros((legend_h, canvas.shape[1], 3), dtype=np.uint8)
    items = [
        ("TP",  (255, 255, 255)),
        ("FN",  (0,   255, 255)),
        ("FP",  (255, 0,   255)),
        ("symbol", (255, 255, 0)),
    ]
    x = 12
    for name, color in items:
        cv2.rectangle(legend, (x, 8), (x + 22, 28), color, -1)
        cv2.putText(legend, name, (x + 28, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        x += 96
    return np.concatenate([legend, canvas], axis=0)


# ---------------------------------------------------------------------------
# Per-image metrics (compact recap of what training-time accumulators report)
# ---------------------------------------------------------------------------


def _image_metrics(gt_bin: np.ndarray, pred_bin: np.ndarray) -> dict:
    """Per-image dice + length-skel ratio + FP/FN/TP pixel counts."""
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
    return dict(
        dice=dice,
        len_skel_ratio=len_skel_ratio,
        gt_count=gt_count, pred_count=pred_count,
        gt_skel=gt_skel,  pred_skel=pred_skel,
        tp=tp, fp=fp, fn=fn,
    )


# ---------------------------------------------------------------------------
# Symbol detection (optional)
# ---------------------------------------------------------------------------


def _build_symbol_client():
    """Lazy import + construct the localizer client. None on any failure."""
    try:
        from symbols.call_symbol_localizer import (
            DEFAULT_CLASSIFICATION_ENDPOINT_ID,
            DEFAULT_GCS_BUCKET,
            DEFAULT_LOCALIZATION_ENDPOINT_ID,
            DEFAULT_LOCATION,
            DEFAULT_PROJECT_ID,
            IsolatedSymbolClient,
            _resolve_credentials_file,
        )
    except Exception as exc:
        print(f"[failure_analysis] could not import symbol client: {exc}")
        return None

    try:
        creds = _resolve_credentials_file("")
        return IsolatedSymbolClient(
            credentials_file=creds,
            project_id=DEFAULT_PROJECT_ID,
            location=DEFAULT_LOCATION,
            localization_endpoint_id=DEFAULT_LOCALIZATION_ENDPOINT_ID,
            classification_endpoint_id=DEFAULT_CLASSIFICATION_ENDPOINT_ID,
            gcs_bucket=DEFAULT_GCS_BUCKET,
            nms_iou_thresh=0.5,
            conf_thresh=0.3,
        )
    except Exception as exc:
        print(f"[failure_analysis] could not init symbol client: {exc}")
        return None


def _fetch_symbol_boxes(client, image_path: Path) -> list[tuple[int, int, int, int, float, str]]:
    """Return [(x1, y1, x2, y2, conf, label), ...]. Empty list on failure."""
    if client is None:
        return []
    try:
        dets = client.localize(str(image_path))
    except Exception as exc:
        print(f"  [warn] symbol API failed for {image_path.name}: {exc}")
        return []
    out = []
    for d in dets:
        try:
            x1 = float(d["x1"]); y1 = float(d["y1"])
            x2 = float(d["x2"]); y2 = float(d["y2"])
            conf = float(d.get("conf", 0.0) or 0.0)
            label = str(d.get("class", d.get("class_id", d.get("label", ""))) or "")
            out.append((x1, y1, x2, y2, conf, label))
        except (KeyError, ValueError, TypeError):
            continue
    return out


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to model_state_dict checkpoint (e.g. best.pth).")
    parser.add_argument("--out-dir",     required=True,
                        help="Output directory (will be created).")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config",      default="configs/train.yaml")
    parser.add_argument("--overlay",     default="configs/train_v2b_6k.yaml",
                        help="Per-experiment overlay so model arch + normalization "
                             "match the checkpoint's training config.")
    parser.add_argument("--splits",      nargs="+", default=["valid"])
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--prob-thresh", type=float, default=0.5,
                        help="Threshold the probability map at this value to get the binary "
                             "prediction mask.")
    parser.add_argument("--no-symbols",  action="store_true",
                        help="Skip the symbol API call (no valve overlay, no cost).")
    parser.add_argument("--max-long-side", type=int, default=1800,
                        help="Resize composite so the long side is at most this many px.")
    args = parser.parse_args()

    cfg = merge_configs(yaml.safe_load(open(args.base_config)),
                        yaml.safe_load(open(args.config)))
    cfg = merge_configs(cfg, yaml.safe_load(open(args.overlay)))

    device = _resolve_device(args.device)
    print(f"[failure_analysis] device: {device}")
    print(f"[failure_analysis] checkpoint: {args.checkpoint}")

    # Build + load model.
    model = build_model(cfg["model"]).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    print(f"[failure_analysis] loaded epoch={ckpt.get('epoch', '?')}, "
          f"trained_val_dice={float(ckpt.get('val_dice', float('nan'))):.4f}")

    norm = cfg["normalization"]
    mean = tuple(norm["mean"]); std = tuple(norm["std"])
    print(f"[failure_analysis] normalization mean={mean}  std={std}")

    sym_client = None if args.no_symbols else _build_symbol_client()
    print(f"[failure_analysis] symbol API: {'ENABLED' if sym_client else 'disabled'}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    fp_summary = open(summary_path, "w", newline="")
    writer = csv.DictWriter(fp_summary, fieldnames=[
        "split", "image_id", "image_file",
        "dice", "len_skel_ratio",
        "gt_count", "pred_count", "gt_skel", "pred_skel",
        "tp", "fp", "fn", "n_symbols",
    ])
    writer.writeheader()

    dataset_root = (Path.cwd() / cfg["data"]["dataset_root"]).resolve()
    lateral_cid  = int(cfg["lateral_category_id"])
    merge_radius = float(cfg["polyline"]["merge_radius"])
    thickness    = int(cfg["rasterize"]["thickness"])

    split_dir_keys = {"train": "train_dir", "valid": "valid_dir", "test": "test_dir"}
    for split in args.splits:
        if split not in split_dir_keys:
            print(f"[failure_analysis] unknown split {split!r} — skipping")
            continue
        split_dir = dataset_root / cfg["data"][split_dir_keys[split]]
        if not split_dir.is_dir():
            print(f"[failure_analysis] split dir not found: {split_dir} — skipping")
            continue
        out_split = out_dir / split; out_split.mkdir(parents=True, exist_ok=True)

        # load_split returns (images_by_id, chords_by_image_id)
        images, chords_by_image = load_split(split_dir, category_id=lateral_cid)
        print(f"\n=== split: {split} ({len(images)} images) ===")
        for image_id, record in tqdm(sorted(images.items()), desc=split):
            img_path = record.path
            if not img_path.is_file():
                print(f"  [warn] missing image: {img_path}")
                continue

            # GT mask from this image's chords
            chords = chords_by_image.get(image_id, [])
            polylines = build_polylines(chords, merge_radius=merge_radius)
            gt_mask = rasterize_polylines(
                polylines=polylines,
                height=record.height,
                width=record.width,
                thickness=thickness,
            )

            # Predicted probability + mask
            prob = predict_full_image(
                model, img_path,
                tile_size=int(cfg["data"]["tile_size"]),
                stride=int(cfg["data"]["stride"]),
                device=device,
                mean=mean, std=std,
                batch_size=4,
                show_progress=False,
            )
            pred_mask = (prob >= args.prob_thresh).astype(np.uint8) * 255

            metrics = _image_metrics(gt_mask, pred_mask)
            sym_boxes = _fetch_symbol_boxes(sym_client, img_path)

            image_rgb = np.array(Image.open(img_path).convert("RGB"))
            viz = _build_composite(
                image_rgb=image_rgb,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                symbol_boxes=sym_boxes,
                max_long_side=args.max_long_side,
            )
            out_jpg = out_split / f"{Path(record.file_name).stem}.jpg"
            cv2.imwrite(str(out_jpg), cv2.cvtColor(viz, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])

            writer.writerow({
                "split":       split,
                "image_id":    image_id,
                "image_file":  record.file_name,
                "dice":        f"{metrics['dice']:.4f}",
                "len_skel_ratio": f"{metrics['len_skel_ratio']:.4f}",
                "gt_count":    metrics["gt_count"],
                "pred_count":  metrics["pred_count"],
                "gt_skel":     metrics["gt_skel"],
                "pred_skel":   metrics["pred_skel"],
                "tp":          metrics["tp"],
                "fp":          metrics["fp"],
                "fn":          metrics["fn"],
                "n_symbols":   len(sym_boxes),
            })
            fp_summary.flush()

    fp_summary.close()
    if sym_client is not None and hasattr(sym_client, "cleanup_uploads"):
        sym_client.cleanup_uploads()

    print(f"\n[failure_analysis] wrote {summary_path}")
    print(f"[failure_analysis] per-image visualizations in {out_dir}/<split>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
