"""Find lateral lines the model predicts with high confidence but that are
absent from the ground-truth mask. These are candidates for re-labeling.

Workflow
--------
For each image across the chosen splits:
  1. Build the full-resolution GT mask from the COCO polylines.
  2. Tile + predict with the best checkpoint → full-resolution probability map.
  3. Extract connected components where the probability is high AND the GT
     (dilated slightly to tolerate minor positional drift) is zero.
  4. Apply size + mean-probability filters.
  5. If any candidates survive, save a side-by-side review PNG with one
     row per candidate: ``image (with bbox) | GT crop | probability heatmap``.

The output filename mirrors the user's spec:

    {split}_{stem}_suggested_modification.png

Where ``stem`` is the source filename with the extension stripped, and
``split`` is the dataset split directory name (``train`` / ``valid`` / ``test``).

This script is read-only with respect to the dataset — it never mutates any
labels. The reviewer's job is to look at each saved PNG and decide whether
each highlighted region is a real missed lateral or a model false positive.

Usage
-----
    python -m scripts.find_missed_labels \
        --checkpoint runs/v1_resnet34_bcedice/best.pth \
        --out-dir    results/missed_label_candidates

    # restrict to splits, change confidence threshold, etc.
    python -m scripts.find_missed_labels \
        --checkpoint runs/v1_resnet34_bcedice/best.pth \
        --out-dir    results/missed_label_candidates \
        --splits     valid test \
        --prob-thresh 0.85 \
        --min-area   100 \
        --device     cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Make repo modules importable when running the file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.coco_loader import load_split
from data.polyline_builder import build_polylines
from data.rasterize import rasterize_polylines
from inference.tiled_predict import predict_full_image
from models.unet import build_model
from train import merge_configs

Image.MAX_IMAGE_PIXELS = None


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------


def find_candidates(
    prob_map: np.ndarray,
    gt_mask: np.ndarray,
    *,
    prob_thresh: float = 0.8,
    gt_dilation_px: int = 5,
    min_area: int = 80,
    min_mean_prob: float = 0.85,
) -> list[dict]:
    """Return connected components of high-probability prediction NOT in GT.

    Parameters
    ----------
    prob_map
        Full-image probability map, float32 in [0, 1].
    gt_mask
        Full-image GT mask, uint8 {0, 255}.
    prob_thresh
        Only pixels with ``prob >= prob_thresh`` are considered.
    gt_dilation_px
        Dilate the GT by this many pixels before checking overlap. This
        tolerates minor positional drift (model predicting a 4-px line one
        pixel off from where the label sits is *not* a missed-label
        candidate; it's just normal prediction noise).
    min_area
        Minimum connected-component area in pixels. Default 80 ≈ a 4-px-wide
        line of length 20 px.
    min_mean_prob
        Mean probability across the component must be at least this. Filters
        out components where only a few pixels barely crossed ``prob_thresh``.

    Returns
    -------
    list[dict]
        Each entry has keys ``bbox=(x, y, w, h)``, ``area``, ``mean_prob``.
        Sorted by ``mean_prob`` descending (highest-confidence first).
    """

    pred_high = (prob_map >= prob_thresh).astype(np.uint8)
    gt_bin    = (gt_mask > 0).astype(np.uint8)

    if gt_dilation_px > 0:
        k = gt_dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        gt_compare = cv2.dilate(gt_bin, kernel, iterations=1)
    else:
        gt_compare = gt_bin

    candidate_mask = pred_high & (gt_compare == 0)

    num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(
        candidate_mask, connectivity=8
    )

    results: list[dict] = []
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < min_area:
            continue
        comp_pixels = labels_img == label_id
        mean_prob = float(prob_map[comp_pixels].mean())
        if mean_prob < min_mean_prob:
            continue
        results.append({
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(area),
            "mean_prob": mean_prob,
        })

    results.sort(key=lambda d: d["mean_prob"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _put_label(img: np.ndarray, text: str, scale: float = 0.6) -> np.ndarray:
    """Draw a small black-on-white text label in the top-left corner."""
    h = int(28 * scale / 0.6)
    out = img.copy()
    cv2.rectangle(out, (0, 0), (max(180, len(text) * 11), h), (255, 255, 255), -1)
    cv2.putText(out, text, (6, int(20 * scale / 0.6)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _to_rgb(mask_2d: np.ndarray) -> np.ndarray:
    if mask_2d.dtype != np.uint8:
        mask_2d = (np.clip(mask_2d, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(mask_2d, cv2.COLOR_GRAY2RGB)


def _prob_to_rgb(prob_2d: np.ndarray) -> np.ndarray:
    u8 = (np.clip(prob_2d, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def render_review_panel(
    image: np.ndarray,
    gt_mask: np.ndarray,
    prob_map: np.ndarray,
    candidates: list[dict],
    *,
    pad: int = 80,
    max_candidates: int = 50,
) -> np.ndarray:
    """Compose a stacked RGB PNG with one row per candidate.

    Each row has three columns:
        1. image crop with the candidate bbox in red
        2. GT mask crop (white = labelled lateral, black = nothing)
        3. probability heatmap crop (JET: blue 0 → red 1)

    All rows are padded to the same width so they stack cleanly.
    """

    H, W = image.shape[:2]
    rows: list[np.ndarray] = []

    if not candidates:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    for i, cand in enumerate(candidates[:max_candidates]):
        x, y, w, h = cand["bbox"]
        x0 = max(0, x - pad);            y0 = max(0, y - pad)
        x1 = min(W, x + w + pad);        y1 = min(H, y + h + pad)

        img_crop  = image[y0:y1, x0:x1].copy()
        gt_crop   = gt_mask[y0:y1, x0:x1]
        prob_crop = prob_map[y0:y1, x0:x1]

        # Draw bbox in red on the image crop.
        cv2.rectangle(
            img_crop,
            (x - x0, y - y0),
            (x + w - x0, y + h - y0),
            (255, 0, 0),
            thickness=2,
        )

        img_panel  = _put_label(img_crop, f"img  cand {i:02d}  area={cand['area']}px  p={cand['mean_prob']:.2f}")
        gt_panel   = _put_label(_to_rgb(gt_crop),       "GT mask (dilated check)")
        prob_panel = _put_label(_prob_to_rgb(prob_crop), "predicted probability")

        row = np.concatenate([img_panel, gt_panel, prob_panel], axis=1)
        rows.append(row)

    # Pad rows to equal width and stack vertically.
    max_w = max(r.shape[1] for r in rows)
    padded: list[np.ndarray] = []
    for r in rows:
        if r.shape[1] < max_w:
            pad_w = max_w - r.shape[1]
            r = np.concatenate([r, np.full((r.shape[0], pad_w, 3), 220, dtype=np.uint8)], axis=1)
        padded.append(r)
    return np.concatenate(padded, axis=0)


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------


def _full_gt_mask(
    polylines_by_image: dict[int, list],
    record_lookup: dict[int, "ImageRecord"],
    img_id: int,
    thickness: int,
) -> np.ndarray:
    """Rasterize the GT polylines for one image at native resolution."""
    rec = record_lookup[img_id]
    return rasterize_polylines(
        polylines_by_image.get(img_id, []),
        height=rec.height,
        width=rec.width,
        thickness=thickness,
    )


def process_split(
    split_dir: Path,
    split_name: str,
    model: torch.nn.Module,
    cfg: dict,
    out_dir: Path,
    *,
    prob_thresh: float,
    gt_dilation_px: int,
    min_area: int,
    min_mean_prob: float,
    pad: int,
    max_candidates_per_image: int,
    device: str,
    batch_size: int,
) -> dict:
    """Run the missed-label scan over one COCO split.

    Returns a small stats dict for the caller to print.
    """

    images, chords_by_image = load_split(split_dir)
    polylines_by_image = {
        img_id: build_polylines(chords, merge_radius=cfg["polyline"]["merge_radius"])
        for img_id, chords in chords_by_image.items()
    }
    thickness = int(cfg["rasterize"]["thickness"])
    mean = tuple(cfg["normalization"]["mean"])
    std  = tuple(cfg["normalization"]["std"])

    out_dir.mkdir(parents=True, exist_ok=True)

    images_with_candidates = 0
    total_candidates = 0

    for img_id, rec in tqdm(sorted(images.items()), desc=f"{split_name}", leave=False):
        if not rec.path.is_file():
            tqdm.write(f"[warn] missing image file: {rec.path}")
            continue

        # GT mask at native resolution.
        gt_mask = _full_gt_mask(polylines_by_image, images, img_id, thickness=thickness)

        # Full-image probability map.
        prob_map = predict_full_image(
            model,
            rec.path,
            tile_size=cfg["data"]["tile_size"],
            stride=cfg["data"]["stride"],
            device=device,
            mean=mean,
            std=std,
            batch_size=batch_size,
            show_progress=False,
        )

        candidates = find_candidates(
            prob_map=prob_map,
            gt_mask=gt_mask,
            prob_thresh=prob_thresh,
            gt_dilation_px=gt_dilation_px,
            min_area=min_area,
            min_mean_prob=min_mean_prob,
        )

        if not candidates:
            continue

        images_with_candidates += 1
        total_candidates       += len(candidates)

        # Load the source image once for the review panel.
        with Image.open(rec.path) as im:
            image_np = np.array(im.convert("RGB"), dtype=np.uint8)

        panel = render_review_panel(
            image=image_np,
            gt_mask=gt_mask,
            prob_map=prob_map,
            candidates=candidates,
            pad=pad,
            max_candidates=max_candidates_per_image,
        )

        stem = Path(rec.file_name).stem
        out_path = out_dir / f"{split_name}_{stem}_suggested_modification.png"
        # cv2 wants BGR
        cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    return {
        "n_images":                len(images),
        "images_with_candidates":  images_with_candidates,
        "total_candidates":        total_candidates,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_model(checkpoint_path: Path, cfg: dict, device: str) -> torch.nn.Module:
    model = build_model(cfg["model"])
    ckpt  = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return model.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best.pth (or any model_state_dict checkpoint).")
    parser.add_argument("--out-dir", required=True,
                        help="Where to write the per-image review PNGs.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config",      default="configs/train.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                        help="Which splits to process. Default: all three.")
    parser.add_argument("--device", default="cuda:0",
                        help="Inference device (cuda:0, cpu, mps).")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Tiles per forward pass at inference time.")

    # Candidate-filtering knobs.
    parser.add_argument("--prob-thresh", type=float, default=0.80,
                        help="Min predicted probability to consider a pixel as candidate.")
    parser.add_argument("--gt-dilation-px", type=int, default=5,
                        help="Dilate GT by this many pixels before checking overlap; "
                             "tolerates minor positional drift of the predicted line.")
    parser.add_argument("--min-area", type=int, default=80,
                        help="Min connected-component area in pixels to keep a candidate.")
    parser.add_argument("--min-mean-prob", type=float, default=0.85,
                        help="Min mean probability across the component.")

    # Output-rendering knobs.
    parser.add_argument("--pad", type=int, default=80,
                        help="Pixels of context to pad around each candidate bbox in the PNG.")
    parser.add_argument("--max-candidates-per-image", type=int, default=50,
                        help="Cap on candidate rows per PNG; further candidates are dropped.")
    args = parser.parse_args()

    # Resolve device, falling back to CPU if CUDA isn't available.
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] CUDA not available; falling back from {device!r} to 'cpu'.")
        device = "cpu"

    # Merge configs and locate the dataset root.
    base_cfg  = yaml.safe_load(open(args.base_config))
    train_cfg = yaml.safe_load(open(args.config))
    cfg = merge_configs(base_cfg, train_cfg)
    dataset_root = (Path.cwd() / cfg["data"]["dataset_root"]).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # Load model.
    print(f"[load] checkpoint   = {args.checkpoint}")
    print(f"[load] device       = {device}")
    model = _load_model(Path(args.checkpoint), cfg, device)

    # Resolve and validate splits.
    split_dir_map = {
        "train": dataset_root / cfg["data"]["train_dir"],
        "valid": dataset_root / cfg["data"]["valid_dir"],
        "test":  dataset_root / cfg["data"]["test_dir"],
    }
    for s in args.splits:
        if s not in split_dir_map:
            raise ValueError(f"Unknown split {s!r}. Pick from {list(split_dir_map)}.")

    out_dir = Path(args.out_dir).resolve()
    print(f"[out ] {out_dir}")
    print(f"[args] prob_thresh={args.prob_thresh}  gt_dilation_px={args.gt_dilation_px}  "
          f"min_area={args.min_area}  min_mean_prob={args.min_mean_prob}")

    # Process.
    overall = {"n_images": 0, "images_with_candidates": 0, "total_candidates": 0}
    for split_name in args.splits:
        split_dir = split_dir_map[split_name]
        if not split_dir.is_dir():
            print(f"[warn] split dir not found: {split_dir}  -- skipping")
            continue
        print(f"\n[process] split={split_name}  ({split_dir})")
        stats = process_split(
            split_dir=split_dir,
            split_name=split_name,
            model=model,
            cfg=cfg,
            out_dir=out_dir,
            prob_thresh=args.prob_thresh,
            gt_dilation_px=args.gt_dilation_px,
            min_area=args.min_area,
            min_mean_prob=args.min_mean_prob,
            pad=args.pad,
            max_candidates_per_image=args.max_candidates_per_image,
            device=device,
            batch_size=args.batch_size,
        )
        print(f"          n_images={stats['n_images']}  "
              f"images_with_candidates={stats['images_with_candidates']}  "
              f"total_candidates={stats['total_candidates']}")
        for k, v in stats.items():
            overall[k] = overall[k] + v

    print()
    print(f"[done] {overall['images_with_candidates']}/{overall['n_images']} images "
          f"had candidates;  {overall['total_candidates']} candidate components total.")
    print(f"[done] PNGs written to: {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
