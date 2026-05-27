"""COCO loader for the lateral_detection dataset.

The Roboflow export uses a keypoint schema where every annotation is a single
two-point chord:

    keypoints = [x1, y1, v1, x2, y2, v2]

The only foreground category we care about is ``lateral`` (``category_id == 1``).
This loader is deliberately self-contained and avoids ``pycocotools`` — the
format is simple JSON and decoding it directly keeps the data layer
transparent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LATERAL_CATEGORY_ID = 1


@dataclass(frozen=True)
class ImageRecord:
    """Metadata for a single image in a split."""

    id: int
    file_name: str
    path: Path
    height: int
    width: int


@dataclass(frozen=True)
class Chord:
    """A single 2-point annotation (one straight line segment)."""

    image_id: int
    p1: np.ndarray  # shape (2,), (x_col, y_row) in pixel coords
    p2: np.ndarray  # shape (2,), (x_col, y_row) in pixel coords
    visibility: tuple[int, int]  # COCO visibility flag per endpoint


def load_split(
    split_dir: Path | str,
    category_id: int = LATERAL_CATEGORY_ID,
    min_visibility: int = 1,
) -> tuple[dict[int, ImageRecord], dict[int, list[Chord]]]:
    """Load a COCO keypoint split.

    Parameters
    ----------
    split_dir
        Directory containing ``_annotations.coco.json`` and the image files.
    category_id
        Only keep annotations with this category id. Default ``1`` = lateral.
    min_visibility
        Drop chords whose endpoints have COCO visibility below this. Both
        endpoints must clear the threshold for the chord to be kept.

    Returns
    -------
    images : dict[int, ImageRecord]
        image id → record
    chords_by_image : dict[int, list[Chord]]
        image id → list of chord annotations (possibly empty)
    """

    split_dir = Path(split_dir)
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {ann_path}")

    with open(ann_path) as f:
        raw = json.load(f)

    images: dict[int, ImageRecord] = {}
    for img in raw.get("images", []):
        images[img["id"]] = ImageRecord(
            id=int(img["id"]),
            file_name=str(img["file_name"]),
            path=split_dir / img["file_name"],
            height=int(img["height"]),
            width=int(img["width"]),
        )

    chords_by_image: dict[int, list[Chord]] = {img_id: [] for img_id in images}
    for ann in raw.get("annotations", []):
        if ann.get("category_id") != category_id:
            continue
        kps = ann.get("keypoints", [])
        if len(kps) < 6:
            continue  # malformed entry
        x1, y1, v1, x2, y2, v2 = kps[:6]
        if int(v1) < min_visibility or int(v2) < min_visibility:
            continue
        chords_by_image.setdefault(ann["image_id"], []).append(
            Chord(
                image_id=int(ann["image_id"]),
                p1=np.array([float(x1), float(y1)], dtype=np.float64),
                p2=np.array([float(x2), float(y2)], dtype=np.float64),
                visibility=(int(v1), int(v2)),
            )
        )

    return images, chords_by_image


def summarize_split(
    images: dict[int, ImageRecord],
    chords_by_image: dict[int, list[Chord]],
) -> dict:
    """Compute a quick numeric summary of a loaded split."""

    sizes = [(rec.width, rec.height) for rec in images.values()]
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    chord_counts = [len(chords_by_image.get(img_id, [])) for img_id in images]

    return {
        "n_images": len(images),
        "n_chords": int(sum(chord_counts)),
        "chords_per_image": {
            "min": int(np.min(chord_counts)) if chord_counts else 0,
            "median": float(np.median(chord_counts)) if chord_counts else 0.0,
            "max": int(np.max(chord_counts)) if chord_counts else 0,
        },
        "image_size": {
            "min_width":  int(np.min(widths))  if widths  else 0,
            "max_width":  int(np.max(widths))  if widths  else 0,
            "min_height": int(np.min(heights)) if heights else 0,
            "max_height": int(np.max(heights)) if heights else 0,
        },
    }
