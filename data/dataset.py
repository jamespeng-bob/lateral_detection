"""Tile-based PyTorch dataset for lateral_detection.

Strategy
--------
- Annotations are loaded once at ``__init__`` and converted into one
  ``list[Polyline]`` per image via :mod:`data.polyline_builder`.
- Images stay on disk at native resolution. Each ``__getitem__`` opens the
  file with PIL, crops the required tile, and rasterizes the affected
  polylines into a tile-local mask. We do **not** preload the full-resolution
  images (a 14400 × 10800 RGB image is ~470 MB; the train split would be tens
  of gigabytes).

Modes
-----
``random``
    Pick a random training image, then a random point along a random
    polyline, then a tile centered (with jitter) on that point. Foreground
    pixels are guaranteed in every tile. Use for training.
``grid``
    Deterministic sliding-window over every image with the configured
    stride. Use for full-image evaluation.
``pos_only_grid``
    Same as ``grid`` but tiles with zero foreground are skipped at
    ``__init__`` time. Use for validation feedback during training.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import TileAugmenter
from .coco_loader import load_split
from .polyline_builder import Polyline, build_polylines

# Allow PIL to open the very large plan images.
Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class TileSample:
    """One training/validation tile."""

    image: torch.Tensor          # [3, T, T], normalized float32
    mask:  torch.Tensor          # [1, T, T], float32 in {0, 1}
    img_id: int
    tile_origin: tuple[int, int] # (row, col) in original image pixels


class TileDataset(Dataset):
    """Tile-level binary segmentation dataset for lateral pipes.

    Parameters
    ----------
    split_dir
        Directory containing ``_annotations.coco.json`` and the images.
    tile_size, stride
        Square tile side length, and sliding-window stride for grid modes.
    mode
        ``random`` | ``grid`` | ``pos_only_grid`` (see module docstring).
    merge_radius
        Pixel radius used by polyline reconstruction (default 10 px).
    thickness
        Stroke thickness in pixels when rasterizing polylines into the mask.
    augmenter
        Optional :class:`TileAugmenter` applied to ``(image, mask)``.
    mean, std
        Per-channel normalization stats applied to the image tensor.
    samples_per_epoch_per_image
        Used only in ``random`` mode: epoch length =
        ``samples_per_epoch_per_image * num_images_with_polylines``.
    jitter_frac
        In ``random`` mode, jitter the tile center by up to
        ``jitter_frac * tile_size`` pixels in each direction so foreground
        isn't always at the exact tile center. Default ``0.25``.
    seed
        Optional seed for the ``random``-mode sampler.
    """

    def __init__(
        self,
        split_dir: Path | str,
        tile_size: int = 1024,
        stride: int = 768,
        mode: str = "random",
        merge_radius: float = 10.0,
        thickness: int = 4,
        augmenter: Optional[TileAugmenter] = None,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std:  tuple[float, float, float] = (0.229, 0.224, 0.225),
        samples_per_epoch_per_image: int = 8,
        jitter_frac: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        if mode not in ("random", "grid", "pos_only_grid"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.split_dir = Path(split_dir)
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.mode = mode
        self.merge_radius = float(merge_radius)
        self.thickness = int(thickness)
        self.augmenter = augmenter
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std  = np.array(std,  dtype=np.float32).reshape(3, 1, 1)
        self.jitter_frac = float(jitter_frac)
        self.rng = random.Random(seed)

        # Load annotations and pre-compute polylines.
        self.images, self.chords_by_image = load_split(self.split_dir)
        self.polylines: dict[int, list[Polyline]] = {
            img_id: build_polylines(
                self.chords_by_image.get(img_id, []),
                merge_radius=self.merge_radius,
            )
            for img_id in self.images
        }

        if mode == "random":
            n_with_polys = sum(1 for p in self.polylines.values() if p)
            self._length = max(1, samples_per_epoch_per_image * max(1, n_with_polys))
            self._grid: Optional[list[tuple[int, int, int]]] = None
        else:
            self._grid = self._build_grid(skip_empty=(mode == "pos_only_grid"))
            self._length = len(self._grid)

    # ------------------------------------------------------------------
    # Standard Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> TileSample:
        if self.mode == "random":
            img_id, row, col = self._sample_random_tile()
        else:
            assert self._grid is not None
            img_id, row, col = self._grid[idx % len(self._grid)]

        image_np, mask_np = self._extract_tile(img_id, row, col)

        if self.augmenter is not None:
            image_np, mask_np = self.augmenter(image_np, mask_np)

        # Normalize image: HWC uint8 → CHW float32, ImageNet stats.
        image_t = (image_np.astype(np.float32) / 255.0).transpose(2, 0, 1)
        image_t = (image_t - self.mean) / self.std

        mask_t = (mask_np.astype(np.float32) / 255.0)[None, :, :]  # [1, T, T]

        return TileSample(
            image=torch.from_numpy(image_t.copy()),
            mask=torch.from_numpy(mask_t.copy()),
            img_id=int(img_id),
            tile_origin=(int(row), int(col)),
        )

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_grid(self, skip_empty: bool) -> list[tuple[int, int, int]]:
        """Build the (img_id, row, col) index for grid / pos_only_grid mode."""
        grid: list[tuple[int, int, int]] = []

        # For pos_only_grid we need to know which tiles contain foreground.
        # Rasterize the full-image mask once per image (O(H * W)) and reuse.
        full_masks: dict[int, np.ndarray] = {}
        if skip_empty:
            for img_id, rec in self.images.items():
                full_masks[img_id] = self._rasterize_full(img_id, rec.height, rec.width)

        for img_id, rec in self.images.items():
            n_rows = max(1, math.ceil(max(rec.height - self.tile_size, 0) / self.stride) + 1)
            n_cols = max(1, math.ceil(max(rec.width  - self.tile_size, 0) / self.stride) + 1)
            full = full_masks.get(img_id) if skip_empty else None
            for tr in range(n_rows):
                for tc in range(n_cols):
                    row = min(tr * self.stride, max(0, rec.height - self.tile_size))
                    col = min(tc * self.stride, max(0, rec.width  - self.tile_size))
                    if full is not None:
                        sub = full[row : row + self.tile_size, col : col + self.tile_size]
                        if not sub.any():
                            continue
                    grid.append((img_id, row, col))
        return grid

    def _rasterize_full(self, img_id: int, height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        for pl in self.polylines.get(img_id, []):
            pts_i = pl.points.astype(np.int32).reshape(-1, 1, 2)
            if pts_i.shape[0] < 2:
                continue
            cv2.polylines(
                mask, [pts_i], isClosed=False, color=255,
                thickness=self.thickness, lineType=cv2.LINE_8,
            )
        return mask

    # ------------------------------------------------------------------
    # Random sampling
    # ------------------------------------------------------------------

    def _sample_random_tile(self) -> tuple[int, int, int]:
        """Pick a tile centered (with jitter) on a random GT polyline point."""
        ids_with_polys = [iid for iid, p in self.polylines.items() if p]
        if not ids_with_polys:
            # Fallback: any image, uniform random origin.
            img_id = self.rng.choice(list(self.images.keys()))
            rec = self.images[img_id]
            row = self.rng.randint(0, max(0, rec.height - self.tile_size))
            col = self.rng.randint(0, max(0, rec.width  - self.tile_size))
            return img_id, row, col

        img_id = self.rng.choice(ids_with_polys)
        rec = self.images[img_id]
        polys = self.polylines[img_id]
        pl = self.rng.choice(polys)
        pt = pl.points[self.rng.randint(0, len(pl.points) - 1)]
        col_center = int(round(pt[0]))
        row_center = int(round(pt[1]))

        half = self.tile_size // 2
        jitter = int(self.jitter_frac * self.tile_size)
        rj = self.rng.randint(-jitter, jitter) if jitter > 0 else 0
        cj = self.rng.randint(-jitter, jitter) if jitter > 0 else 0
        row = max(0, min(max(0, rec.height - self.tile_size), row_center - half + rj))
        col = max(0, min(max(0, rec.width  - self.tile_size), col_center - half + cj))
        return img_id, row, col

    # ------------------------------------------------------------------
    # Tile extraction
    # ------------------------------------------------------------------

    def _extract_tile(self, img_id: int, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
        """Read a tile from disk (RGB) and rasterize the per-tile mask."""
        rec = self.images[img_id]
        T = self.tile_size

        with Image.open(rec.path) as im:
            im = im.convert("RGB")
            # PIL crop uses (left, upper, right, lower) → (col0, row0, col1, row1).
            # Out-of-bounds regions are padded with zeros by PIL.
            crop = im.crop((col, row, col + T, row + T))
        image_np = np.array(crop, dtype=np.uint8)  # [T, T, 3]

        mask_np = self._rasterize_tile(img_id, row, col)

        # Safety pad to (T, T) if PIL gave back a smaller crop (rare).
        if image_np.shape[:2] != (T, T):
            padded = np.zeros((T, T, 3), dtype=np.uint8)
            h, w = image_np.shape[:2]
            padded[:h, :w] = image_np
            image_np = padded
        if mask_np.shape != (T, T):
            padded_m = np.zeros((T, T), dtype=np.uint8)
            h, w = mask_np.shape
            padded_m[:h, :w] = mask_np
            mask_np = padded_m

        return image_np, mask_np

    def _rasterize_tile(self, img_id: int, row: int, col: int) -> np.ndarray:
        """Rasterize only the polylines into a tile-local mask.

        Polylines are translated by ``(-col, -row)`` so original-image
        coordinates become tile-local; cv2 clips against the canvas bounds.
        """
        T = self.tile_size
        mask = np.zeros((T, T), dtype=np.uint8)
        for pl in self.polylines.get(img_id, []):
            pts = pl.points.copy()
            pts[:, 0] -= col
            pts[:, 1] -= row
            pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
            if pts_i.shape[0] < 2:
                continue
            cv2.polylines(
                mask, [pts_i], isClosed=False, color=255,
                thickness=self.thickness, lineType=cv2.LINE_8,
            )
        return mask


# ---------------------------------------------------------------------------
# DataLoader collation + worker initialization
# ---------------------------------------------------------------------------

def collate_tile_samples(samples: list[TileSample]) -> dict[str, torch.Tensor]:
    """Custom collate that stacks :class:`TileSample` instances into a dict."""
    return {
        "image": torch.stack([s.image for s in samples], dim=0),
        "mask":  torch.stack([s.mask  for s in samples], dim=0),
        "img_ids": torch.tensor([s.img_id for s in samples], dtype=torch.long),
        "tile_origins": torch.tensor([s.tile_origin for s in samples], dtype=torch.long),
    }


def worker_init_fn(worker_id: int) -> None:
    """Re-seed per-worker RNGs after fork (Linux) or spawn (macOS).

    On Linux ``DataLoader`` workers fork from the main process and inherit the
    full Python ``random``, NumPy, and our :class:`TileDataset`'s ``self.rng``
    state. Without this hook every worker would produce the same sequence of
    "random" tiles, collapsing effective batch diversity. Each epoch torch
    rolls a fresh base seed, so this also re-seeds across epochs.
    """
    import random as _py_random

    import numpy as _np

    info = torch.utils.data.get_worker_info()
    base = torch.initial_seed() % (2**31)
    seed = (base + worker_id) % (2**31)
    _py_random.seed(seed)
    _np.random.seed(seed)
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = _py_random.Random(seed)
