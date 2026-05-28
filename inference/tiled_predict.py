"""Tile-based full-image inference for the lateral_detection UNet.

Plans are too large to fit in GPU memory (typically 7200 x 10800 to
10800 x 14400). We tile, predict each tile, and average overlapping
predictions back into a full-resolution probability map.

Reusable for:
  - Finding missed labels (scripts/find_missed_labels.py)
  - Whole-image Dice evaluation
  - Production inference
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Plans can be very large; lift PIL's safety limit.
Image.MAX_IMAGE_PIXELS = None


def _tile_origins(height: int, width: int, tile: int, stride: int) -> list[tuple[int, int]]:
    """Sliding-window origins. Always includes the right/bottom edge.

    Each origin is a ``(row, col)`` top-left corner. Tiles that extend past
    the image edge are right/bottom-padded later by ``predict_full_image``.
    """
    def axis(size: int) -> list[int]:
        if size <= tile:
            return [0]
        starts = list(range(0, size - tile + 1, stride))
        if starts[-1] != size - tile:
            starts.append(size - tile)
        return starts

    return [(r, c) for r in axis(height) for c in axis(width)]


def predict_full_image(
    model: nn.Module,
    image: np.ndarray | Image.Image | str | Path,
    *,
    tile_size: int = 1024,
    stride: int = 768,
    device: str = "cuda:0",
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std:  tuple[float, float, float] = (0.229, 0.224, 0.225),
    batch_size: int = 4,
    show_progress: bool = True,
) -> np.ndarray:
    """Run tile-based inference and return a full-resolution probability map.

    Parameters
    ----------
    model
        A trained binary segmentation model returning logits of shape
        ``(B, 1, T, T)`` for input ``(B, 3, T, T)``. The model is moved to
        ``device`` and put in ``eval`` mode here.
    image
        One of: numpy ``uint8`` HxWx3 RGB array, ``PIL.Image``, file path
        (``str`` or ``Path``).
    tile_size, stride
        Sliding-window parameters. Default matches ``configs/train.yaml``.
    device
        Inference device (``cuda:0``, ``cpu``, etc.).
    mean, std
        Per-channel ImageNet normalization, matching training-time stats in
        ``configs/train.yaml``.
    batch_size
        Tiles per forward pass. 4 is conservative for a 1024-px tile; bump
        if VRAM allows.
    show_progress
        Show a tqdm bar across batches.

    Returns
    -------
    np.ndarray
        Probability map of shape ``(H, W)``, ``float32`` in ``[0, 1]``.
    """

    # ── Load image as uint8 HxWx3 numpy ─────────────────────────────────
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            image = np.array(im.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB array, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    H, W = image.shape[:2]

    # ── Pre-build normalization tensors on the right device ────────────
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1, 3, 1, 1)

    # ── Prepare model ───────────────────────────────────────────────────
    model = model.to(device)
    model.eval()

    # ── Prepare accumulators (CPU float32) ──────────────────────────────
    prob_sum     = np.zeros((H, W), dtype=np.float32)
    pixel_counter = np.zeros((H, W), dtype=np.float32)

    # ── Compute tile origins, then batch them ──────────────────────────
    origins = _tile_origins(H, W, tile_size, stride)
    batches = [origins[i:i + batch_size] for i in range(0, len(origins), batch_size)]

    iterator = batches
    if show_progress:
        iterator = tqdm(batches, desc="predict", leave=False)

    with torch.no_grad():
        for batch_origins in iterator:
            # Build the batch of tiles, padding edge tiles with zeros.
            B = len(batch_origins)
            tile_np = np.zeros((B, tile_size, tile_size, 3), dtype=np.uint8)
            for i, (r, c) in enumerate(batch_origins):
                h = min(tile_size, H - r)
                w = min(tile_size, W - c)
                tile_np[i, :h, :w] = image[r:r + h, c:c + w]

            # → torch tensor on device, normalized
            tile_t = torch.from_numpy(tile_np).to(device, non_blocking=True)
            tile_t = tile_t.permute(0, 3, 1, 2).contiguous().float() / 255.0
            tile_t = (tile_t - mean_t) / std_t

            logits = model(tile_t)                              # [B, 1, T, T]
            probs  = torch.sigmoid(logits[:, 0]).cpu().numpy()  # [B, T, T] float32

            # Splat predictions back into the full-resolution accumulator.
            for (r, c), prob in zip(batch_origins, probs):
                h = min(tile_size, H - r)
                w = min(tile_size, W - c)
                prob_sum[r:r + h, c:c + w]     += prob[:h, :w]
                pixel_counter[r:r + h, c:c + w] += 1.0

    # ── Average overlapping predictions ─────────────────────────────────
    np.maximum(pixel_counter, 1e-6, out=pixel_counter)
    return prob_sum / pixel_counter
