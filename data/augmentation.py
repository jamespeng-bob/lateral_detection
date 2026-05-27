"""Geometric augmentations for (image, mask) tile pairs.

We deliberately restrict ourselves to augmentations that preserve thin lines
exactly — horizontal flip, vertical flip, and rotations by multiples of 90°.
Anything else (elastic transforms, arbitrary rotation, perspective) risks
sub-pixelating 4 px lateral strokes into invisibility.

Image and mask must share the same spatial shape ``(H, W)``.
"""

from __future__ import annotations

import random

import numpy as np


class TileAugmenter:
    """Stateful augmenter with its own RNG so it's reproducible from a seed."""

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_90_prob: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.rotate_90_prob = float(rotate_90_prob)
        self.rng = random.Random(seed)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Augment a single ``(image, mask)`` pair.

        Parameters
        ----------
        image : np.ndarray, shape (H, W, 3), uint8
        mask  : np.ndarray, shape (H, W),    uint8

        Returns
        -------
        (image, mask) with the same dtypes and shapes, both contiguous.
        """
        if self.hflip_prob > 0 and self.rng.random() < self.hflip_prob:
            image = image[:, ::-1]
            mask  = mask[:, ::-1]
        if self.vflip_prob > 0 and self.rng.random() < self.vflip_prob:
            image = image[::-1, :]
            mask  = mask[::-1, :]
        if self.rotate_90_prob > 0 and self.rng.random() < self.rotate_90_prob:
            k = self.rng.randint(1, 3)  # 90, 180, or 270 degrees
            image = np.rot90(image, k=k)
            mask  = np.rot90(mask,  k=k)
        return np.ascontiguousarray(image), np.ascontiguousarray(mask)
