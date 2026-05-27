"""Rasterize polylines into a binary segmentation mask."""

from __future__ import annotations

import cv2
import numpy as np

from .polyline_builder import Polyline


def rasterize_polylines(
    polylines: list[Polyline],
    height: int,
    width: int,
    thickness: int = 4,
    value: int = 255,
) -> np.ndarray:
    """Paint polylines into a ``uint8`` mask of shape ``(height, width)``.

    Parameters
    ----------
    polylines
        Polylines to rasterize. Empty list returns an all-zero mask.
    height, width
        Output mask shape in pixels (typically the source image resolution).
    thickness
        Stroke thickness in pixels. Default ``4`` matches the typical
        lateral-line stroke width on the blueprints.
    value
        Foreground value used for the painted pixels (default ``255``).

    Returns
    -------
    np.ndarray
        Mask of shape ``(height, width)``, dtype ``uint8``.
    """

    mask = np.zeros((height, width), dtype=np.uint8)
    for pl in polylines:
        pts = pl.points.astype(np.int32).reshape(-1, 1, 2)
        if pts.shape[0] < 2:
            continue
        cv2.polylines(
            mask,
            [pts],
            isClosed=False,
            color=int(value),
            thickness=int(thickness),
            lineType=cv2.LINE_8,
        )
    return mask
