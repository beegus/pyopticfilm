# SPDX-License-Identifier: GPL-3.0-or-later
"""Integer block-mean pyramid (AREA-like, deterministic)."""

from __future__ import annotations

import numpy as np


def block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping ``factor×factor`` blocks to float32.

    Trims trailing rows/cols not divisible by ``factor``. Overflow-safe for
    ``uint16`` via ``uint64`` accumulation.
    """
    if factor <= 1:
        return np.asarray(image, dtype=np.float32)
    arr = np.asarray(image)
    h = (arr.shape[0] // factor) * factor
    w = (arr.shape[1] // factor) * factor
    if h == 0 or w == 0:
        return np.asarray(arr, dtype=np.float32)
    x = arr[:h, :w]
    if x.ndim == 2:
        blocks = x.reshape(h // factor, factor, w // factor, factor)
        summed = blocks.sum(axis=(1, 3), dtype=np.float64)
        return (summed / float(factor * factor)).astype(np.float32)
    # RGB: reshape once; mean over block + channel axes.
    c = x.shape[2]
    out_h, out_w = h // factor, w // factor
    blocks = x.reshape(out_h, factor, out_w, factor, c)
    return blocks.mean(axis=(1, 3, 4), dtype=np.float64).astype(np.float32)


def downsample_to_width(image: np.ndarray, target_w: int) -> tuple[np.ndarray, float]:
    """Block-mean so the long side / width is near ``target_w``.

    Returns ``(plane, scale)`` where ``scale = full_w / coarse_w`` (≈ factor).
    """
    arr = np.asarray(image)
    _h, w = arr.shape[:2]
    if w <= target_w:
        if arr.ndim == 3:
            return arr.astype(np.float32).mean(axis=2), 1.0
        return arr.astype(np.float32), 1.0
    factor = max(1, round(w / target_w))
    # Prefer integer factors that land near target_w.
    while factor > 1 and (w // factor) < max(32, target_w // 4):
        factor -= 1
    coarse = block_mean(arr, factor)
    if arr.ndim == 3 and coarse.ndim == 3:
        coarse = coarse.mean(axis=2)
    scale = float(w) / float(coarse.shape[1])
    return coarse, scale


def downsample_plane_to_width(plane: np.ndarray, target_w: int) -> tuple[np.ndarray, float]:
    """Like :func:`downsample_to_width` but for a 2-D float/uint plane."""
    arr = np.asarray(plane)
    if arr.ndim != 2:
        raise ValueError("downsample_plane_to_width expects a 2-D plane")
    _h, w = arr.shape
    if w <= target_w:
        return arr.astype(np.float32), 1.0
    factor = max(1, round(w / target_w))
    while factor > 1 and (w // factor) < max(32, target_w // 4):
        factor -= 1
    coarse = block_mean(arr, factor)
    scale = float(w) / float(coarse.shape[1])
    return coarse, scale
