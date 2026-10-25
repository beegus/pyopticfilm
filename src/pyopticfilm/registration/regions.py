# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial patch selection and regional shift consensus."""

from __future__ import annotations

import numpy as np

from pyopticfilm.registration.phase import phase_correlate_shift
from pyopticfilm.registration.types import RegionalShift


def select_patch_grid(
    h: int,
    w: int,
    *,
    n_y: int = 3,
    n_x: int = 3,
    patch: int = 256,
    margin: float = 0.08,
) -> list[tuple[int, int, int, int]]:
    """Return ``(y0, x0, y1, x1)`` patches on a spatial grid."""
    patch = min(patch, h, w)
    if patch < 32:
        return []
    y_lo = int(h * margin)
    y_hi = int(h * (1.0 - margin)) - patch
    x_lo = int(w * margin)
    x_hi = int(w * (1.0 - margin)) - patch
    if y_hi < y_lo or x_hi < x_lo:
        y0 = max(0, (h - patch) // 2)
        x0 = max(0, (w - patch) // 2)
        return [(y0, x0, y0 + patch, x0 + patch)]
    ys = np.linspace(y_lo, max(y_lo, y_hi), n_y)
    xs = np.linspace(x_lo, max(x_lo, x_hi), n_x)
    boxes: list[tuple[int, int, int, int]] = []
    for y0f in ys:
        for x0f in xs:
            y0 = round(y0f)
            x0 = round(x0f)
            y0 = min(max(0, y0), h - patch)
            x0 = min(max(0, x0), w - patch)
            boxes.append((y0, x0, y0 + patch, x0 + patch))
    return boxes


def patch_texture(plane: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> float:
    roi = plane[y0:y1, x0:x1]
    return float(roi.std())


def estimate_regional_shifts(
    ref_plane: np.ndarray,
    mov_plane: np.ndarray,
    *,
    min_response: float = 0.05,
    min_texture: float = 1.0,
    n_y: int = 3,
    n_x: int = 3,
    patch: int = 256,
) -> list[RegionalShift]:
    """Independent phase-correlate shifts on a spatial patch grid."""
    h, w = ref_plane.shape[:2]
    boxes = select_patch_grid(h, w, n_y=n_y, n_x=n_x, patch=patch)
    out: list[RegionalShift] = []
    for y0, x0, y1, x1 in boxes:
        if patch_texture(ref_plane, y0, x0, y1, x1) < min_texture:
            out.append(
                RegionalShift(
                    (x0 + x1) / 2.0,
                    (y0 + y1) / 2.0,
                    0.0,
                    0.0,
                    0.0,
                    False,
                )
            )
            continue
        dx, dy, resp = phase_correlate_shift(
            ref_plane[y0:y1, x0:x1],
            mov_plane[y0:y1, x0:x1],
            scale=1.0,
        )
        ok = resp >= min_response
        out.append(
            RegionalShift(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                dx,
                dy,
                resp,
                ok,
            )
        )
    return out


def consensus_translation(
    regions: list[RegionalShift],
) -> tuple[float, float, float, int] | None:
    """Median (dx, dy) of accepted regions + MAD; or None if too few."""
    good = [r for r in regions if r.accepted]
    if len(good) < max(2, (len(regions) + 1) // 3):
        return None
    dxs = np.array([r.dx for r in good], dtype=np.float64)
    dys = np.array([r.dy for r in good], dtype=np.float64)
    dx = float(np.median(dxs))
    dy = float(np.median(dys))
    mad = float(np.median(np.hypot(dxs - dx, dys - dy)))
    return dx, dy, mad, len(good)
