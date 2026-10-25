# SPDX-License-Identifier: GPL-3.0-or-later
"""Geometric models: global translation and Multi-Pass banded Y profiles."""

from __future__ import annotations

import numpy as np

from pyopticfilm.registration.phase import phase_correlate_shift
from pyopticfilm.registration.pyramid import downsample_plane_to_width
from pyopticfilm.registration.types import LineWarpTransform, RegistrationOptions


def shift_is_pathological(dx: float, dy: float, full_w: int, full_h: int, *, max_frac: float = 0.5) -> bool:
    return abs(dx) > max_frac * full_w or abs(dy) > max_frac * full_h


def band_shift_profile(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    options: RegistrationOptions | None = None,
    phase_fn=None,
) -> tuple[float, np.ndarray] | None:
    """Per-row-band ``(dx, dy)`` fit to a per-row ``dy(y)`` line.

    Port of ``pass_align._band_shift_profile``: splits into ``n_bands``
    horizontal strips, phase-correlates each, fits a line, repeatedly drops
    outlier bands, returns ``(dx_median, dy_per_row)`` or ``None``.
    """
    opt = options or RegistrationOptions()
    n_bands = opt.band_count
    ref = np.asarray(reference)
    mov = np.asarray(moving)
    if ref.shape[:2] != mov.shape[:2]:
        return None
    h, w = ref.shape[:2]
    if h < n_bands * opt.band_min_height:
        return None

    correlate = phase_fn or phase_correlate_shift
    band_h = h // n_bands
    centers: list[float] = []
    dxs: list[float] = []
    dys: list[float] = []

    for i in range(n_bands):
        y0 = i * band_h
        y1 = h if i == n_bands - 1 else y0 + band_h
        ref_band = _band_luma(ref[y0:y1], opt.coarse_target_px)
        mov_band = _band_luma(mov[y0:y1], opt.coarse_target_px)
        scale = max(1.0, w / float(opt.coarse_target_px))
        # If already at full width, scale is 1.
        if ref_band.shape[1] == w:
            scale = 1.0
        else:
            scale = float(w) / float(ref_band.shape[1])
        dx, dy, resp = correlate(ref_band, mov_band, scale=scale)
        if resp < opt.min_response:
            continue
        centers.append((y0 + y1) / 2.0)
        dxs.append(dx)
        dys.append(dy)

    if len(centers) < max(3, n_bands // 2):
        return None

    c = np.asarray(centers, dtype=np.float64)
    d = np.asarray(dys, dtype=np.float64)
    dx_arr = np.asarray(dxs, dtype=np.float64)
    slope, intercept = np.polyfit(c, d, 1)
    resid = np.abs(d - (slope * c + intercept))
    while len(c) > 3 and resid.max() > opt.band_outlier_px:
        keep = resid < resid.max()
        if keep.sum() < 3:
            break
        c, d = c[keep], d[keep]
        dx_arr = dx_arr[keep]
        slope, intercept = np.polyfit(c, d, 1)
        resid = np.abs(d - (slope * c + intercept))

    dy_per_row = slope * np.arange(h, dtype=np.float64) + intercept
    if float(np.max(dy_per_row) - np.min(dy_per_row)) > 0.5 * h:
        return None
    dx_med = float(np.median(dx_arr))
    if shift_is_pathological(dx_med, float(np.max(np.abs(dy_per_row))), w, h, max_frac=opt.sanity_max_frac):
        return None
    return dx_med, dy_per_row


def _band_luma(image: np.ndarray, target_w: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        plane = arr.astype(np.float32).mean(axis=2)
    else:
        plane = arr.astype(np.float32)
    coarse, _ = downsample_plane_to_width(plane, target_w)
    return coarse


def line_warp_from_profile(dx: float, dy_per_row: np.ndarray) -> LineWarpTransform:
    return LineWarpTransform(dx=float(dx), dy_per_row=np.asarray(dy_per_row, dtype=np.float64))
