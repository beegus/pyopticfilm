# SPDX-License-Identifier: GPL-3.0-or-later
"""Held-out residual checks and accept/reject helpers."""

from __future__ import annotations

import numpy as np

from pyopticfilm.registration.preprocess import structural_luminance
from pyopticfilm.registration.types import RegistrationFailure, RegistrationMode


def luma_residual(a: np.ndarray, b: np.ndarray, *, inset: float = 0.1) -> float:
    """Mean absolute luma difference on a central inset."""
    ha, wa = a.shape[:2]
    y0, y1 = int(ha * inset), int(ha * (1.0 - inset))
    x0, x1 = int(wa * inset), int(wa * (1.0 - inset))
    if y1 <= y0 or x1 <= x0:
        la = structural_luminance(a)
        lb = structural_luminance(b)
        return float(np.mean(np.abs(la - lb)))
    la = structural_luminance(a[y0:y1, x0:x1])
    lb = structural_luminance(b[y0:y1, x0:x1])
    return float(np.mean(np.abs(la.astype(np.float64) - lb.astype(np.float64))))


def should_snap_to_integer(
    dx: float,
    dy: float,
    *,
    eps: float = 0.08,
    regional_frac_consistent: bool = True,
) -> bool:
    if not regional_frac_consistent:
        return False
    return abs(dx - round(dx)) < eps and abs(dy - round(dy)) < eps


def classify_failure(
    *,
    response: float,
    min_response: float,
    pathological: bool,
    regional_mad: float | None,
    shape_ok: bool,
    mode: RegistrationMode,
) -> RegistrationFailure:
    if not shape_ok:
        return RegistrationFailure.SHAPE_MISMATCH
    if pathological:
        return RegistrationFailure.PATHLOGICAL_SHIFT
    if response < min_response:
        return RegistrationFailure.LOW_TEXTURE
    if regional_mad is not None and regional_mad > 2.0:
        if mode is RegistrationMode.RGB_IR:
            return RegistrationFailure.CROSS_MODAL_MISMATCH
        return RegistrationFailure.REGIONAL_DISAGREEMENT
    return RegistrationFailure.NONE
