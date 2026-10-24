# SPDX-License-Identifier: GPL-3.0-or-later
"""RGB↔IR refine: fractional Y recovery on tone-mapped synthetic IR."""

from __future__ import annotations

import numpy as np

from pyopticfilm.registration import RegistrationMode, register
from pyopticfilm.registration.resample import warp_shift


def test_rgb_ir_recovers_fractional_y_shift():
    """Tone-mapped mono 'IR' with known fractional Y; refine should land ≤0.15 px."""
    rng = np.random.default_rng(11)
    h, w = 512, 640
    rgb = rng.integers(2000, 45000, (h, w, 3), dtype=np.uint16)
    # IR-like: nonlinear map of luma (cross-modal intensity, shared structure).
    luma = (rgb[:, :, 0].astype(np.float64) + 2 * rgb[:, :, 1] + rgb[:, :, 2]) * 0.25
    ir = np.clip(np.sqrt(luma / 65535.0) * 65535.0, 0, 65535).astype(np.uint16)

    true_dx, true_dy = 2.35, -1.40
    mis = warp_shift(ir, true_dx, true_dy)
    result = register(rgb, mis, RegistrationMode.RGB_IR)
    assert result.accepted
    dx, dy = result.shift
    assert abs(dx - true_dx) <= 0.15, f"dx={dx}"
    assert abs(dy - true_dy) <= 0.15, f"dy={dy}"


def test_rgb_ir_near_zero_on_aligned_pair():
    rng = np.random.default_rng(12)
    # Large enough that refine runs (engine skips refine below 512²).
    h, w = 640, 768
    rgb = rng.integers(3000, 40000, (h, w, 3), dtype=np.uint16)
    luma = (rgb[:, :, 0].astype(np.float64) + 2 * rgb[:, :, 1] + rgb[:, :, 2]) * 0.25
    ir = np.clip(luma * 0.9 + 1000, 0, 65535).astype(np.uint16)
    result = register(rgb, ir, RegistrationMode.RGB_IR)
    assert result.accepted
    dx, dy = result.shift
    assert abs(dx) <= 0.15, f"dx={dx}"
    assert abs(dy) <= 0.15, f"dy={dy}"
