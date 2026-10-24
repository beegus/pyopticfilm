# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end synthetic registration (no OpenCV)."""

from __future__ import annotations

import numpy as np

from pyopticfilm.pass_align import (
    align_pass_to_reference,
    align_pass_to_reference_banded,
    estimate_pass_shift,
)
from pyopticfilm.registration import RegistrationMode, register, register_and_align
from pyopticfilm.registration.resample import warp_shift


def test_estimate_pass_shift_recovers_known_translation():
    rng = np.random.default_rng(1)
    base = rng.integers(1000, 20000, (96, 96, 3), dtype=np.uint16)
    shifted = warp_shift(base, 3.0, -2.0)
    dx, dy = estimate_pass_shift(base, shifted)
    assert abs(dx - 3.0) < 0.6
    assert abs(dy - (-2.0)) < 0.6


def test_register_rejects_featureless():
    blank = np.full((128, 128, 3), 1000, dtype=np.uint16)
    result = register(blank, blank, RegistrationMode.MULTI_EXPOSURE)
    assert not result.accepted or result.shift == (0.0, 0.0)


def test_register_and_align_me_gain_change():
    rng = np.random.default_rng(7)
    short = rng.integers(2000, 12000, (128, 128, 3), dtype=np.uint16)
    long = np.clip(short.astype(np.float64) * 3.0, 0, 65535).astype(np.uint16)
    long = warp_shift(long, 2.0, 1.0)
    aligned, result = register_and_align(
        short, long, RegistrationMode.MULTI_EXPOSURE
    )
    assert result.accepted
    assert abs(result.shift[0] - 2.0) < 0.75
    assert abs(result.shift[1] - 1.0) < 0.75
    assert aligned.shape == short.shape


def test_banded_beats_whole_frame_on_progressive_drift():
    rng = np.random.default_rng(5)
    h, w = 2400, 300
    base = rng.integers(1000, 40000, (h, w, 3), dtype=np.uint16).astype(np.float64)
    true_dy = np.linspace(0.0, 40.0, h)
    row_idx = np.clip(np.arange(h) - np.round(true_dy).astype(int), 0, h - 1)
    moving = base[row_idx].astype(np.uint16)
    reference = base.astype(np.uint16)
    warped, (dx, dy_center) = align_pass_to_reference_banded(reference, moving)
    assert abs(dx) < 1.0
    assert abs(abs(dy_center) - 20.0) < 3.0
    whole, _ = align_pass_to_reference(reference, moving)

    def residual(a, b, y0, y1):
        return np.abs(a[y0:y1].astype(np.float64) - b[y0:y1].astype(np.float64)).mean()

    band = 200
    assert residual(reference, warped, 0, band) < 0.75 * residual(
        reference, whole, 0, band
    )
    assert residual(reference, warped, h - band, h) < 0.75 * residual(
        reference, whole, h - band, h
    )
