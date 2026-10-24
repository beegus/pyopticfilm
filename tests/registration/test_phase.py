# SPDX-License-Identifier: GPL-3.0-or-later
"""Phase-correlation unit tests."""

from __future__ import annotations

import numpy as np

from pyopticfilm.registration.phase import (
    hanning_window,
    optimal_dft_size,
    phase_correlate,
    phase_correlate_shift,
)
from pyopticfilm.registration.resample import warp_shift


def test_optimal_dft_size_powers():
    assert optimal_dft_size(1024) == 1024
    assert optimal_dft_size(7) == 8
    assert optimal_dft_size(1025) == 1080


def test_hanning_window_shape_and_peak():
    win = hanning_window(32, 48)
    assert win.shape == (32, 48)
    assert win.dtype == np.float32
    assert float(win.max()) <= 1.0 + 1e-5
    assert float(win.min()) >= 0.0
    assert win[16, 24] > win[0, 0]


def test_phase_correlate_shift_recovers_warp():
    rng = np.random.default_rng(1)
    base = rng.integers(1000, 20000, (96, 96), dtype=np.uint16)
    mov = warp_shift(base, 3.0, -2.0)
    dx, dy, resp = phase_correlate_shift(
        base.astype(np.float32), mov.astype(np.float32)
    )
    assert resp > 0.05
    assert abs(dx - 3.0) < 0.5
    assert abs(dy - (-2.0)) < 0.5


def test_phase_correlate_rejects_empty():
    dx, dy, resp, _ = phase_correlate(np.zeros((0, 0)), np.zeros((0, 0)))
    assert (dx, dy, resp) == (0.0, 0.0, 0.0)
