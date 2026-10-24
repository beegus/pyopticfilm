# SPDX-License-Identifier: GPL-3.0-or-later
"""Integer / cubic / banded resample tests."""

from __future__ import annotations

import numpy as np

from pyopticfilm.registration.resample import (
    fill_shift_border,
    warp_cubic,
    warp_integer,
    warp_row_shifts,
    warp_shift,
)


def test_integer_warp_exact_overlap():
    rng = np.random.default_rng(3)
    src = rng.integers(0, 65535, (32, 40, 3), dtype=np.uint16)
    dx, dy = 3, -2
    out = warp_integer(src, dx, dy)
    # out[y,x] = src[y+dy, x+dx] in the interior away from border-fill strips.
    bx, by = abs(dx), abs(dy)
    y0, y1 = by, 32 - by
    x0, x1 = bx, 40 - bx
    assert np.array_equal(
        out[y0:y1, x0:x1], src[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    )


def test_cubic_preserves_constant_field():
    src = np.full((48, 48), 12345, dtype=np.uint16)
    out = warp_cubic(src, 1.3, -0.7)
    assert out.dtype == np.uint16
    assert int(out[10:40, 10:40].min()) >= 12345 - 1
    assert int(out[10:40, 10:40].max()) <= 12345 + 1


def test_cubic_clips_to_uint16_rails():
    src = np.full((32, 32), 65535, dtype=np.uint16)
    out = warp_cubic(src, 0.5, 0.5)
    assert out.max() <= 65535
    assert out.min() >= 0


def test_fill_shift_border_uses_interior():
    out = np.zeros((10, 10), dtype=np.uint16)
    out[:, 2:] = 100
    out[:, 2] = 50
    filled = fill_shift_border(out, dx=2.0, dy=0.0)
    assert np.all(filled[:, :2] == 50)


def test_warp_row_shifts_identity():
    rng = np.random.default_rng(4)
    src = rng.integers(0, 1000, (64, 32), dtype=np.uint16)
    dy = np.zeros(64, dtype=np.float64)
    out = warp_row_shifts(src, 0.0, dy)
    assert np.array_equal(out, src)


def test_warp_shift_zero_is_identity():
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    assert np.array_equal(warp_shift(arr, 0.0, 0.0), arr)
