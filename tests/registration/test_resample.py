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
    # Remap: out[y,x] = src[y-dy, x-dx] away from IR-safe border-fill strips.
    # fill uses ~2*|shift| as the donor, so keep a deeper interior crop.
    y0, y1 = 2 * abs(dy), 32 - 2 * abs(dy)
    x0, x1 = 2 * abs(dx), 40 - 2 * abs(dx)
    assert np.array_equal(out[y0:y1, x0:x1], src[y0 - dy : y1 - dy, x0 - dx : x1 - dx])


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
    out[:, 4] = 50  # donor at 2*|dx|
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


def test_register_apply_round_trip_reduces_residual():
    """warp_shift create with -S; estimate ≈ S; apply(+S) aligns (remap)."""
    from pyopticfilm.registration import RegistrationMode, register_and_align
    from pyopticfilm.registration.validate import luma_residual

    rng = np.random.default_rng(9)
    ref = rng.integers(2000, 40000, (256, 320, 3), dtype=np.uint16)
    true = (4.0, -3.0)
    # Inverse of the apply shift: mis is what apply(+true) undoes.
    mis = warp_shift(ref, -true[0], -true[1])
    before = luma_residual(ref, mis)
    aligned, result = register_and_align(ref, mis, RegistrationMode.MULTI_EXPOSURE)
    assert result.accepted
    assert abs(result.shift[0] - true[0]) < 0.5
    assert abs(result.shift[1] - true[1]) < 0.5
    after = luma_residual(ref, aligned)
    assert after < 0.25 * before, (before, after)
