# SPDX-License-Identifier: GPL-3.0-or-later
"""RGB↔IR coarse seed merge: reject the larger cross-modal lobe per axis."""

from __future__ import annotations

from pyopticfilm.registration.engine import _rgb_ir_keep_seed_axes, _rgb_ir_merge_seed_axes


def test_merge_seed_prefers_luma_when_axes_agree():
    sx, sy = _rgb_ir_merge_seed_axes(-0.40, -0.12, -0.35, -0.05)
    assert abs(sx - (-0.35)) < 1e-9
    assert abs(sy - (-0.05)) < 1e-9


def test_merge_seed_prefers_smaller_abs_when_x_disagrees():
    # 1800 IR case: luma pulled to −1.95, gradient nearer true −1.5.
    sx, sy = _rgb_ir_merge_seed_axes(-1.43, -0.13, -1.95, -0.01)
    assert abs(sx - (-1.43)) < 1e-9
    assert abs(sy - (-0.01)) < 1e-9


def test_merge_seed_prefers_smaller_abs_when_luma_is_better():
    # 3600 IR case: luma residual small, gradient still biased.
    sx, sy = _rgb_ir_merge_seed_axes(-1.03, 0.09, -0.27, -0.08)
    assert abs(sx - (-0.27)) < 1e-9
    assert abs(sy - (-0.08)) < 1e-9


def test_keep_seed_axis_when_refine_grows_magnitude():
    dx, dy = _rgb_ir_keep_seed_axes(-1.43, -0.01, -1.90, -0.02)
    assert abs(dx - (-1.43)) < 1e-9
    assert abs(dy - (-0.02)) < 1e-9
