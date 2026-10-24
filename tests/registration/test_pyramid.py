# SPDX-License-Identifier: GPL-3.0-or-later
"""Block-mean pyramid tests."""

from __future__ import annotations

import numpy as np

from pyopticfilm.registration.pyramid import block_mean, downsample_plane_to_width


def test_block_mean_all_65535_no_overflow():
    img = np.full((64, 64), 65535, dtype=np.uint16)
    out = block_mean(img, 4)
    assert out.shape == (16, 16)
    assert np.allclose(out, 65535.0)


def test_block_mean_trims_remainder():
    img = np.arange(70 * 65, dtype=np.uint16).reshape(70, 65)
    out = block_mean(img, 4)
    assert out.shape == (17, 16)


def test_downsample_plane_to_width_identity_when_small():
    plane = np.ones((40, 50), dtype=np.float32)
    coarse, scale = downsample_plane_to_width(plane, 1024)
    assert scale == 1.0
    assert coarse.shape == (40, 50)
