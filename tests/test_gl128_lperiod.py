# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 ``LPERIOD`` (``0x28``) follows SilverFast's window rule on every model.

Values are SilverFast's own writes from USB captures; the window is the pass's
``END - STR`` in native pixels. Image passes and shading strips of a scan use
the same value.
"""

from __future__ import annotations

import pytest

from pyopticfilm.device.gl128_common import lperiod_for_line, lperiod_for_window
from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE
from pyopticfilm.scan.calib_gl128 import AFE_STRIP_PIXELS, AFE_WIDE_PIXELS
from pyopticfilm.scan.geometry import compute_geometry

GL128_MODELS = (MODEL_8200I_SE, MODEL_8100_V2, MODEL_8300I_SE)

#: (ppi, span, SilverFast LPERIOD) from captures.
SF_WINDOWS = [
    # Full frames: 8100 V2 06_ppi_ladder / 04_color_7200, 8300i SE PPI ladder,
    # 8200i SE Alignment_test / sf_repeat (1800) and session 03 (1200).
    (600, 10368, 11067),
    (720, 10360, 11110),
    (900, 10368, 11175),
    (1200, 10368, 11283),
    (1440, 10370, 11369),
    (1800, 10368, 11499),
    (2400, 10368, 11715),
    (3600, 10368, 13443),
    (7200, 10368, 16035),
    # 8200i SE session 13 ladder crop (STR 386).
    (600, 10224, 11064),
    (720, 10220, 11106),
    (900, 10224, 11170),
    (1200, 10224, 11277),
    (1440, 10224, 11362),
    (1800, 10224, 11490),
    (2400, 10224, 11703),
    (3600, 10224, 13407),
    (7200, 10224, 15963),
    # 8200i SE session 04 crop and session 14 crop (1800).
    (1800, 9912, 11470),
    (1800, 9960, 11473),
]


@pytest.mark.parametrize(("ppi", "span", "expected"), SF_WINDOWS)
def test_lperiod_matches_silverfast_window(ppi: int, span: int, expected: int) -> None:
    assert lperiod_for_window(ppi, span) == expected
    for model in GL128_MODELS:
        assert model.line_period_for(ppi, span=span) == expected


@pytest.mark.parametrize(
    ("scan_dpi", "narrow", "wide"),
    [(600, 10979, 13445), (2400, 10979, 13445), (3600, 11107, 16040), (7200, 11107, 16040)],
)
def test_lperiod_matches_silverfast_afe_strips(scan_dpi: int, narrow: int, wide: int) -> None:
    assert lperiod_for_line(AFE_STRIP_PIXELS, scan_dpi) == narrow
    assert lperiod_for_line(AFE_WIDE_PIXELS, scan_dpi) == wide


@pytest.mark.parametrize("model", GL128_MODELS, ids=lambda m: m.name)
def test_default_full_frame_uses_the_full_frame_value(model) -> None:
    for ppi, expected in ((1200, 11283), (1800, 11499), (3600, 13443), (7200, 16035)):
        geo = compute_geometry(ppi, model=model)
        assert model.line_period_for(ppi, span=geo.pixel_endx - geo.pixel_startx) == expected
        assert model.lperiod_by_dpi[ppi] == expected


def test_crops_scale_with_the_window() -> None:
    full = compute_geometry(1800, model=MODEL_8200I_SE)
    crop = compute_geometry(1800, model=MODEL_8200I_SE, area=(0.2, 0.0, 0.8, 1.0))
    assert crop.pixel_endx - crop.pixel_startx < full.pixel_endx - full.pixel_startx
    assert crop.exposure_lperiod < full.exposure_lperiod
    assert crop.exposure_lperiod == lperiod_for_window(1800, crop.pixel_endx - crop.pixel_startx)
