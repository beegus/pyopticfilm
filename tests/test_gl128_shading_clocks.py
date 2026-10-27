# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 shading-strip clocks (``0x2B``, ``0xA5``, ``0xAB``) follow SilverFast's window rule.

Values are SilverFast's own writes from USB captures, one row per distinct
(strip, ppi, window span): 8200i SE sessions 03-14 and 8100 V2 sessions 03-07
(66 strips), and the 8300i SE PPI-ladder dark strips. The 8300i SE white strip
uses its own clocks and keeps its measured table.
"""

from __future__ import annotations

import pytest

from pyopticfilm.device.gl128_common import shading_clocks_for_window
from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE

#: (strip, ppi, span, 0x2B, 0xA5, 0xAB) on the 8200i SE and 8100 V2.
SF_STRIPS_SE_V2 = [
    ("dark", 600, 10224, 0x02, 0x01, 0x30),
    ("dark", 600, 10368, 0x02, 0x01, 0x30),
    ("dark", 720, 10220, 0x03, 0x01, 0x30),
    ("dark", 720, 10360, 0x03, 0x01, 0x30),
    ("dark", 900, 10224, 0x03, 0x01, 0x30),
    ("dark", 900, 10368, 0x03, 0x01, 0x30),
    ("dark", 1200, 10224, 0x04, 0x01, 0x30),
    ("dark", 1200, 10368, 0x04, 0x01, 0x30),
    ("dark", 1440, 10220, 0x05, 0x01, 0x30),
    ("dark", 1440, 10370, 0x05, 0x01, 0x30),
    ("dark", 1800, 6552, 0x04, 0x01, 0x30),
    ("dark", 1800, 9912, 0x06, 0x01, 0x30),
    ("dark", 1800, 9960, 0x06, 0x01, 0x30),
    ("dark", 1800, 10224, 0x06, 0x01, 0x30),
    ("dark", 1800, 10344, 0x06, 0x01, 0x30),
    ("dark", 1800, 10368, 0x06, 0x01, 0x30),
    ("dark", 2400, 10224, 0x08, 0x01, 0x30),
    ("dark", 2400, 10368, 0x08, 0x01, 0x30),
    ("dark", 3600, 9912, 0x0B, 0x01, 0x30),
    ("dark", 3600, 10224, 0x0C, 0x01, 0x30),
    ("dark", 3600, 10368, 0x0C, 0x01, 0x30),
    ("dark", 7200, 10224, 0x17, 0x01, 0x30),
    ("dark", 7200, 10368, 0x17, 0x01, 0x30),
    ("white", 600, 10224, 0x01, 0x02, 0x02),
    ("white", 600, 10368, 0x01, 0x02, 0x02),
    ("white", 720, 10220, 0x01, 0x02, 0x02),
    ("white", 720, 10360, 0x01, 0x02, 0x02),
    ("white", 900, 10224, 0x01, 0x02, 0x02),
    ("white", 900, 10368, 0x01, 0x02, 0x02),
    ("white", 1200, 10224, 0x02, 0x02, 0x02),
    ("white", 1200, 10368, 0x02, 0x02, 0x02),
    ("white", 1440, 10220, 0x02, 0x02, 0x02),
    ("white", 1440, 10370, 0x02, 0x02, 0x02),
    ("white", 1800, 6552, 0x02, 0x02, 0x02),
    ("white", 1800, 9912, 0x02, 0x02, 0x02),
    ("white", 1800, 9960, 0x02, 0x02, 0x02),
    ("white", 1800, 10224, 0x02, 0x02, 0x02),
    ("white", 1800, 10344, 0x02, 0x02, 0x02),
    ("white", 1800, 10368, 0x02, 0x02, 0x02),
    ("white", 2400, 10224, 0x03, 0x02, 0x02),
    ("white", 2400, 10368, 0x03, 0x02, 0x02),
    ("white", 3600, 9912, 0x08, 0x01, 0x01),
    ("white", 3600, 10224, 0x08, 0x01, 0x01),
    ("white", 3600, 10368, 0x08, 0x01, 0x01),
    ("white", 7200, 10224, 0x0F, 0x01, 0x01),
    ("white", 7200, 10368, 0x10, 0x01, 0x01),
]
#: 8300i SE dark strips (full frame).
SF_DARK_STRIPS_8300I = [
    ("dark", 600, 10368, 0x02, 0x01, 0x30),
    ("dark", 900, 10368, 0x03, 0x01, 0x30),
    ("dark", 1200, 10368, 0x04, 0x01, 0x30),
    ("dark", 1800, 10368, 0x06, 0x01, 0x30),
    ("dark", 2400, 10368, 0x08, 0x01, 0x30),
    ("dark", 3600, 10368, 0x0C, 0x01, 0x30),
    ("dark", 7200, 10368, 0x17, 0x01, 0x30),
]


@pytest.mark.parametrize(("strip", "ppi", "span", "dummy", "clk_a", "clk_b"), SF_STRIPS_SE_V2)
def test_se_v2_strip_clocks_match_silverfast(
    strip: str, ppi: int, span: int, dummy: int, clk_a: int, clk_b: int
) -> None:
    dvdset = strip == "white"
    assert shading_clocks_for_window(ppi, span, dvdset=dvdset) == (dummy, clk_a, clk_b)
    for model in (MODEL_8200I_SE, MODEL_8100_V2):
        assert model.shading_strip_clocks(ppi, dvdset=dvdset, span=span) == (dummy, clk_a, clk_b)


@pytest.mark.parametrize(("strip", "ppi", "span", "dummy", "clk_a", "clk_b"), SF_DARK_STRIPS_8300I)
def test_8300i_dark_strip_clocks_match_silverfast(
    strip: str, ppi: int, span: int, dummy: int, clk_a: int, clk_b: int
) -> None:
    assert MODEL_8300I_SE.shading_strip_clocks(ppi, dvdset=False, span=span) == (dummy, clk_a, clk_b)


def test_8300i_white_strip_keeps_its_own_clocks() -> None:
    assert MODEL_8300I_SE.shading_strip_clocks(1800, dvdset=True) == (0x08, 0x03, 0x03)
    assert MODEL_8300I_SE.shading_strip_clocks(7200, dvdset=True) == (0x26, 0x02, 0x02)


def test_default_span_is_the_full_frame() -> None:
    for model in (MODEL_8200I_SE, MODEL_8100_V2, MODEL_8300I_SE):
        for dvdset in (False, True):
            assert model.shading_strip_clocks(3600, dvdset=dvdset) == model.shading_strip_clocks(
                3600, dvdset=dvdset, span=10368
            )
