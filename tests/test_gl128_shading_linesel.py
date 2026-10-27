# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 shading strips use SilverFast's ``0x1D`` (LINESEL) and white-strip motor table.

Values are SilverFast's own writes from USB captures: 8200i SE ``13_ppi_ladder``
and ``06_color_high_dpi``, 8100 V2 ``03``/``04``/``06``/``07``, and the 8300i SE
PPI ladder (colour and IR). Dark and white strips carry the same ``0x1D``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pyopticfilm.asic.gl128 import Gl128
from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE

#: (ppi, strip 0x1D, white-strip motor period) on the 8200i SE and 8100 V2.
SF_SE_V2 = [
    (150, 0x80, 3500),
    (600, 0x80, 3500),
    (720, 0x80, 3500),
    (900, 0x80, 3500),
    (1200, 0x80, 3500),
    (1440, 0x80, 3500),
    (1800, 0x80, 3500),
    (2400, 0x80, 3500),
    (3600, 0x81, 7000),
    (7200, 0x82, 10500),
]

#: (ppi, strip 0x1D, white-strip motor period) on the 8300i SE.
SF_8300I = [
    (150, 0x80, 3750),
    (600, 0x80, 3750),
    (900, 0x80, 3750),
    (1200, 0x80, 3750),
    (1800, 0x80, 3750),
    (2400, 0x81, 7500),
    (3600, 0x81, 7500),
    (7200, 0x83, 15000),
]


@pytest.mark.parametrize("model", [MODEL_8200I_SE, MODEL_8100_V2], ids=["se", "v2"])
@pytest.mark.parametrize(("ppi", "reg1d", "period"), SF_SE_V2)
def test_se_v2_match_silverfast(model, ppi: int, reg1d: int, period: int) -> None:
    assert 0x80 | model.shading_linesel_for(ppi) == reg1d
    assert model.shading_white_motor_period(ppi) == period


@pytest.mark.parametrize(("ppi", "reg1d", "period"), SF_8300I)
def test_8300i_matches_silverfast(ppi: int, reg1d: int, period: int) -> None:
    assert 0x80 | MODEL_8300I_SE.shading_linesel_for(ppi) == reg1d
    assert MODEL_8300I_SE.shading_white_motor_period(ppi) == period


def _strip_reg1d(model, ppi: int, *, dvdset: bool, cached: int) -> int:
    proto = MagicMock()
    asic = Gl128(proto, model)
    asic._reg_cache[0x1D] = cached
    asic._setup_shading_strip_regs(
        pixels=8, lines=128, resolution=ppi, strpixel=240, endpixel=240 + 8 * 4, dvdset=dvdset
    )
    return asic._reg_cache[0x1D]


@pytest.mark.parametrize("dvdset", [False, True], ids=["dark", "white"])
def test_strip_writes_linesel(dvdset: bool) -> None:
    assert _strip_reg1d(MODEL_8200I_SE, 7200, dvdset=dvdset, cached=0x80) == 0x82
    assert _strip_reg1d(MODEL_8300I_SE, 7200, dvdset=dvdset, cached=0x80) == 0x83
    assert _strip_reg1d(MODEL_8300I_SE, 2400, dvdset=dvdset, cached=0x80) == 0x81


def test_strip_replaces_leftover_image_linesel() -> None:
    # A 7200 image pass leaves 0x81; a 1800 strip must write 0x80, not keep it.
    assert _strip_reg1d(MODEL_8200I_SE, 1800, dvdset=True, cached=0x81) == 0x80


def test_afe_strip_keeps_linesel_zero() -> None:
    proto = MagicMock()
    asic = Gl128(proto, MODEL_8200I_SE)
    asic._reg_cache[0x1D] = 0x82
    asic._setup_afe_strip_regs()
    assert asic._reg_cache[0x1D] == 0x80
