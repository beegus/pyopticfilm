# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 calibration strips write SilverFast's ``0xA3`` and ``0xA6``-``0xA9`` themselves.

SilverFast (every capture, all three models, 600-7200 ppi, colour and IR):
AFE ``A3=00``, dark ``A3=01``, white ``A3=00``, and ``A6``-``A9 = 00 01 00 01``
on all three; feeds and image ``00 00 00 00``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pyopticfilm.asic.gl128 import Gl128
from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE

_REGS = (0xA3, 0xA6, 0xA7, 0xA8, 0xA9)


def _after_image_pass(model) -> Gl128:
    asic = Gl128(MagicMock(), model)
    # An image pass leaves the boot-map zeros behind (no AFE strip since).
    asic._reg_cache.update(dict.fromkeys(_REGS, 0x00))
    return asic


@pytest.mark.parametrize("model", [MODEL_8200I_SE, MODEL_8100_V2, MODEL_8300I_SE], ids=["se", "v2", "8300i"])
@pytest.mark.parametrize(("dvdset", "a3"), [(False, 0x01), (True, 0x00)], ids=["dark", "white"])
def test_shading_strip_writes_silverfast_values(model, dvdset: bool, a3: int) -> None:
    asic = _after_image_pass(model)
    asic._setup_shading_strip_regs(
        pixels=8, lines=128, resolution=1800, strpixel=240, endpixel=240 + 8 * 4, dvdset=dvdset
    )
    assert [asic._reg_cache[a] for a in _REGS] == [a3, 0x00, 0x01, 0x00, 0x01]


def test_afe_strip_writes_silverfast_values() -> None:
    asic = _after_image_pass(MODEL_8200I_SE)
    asic._setup_afe_strip_regs()
    assert [asic._reg_cache[a] for a in _REGS] == [0x00, 0x00, 0x01, 0x00, 0x01]
