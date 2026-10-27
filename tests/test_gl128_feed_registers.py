# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 positioning feeds run under the image pass's registers, like SilverFast's.

Every SilverFast capture (8200i SE, 8100 V2, 8300i SE, 150-7200 ppi) programs the
image pass's DPISET, pixel clocks, LPERIOD, window, exposure, ``0x1D`` and depth
``0x33`` before both positioning feeds. The feed itself sets only ``0x01=0x22``,
``0x05=0x48``, ``0x02=0x18``, FEEDL and ``0xAF`` (``0x7F`` reference, ``0xFF``
scan start); ``0x2B`` keeps the white shading strip's dummy.
"""

from __future__ import annotations

import pytest

from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE
from pyopticfilm.device.select import create_asic
from pyopticfilm.scan.geometry import compute_geometry
from pyopticfilm.scan.session_gl128 import Gl128ScanSession
from pyopticfilm.usb.fake import MockScannerTransport
from pyopticfilm.usb.protocol import GenesysUsbProtocol

#: Registers whose image-pass value the feeds must carry.
CARRIED = (
    0x2C,
    0x2D,
    0x28,
    0x29,
    0x2A,
    0x82,
    0x83,
    0x84,
    0x85,
    0x86,
    0x87,
    0x7D,
    0x7E,
    0x7F,
    0x1D,
    0x33,
    0xA5,
    0xAB,
    0x3B,
)


def _starts(usb: MockScannerTransport) -> list[dict[int, int]]:
    """Register state at every START (``0x0F <- 1``), in wire order."""
    regs: dict[int, int] = {}
    out = []
    for t in usb.transactions:
        if t.operation != "control_write" or (t.value & 0xFF) != 0x83:
            continue
        data = t.data or b""
        high = 0x100 if t.value & 0x100 else 0
        for j in range(0, len(data) - 1, 2):
            regs[data[j] | high] = data[j + 1]
            if (data[j] | high) == 0x0F and data[j + 1] == 1:
                out.append(dict(regs))
    return out


@pytest.mark.parametrize("model", (MODEL_8200I_SE, MODEL_8100_V2, MODEL_8300I_SE), ids=lambda m: m.name)
@pytest.mark.parametrize("dpi", (600, 1800, 3600, 7200))
def test_feeds_carry_the_image_registers(model, dpi: int) -> None:
    usb = MockScannerTransport()
    asic = create_asic(GenesysUsbProtocol(usb), model)
    asic._motor_moves_enabled = True
    asic.init()
    session = Gl128ScanSession(asic, model)
    session._configure(compute_geometry(dpi, model=model))
    image = dict(asic._reg_cache)

    feeds = _starts(usb)
    assert len(feeds) == 2
    for feed, af in zip(feeds, (0x7F, 0xFF), strict=True):
        for reg in CARRIED:
            assert feed[reg] == image[reg], (hex(reg), feed[reg], image[reg])
        assert (feed[0x01], feed[0x05], feed[0x02], feed[0xAF]) == (0x22, 0x48, 0x18, af)
        assert feed[0x2B] == model.shading_strip_clocks(dpi, dvdset=True)[0]
    # The image pass still gets its own depth pair and AGOHOME.
    assert (image[0x33], image[0xAF], image[0x02]) == (0x1F, 0xFF, 0x30)


def test_8300i_feeds_use_its_exposure() -> None:
    """SF's 8300i SE feeds run at its 15000 exposure, not the old fixed 14000."""
    usb = MockScannerTransport()
    asic = create_asic(GenesysUsbProtocol(usb), MODEL_8300I_SE)
    asic._motor_moves_enabled = True
    asic.init()
    Gl128ScanSession(asic, MODEL_8300I_SE)._configure(compute_geometry(1800, model=MODEL_8300I_SE))
    for feed in _starts(usb):
        assert (feed[0x7D] << 16) | (feed[0x7E] << 8) | feed[0x7F] == 15000


def test_feeds_without_scan_registers_keep_the_fixed_recipe() -> None:
    from pyopticfilm.asic.gl128 import _FEED_SETUP_REGS, _feed_setup_regs

    assert _feed_setup_regs(None, final_positioning=False) == _FEED_SETUP_REGS
    assert _feed_setup_regs(None, final_positioning=True)[0xAF] == 0xFF
