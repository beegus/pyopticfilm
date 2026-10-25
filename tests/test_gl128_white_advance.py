# SPDX-License-Identifier: GPL-3.0-or-later
"""SilverFast white-strip carriage advance before the positioning feeds (#102)."""

from __future__ import annotations

import pytest

from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE
from pyopticfilm.device.select import create_asic
from pyopticfilm.exceptions import AsicError
from pyopticfilm.usb.fake import MockScannerTransport
from pyopticfilm.usb.protocol import GenesysUsbProtocol


def _words(data: bytes) -> tuple[int, ...]:
    return tuple(data[i] | (data[i + 1] << 8) for i in range(0, len(data), 2))


def _feedl(regs: dict[int, int]) -> int:
    return (regs[0x3D] << 16) | (regs[0x3E] << 8) | regs[0x3F]


def test_white_origin_advance_matches_silverfast_white_strips():
    """White-strip period is exposure × (0x1D low + 1) / 4 in every SF capture."""
    dpi, low, table = MODEL_8200I_SE.white_origin_advance()
    assert (dpi, low) == (1800, 0)
    assert table[:2] == (3516, 3500) and set(table[1:]) == {3500}

    dpi, low, table = MODEL_8300I_SE.white_origin_advance()
    assert (dpi, low) == (1800, 0)
    assert table[:2] == (40404, 4028) and table[-1] == 3750

    dpi, low, table = MODEL_8100_V2.white_origin_advance()
    assert (dpi, low) == (7200, 2)
    assert set(table) == {10500} and len(table) == 256


@pytest.mark.parametrize("model", [MODEL_8200I_SE, MODEL_8300I_SE, MODEL_8100_V2], ids=lambda m: m.model)
def test_positioning_runs_the_white_advance_before_feed1(model):
    usb = MockScannerTransport()
    protocol = GenesysUsbProtocol(usb)
    asic = create_asic(protocol, model)
    asic.init()
    asic._motor_moves_enabled = True
    r = asic.registers
    tables: dict[int, bytes] = {}
    starts: list[dict[int, int]] = []
    orig_ahb = protocol.write_ahb

    def track_ahb(addr, data):
        tables[int(addr)] = bytes(data)
        return orig_ahb(addr, data)

    protocol.write_ahb = track_ahb
    orig_after = usb._after_register_write

    def after(addr, value):
        if addr == r.REG_START and value == r.START_GO:
            snap = dict(usb.registers)
            snap["table"] = tables.get(r.AHB_CHANNEL_R)
            starts.append(snap)
        return orig_after(addr, value)

    usb._after_register_write = after

    asic.position_for_full_frame_scan(scan_steps=13128)

    assert [s[r.REG_0x02] & (r.MTRPWR | r.AGOHOME) for s in starts] == [r.MTRPWR, r.MTRPWR, r.MTRPWR]
    white, feed1, feed2 = starts
    _, low, table = model.white_origin_advance()
    assert white[r.REG_0x02] == r.MTRPWR
    assert _words(white["table"]) == table
    assert white[0x1D] & 0x1F == low
    assert (white[0x25], white[0x26], white[0x27]) == (0, 0, 128)
    assert white[0xA7] == white[0xA9] == 1
    assert feed1[0xA7] == feed1[0xA9] == 0
    assert _feedl(feed1) == model.feed_to_reference_steps
    assert _feedl(feed2) == 13128
    assert not usb.at_home


def test_white_advance_refuses_when_carriage_never_left_home(monkeypatch):
    usb = MockScannerTransport()
    asic = create_asic(GenesysUsbProtocol(usb), MODEL_8200I_SE)
    asic.init()
    asic._motor_moves_enabled = True
    orig_after = usb._after_register_write

    def stuck(addr, value):
        orig_after(addr, value)
        usb.at_home = True
        usb._sync_status()

    usb._after_register_write = stuck
    with pytest.raises(AsicError, match="white-strip advance"):
        asic.advance_white_origin()
