# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 ``0x8C`` clock: boot value at boot, boot then scan value before each pass.

Vendor captures: 8200i SE ``0x0C``/``0x0C``, 8100 V2 ``0x0E``/``0x0C``,
8300i SE ``0x0E``/``0x0D``, always written to indices ``0x10`` then ``0x13``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyopticfilm.asic.gl128 import Gl128
from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE
from pyopticfilm.scan.session import ScanSession
from pyopticfilm.scan.session_gl128 import Gl128ScanSession

_CLOCKS = [
    (MODEL_8200I_SE, 0x0C, 0x0C),
    (MODEL_8100_V2, 0x0E, 0x0C),
    (MODEL_8300I_SE, 0x0E, 0x0D),
]


def _clock_writes(protocol: MagicMock) -> list[tuple[int, int]]:
    return [tuple(c.args) for c in protocol.write_0x8c.call_args_list]


@pytest.mark.parametrize(("model", "boot", "scan"), _CLOCKS, ids=lambda v: getattr(v, "name", ""))
def test_asic_boot_writes_boot_clock(model, boot, scan):
    asic = Gl128(MagicMock(), model)
    asic.asic_boot()
    assert _clock_writes(asic.protocol) == [(0x10, boot), (0x13, boot)]


@pytest.mark.parametrize(("model", "boot", "scan"), _CLOCKS, ids=lambda v: getattr(v, "name", ""))
def test_enter_scan_clock_writes_boot_then_scan(model, boot, scan):
    asic = Gl128(MagicMock(), model)
    asic._initialized = True
    asic.enter_scan_clock()
    assert _clock_writes(asic.protocol) == [
        (0x10, boot),
        (0x13, boot),
        (0x10, scan),
        (0x13, scan),
    ]


def test_enter_scan_clock_boots_first_so_scan_clock_is_last():
    asic = Gl128(MagicMock(), MODEL_8300I_SE)
    asic.enter_scan_clock()
    assert asic._initialized
    assert _clock_writes(asic.protocol)[-2:] == [(0x10, 0x0D), (0x13, 0x0D)]


def test_single_pass_run_enters_scan_clock_before_base_run():
    asic = MagicMock()
    asic._motor_moves_enabled = True
    order: list[str] = []
    asic.enter_scan_clock.side_effect = lambda: order.append("clock")
    session = Gl128ScanSession(asic, MODEL_8300I_SE, None)
    with patch.object(ScanSession, "run", lambda self, *a, **k: order.append("run")):
        session.run(resolution=1200)
    assert order == ["clock", "run"]
