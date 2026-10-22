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


def _clock_and_dram_writes(protocol: MagicMock) -> list[tuple]:
    """``0x8C`` writes and single ``0x0B``/``0x13`` register writes, in wire order."""
    out = []
    for name, args, _kw in protocol.mock_calls:
        if name == "write_0x8c":
            out.append(("8C", *args))
        elif name == "write_register" and args[0] in (0x0B, 0x13):
            out.append(args)
    return out


@pytest.mark.parametrize(
    ("model", "boot", "scan", "dram"),
    [(m, b, s, d) for (m, b, s), d in zip(_CLOCKS, (0x4C, 0x8C, 0x8C), strict=True)],
    ids=lambda v: getattr(v, "name", ""),
)
def test_asic_boot_writes_boot_clock_then_toggles_dram(model, boot, scan, dram):
    asic = Gl128(MagicMock(), model)
    asic.asic_boot()
    assert _clock_and_dram_writes(asic.protocol) == [
        ("8C", 0x10, boot),
        ("8C", 0x13, boot),
        (0x0B, dram & ~0x08),
        (0x13, 0x0F),
        (0x0B, dram),
    ]


@pytest.mark.parametrize(("model", "boot", "scan"), _CLOCKS, ids=lambda v: getattr(v, "name", ""))
def test_enter_scan_clock_writes_boot_then_scan(model, boot, scan):
    asic = Gl128(MagicMock(), model)
    asic._initialized = True
    asic.enter_scan_clock()
    assert _clock_and_dram_writes(asic.protocol) == [
        ("8C", 0x10, boot),
        ("8C", 0x13, boot),
        ("8C", 0x10, scan),
        ("8C", 0x13, scan),
        (0x0B, model.sensor_custom_regs[0x0B]),
        (0x13, 0x08),
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
