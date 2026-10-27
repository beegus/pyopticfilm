# SPDX-License-Identifier: GPL-3.0-or-later
"""The image-pass register blast must not overwrite calibrated AFE codes.

The GL128 frontend is written through ports: ``0x51`` selects the FE index and a
write to ``0x5E`` (low byte, with ``0x5D`` as high byte) commits the value. The
boot register map carries ``0x5E=0x1F``; blasting it after ``apply_frontend``
left ``0x51=7`` and so set the blue offset to ``0x1F`` (seen on the wire in an
8200i SE pyopticfilm capture: ``FE[7]`` 0x10 -> 0x1F before the image pass).
"""

from __future__ import annotations

import pytest

from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE
from pyopticfilm.device.select import create_asic
from pyopticfilm.scan.calib_gl128 import AfeFrontend
from pyopticfilm.scan.geometry import compute_geometry
from pyopticfilm.scan.session_gl128 import Gl128ScanSession
from pyopticfilm.usb.fake import MockScannerTransport
from pyopticfilm.usb.protocol import GenesysUsbProtocol

FE_INDEX, FE_HIGH, FE_LOW = 0x51, 0x5D, 0x5E

#: SilverFast 8200i SE image-pass AFE (1800 ppi full frame).
CALIBRATED = AfeFrontend(offsets=(0x15, 0x1F, 0x1A), gains=(0x26, 0x1D, 0x24))


def _register_writes(usb: MockScannerTransport) -> list[tuple[int, int]]:
    """Every register write in wire order, single and batched."""
    writes: list[tuple[int, int]] = []
    for t in usb.transactions:
        if t.operation != "control_write" or (t.value & 0xFF) != 0x83:
            continue
        data = t.data or b""
        high = 0x100 if t.value & 0x100 else 0
        writes.extend((data[j] | high, data[j + 1]) for j in range(0, len(data) - 1, 2))
    return writes


def _frontend_state(writes: list[tuple[int, int]]) -> dict[int, int]:
    """FE index -> value as the AFE latches it (commit on each ``0x5E`` write)."""
    regs: dict[int, int] = {}
    fe: dict[int, int] = {}
    for addr, val in writes:
        regs[addr] = val
        if addr == FE_LOW:
            fe[regs.get(FE_INDEX, 0)] = (regs.get(FE_HIGH, 0) << 8) | val
    return fe


@pytest.mark.parametrize("model", (MODEL_8200I_SE, MODEL_8100_V2, MODEL_8300I_SE), ids=lambda m: m.name)
@pytest.mark.parametrize("dpi", (1200, 1800, 7200))
def test_image_configure_keeps_calibrated_afe(model, dpi):
    usb = MockScannerTransport()
    asic = create_asic(GenesysUsbProtocol(usb), model)
    asic._motor_moves_enabled = False
    asic.init()
    asic.last_afe = CALIBRATED
    geometry = compute_geometry(dpi, model=model, area=(0.0, 0.0, 1.0, 0.12))
    Gl128ScanSession(asic, model)._configure(geometry)

    fe = _frontend_state(_register_writes(usb))
    assert dict(CALIBRATED.as_fe_writes()) == {i: fe[i] for i in range(2, 8)}
