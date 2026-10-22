# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-channel exposure table for GL128 image passes at native 7200 dpi."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pyopticfilm.asic.registers import Gl128Registers
from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE
from pyopticfilm.scan.geometry import compute_geometry
from pyopticfilm.scan.session_gl128 import Gl128ScanSession


def _image_regs(model, dpi: int) -> dict[int, int]:
    asic = MagicMock()
    asic._motor_moves_enabled = False
    asic.last_afe = None
    asic.asic_shading_ready = True
    session = Gl128ScanSession(asic, model)
    session.se_regs = Gl128Registers()
    session._lamp_requested = True
    session._configure(compute_geometry(dpi, model=model, area=None))
    return dict(asic.protocol.write_registers_batched.call_args[0][0])


@pytest.mark.parametrize("model", [MODEL_8200I_SE, MODEL_8100_V2, MODEL_8300I_SE])
@pytest.mark.parametrize(("dpi", "reg_1d"), [(3600, 0x80), (7200, 0x81)])
def test_image_reg_0x1d_matches_vendor(model, dpi, reg_1d):
    assert _image_regs(model, dpi)[0x1D] == reg_1d


def _configure(model, dpi: int, **pass_state) -> tuple[int, int]:
    asic = MagicMock()
    asic._motor_moves_enabled = False
    asic.last_afe = None
    asic.asic_shading_ready = True
    asic._scan_method = "color"
    session = Gl128ScanSession(asic, model)
    session.se_regs = Gl128Registers()
    session._lamp_requested = True
    for key, value in pass_state.items():
        setattr(session, key, value)
    session._configure(compute_geometry(dpi, model=model, area=None))
    regs = dict(asic.protocol.write_registers_batched.call_args[0][0])
    reg_exposure = (regs[0x7D] << 16) | (regs[0x7E] << 8) | regs[0x7F]
    return asic.upload_tables.call_args.kwargs["channel_exposure"], reg_exposure


@pytest.mark.parametrize(("dpi", "expected"), [(1200, 2333), (2400, 4666), (3600, 7000), (7200, 28000)])
def test_se_image_table_matches_silverfast_ladder(dpi, expected):
    channel, reg_exposure = _configure(MODEL_8200I_SE, dpi)
    assert channel == expected
    assert reg_exposure == 14000


def test_se_manual_override_is_not_scaled():
    channel, reg_exposure = _configure(MODEL_8200I_SE, 7200, _pass_exposure=28000, _pass_manual=True)
    assert (channel, reg_exposure) == (28000, 28000)


def test_se_me_bracket_keeps_unscaled_table():
    channel, _ = _configure(MODEL_8200I_SE, 7200, _pass_native_exposure_scale=False)
    assert channel == 14000


def test_8300i_se_image_table_at_7200():
    channel, reg_exposure = _configure(MODEL_8300I_SE, 7200)
    assert (channel, reg_exposure) == (30000, 15000)


def test_8300i_se_below_7200_unchanged():
    channel, _ = _configure(MODEL_8300I_SE, 3600)
    assert channel == 15000 // 2


def test_8100_v2_image_table_at_7200():
    channel, reg_exposure = _configure(MODEL_8100_V2, 7200)
    assert (channel, reg_exposure) == (28000, 14000)
