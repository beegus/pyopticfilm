# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 infrared shading follows SilverFast.

SilverFast (8300i SE PPI ladder, 8200i SE session 05) reads the IR dark strip at
home, uploads a unity table with zero dark terms, and reads the IR white strip
with DVDSET armed and the motor running (``0x01=0x23``, ``0x02=0x10``), like the
colour white strip.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pyopticfilm.asic.gl128 as gl128_mod
from pyopticfilm.asic.gl128 import Gl128
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.scan.calib_gl128 import AfeFrontend

N = 64


def _ir_asic(monkeypatch, white_cols):
    """GL128 with strip I/O mocked; returns (asic, acquire mock)."""
    monkeypatch.setattr(gl128_mod.time, "sleep", lambda *_a, **_k: None)
    proto = MagicMock()
    proto.read_register = MagicMock(side_effect=lambda addr: {0x03: 0x20, 0x37: 0xB4}.get(int(addr), 0))
    asic = Gl128(proto, MODEL_8200I_SE)
    asic._initialized = True
    asic._motor_moves_enabled = False
    asic.last_afe = AfeFrontend(gains=(2, 12, 8), offsets=(0x130, 0x128, 0x12F))
    for name in (
        "apply_frontend",
        "_require_carriage_at_home",
        "lamp_off",
        "lamp_on",
        "_reassert_lamp_off",
        "_log_lamp_off_state",
        "_log_lamp_state",
        "_apply_infrared",
        "upload_tables",
    ):
        setattr(asic, name, MagicMock())
    asic.upload_shading_table = MagicMock()
    strip = b"\x00\x00" * (N * 128 * 3)
    acquire = MagicMock(side_effect=[strip, strip])
    asic.acquire_shading_strip = acquire
    monkeypatch.setattr(gl128_mod, "choose_usb_planar", lambda *_a, **_k: False)
    monkeypatch.setattr(
        gl128_mod,
        "average_rgb16_columns",
        MagicMock(side_effect=[[(300, 310, 420)] * N, list(white_cols)]),
    )
    return asic, acquire


def _run(asic):
    asic.run_asic_shading(resolution=1800, method="infrared", strpixel=240, endpixel=240 + N * 4, dpiset=300)


def test_ir_white_strip_moves_with_dvdset_like_silverfast(monkeypatch) -> None:
    asic, acquire = _ir_asic(monkeypatch, [(43000, 53000, 55000)] * N)
    _run(asic)
    dark_call, white_call = acquire.call_args_list
    assert not dark_call.kwargs.get("dvdset", False)  # dark: stationary, DVDSET off
    assert white_call.kwargs["dvdset"] is True  # white: DVDSET + motor, parks after
    # DVDSET is armed on the unity table before the white strip.
    assert asic._reg_cache[0x01] & 0x20
    # The unity upload carries zero dark terms (SF session 05 / 8300i ladder).
    unity = asic.upload_shading_table.call_args_list[0].args[0]
    assert unity[0:2] == b"\x00\x00"


def _table_gains(blob: bytes, pixel: int) -> tuple[int, int, int]:
    """(R, G, B) gain words of ``pixel`` in a packed table (126 pairs per 512-byte block)."""
    out = []
    for c in range(3):
        pair = pixel * 3 + c
        block, slot = divmod(pair, 126)
        off = block * 512 + slot * 4
        out.append(int.from_bytes(blob[off + 2 : off + 4], "little"))
    return out[0], out[1], out[2]


def test_ir_table_is_per_channel_and_arms_dvdset(monkeypatch) -> None:
    """SF's IR gains are per channel (8300i SE ladder); so is this table."""
    from pyopticfilm.scan.calib_gl128 import SHADING_GAIN_TARGET, SHADING_GAIN_UNITY

    asic, _ = _ir_asic(monkeypatch, [(43000, 53000, 55000)] * N)
    _run(asic)
    measured = asic.upload_shading_table.call_args_list[-1].args[0]
    expected = tuple(round(SHADING_GAIN_TARGET * SHADING_GAIN_UNITY / w) for w in (43000, 53000, 55000))
    assert _table_gains(measured, 0) == expected
    assert asic.ir_asic_shading_ready is True
    assert asic.ir_host_flatten_ready is True  # host profile kept as the fallback


def test_ir_table_with_out_of_band_gains_keeps_the_host_flatten(monkeypatch) -> None:
    """A white far too dim for a sane table: no IR DVDSET, host flatten still ready."""
    asic, _ = _ir_asic(monkeypatch, [(12000, 12500, 13000)] * N)
    _run(asic)
    assert asic.ir_asic_shading_ready is False
    assert asic.ir_host_flatten_ready is True


def test_any_other_upload_clears_the_ir_table_flag(monkeypatch) -> None:
    asic, _ = _ir_asic(monkeypatch, [(43000, 53000, 55000)] * N)
    _run(asic)
    assert asic.ir_asic_shading_ready is True
    del asic.upload_shading_table  # real upload: e.g. the colour table for the next pass
    asic.upload_shading_table(b"\x00" * 512)
    assert asic.ir_asic_shading_ready is False


def test_ir_plane_skips_host_divide_when_the_asic_applied_the_table(monkeypatch) -> None:
    import numpy as np

    import pyopticfilm.scan.calib_gl128 as calib_mod
    from pyopticfilm.scan.session import ScanSession

    divides: list[int] = []
    real = calib_mod.flatten_ir_columns
    monkeypatch.setattr(
        calib_mod, "flatten_ir_columns", lambda plane, white, **kw: divides.append(1) or real(plane, white, **kw)
    )
    session = ScanSession.__new__(ScanSession)
    session.model = MODEL_8200I_SE
    session.asic = MagicMock()
    session.asic.ir_host_flatten_ready = True
    session.asic.last_ir_host_white = [30000] * 16
    rgb = np.full((16, 16, 3), 30000, dtype=np.uint16)

    session.asic.ir_asic_shading_ready = True
    session._infrared_plane(rgb)
    assert divides == []  # the ASIC already applied the IR table
    session.asic.ir_asic_shading_ready = False
    session._infrared_plane(rgb)
    assert divides == [1]  # host flatten when DVDSET stayed off
