# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 IR: AFE search + IR white strip at home before every IR pass."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.exceptions import ScanError
from pyopticfilm.scan.calib_gl128 import AfeFrontend
from pyopticfilm.scan.calibrate import Calibrator
from pyopticfilm.scan.geometry import compute_geometry
from pyopticfilm.scan.session import ScanSession

COLOUR_AFE = AfeFrontend(gains=(38, 30, 36), offsets=(19, 31, 23))
IR_AFE = AfeFrontend(gains=(6, 16, 12), offsets=(0x30, 0x29, 0x31))


def _geometry(dpi: int = 1800):
    return compute_geometry(dpi, model=MODEL_8200I_SE, area=(0.0, 0.0, 1.0, 0.5))


def _asic() -> MagicMock:
    asic = MagicMock()
    asic._initialized = True
    asic._motor_moves_enabled = True
    asic.asic_shading_ready = True
    asic.is_at_home = MagicMock(return_value=True)
    asic.last_afe = COLOUR_AFE
    asic.ir_host_flatten_ready = False
    asic.last_ir_host_white = None
    asic.usb_planar_rgb = False
    asic._reg_cache = {}
    return asic


def _wire(asic: MagicMock, *, flatten_ok: bool, order: list[str] | None = None) -> None:
    def search_afe(**kw):
        assert kw == {"method": "infrared"}
        assert asic._motor_moves_enabled is False
        if order is not None:
            order.append("ir_afe")
        asic.last_afe = IR_AFE
        return IR_AFE

    def run_asic_shading(**kw):
        assert kw["method"] == "infrared"
        assert asic._motor_moves_enabled is False
        if order is not None:
            order.append("ir_white")
        asic.asic_shading_ready = False
        asic.ir_host_flatten_ready = flatten_ok
        asic.last_ir_host_white = [50000] * 16 if flatten_ok else None
        return b"\x00" * 16

    asic.search_afe.side_effect = search_afe
    asic.run_asic_shading.side_effect = run_asic_shading


def test_measure_ir_keeps_ir_afe_and_uses_scan_window(tmp_path: Path):
    asic = _asic()
    _wire(asic, flatten_ok=True)
    cal = Calibrator(asic, cache_path=tmp_path / "calib.json", model=MODEL_8200I_SE)
    geometry = _geometry()

    assert cal.measure_ir_host_flatten(geometry) is True

    kw = asic.run_asic_shading.call_args.kwargs
    assert kw["resolution"] == geometry.resolution
    assert (kw["strpixel"], kw["endpixel"], kw["dpiset"]) == (
        geometry.pixel_startx,
        geometry.pixel_endx,
        geometry.register_dpiset,
    )
    assert asic.last_afe == IR_AFE
    assert asic._motor_moves_enabled is True


def test_rejected_ir_profile_restores_colour_afe(tmp_path: Path):
    asic = _asic()
    _wire(asic, flatten_ok=False)
    cal = Calibrator(asic, cache_path=tmp_path / "calib.json", model=MODEL_8200I_SE)

    assert cal.measure_ir_host_flatten(_geometry()) is False

    assert asic.last_afe == COLOUR_AFE
    asic.apply_frontend.assert_called_with(COLOUR_AFE)
    assert asic.ir_host_flatten_ready is False


def test_ir_calibration_error_is_not_fatal(tmp_path: Path):
    asic = _asic()
    asic.search_afe.side_effect = ScanError("AFE strip short read")
    cal = Calibrator(asic, cache_path=tmp_path / "calib.json", model=MODEL_8200I_SE)

    assert cal.measure_ir_host_flatten(_geometry()) is False

    asic.run_asic_shading.assert_not_called()
    assert asic.last_afe == COLOUR_AFE
    assert asic.ir_host_flatten_ready is False
    assert asic._motor_moves_enabled is True


def test_infrared_scan_calibrates_ir_before_acquire(tmp_path: Path, monkeypatch):
    asic = _asic()
    order: list[str] = []
    _wire(asic, flatten_ok=True, order=order)
    cal = Calibrator(asic, cache_path=tmp_path / "calib.json", model=MODEL_8200I_SE)
    geometry = _geometry()
    session = ScanSession(asic, MODEL_8200I_SE, cal)
    monkeypatch.setattr(session, "acquire_raw", lambda *_a, **_k: (order.append("acquire") or b"\x00" * 64))
    monkeypatch.setattr(
        session.pipeline,
        "assemble",
        lambda *_a, **_k: np.full((4, 16, 3), 40000, dtype=np.uint16),
    )

    session.run(resolution=geometry.resolution, mode="infrared", geometry=geometry)

    assert order == ["ir_afe", "ir_white", "acquire"]
