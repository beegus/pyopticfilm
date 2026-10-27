# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 image-pass motor gate: default full frames must pass, overruns must not.

SilverFast's own full-frame LINCNTs (8100 V2 ``06_ppi_ladder`` / ``04_color_7200``,
8300i SE PPI ladder) run up to 12 steps past ``scan_window_end_steps``; the gate
allows that and nothing more.
"""

from __future__ import annotations

import pytest

from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE
from pyopticfilm.device.select import create_asic
from pyopticfilm.exceptions import ScanError
from pyopticfilm.scan.bringup import apply_target_lincnt
from pyopticfilm.scan.geometry import compute_geometry
from pyopticfilm.scan.session_gl128 import Gl128ScanSession, image_feed2_steps
from pyopticfilm.usb.fake import MockScannerTransport
from pyopticfilm.usb.protocol import GenesysUsbProtocol

GL128_MODELS = (MODEL_8200I_SE, MODEL_8100_V2, MODEL_8300I_SE)
LADDER_DPI = (600, 720, 900, 1200, 1440, 1800, 2400, 3600, 7200)

#: SilverFast full-frame image LINCNT at feed2 13128 (8100 V2 captures).
SF_LINCNT_SE_V2 = {
    600: 2420,
    720: 2904,
    900: 3628,
    1200: 4836,
    1440: 5804,
    1800: 7252,
    2400: 9668,
    3600: 14500,
    7200: 29012,
}
#: SilverFast full-frame image LINCNT on the 8300i SE (PPI ladder).
SF_LINCNT_8300I = {
    600: 605,
    900: 907,
    1200: 1209,
    1800: 1813,
    2400: 2417,
    3600: 3625,
    7200: 7253,
}


def _armed_session(model):
    asic = create_asic(GenesysUsbProtocol(MockScannerTransport()), model)
    asic._motor_moves_enabled = True
    asic.init()
    return Gl128ScanSession(asic, model)


@pytest.mark.parametrize("model", GL128_MODELS, ids=lambda m: m.name)
@pytest.mark.parametrize("dpi", LADDER_DPI)
def test_default_full_frame_fits_the_motor_gate(model, dpi):
    geometry = compute_geometry(dpi, model=model)
    feed2 = image_feed2_steps(model, geometry, long_exposure=False)
    assert geometry.lincnt_register <= model.max_safe_lincnt_for(feed2, dpi)


@pytest.mark.parametrize("model", (MODEL_8200I_SE, MODEL_8100_V2), ids=lambda m: m.name)
@pytest.mark.parametrize("dpi", sorted(SF_LINCNT_SE_V2))
def test_silverfast_full_frame_lincnt_fits_se_v2(model, dpi):
    assert SF_LINCNT_SE_V2[dpi] <= model.max_safe_lincnt_for(13128, dpi)


@pytest.mark.parametrize("dpi", sorted(SF_LINCNT_8300I))
def test_silverfast_full_frame_lincnt_fits_8300i(dpi):
    feed2 = MODEL_8300I_SE.feed_to_scan_steps_for_dpi(dpi)
    assert SF_LINCNT_8300I[dpi] <= MODEL_8300I_SE.max_safe_lincnt_for(feed2, dpi)


def test_safe_limit_is_only_the_silverfast_overrun():
    """12 steps past the window end; crop sizing keeps the window end."""
    m = MODEL_8100_V2
    assert m.max_travel_end_steps - m.scan_window_end_steps == 12
    assert m.max_lincnt_for(13128, 1440) == 5800
    assert m.max_safe_lincnt_for(13128, 1440) == 5808


@pytest.mark.parametrize(
    ("model", "dpi"),
    [(MODEL_8200I_SE, 1440), (MODEL_8100_V2, 1440), (MODEL_8100_V2, 7200)],
    ids=["se-1440", "v2-1440", "v2-7200"],
)
def test_configure_accepts_default_full_frame(model, dpi):
    """These raised ``ScanError`` before the gate used SilverFast's travel."""
    _armed_session(model)._configure(compute_geometry(dpi, model=model))


def test_configure_still_refuses_an_overrun():
    model = MODEL_8100_V2
    geometry = compute_geometry(1440, model=model)
    feed2 = image_feed2_steps(model, geometry, long_exposure=False)
    too_long = model.max_safe_lincnt_for(feed2, 1440) + model.image_lincnt_per_line
    with pytest.raises(ScanError, match="travel limit"):
        _armed_session(model)._configure(apply_target_lincnt(geometry, too_long))
