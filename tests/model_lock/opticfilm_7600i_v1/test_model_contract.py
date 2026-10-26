# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen 7600i v1 model flags and job geometry — do not retarget to match new code."""

from __future__ import annotations

import hashlib

import pytest

from pyopticfilm.device.model_7600i_v1 import CALIBRATED_SHIFTS, MODEL_7600I_V1
from pyopticfilm.scan.job_gl843_v1 import build_job, host_program, job_for, measured_from


def test_identity_and_scan_flags():
    assert MODEL_7600I_V1.asic == "GL843"
    assert (MODEL_7600I_V1.usb_vendor_id, MODEL_7600I_V1.usb_product_id) == (0x07B3, 0x0C3B)
    assert MODEL_7600I_V1.scan_ready is True
    assert MODEL_7600I_V1.supports_infrared is True
    assert MODEL_7600I_V1.mirror_x is True
    assert MODEL_7600I_V1.resolutions_dpi == (7200, 3600, 1440)
    assert MODEL_7600I_V1.infrared_resolutions_dpi == (7200, 3600)
    assert MODEL_7600I_V1.dummy_lines == "fewer"
    assert MODEL_7600I_V1.single_sample is True
    assert MODEL_7600I_V1.window_start == 80
    assert MODEL_7600I_V1.stagger_y_by_dpi[7200] == (4, 0)


@pytest.mark.parametrize(
    ("mode", "dpi", "pixels", "lines", "yres", "stop_ms"),
    [
        ("color", 1440, 2050, 2824, 2880, 2571.9),
        ("color", 3600, 5124, 7058, 7200, 2559.7),
        ("color", 7200, 10248, 14122, 14400, 2571.8),
        ("infrared", 3600, 5124, 7058, 7200, 2571.8),
    ],
)
def test_job_frames_and_positioning_stop(mode, dpi, pixels, lines, yres, stop_ms):
    p = MODEL_7600I_V1.recorded_job(mode, dpi)
    main = p["frames"][p["mainFrame"]]
    assert p["mainFrame"] == 7
    assert (main["pixels"], main["lines"], main["bytes"]) == (pixels, lines, pixels * lines * 6)
    assert p["yres"] == yres
    assert next(o["ms"] for o in p["ops"] if o.get("timedStop")) == stop_ms


def test_calibrated_shifts():
    assert CALIBRATED_SHIFTS == {
        2880: (0.0, 9.84, 19.30),
        7200: (0.0, 24.22, 48.21),
        14400: (0.0, 48.44, 96.42),
    }


@pytest.mark.parametrize(
    ("dpi", "setting", "kwargs", "cruise"),
    [
        (3600, "recorded", {}, (14000, 14000)),
        (3600, "none", {}, (14000, 7000)),
        (7200, "fewer", {}, (42000, 28000)),
        (7200, "none", {}, (42000, 14000)),
        (3600, "recorded", {"exposure_multiplier": 3}, (14000, 21000)),
    ],
)
def test_main_scan_motor_cruise(dpi, setting, kwargs, cruise):
    measured = measured_from(MODEL_7600I_V1.recorded_job("color", dpi))
    got = build_job("color", dpi, measured, dummy_lines=setting, **kwargs)["motorCruise"]
    assert (got["silverfast"], got["used"]) == cruise


#: The default scan jobs as validated on hardware, when they were SilverFast's recorded jobs with the
#: window widened to sensor pixel 80 and the main scan changed for one dummy line fewer and single
#: sampling: main frame pixels, lines, vertical sampling, and a fingerprint of the host program.
DEFAULT_JOBS = [
    ("color", 1440, 2076, 1412, 1440, "1e567811229d273006e330404a5de3335254c5952f4011f7dc6bd4b50d9a83b5"),
    ("color", 3600, 5188, 3529, 3600, "663025c1bd575ccf1f0c8f25dbc0c5dca7b2e16fdf9a6cef36ed871b9c7a539f"),
    ("color", 7200, 10378, 7061, 7200, "47a0550ea7f1c4d8cca2d9bbedc8a6e2a23081af8d7d72d69f846fd4bd9cc86e"),
    ("infrared", 3600, 5188, 3529, 3600, "15a931caa37e50b99b940d30ac6d3465427a6dc6ea36aef7d02def070b03d8b6"),
    ("infrared", 7200, 10378, 7061, 7200, "0ff47d51cfae101682e9040a84ffbf7dede8d3238dccf91e397fb2b89664374d"),
]


@pytest.mark.parametrize(("mode", "dpi", "pixels", "lines", "yres", "program"), DEFAULT_JOBS)
def test_default_jobs_are_the_hardware_validated_jobs(mode, dpi, pixels, lines, yres, program):
    p = job_for(MODEL_7600I_V1, mode, dpi)
    main = p["frames"][p["mainFrame"]]
    assert (main["pixels"], main["lines"], p["yres"]) == (pixels, lines, yres)
    assert hashlib.sha256(repr(host_program(p)).encode()).hexdigest() == program
