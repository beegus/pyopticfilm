# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests must never read or write the developer's real calibration cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pyopticfilm import Scanner
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.scan.calibrate import Calibrator

REAL_CACHE_DIR = Path.home() / ".cache" / "pyopticfilm"


def test_calibrator_default_cache_is_per_test(_isolated_calib_cache: Path) -> None:
    cal = Calibrator(MagicMock(), cache_path=None, model=MODEL_8200I_SE)
    assert cal.cache.path == _isolated_calib_cache
    assert REAL_CACHE_DIR not in cal.cache.path.parents


def test_scanner_default_cache_is_per_test(_isolated_calib_cache: Path) -> None:
    with Scanner.open_fake(MODEL_8200I_SE) as scanner:
        assert scanner.calibrator.cache.path == _isolated_calib_cache
