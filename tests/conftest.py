# SPDX-License-Identifier: GPL-3.0-or-later
"""Make ``tests/scanners`` importable as ``scanners``; keep tests off the real calib cache."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent
for _path in (_TESTS, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@pytest.fixture(autouse=True)
def _isolated_calib_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the default calibration cache at a per-test file.

    ``Scanner`` and ``Calibrator(cache_path=None)`` otherwise read and write
    ``~/.cache/pyopticfilm/calib_v2.json``: a test's fake calibration then
    matched the developer's real scanner by fingerprint and was applied to
    real scans, and tests could pick up whatever that machine had cached.
    """
    from pyopticfilm import scanner
    from pyopticfilm.scan import calibrate

    path = tmp_path / "calib_cache" / "calib_v2.json"
    monkeypatch.setattr(calibrate, "default_cache_path", lambda: path)
    monkeypatch.setattr(scanner, "default_cache_path", lambda: path)
    return path
