# SPDX-License-Identifier: GPL-3.0-or-later
"""Frozen 8200i SE USB image-read chunking — do not retarget to match new code."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.scan.geometry import compute_geometry
from pyopticfilm.scan.session_gl128 import Gl128ScanSession


def _geometry(*, dpi: int = 7200, lines: int = 4) -> object:
    return compute_geometry(dpi, model=MODEL_8200I_SE, area=(0.0, 0.0, 1.0, 0.05))


def test_chunk_bytes_prefers_line_alignment():
    geometry = _geometry()
    line = geometry.line_bytes
    assert Gl128ScanSession._chunk_bytes(geometry, line * 3) == line
    assert Gl128ScanSession._chunk_bytes(geometry, line + 10) == line


def test_verify_geometry_usb_span_rejects_bad_line_bytes():
    geometry = _geometry(dpi=1800)
    session = Gl128ScanSession(MagicMock(), MODEL_8200I_SE)
    bad = replace(geometry, line_bytes=geometry.line_bytes + 1)
    with pytest.raises(Exception, match="line_bytes"):
        session._verify_geometry_usb_span(bad)
