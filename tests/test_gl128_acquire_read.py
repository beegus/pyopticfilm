# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 image read: preallocated buffer, unpaced drain, timing report."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pyopticfilm.asic.registers import Gl128Registers
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.exceptions import ScanError
from pyopticfilm.scan.session_gl128 import ACQUIRE_TIMING_ENV, Gl128ScanSession


def _session(reads):
    proto = MagicMock()
    proto.bulk_read_exact.side_effect = reads
    asic = MagicMock()
    asic.protocol = proto
    session = Gl128ScanSession(asic, MODEL_8200I_SE)
    session.se_regs = Gl128Registers()
    return session


def _geometry(total: int, line: int) -> MagicMock:
    geometry = MagicMock()
    geometry.total_bytes = total
    geometry.line_bytes = line
    geometry.disable_buffer_full_move = False
    geometry.resolution = 7200
    return geometry


def test_short_reads_fill_the_preallocated_buffer_in_order():
    session = _session([b"\x01" * 3, b"\x02" * 5, b"\x03" * 2])
    with patch.object(Gl128ScanSession, "_wait_data"):
        raw = session._acquire(_geometry(10, 4), progress=None, cancel=None)
    assert bytes(raw) == b"\x01" * 3 + b"\x02" * 5 + b"\x03" * 2


def test_overlong_last_read_is_truncated_to_total():
    session = _session([b"\x07" * 8])
    with patch.object(Gl128ScanSession, "_wait_data"):
        raw = session._acquire(_geometry(6, 6), progress=None, cancel=None)
    assert bytes(raw) == b"\x07" * 6


def test_empty_read_raises():
    session = _session([b"\x00" * 4, b""])
    with patch.object(Gl128ScanSession, "_wait_data"), pytest.raises(ScanError, match="4 of 10"):
        session._acquire(_geometry(10, 4), progress=None, cancel=None)


def test_acquire_never_sleeps():
    session = _session([b"\x00" * 4096] * 3)
    with (
        patch.object(Gl128ScanSession, "_wait_data"),
        patch("pyopticfilm.scan.session_gl128.time.sleep") as sleep,
    ):
        session._acquire(_geometry(3 * 4096, 4096), progress=None, cancel=None)
    sleep.assert_not_called()


def test_timing_summary_logged_and_csv_appended(tmp_path, monkeypatch, caplog):
    out = tmp_path / "timing.csv"
    monkeypatch.setenv(ACQUIRE_TIMING_ENV, str(out))
    for _ in range(2):
        session = _session([b"\x00" * 4, b"\x00" * 4])
        with patch.object(Gl128ScanSession, "_wait_data"), caplog.at_level("INFO"):
            session._acquire(_geometry(8, 4), progress=None, cancel=None)
    assert "GL128 image read: 7200 dpi" in caplog.text
    assert "chunks=2" in caplog.text
    assert "quiet_drain" not in caplog.text
    rows = out.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "dpi,offset,bytes,read_s,gap_s"
    assert len(rows) == 1 + 4
    assert [r.split(",")[1:3] for r in rows[1:3]] == [["0", "4"], ["4", "4"]]
