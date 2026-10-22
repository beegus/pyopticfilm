# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 image read: preallocated buffer, unpaced drain, timing report."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from pyopticfilm.asic.registers import Gl128Registers
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE
from pyopticfilm.exceptions import ScanError
from pyopticfilm.scan.session_gl128 import ACQUIRE_TIMING_ENV, Gl128ScanSession
from pyopticfilm.usb.protocol import GenesysUsbProtocol


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


def _announces(model, geometry):
    proto = MagicMock()
    begins: list[int] = []
    proto.bulk_read_begin.side_effect = lambda size, **_kw: begins.append(int(size))
    proto.bulk_read_exact.side_effect = lambda size, **_kw: b"\x00" * int(size)
    asic = MagicMock()
    asic.protocol = proto
    session = Gl128ScanSession(asic, model)
    session.se_regs = Gl128Registers()
    with patch.object(Gl128ScanSession, "_wait_data"):
        session._acquire(geometry, progress=None, cancel=None)
    return begins, proto.bulk_read_exact.call_args_list


@pytest.mark.parametrize("model", [MODEL_8200I_SE, MODEL_8300I_SE])
def test_one_announce_per_full_width_7200_line(model):
    line = 62208  # 10368 px x 3 channels x 2 bytes at 7200 dpi
    begins, reads = _announces(model, _geometry(3 * line, line))
    assert begins == [line] * 3
    assert [c.kwargs for c in reads] == [{}] * 3


def test_large_chunk_model_passes_max_transfer():
    line = 62208
    model = replace(MODEL_8300I_SE, image_read_chunk_bytes=0x80000)
    begins, reads = _announces(model, _geometry(20 * line, line))
    assert begins == [8 * line, 8 * line, 4 * line]
    assert [c.kwargs for c in reads] == [{"max_transfer": 0x80000}] * 3


def test_bulk_read_exact_max_transfer_sets_request_size():
    usb = MagicMock()
    usb.bulk_read.side_effect = lambda n, **_kw: b"\x00" * n
    GenesysUsbProtocol(usb).bulk_read_exact(500_000, max_transfer=0x80000)
    assert [c.args[0] for c in usb.bulk_read.call_args_list] == [500_000]
    usb.bulk_read.reset_mock()
    GenesysUsbProtocol(usb).bulk_read_exact(62208)
    assert [c.args[0] for c in usb.bulk_read.call_args_list] == [0xF000, 62208 - 0xF000]


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


def test_timing_csv_written_when_the_read_fails(tmp_path, monkeypatch):
    out = tmp_path / "timing.csv"
    monkeypatch.setenv(ACQUIRE_TIMING_ENV, str(out))
    session = _session([b"\x00" * 4, OSError("scanner powered off")])
    with patch.object(Gl128ScanSession, "_wait_data"), pytest.raises(OSError):
        session._acquire(_geometry(12, 4), progress=None, cancel=None)
    rows = out.read_text(encoding="utf-8").splitlines()
    assert rows == ["dpi,offset,bytes,read_s,gap_s", rows[1]]
    assert rows[1].split(",")[1:3] == ["0", "4"]
