# SPDX-License-Identifier: GPL-3.0-or-later
"""The second positioning feed ends on FEEDFSH and motor stop, not on the vendor probe.

On the 8100 V2 the probe (0x21) reads 0x04 about 50 ms after START while the carriage is still
moving. SilverFast never reads the probe on that feed, so a probe-driven end cuts the motor
mid-ramp.
"""

from __future__ import annotations

from pyopticfilm.asic.gl128 import _FEED_PROBE_DONE
from pyopticfilm.asic.status import ScannerStatus
from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.select import create_asic
from pyopticfilm.usb.fake import MockScannerTransport
from pyopticfilm.usb.protocol import GenesysUsbProtocol

RUNNING = 0xC1  # PWRBIT | BUFEMPTY | MOTORENB
FINISHED = 0xE0  # PWRBIT | BUFEMPTY | FEEDFSH, motor stopped
MOVE_POLLS = 6


def _asic_with_feed(monkeypatch, *, probe_value: int):
    asic = create_asic(GenesysUsbProtocol(MockScannerTransport()), MODEL_8100_V2)
    polls = {"n": 0}

    def status():
        polls["n"] += 1
        if polls["n"] <= MOVE_POLLS:
            return ScannerStatus.from_reg41(RUNNING)
        return ScannerStatus.from_reg41(FINISHED)

    monkeypatch.setattr(asic, "read_status_reliable", status)
    monkeypatch.setattr(asic, "_read_feed_probe", lambda: probe_value)
    monkeypatch.setattr("pyopticfilm.asic.gl128.HOME_POLL_S", 0.0)
    monkeypatch.setattr("pyopticfilm.asic.gl128._FEED_PROBE_POLL_S", 0.0)
    return asic, polls


def test_probe_ends_feed_early_by_default(monkeypatch):
    asic, polls = _asic_with_feed(monkeypatch, probe_value=_FEED_PROBE_DONE)

    asic._wait_feed_probe_done(steps=13548, timeout_s=5.0, stale_done=False, require_motion=True)

    assert polls["n"] < MOVE_POLLS  # taken as done while the motor still runs


def test_ignore_probe_waits_for_motor_stop(monkeypatch):
    asic, polls = _asic_with_feed(monkeypatch, probe_value=_FEED_PROBE_DONE)

    asic._wait_feed_probe_done(
        steps=13548, timeout_s=5.0, stale_done=False, require_motion=True, ignore_probe=True
    )

    assert polls["n"] > MOVE_POLLS  # returned only after FEEDFSH with motor stopped
