# SPDX-License-Identifier: GPL-3.0-or-later
"""7600i v1 jobs built from the decoded tables must match SilverFast's recorded jobs."""

from __future__ import annotations

import base64
from collections import Counter

import pytest

from pyopticfilm.asic.gl843_v1 import Gl843V1Usb
from pyopticfilm.device.model_7600i_v1 import MODEL_7600I_V1
from pyopticfilm.device.tables_7600i_v1 import POSITION_STOP_MS, RESOLUTIONS, scan_table, window_pixels
from pyopticfilm.scan.job_gl843_v1 import build_job, host_program, measured_from, vendor_measured
from pyopticfilm.scan.replay_gl843_v1 import ReplayHooks, run_profile
from pyopticfilm.usb.fake_gl843_v1 import SimulatedGl843V1Transport

RECORDED = [("color", 1440), ("color", 3600), ("color", 7200), ("infrared", 3600)]


def _frame(f: dict) -> tuple:
    return f["pixels"], f["lines"], f["bytes"], f["regs"]


@pytest.mark.parametrize(("mode", "dpi"), RECORDED)
def test_built_job_is_the_recorded_program(mode, dpi):
    recorded = MODEL_7600I_V1.recorded_job(mode, dpi)
    built = build_job(mode, dpi, measured_from(recorded))
    assert host_program(built) == host_program(recorded)
    assert [_frame(f) for f in built["frames"]] == [_frame(f) for f in recorded["frames"]]
    assert built["mainFrame"] == recorded["mainFrame"] and built["yres"] == recorded["yres"]
    assert built["moves"] == recorded["moves"]
    assert {k: built["scan"][k] for k in recorded["scan"]} == recorded["scan"]
    assert built["scan"]["backtracking"] and built["motorCruise"]["used"] == built["motorCruise"]["silverfast"]


def test_built_7200_infrared_is_the_7200_colour_job_with_the_infrared_changes():
    """The uncaptured job: what differs from the 7200 dpi colour job is what differs at 3600 dpi."""
    def writes(mode: str, dpi: int) -> Counter:
        job = build_job(mode, dpi, vendor_measured(MODEL_7600I_V1, mode, dpi))
        # register writes and reads; not the AFE codes, which are calibration results
        return Counter(x for x in host_program(job)
                       if (x[0] == "write" and x[4][:1] != (0x51,)) or x[0] == "read register")

    for dpi in (3600, 7200):
        added = writes("infrared", dpi) - writes("color", dpi)
        dropped = writes("color", dpi) - writes("infrared", dpi)
        if dpi == 3600:
            want = (added, dropped)
        else:
            assert (added, dropped) == want
    assert {x[4] for x in want[0] if x[0] == "write"} >= {(0xA8, 0x27), (0xA8, 0x23), (0x03, 0x8F)}


def test_host_program_sees_every_host_decision():
    recorded = MODEL_7600I_V1.recorded_job("color", 3600)
    reference = host_program(build_job("color", 3600, measured_from(recorded)))

    def changed(edit) -> bool:
        job = build_job("color", 3600, measured_from(recorded))
        edit(job["ops"])
        return host_program(job) != reference

    def nth(ops, pred, n=0):
        return [o for o in ops if pred(o)][n]

    def batch(o):
        return o["kind"] == "control" and o["rt"] == 0x40 and o["value"] == 0x83 and len(o["data"]) > 8

    def bump_value(ops):
        nth(ops, batch, 10)["data"][-1] ^= 1

    def swap_pairs(ops):
        d = nth(ops, batch, 10)["data"]
        d[0:2], d[2:4] = d[2:4], d[0:2]

    def drop_afe_write(ops):
        ops.remove(nth(ops, lambda o: o["kind"] == "control" and o.get("data", [None])[0] == 0x51, 12))

    def bump_payload(ops):
        op = nth(ops, lambda o: o["kind"] == "write", 3)
        data = bytearray(base64.b64decode(op["data"]))
        data[7] ^= 1
        op["data"] = base64.b64encode(bytes(data)).decode()

    def shorten_read(ops):
        nth(ops, lambda o: o["kind"] == "read", 2)["length"] -= 6

    def drop_wait(ops):
        i = ops.index(nth(ops, lambda o: o.get("register") == 0x41 and o["expected"] == [0xB5]))
        del ops[i - 2 : i + 1]  # address, acknowledge, status read

    for edit in (bump_value, swap_pairs, drop_afe_write, bump_payload, shorten_read, drop_wait):
        assert changed(edit), edit.__name__


def test_poll_counts_and_untimed_delays_are_not_the_program():
    recorded = MODEL_7600I_V1.recorded_job("color", 3600)
    built = build_job("color", 3600, measured_from(recorded))
    ops = built["ops"]
    i = next(i for i, o in enumerate(ops) if o.get("register") == 0x41)
    ops[i + 1 : i + 1] = ops[i - 2 : i + 1] * 4  # the same status poll four more times
    ops.insert(0, {"kind": "delay", "ms": 50})
    assert host_program(built) == host_program(recorded)


@pytest.mark.parametrize("dpi", sorted(RESOLUTIONS))
def test_tables_give_whole_motor_periods_and_even_windows(dpi):
    assert window_pixels(dpi) % 2 == 0  # with an odd count the scanner sends a pixel less per line
    res = RESOLUTIONS[dpi]
    assert res.maxwd[2] == window_pixels(dpi) + 5450 or dpi == 1440
    assert len(scan_table(14000)) == 256


@pytest.mark.parametrize("calibrate", [False, True])
@pytest.mark.parametrize(("mode", "dpi"), [("color", 1440), ("infrared", 3600)])
def test_built_job_plays_on_the_simulated_scanner(mode, dpi, calibrate):
    recorded = MODEL_7600I_V1.recorded_job(mode, dpi)
    built = build_job(mode, dpi, measured_from(recorded))
    clock = {"t": 0.0}

    def sleep(s: float) -> None:
        clock["t"] += s

    sim = SimulatedGl843V1Transport(position=0)
    result = run_profile(built, Gl843V1Usb(sim), ReplayHooks(sleep=sleep, now=lambda: clock["t"]),
                         calibrate=calibrate)
    main = built["frames"][built["mainFrame"]]
    assert len(result.main) == main["bytes"]
    assert result.positioning_stop_ms == pytest.approx(POSITION_STOP_MS[(mode, dpi)], abs=1)
