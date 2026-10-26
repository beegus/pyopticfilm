# SPDX-License-Identifier: GPL-3.0-or-later
"""OpticFilm 7600i v1 scan jobs: SilverFast's job as named steps over the decoded tables.

:func:`build_job` writes a job in the replay format of :mod:`pyopticfilm.scan.replay_gl843_v1`
(``control`` / ``write`` / ``read`` / ``delay`` ops) from :mod:`pyopticfilm.device.tables_7600i_v1`
and the calibration results the job writes (:class:`Measured`). The steps are those of every
recorded job: two AFE offset probes, the gain line, two more offset probes, the dark and white
frames, two positioning moves, the main scan, and the park. Options change the window, the
main scan's dummy lines, its vertical sampling and its exposure; with none, the job is
SilverFast's. :func:`job_for` is the job a scan runs, with the model's options.

:func:`host_program` reduces a job to what the host decides: every control write and its
payload, every control read request, every bulk payload, every frame read, and where it waits
and for what. Poll counts, untimed delays and the scanner's replies are left out: they are
SilverFast's timing and the scanner's state, not the program. A job built from the tables gives
the same host program as the recorded job it reproduces.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

from pyopticfilm.device.tables_7600i_v1 import (
    AFE_RESET,
    BOOT_CLOCK,
    CALIBRATION_LINES,
    FAST_RAMP,
    FMOVNO_CALIBRATION,
    FMOVNO_SCAN,
    GAIN_LINE_PIXELS,
    LPERIOD,
    PIXEL_CLOCK_HZ,
    POSITION_FEEDS,
    POSITION_RAMP,
    POSITION_STOP_MS,
    POWER_ON_REGS,
    PROBE_DPISET,
    PROBE_PIXELS,
    PROBE_START,
    RESOLUTIONS,
    SCAN_RAMP_START,
    SCAN_REGS,
    STATIONARY_TABLE,
    WHITE_STEPS_PER_LINE,
    cruise,
    dpiset,
    main_steps_per_line,
    scan_table,
    scan_window,
    window_pixels,
)
from pyopticfilm.scan.replay_gl843_v1 import SHADING_UNITY, shading_blocks, shading_table, widen_words

STATUS = 0x41
BUFEMPTY = 0x40
FEEDFSH = 0x20
LAMPSTS = 0x04
MOTORENB = 0x01
ACDCDIS = 0x40
PACKET = 512
#: shading words for 42 pixels fill one 512-byte block; a transfer holds at most 127 blocks
SHADING_PIXELS_PER_BLOCK = 42
SHADING_BLOCKS_PER_TRANSFER = 127
READ_HEADER = [0x00, 0x00, 0x82, 0x00]
UPLOAD_HEADER = [0x01, 0x00, 0x00, 0x00]
IR_LED_ON, IR_LED_OFF = 0x27, 0x23
FRAMES = ("probe_1a", "probe_1b", "gain_line", "probe_2a", "probe_2b", "dark", "white", "main")
WHITE_FRAME = FRAMES.index("white")
DUMMY_LINES = ("recorded", "fewer", "none")


@dataclass(frozen=True)
class LiveShading:
    """A shading table computed from the job's white frame while it runs.

    ``widen``: a recorded table and the pixel count added in front of it; the gains are then
    computed for the recorded table's per-channel target. Without it, SilverFast's infrared
    shading (``0x13000 × 0x2000 / white``).
    """

    widen: tuple[bytes, int] | None = None


@dataclass(frozen=True)
class Measured:
    """Calibration results a job writes: 9-bit AFE codes per channel, and shading uploads.

    ``dark_offsets`` and ``final_gains`` are written in colour jobs only. Live calibration
    replaces all of them while the job runs; these are the values written without it.
    """

    zero_low: tuple[int, int, int]
    gains: tuple[int, int, int]
    zero_high: tuple[int, int, int]
    dark_offsets: tuple[int, int, int] | None
    white_offsets: tuple[int, int, int]
    final_gains: tuple[int, int, int] | None
    dark_shading: bytes
    shading: bytes | LiveShading


def _field(first: int, value: int, size: int) -> list[tuple[int, int]]:
    """A big-endian multi-byte register field as (register, byte) pairs."""
    return [(first + i, (value >> (8 * (size - 1 - i))) & 0xFF) for i in range(size)]


def feedl(n: int) -> list[tuple[int, int]]:
    return _field(0x3D, n, 3)


def lincnt(n: int) -> list[tuple[int, int]]:
    return _field(0x25, n, 3)


def maxwd(n: int) -> list[tuple[int, int]]:
    return _field(0x35, n, 3)


def lperiod(n: int = LPERIOD) -> list[tuple[int, int]]:
    return _field(0x38, n, 2)


def dpiset_regs(n: int) -> list[tuple[int, int]]:
    return _field(0x2C, n, 2)


def window(start: int, end: int) -> list[tuple[int, int]]:
    return _field(0x30, start, 2) + _field(0x32, end, 2)


CLEAR_10_15 = [(r, 0) for r in range(0x10, 0x16)]


class _Job:
    """Op writer: every control write is followed by its acknowledge poll, as the vendor sends it."""

    def __init__(self) -> None:
        self.ops: list[dict] = []
        self.regs: dict[int, int] = {}
        self.frames: list[dict] = []

    # --- transfers

    def _out(self, value: int, data: list[int], index: int = 0) -> None:
        self.ops.append({"kind": "control", "rt": 0x40, "request": 0x04 if len(data) > 1 else 0x0C,
                         "value": value, "index": index, "data": list(data)})
        self._in(0x8E, 0x20, 1)

    def _in(self, value: int, index: int, expected: int, register: int | None = None) -> None:
        op = {"kind": "control", "rt": 0xC0, "request": 0x0C, "value": value, "index": index, "length": 1,
              "expected": [expected]}
        if register is not None:
            op["register"] = register
        self.ops.append(op)

    def write(self, *pairs: tuple[int, int]) -> None:
        """One register transfer."""
        self._out(0x83, [b for pair in pairs for b in pair])
        self.regs.update(pairs)

    def write_file(self, regs: dict[int, int]) -> None:
        pairs = sorted(regs.items())
        for i in range(0, len(pairs), 32):
            self.write(*pairs[i : i + 32])

    def afe(self, addr: int, value: int) -> None:
        self.write((0x51, addr), (0x3A, value >> 8), (0x3B, value & 0xFF))

    def afe_channels(self, first: int, values: Iterable[int]) -> None:
        for c, v in enumerate(values):
            self.afe(first + c, v)

    def clock(self, index: int, value: int) -> None:
        self._out(0x8C, [value], index)

    def read_register(self, reg: int) -> None:
        self._out(0x83, [reg])
        self._in(0x84, 0, self.regs.get(reg, 0), reg)

    def probe(self) -> None:
        """``0x8E`` read at index 0, as the vendor sends it before a boot and before each frame."""
        self._in(0x8E, 0, 0)

    def wait(self, status: int) -> None:
        """Poll status until ``status``; its lamp bit follows the white LED (``0x03`` bit 4)."""
        status = (status & ~LAMPSTS) | (LAMPSTS if self.regs.get(0x03, 0) & 0x10 else 0)
        self._out(0x83, [STATUS])
        self._in(0x84, 0, status, STATUS)

    def bulk_done(self) -> None:
        self._in(0x8E, 0x18, 0)

    def delay(self, ms: float, *, timed: bool = False) -> None:
        self.ops.append({"kind": "delay", "ms": ms, **({"timedStop": True} if timed else {})})

    def _setup(self, header: list[int], n: int) -> None:
        self._out(0x82, header + list(n.to_bytes(4, "little")))

    def _bulk(self, payload: bytes) -> None:
        self.ops.append({"kind": "write", "data": base64.b64encode(payload).decode()})

    # --- vendor primitives

    def motor_table(self, slot: int, table: tuple[int, ...]) -> None:
        bank = 0x40 | slot << 3
        data = b"".join(v.to_bytes(2, "little") for v in table)
        self.write((0x5B, bank), (0x5C, 0x00))
        self._out(0x83, [0x28])
        self._setup(UPLOAD_HEADER, len(data))
        self._bulk(data + bytes(-len(data) % PACKET))
        self.bulk_done()
        self.write((0x5B, bank & ~0x40))

    def scan_tables(self, table: tuple[int, ...]) -> None:
        for slot in range(3):
            self.motor_table(slot, table)

    def shading(self, table: bytes | LiveShading, pixels: int) -> None:
        """Shading RAM, at most 127 blocks per set-up; a set-up counts the bytes up to the last pixel."""
        blocks = -(-pixels // SHADING_PIXELS_PER_BLOCK)
        used = pixels // SHADING_PIXELS_PER_BLOCK * PACKET + pixels % SHADING_PIXELS_PER_BLOCK * 12
        if isinstance(table, bytes) and len(table) != blocks * PACKET:
            raise ValueError(f"shading table of {len(table)} bytes, need {blocks * PACKET}")
        self.write((0x29, 0), (0x2A, 0), (0x2B, 0))
        for first in range(0, blocks, SHADING_BLOCKS_PER_TRANSFER):
            at = first * PACKET
            n = min(SHADING_BLOCKS_PER_TRANSFER, blocks - first) * PACKET
            self._out(0x83, [0x3C])
            self._setup(UPLOAD_HEADER, min(used - at, n))
            if isinstance(table, bytes):
                self._bulk(table[at : at + n])
            else:
                live: dict = {"frame": WHITE_FRAME, "offset": at}
                if table.widen is not None:
                    recorded, extra = table.widen
                    live["widen"] = {"table": base64.b64encode(recorded).decode(), "extra": extra}
                self.ops.append({"kind": "write", "length": n, "shading": live})
            self.bulk_done()

    def ir_led(self, value: int) -> None:
        """``0xA8`` read-modify-write: GPIO27 drives the infrared LED."""
        self.read_register(0xA8)
        self.write((0xA8, value))
        self.read_register(0xA8)

    def acquire(self, *, pixels: int, lines: int, scan: int, status: int,
                prefix: Iterable[tuple[int, int]] = (), start: Iterable[tuple[int, int]] = ()) -> None:
        """Start a frame (``start``: registers written with the start), wait for data, read it."""
        for pair in prefix:
            self.write(pair)
        self.write((0x01, scan))
        self.write(*start, (0x0F, 0x01))
        self.wait(status)
        for reg in (0x42, 0x43, 0x44):
            self.read_register(reg)
        self._out(0x83, [0x45])
        self.probe()
        self.clock(0x0F, 0x02)
        n = pixels * lines * 6
        self._setup(READ_HEADER, n)
        frame = len(self.frames)
        self.ops.append({"kind": "read", "frame": frame, "length": n})
        self.frames.append({"pixels": pixels, "lines": lines, "bytes": n,
                            "regs": {str(r): v for r, v in sorted(self.regs.items())}})

    def next_frame(self) -> None:
        """Transfer complete, scan off, then the vendor's ``0x40`` read and status."""
        self.bulk_done()
        self.write((0x01, 0x02))
        self.read_register(0x40)
        self.wait(0xD4)


def build_job(mode: str, dpi: int, measured: Measured, *, first_pixel: int | None = None,
              dummy_lines: str = "recorded", single_sample: bool = False, exposure_multiplier: int = 1) -> dict:
    """The 7600i v1 job for ``mode`` ("color" / "infrared") at ``dpi``, in the replay format.

    ``first_pixel``: start the dark, white and main frames there (see
    :func:`~pyopticfilm.device.tables_7600i_v1.scan_window`) instead of at SilverFast's STRPIXEL.
    ``dummy_lines``: the main scan's CCD dummy lines, SilverFast's (``recorded``), one fewer
    (``fewer``) or ``none``. ``single_sample``: sample the nominal resolution vertically, not
    twice it (half the lines, twice the motor steps per line). ``exposure_multiplier``: LPERIOD
    of the main scan ×k without dummy lines (k = 3 is SilverFast's multi-exposure pass).
    """
    if mode not in ("color", "infrared"):
        raise ValueError("mode must be color or infrared")
    if dummy_lines not in DUMMY_LINES or exposure_multiplier not in (1, 2, 3, 4):
        raise ValueError("dummy_lines must be recorded/fewer/none and exposure_multiplier 1-4")
    ir = mode == "infrared"
    res = RESOLUTIONS[dpi]
    start, end = scan_window(dpi, first_pixel)
    pixels = window_pixels(dpi, first_pixel)
    probe_maxwd, gain_maxwd, scan_maxwd = res.maxwd
    calib_linesel = 0x10 | res.linesel_calibration
    j = _Job()

    # -- boot: the power-on register file, the clock, the scan register file, AFE reset
    j.probe()
    j.write_file(POWER_ON_REGS)
    j.clock(0x10, BOOT_CLOCK)
    j.read_register(0xA2)
    j.read_register(0x0B)
    j.wait(0xFC)  # motor idle
    j.write((0x0B, 0x4A))
    j.afe_channels(0, AFE_RESET)
    j.wait(0xFC)
    j.write((0x03, 0xBF))
    j.wait(0xFC)
    j.clock(0x10, BOOT_CLOCK)
    j.write_file(SCAN_REGS)
    j.afe_channels(0, AFE_RESET)
    j.wait(0xFC)
    j.delay(81)
    j.wait(0xFC)

    # -- one forward step on the fast ramp; infrared switches the white LED off for good
    wake = [(0x02, 0x18), (0x6A, FMOVNO_CALIBRATION), *feedl(1)]
    if ir:
        j.write((0x03, 0xAF))
        j.ir_led(IR_LED_ON)
        j.wait(0xFC)
        j.write(*wake)
    else:
        j.write((0x03, 0xBF), *wake)
    j.motor_table(3, FAST_RAMP)
    j.write((0x0F, 0x01))
    j.wait(0xF4)  # move done
    j.scan_tables(STATIONARY_TABLE)

    # -- AFE offset probe 1a: 512 pixels at full resolution, offsets at mid-scale
    j.write((0x02, 0x00), *feedl(1), *lperiod(), (0x1C, 0x20), *CLEAR_10_15, *dpiset_regs(PROBE_DPISET),
            (0x34, 0x14), *window(PROBE_START, PROBE_START + PROBE_PIXELS), (0x04, 0x62), (0x01, 0x02),
            (0x05, 0x40), (0x1F, 0x00), (0x22, 0x01), (0x23, 0x01), (0x1E, 0x10), (0x21, 0x01), (0x69, 0x01),
            *lincnt(1))
    j.write(*maxwd(probe_maxwd), (0x20, 0x02))
    j.afe_channels(2, (0, 0, 0))
    j.afe_channels(5, (128, 128, 128))
    j.delay(107)
    j.acquire(pixels=PROBE_PIXELS, lines=1, scan=0x03, status=0xB5, prefix=[(0x0D, 0x01), (0x0D, 0x03)])
    again = [(0x0D, 0x03), (0x0D, 0x03)]

    # -- probe 1b: offsets at full scale
    j.next_frame()
    j.afe_channels(5, (255, 255, 255))
    j.acquire(pixels=PROBE_PIXELS, lines=1, scan=0x03, status=0xB5, prefix=again)

    # -- gain line: the whole sensor at zero offset, unity gain
    j.next_frame()
    j.afe_channels(5, measured.zero_low)
    j.wait(0xD4)
    j.write((0x01, 0x02), (0x04, 0x62), (0x05, 0x40), *feedl(1), (0x22, 0x01), (0x23, 0x01),
            *dpiset_regs(PROBE_DPISET), (0x02, 0x00), (0x1F, 0x00), (0x1E, 0x10), (0x21, 0x01), (0x69, 0x01),
            (0x34, 0x14), *window(PROBE_START, PROBE_START + GAIN_LINE_PIXELS), *lincnt(1), *maxwd(gain_maxwd),
            (0x20, 0x17))
    j.afe_channels(2, (0, 0, 0))
    j.acquire(pixels=GAIN_LINE_PIXELS, lines=1, scan=0x03, status=0xB5, prefix=again)

    # -- probes 2a / 2b at the coarse gains
    j.next_frame()
    j.afe_channels(2, measured.gains)
    j.write(*window(PROBE_START, PROBE_START + PROBE_PIXELS), *dpiset_regs(PROBE_DPISET))
    j.afe_channels(5, (128, 128, 128))
    j.acquire(pixels=PROBE_PIXELS, lines=1, scan=0x03, status=0xB5, prefix=again)
    j.next_frame()
    j.afe_channels(5, (255, 255, 255))
    j.acquire(pixels=PROBE_PIXELS, lines=1, scan=0x03, status=0xB5, prefix=again)

    # -- dark frame: light off, the scan window, 128 lines without moving
    j.next_frame()
    j.afe_channels(5, measured.zero_high)
    j.wait(0xD4)
    if ir:
        j.ir_led(IR_LED_OFF)
    else:
        j.write((0x03, 0xAF))
    j.delay(316)
    scan_window_regs = [(0x34, 0x14), *window(start, end), *dpiset_regs(dpiset(dpi))]
    j.write((0x01, 0x02), (0x04, 0x62), (0x05, 0x40), *feedl(1), (0x22, 0x01), (0x23, 0x01), (0x02, 0x00),
            *scan_window_regs, (0x1F, 0x01), (0x1E, calib_linesel), (0x69, 0x01), (0x0D, 0x03), (0x21, 0x01),
            *lincnt(CALIBRATION_LINES), *maxwd(scan_maxwd), (0x20, res.bufsel[0]))
    if measured.dark_offsets is not None:
        j.afe_channels(5, measured.dark_offsets)
    j.acquire(pixels=pixels, lines=CALIBRATION_LINES, scan=0x03, status=0xA1)

    # -- white frame: dark-corrected shading, light on, 128 lines while moving
    j.next_frame()
    j.shading(measured.dark_shading, pixels)
    if ir:
        j.ir_led(IR_LED_ON)
    else:
        j.write((0x03, 0xBF))
    j.afe_channels(5, measured.white_offsets)
    j.wait(0xD4)
    j.delay(20)
    j.wait(0xD4)
    j.scan_tables(scan_table(cruise(res.linesel_calibration, WHITE_STEPS_PER_LINE)))
    motor = [(0x1C, 0x20), *lperiod(), *CLEAR_10_15]
    j.write((0x01, 0x22), (0x04, 0x62), (0x05, 0x40), *feedl(1), (0x22, 0x01), (0x23, 0x01), *scan_window_regs,
            (0x02, 0x10), (0x1F, 0x00), (0x1E, calib_linesel), (0x69, 0x01), (0x0D, 0x03), (0x80, 0xFF), *motor,
            (0x21, res.steps), (0x24, res.steps))
    j.write(*maxwd(scan_maxwd), (0x20, res.bufsel[1]), *lincnt(CALIBRATION_LINES))
    j.acquire(pixels=pixels, lines=CALIBRATION_LINES, scan=0x23, status=0xA5)

    # -- main scan: SilverFast's, with pyopticfilm's changes written with the start
    k = exposure_multiplier
    linesel = res.linesel_main
    if k > 1 or dummy_lines == "none":
        linesel = 0
    elif dummy_lines == "fewer":
        linesel = max(0, res.linesel_main - 1)
    lines = res.main_lines // 2 if single_sample else res.main_lines
    yres = dpi if single_sample else 2 * dpi
    vendor_cruise = cruise(res.linesel_main, main_steps_per_line(2 * dpi))
    main_cruise = cruise(linesel, main_steps_per_line(yres), lperiod=LPERIOD * k)
    # the table keeps SilverFast's shape: flat when SilverFast's cruise is slower than the ramp start
    main_table = scan_table(main_cruise, first=SCAN_RAMP_START if vendor_cruise < SCAN_RAMP_START else main_cruise)
    changes: list[tuple[int, int]] = []
    if linesel != res.linesel_main or k > 1 or single_sample:
        changes.append((0x1E, 0x10 | linesel))
        if k > 1:
            changes += [*lperiod(LPERIOD * k), (0x20, 0x08)]
        elif linesel != res.linesel_main:
            # two lines of buffer (MAXWD in 4-byte units), and no backtracking
            changes += [*maxwd(pixels * 6 >> 1), (0x20, 0x10), (0x02, 0x30 | ACDCDIS)]
        if single_sample:
            changes += lincnt(2 * lines)
    backtracking = not (k == 1 and linesel != res.linesel_main)

    j.bulk_done()
    j.delay(107)
    j.write((0x01, 0x22))
    j.read_register(0x40)
    j.wait(0xD4)
    j.shading(measured.shading, pixels)
    j.wait(0xD4)
    j.read_register(0x40)
    j.wait(0xD4)
    if measured.final_gains is not None:
        j.afe_channels(2, measured.final_gains)
        j.delay(111)
    first, second = POSITION_FEEDS
    j.write((0x80, 0xFF), (0x01, 0x22), (0x04, 0x62), (0x05, 0x48), *feedl(first), (0x22, 0x01), (0x23, 0x01),
            (0x1C, 0x20), *lperiod(), *scan_window_regs, (0x1E, 0x10 | res.linesel_main), *CLEAR_10_15,
            (0x21, res.steps), (0x24, res.steps), (0x02, 0x18), (0x6A, FMOVNO_SCAN))
    j.motor_table(3, POSITION_RAMP)
    j.write((0x0F, 0x01))
    j.delay(POSITION_STOP_MS[(mode, dpi)], timed=True)
    j.write((0x02, 0x08))  # the vendor's stop: then FEEDL=1 while the move runs
    j.write(*feedl(1))
    j.wait(0xF4)
    j.write((0x02, 0x18))
    j.wait(0xF4)
    move = [*lperiod(), (0x1C, 0x20), *CLEAR_10_15]  # the same registers as ``motor``, in this order
    j.write(*feedl(second), (0x02, 0x18), *move, (0x6A, FMOVNO_SCAN))
    j.motor_table(3, FAST_RAMP)
    j.write((0x0F, 0x01))
    j.wait(0xF4)
    j.scan_tables(main_table)
    j.write((0x02, 0x30), *feedl(1), *move, (0x6A, FMOVNO_SCAN), (0x69, 0x01), (0x0D, 0x03), *maxwd(scan_maxwd),
            (0x20, res.bufsel[2]), *lincnt(2 * res.main_lines), (0x05, 0x40))
    j.acquire(pixels=pixels, lines=lines, scan=0x23, status=0xA5, start=changes)

    # -- park: the chip returns the carriage (AGOHOME); white LED off, then on again
    j.delay(32)
    j.bulk_done()
    j.wait(0xA5)
    if ir:
        j.write((0x03, 0x8F))
        j.wait(0xA1)
        j.write((0x03, 0xAF))
    j.write((0x03, 0xAF))
    j.write((0x01, 0x22))
    j.read_register(0x40)
    j.delay(30)
    j.wait(0xA1)
    j.delay(30)
    j.wait(0xA1)
    j.write((0x03, 0x8F))
    j.wait(0xA1)
    j.write((0x03, 0xBF))

    line_seconds = LPERIOD * k * (linesel + 1) / PIXEL_CLOCK_HZ
    return {
        "name": f"{'Infrared' if ir else 'Colour'} {dpi} dpi",
        "dpi": dpi,
        "yres": yres,
        "mainFrame": len(j.frames) - 1,
        "frames": j.frames,
        "lamp": {"line": {"frame": FRAMES.index("gain_line")}, "dark": {"frame": FRAMES.index("dark")},
                 "shading": {"frame": WHITE_FRAME}},
        "moves": [1, first, second],
        "motorCruise": {"silverfast": vendor_cruise, "used": main_cruise},
        "scan": {"lPeriod": LPERIOD * k, "lineSel": linesel, "backtracking": backtracking,
                 "lineSeconds": round(line_seconds, 6), "seconds": round(lines * line_seconds, 1),
                 "bytesPerSecond": round(pixels * 6 / line_seconds)},
        "ops": j.ops,
    }


# --- SilverFast's calibration results and references -------------------------------------------


def _frame_steps(profile: dict) -> list[int]:
    """The frame step of each op: the number of frames completely read before it."""
    got: dict[int, int] = {}
    step, out = 0, []
    for op in profile["ops"]:
        out.append(step)
        if op["kind"] == "read":
            got[op["frame"]] = got.get(op["frame"], 0) + op["length"]
            if got[op["frame"]] >= profile["frames"][op["frame"]]["bytes"]:
                step += 1
    return out


def measured_from(profile: dict) -> Measured:
    """The calibration results a recorded (or built) job writes."""
    afe: dict[int, list[tuple[int, int]]] = {}
    shading: dict[int, list[dict]] = {}
    regs: dict[int, int] = {}
    for step, op in zip(_frame_steps(profile), profile["ops"], strict=True):
        if op["kind"] == "control" and op["rt"] == 0x40 and op["value"] == 0x83 and len(op["data"]) > 1:
            d = op["data"]
            if d[0] == 0x51:
                afe.setdefault(step, []).append((d[1], d[3] << 8 | d[5]))
            regs.update(zip(d[::2], d[1::2], strict=True))
        elif op["kind"] == "write" and not regs.get(0x5B, 0) & 0x40:
            shading.setdefault(step, []).append(op)

    def codes(step: int, first: int, nth: int = 0) -> tuple[int, int, int] | None:
        found = [v for a, v in afe.get(step, []) if first <= a < first + 3]
        part = found[3 * nth : 3 * nth + 3]
        return (part[0], part[1], part[2]) if len(part) == 3 else None

    def table(step: int) -> bytes | LiveShading:
        ops = shading.get(step, [])
        live = [op["shading"] for op in ops if "data" not in op]
        if live:
            widen = live[0].get("widen")
            return LiveShading(None if widen is None else (base64.b64decode(widen["table"]), widen["extra"]))
        return b"".join(base64.b64decode(op["data"]) for op in ops)

    dark, main = FRAMES.index("dark"), FRAMES.index("main")
    zero_low = codes(FRAMES.index("gain_line"), 5)
    gains = codes(FRAMES.index("probe_2a"), 2)
    zero_high = codes(dark, 5)
    white_offsets = codes(WHITE_FRAME, 5)
    dark_shading = table(WHITE_FRAME)
    if zero_low is None or gains is None or zero_high is None or white_offsets is None:
        raise ValueError("job has no recognisable calibration writes")
    if not isinstance(dark_shading, bytes):
        raise TypeError("the shading before the white frame must be a table")
    return Measured(zero_low=zero_low, gains=gains, zero_high=zero_high, dark_offsets=codes(dark, 5, 1),
                    white_offsets=white_offsets, final_gains=codes(main, 2), dark_shading=dark_shading,
                    shading=table(main))


def _reference_job(model, mode: str, dpi: int) -> dict:
    """The recorded job whose calibration a job starts from: 7200 dpi infrared was not recorded and
    takes the 3600 dpi infrared job's."""
    return model.recorded_job("infrared", 3600) if (mode, dpi) == ("infrared", 7200) else model.recorded_job(mode, dpi)


def vendor_measured(model, mode: str, dpi: int, *, first_pixel: int | None = None) -> Measured:
    """SilverFast's calibration results for a job (written when live calibration is off).

    The recorded job's AFE codes and shading tables. A window widened to ``first_pixel`` copies
    the first recorded pixels into the added ones before the white frame, then computes every
    gain from it. 7200 dpi infrared (not recorded) takes the 3600 dpi infrared codes, unity
    shading while the white frame is read, and SilverFast's infrared shading from it.
    """
    pixels = window_pixels(dpi, first_pixel)
    extra = pixels - window_pixels(dpi)
    m = measured_from(_reference_job(model, mode, dpi))
    if (mode, dpi) == ("infrared", 7200):
        return replace(m, dark_shading=shading_table(np.full((pixels, 3), SHADING_UNITY)), shading=LiveShading())
    if not extra:
        return m
    shading = m.shading if isinstance(m.shading, LiveShading) else LiveShading((m.shading, extra))
    return replace(m, dark_shading=shading_blocks(widen_words(m.dark_shading, pixels - extra, extra)), shading=shading)


def vendor_lamp(model, mode: str, dpi: int) -> dict:
    """SilverFast's line, dark and white frame levels for a job, which scans are checked against."""
    return _reference_job(model, mode, dpi)["lamp"]


def job_for(model, mode: str, dpi: int, *, dummy_lines: str | None = None) -> dict:
    """The job a scan runs: the model's window, dummy lines and sampling, SilverFast's references."""
    job = build_job(mode, dpi, vendor_measured(model, mode, dpi, first_pixel=model.window_start),
                    first_pixel=model.window_start, dummy_lines=dummy_lines or model.dummy_lines,
                    single_sample=model.single_sample)
    job["lamp"] = vendor_lamp(model, mode, dpi)
    return job


# --- reference ----------------------------------------------------------------------------------


def _wait_kind(status: int, scanning: bool, moved: bool) -> str:
    """What a status read waits for, from the value that ended it.

    Before the job's first move the vendor polls until the motor is idle; whether ``FEEDFSH`` is
    still latched from an earlier move is the scanner's state, not the job's.
    """
    if not scanning and not moved and not status & MOTORENB:
        return "motor idle"
    if not scanning and status & (FEEDFSH | MOTORENB) == FEEDFSH:
        return "move done"
    if scanning and not status & BUFEMPTY:
        return "scan data"
    return "status"


def host_program(profile: dict) -> list[tuple]:
    """What the host decides in a job, for comparing jobs (see the module docstring).

    Consecutive status reads are one wait (the last one ends it); consecutive identical register
    reads and transfer-complete polls are one poll; consecutive bulk writes of whole 512-byte
    packets are one payload (the scanner receives the same packets); write acknowledges and
    untimed delays are left out.
    """
    out: list[tuple] = []
    scanning = moved = False
    address: int | None = None

    def add(item: tuple, *, merge: bool = False) -> None:
        if merge and out and out[-1][0] == item[0] and (item[0] == "wait" or out[-1] == item):
            out[-1] = item
        else:
            out.append(item)

    def bulk(payload: bytes | int) -> None:
        last = out[-1] if out and out[-1][0] == "bulk write" else None
        if last is not None and type(last[1]) is type(payload) and (
                (len(last[1]) if isinstance(last[1], bytes) else last[1]) % PACKET == 0):
            out[-1] = ("bulk write", last[1] + payload)
        else:
            out.append(("bulk write", payload))

    for op in profile["ops"]:
        kind = op["kind"]
        if kind == "delay":
            if op.get("timedStop"):
                add(("timed stop",))
            continue
        if kind == "read":
            add(("read frame", op["frame"], op["length"]))
            continue
        if kind == "write":
            bulk(base64.b64decode(op["data"]) if "data" in op else int(op["length"]))
            continue
        if op["rt"] == 0x40:
            data = list(op["data"])
            if op["value"] == 0x83 and len(data) == 1:
                address = data[0]  # completed by the read that follows
                continue
            if address is not None:
                add(("address", address))
                address = None
            if op["value"] == 0x83:
                pairs = dict(zip(data[::2], data[1::2], strict=True))
                if 0x01 in pairs:
                    scanning = bool(pairs[0x01] & 0x01)
                moved |= pairs.get(0x0F) == 0x01
            add(("write", op["request"], op["value"], op["index"], tuple(data)))
            continue
        if op["value"] == 0x8E and op["index"] == 0x20:
            continue  # write acknowledge
        if op["value"] == 0x84 and address == op.get("register") == STATUS:
            add(("wait", _wait_kind(op["expected"][0], scanning, moved)), merge=True)
        elif op["value"] == 0x84 and address is not None:
            add(("read register", address), merge=True)
        elif op["value"] == 0x8E and op["index"] == 0x18:
            add(("transfer done",), merge=True)
        else:
            if address is not None:
                add(("address", address))
            add(("read", op["request"], op["value"], op["index"], op["length"]))
        address = None
    return out
