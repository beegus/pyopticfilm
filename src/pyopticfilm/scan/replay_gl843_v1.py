# SPDX-License-Identifier: GPL-3.0-or-later
"""Run an OpticFilm 7600i v1 job (``control`` / ``write`` / ``read`` / ``delay`` ops) on the scanner.

Jobs come from :mod:`pyopticfilm.scan.job_gl843_v1`. Status polls are live waits: write
acknowledge, bulk completion, motor idle before a start, FEEDFSH after the job's own move,
BUFEMPTY before image reads. Shading may be computed from the job's white frame as it runs.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from pyopticfilm.exceptions import AsicError, ScanCancelled, ScanError
from pyopticfilm.scan.calib_gl843_v1 import Calibration

STATUS = 0x41
MOTORENB = 0x01
FEEDFSH = 0x20
BUFEMPTY = 0x40
CHUNK = 0x40000
PACKET = 512
THROUGHPUT_GRACE_S = 6.0
THROUGHPUT_MIN = 0.9
SHADING_UNITY = 0x2000
#: SilverFast's infrared shading: no dark offset, gain = target * unity / white (all channels)
IR_SHADING_TARGET = 0x13000

_ADDR_OP = {"kind": "control", "rt": 0x40, "request": 0x0C, "value": 0x83, "index": 0, "data": [STATUS]}
_ACK_OP = {"kind": "control", "rt": 0xC0, "request": 0x0C, "value": 0x8E, "index": 0x20, "length": 1}


def _writes_motor_start(op: dict) -> bool:
    d = op.get("data") or []
    return op.get("rt") == 0x40 and op.get("value") == 0x83 and len(d) > 1 and any(
        d[i] == 0x0F and d[i + 1] == 1 for i in range(0, len(d) - 1, 2)
    )


def shading_table(gains: np.ndarray, dark: tuple[int, int, int] = (0, 0, 0)) -> bytes:
    """GL843 shading upload: dark and gain words for R, G, B per pixel, 42 pixels per 512 bytes."""
    n = -(-len(gains) // 42)
    words = np.zeros((n * 42, 6), "<u2")
    words[:, 0::2] = dark
    words[: len(gains), 1::2] = gains
    blocks = np.zeros((n, 256), "<u2")
    blocks[:, :252] = words.reshape(n, 252)
    return blocks.tobytes()


def white_shading(white: bytes, pixels: int) -> bytes:
    """Infrared shading from a white frame read at unity gain."""
    mean = np.frombuffer(white, "<u2").reshape(-1, pixels, 3).mean(axis=0)
    return shading_table(np.clip(np.rint(IR_SHADING_TARGET * SHADING_UNITY / np.maximum(mean, 1)), 0, 0xFFFF))


def widen_words(table: bytes, pixels: int, extra: int) -> np.ndarray:
    """A shading table's per-pixel words with ``extra`` pixels in front: the median of the first
    recorded pixels of the same column parity (dark words differ between even and odd columns)."""
    words = np.frombuffer(table, "<u2").reshape(-1, 256)[:, :252].reshape(-1, 6)[:pixels]
    ref = np.stack([np.median(words[p:16:2], axis=0) for p in range(2)])
    return np.concatenate([np.rint(ref[(np.arange(extra) - extra) % 2]), words]).astype("<u2")


def shading_blocks(words: np.ndarray) -> bytes:
    n = -(-len(words) // 42)
    blocks = np.zeros((n, 256), "<u2")
    blocks[:, :252] = np.concatenate([words, np.zeros((n * 42 - len(words), 6), "<u2")]).reshape(n, 252)
    return blocks.tobytes()


def widened_shading(table: bytes, extra: int, white: bytes, pixels: int) -> bytes:
    """A recorded final shading table with ``extra`` pixels added in front, every gain computed from
    the white frame (read at unity gain) for the per-channel target the recorded gains give."""
    mean = np.maximum(np.frombuffer(white, "<u2").reshape(-1, pixels, 3).mean(axis=0), 1)
    words = widen_words(table, pixels - extra, extra)
    target = np.median(words[extra:, 1::2] * mean[extra:], axis=0)
    words[:, 1::2] = np.clip(np.rint(target / mean), 0, 0xFFFF)
    return shading_blocks(words)


def validate_profile(profile: dict) -> None:
    """Refuse a profile whose frames are incomplete or whose image reads outlast the scan."""
    planned: dict[int, int] = {}
    frames = profile["frames"]
    main = profile["mainFrame"]
    for op in profile["ops"]:
        if op["kind"] == "read":
            planned[op["frame"]] = planned.get(op["frame"], 0) + op["length"]
            if planned[op["frame"]] > frames[op["frame"]]["bytes"]:
                raise AsicError("invalid frame read budget")
        elif op["kind"] == "control" and 0 < planned.get(main, 0) < frames[main]["bytes"]:
            d = op.get("data") or []
            done = op["rt"] == 0xC0 and op["value"] == 0x8E and op["index"] == 0x18
            shutdown = op["rt"] == 0x40 and op["value"] == 0x83 and len(d) > 1 and any(
                (d[i] == 1 and not d[i + 1] & 1) or (d[i] == 3 and not d[i + 1] & 0x10)
                for i in range(0, len(d) - 1, 2)
            )
            if done or shutdown:
                raise AsicError("main frame incomplete before transfer completion or shutdown")
    if any(planned.get(i) != f["bytes"] for i, f in enumerate(frames)):
        raise AsicError("incomplete frame in profile")


@dataclass
class ReplayHooks:
    check: Callable[[], None] = lambda: None
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    progress: Callable[[int, int, int], None] | None = None
    #: called with (frame, bytes) when a calibration frame (white line, dark, white) is complete
    frame_done: Callable[[int, bytes], None] | None = None
    timeout_s: float = 60.0
    read_chunk: int = CHUNK


@dataclass
class ReplayResult:
    main: bytearray
    frames: dict[int, bytes] = field(default_factory=dict)
    positioning_stop_ms: float | None = None


def run_profile(profile: dict, usb, hooks: ReplayHooks | None = None, *, calibrate: bool = False) -> ReplayResult:
    """Replay a job; ``calibrate`` replaces recorded AFE/shading with live measurements."""
    h = hooks or ReplayHooks()
    validate_profile(profile)
    transport = usb.transport
    frames = profile["frames"]
    main = profile["mainFrame"]
    calibration = Calibration(profile) if calibrate else None
    lamp = profile.get("lamp") or {}
    calib_frames = {lamp[k]["frame"] for k in ("line", "shading", "dark") if k in lamp}
    if calibration:
        calib_frames.update(range(calibration.first, calibration.white + 1))
    # without backtracking a slow host only pauses the carriage, so the rate is not enforced
    need = profile["scan"]["bytesPerSecond"] if profile["scan"].get("backtracking", True) else 0
    regs: dict[int, int] = {}
    st = {"address": 0, "status": None, "moved": None}
    buffers = {i: bytearray(frames[i]["bytes"]) for i in calib_frames | {main}}
    got = dict.fromkeys(range(len(frames)), 0)
    result = ReplayResult(main=buffers[main])
    shading: dict[int, bytes] = {}
    pool = ThreadPoolExecutor(max_workers=1)

    def raw(op: dict) -> bytes:
        if op["rt"] == 0x40:
            transport.control_msg(0x40, op["request"], op["value"], op["index"], bytes(op["data"]))
            if op["value"] == 0x83:
                d = op["data"]
                if len(d) == 1:
                    st["address"] = d[0]
                for i in range(0, len(d) - 1, 2):
                    regs[d[i]] = usb.shadow[d[i]] = d[i + 1]
            return b""
        r = bytes(transport.control_msg(0xC0, op["request"], op["value"], op["index"], op["length"]))
        if len(r) != op["length"]:
            raise ScanError("short control read")
        if op["value"] == 0x84:  # the GL843 auto-increments the address after a read
            if st["address"] == STATUS:
                st["status"] = r[0]
            st["address"] = (st["address"] + 1) & 0xFF
        return r

    def wait(r: bytes, again: Callable[[], bytes], ready: Callable[[int], bool], why: str, pause: float) -> None:
        deadline = h.now() + h.timeout_s
        while not ready(r[0]):
            h.check()
            if h.now() > deadline:
                raise ScanError(f"timeout {why} (status 0x{r[0]:02x})")
            h.sleep(pause)
            r = again()

    def ack() -> bytes:
        return raw(_ACK_OP)

    def poll(op: dict, r: bytes, mask: int, target: int, why: str) -> None:
        def status() -> bytes:
            raw(_ADDR_OP)
            wait(ack(), ack, lambda s: bool(s & 1), "waiting for write acknowledge", 0.005)
            return raw(op)

        wait(r, status, lambda s: s & mask == target, why, 0.015)

    def control(op: dict) -> None:
        if op["rt"] == 0x40:
            start = _writes_motor_start(op)
            if start and st["status"] is not None and st["status"] & MOTORENB:
                poll({**_ACK_OP, "index": 0, "value": 0x84}, bytes([st["status"]]), MOTORENB, 0, "motor before start")
            raw(op)
            if start:
                st["status"], st["moved"] = None, h.now()
            return
        r = raw(op)
        if op["value"] == 0x8E and op["index"] == 0x20:
            wait(r, lambda: raw(op), lambda s: bool(s & 1), "waiting for write acknowledge", 0.005)
        elif op["value"] == 0x8E and op["index"] == 0x18:
            wait(r, lambda: raw(op), lambda s: not s & 0x0C, "waiting for bulk completion", 0.02)
        elif op["value"] == 0x84 and op.get("register") == STATUS:
            expected = op["expected"][0]
            scanning = regs.get(1, 0) & 1
            if not scanning and expected & (FEEDFSH | MOTORENB) == FEEDFSH:
                # FEEDFSH latches the end of a move; before this job's own move it is the vendor's.
                if st["moved"] is None:
                    poll(op, r, MOTORENB, 0, "waiting for the previous move")
                else:
                    poll(op, r, FEEDFSH | MOTORENB, FEEDFSH, "waiting for the positioning move")
            elif scanning and not expected & BUFEMPTY:
                poll(op, r, BUFEMPTY, 0, "waiting for scan data")

    def read(op: dict) -> None:
        i = op["frame"]
        size = frames[i]["bytes"]
        left = asked = op["length"]

        def request():
            # whole 512-byte packets, remainder on its own (a read ending mid-packet overflows)
            nonlocal asked
            n = min(h.read_chunk, asked)
            n = n - n % PACKET if n >= PACKET else n
            asked -= n
            return pool.submit(lambda: (bytes(transport.bulk_read(n)), n))

        pending = request()  # keep one transfer in flight so the scanner's buffer never fills
        t0 = None
        since = 0
        while left:
            h.check()
            part, n = pending.result()
            if not part or len(part) > n or got[i] + len(part) > size:
                raise ScanError(f"frame {i}: bulk read returned {len(part)} of {n} bytes")
            left -= len(part)
            asked += n - len(part)  # a short read is asked for again
            if asked:
                pending = request()
            if i in buffers:
                buffers[i][got[i] : got[i] + len(part)] = part
            got[i] += len(part)
            if t0 is None:
                t0 = h.now()
            elif i == main:
                since += len(part)
                secs = h.now() - t0
                if secs > THROUGHPUT_GRACE_S and since / secs < need * THROUGHPUT_MIN:
                    raise ScanError(f"scan data at {since / secs / 1e6:.2f} MB/s, the scan needs "
                                    f"{need / 1e6:.2f} MB/s")
            if h.progress is not None:
                h.progress(i, got[i], size)
        if i != main and i in buffers and got[i] == size:
            result.frames[i] = bytes(buffers[i])
            if calibration:
                calibration.frame_done(i, result.frames[i])
            if h.frame_done is not None and i in {lamp[k]["frame"] for k in lamp}:
                h.frame_done(i, result.frames[i])

    try:
        for op in profile["ops"]:
            h.check()
            if op["kind"] == "delay":
                if not op.get("timedStop"):
                    h.sleep(op["ms"] / 1000)
                    continue
                # The vendor stops the first positioning move on the position-sensor event,
                # 2.56-2.57 s after the start; stop at the recorded moment.
                if st["moved"] is None:
                    raise AsicError("timed stop without a preceding motor start")
                target = st["moved"] + op["ms"] / 1000
                while h.now() < target:
                    h.check()
                    h.sleep(max(0.001, min(0.02, target - h.now())))
                result.positioning_stop_ms = (h.now() - st["moved"]) * 1000
            elif op["kind"] == "control":
                h.check()
                control(calibration.control(op) if calibration else op)
            elif op["kind"] == "write":
                if calibration and (regs.get(0x5B), regs.get(0x5C)) == (0x10, 0):
                    size = op.get("length") or len(base64.b64decode(op["data"]))
                    payload = calibration.upload(size)
                elif "shading" in op:
                    f, start = op["shading"]["frame"], op["shading"]["offset"]
                    if f not in shading:
                        w = op["shading"].get("widen")
                        shading[f] = (widened_shading(base64.b64decode(w["table"]), w["extra"], result.frames[f],
                                                      frames[f]["pixels"]) if w
                                      else white_shading(result.frames[f], frames[f]["pixels"]))
                    payload = shading[f][start : start + op["length"]]
                else:
                    payload = base64.b64decode(op["data"])
                if transport.bulk_write(payload) != len(payload):
                    raise ScanError("short bulk upload")
            else:
                read(op)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if any(got[i] != f["bytes"] for i, f in enumerate(frames)):
        raise ScanError("incomplete frame")
    return result


def check_cancel(cancel) -> Callable[[], None]:
    def check() -> None:
        if cancel is not None and cancel.is_set():
            raise ScanCancelled("cancelled")

    return check
