# SPDX-License-Identifier: GPL-3.0-or-later
"""Calibration diagnostics bundle for bug reports.

Runs one forced colour calibration (no film needed; the carriage stays at
home) against a throwaway calibration cache, so an existing cached table can
never mask the result, and writes into a new folder:

- ``info.json``: pyopticfilm version and install path, detected model,
  exposure, per-scan sensor registers, calibration cache version.
- ``debug.log``: full DEBUG log (AFE probe / wide strip / search result,
  shading accept or reject).
- ``usb.jsonl``: every USB transfer, including bulk-in payloads (the AFE and
  shading strips, a few MB in total).

The folder is also zipped for attaching to an issue. A rejected calibration
is logged, not raised, so the bundle is complete either way.

``--reg ADDR=VALUE`` (repeatable) overlays a register onto the model's
per-scan sensor registers for this run only, which are written before every
calibration acquire. Use it for A/B experiments; nothing is persisted.

Usage (scanner on, NegPy / other scan apps closed)::

    python tools/diag_calibration.py
    python tools/diag_calibration.py --resolution 3600
    python tools/diag_calibration.py --reg 0x36=0x04
    python tools/diag_calibration.py --fake 8300i  # mock USB, no hardware
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pyopticfilm
from pyopticfilm import Scanner
from pyopticfilm.logging import enable_debug_logging
from pyopticfilm.scan import calibrate as calib_mod
from pyopticfilm.usb.protocol import GenesysUsbProtocol
from pyopticfilm.usb.trace import RecordingTransport, UsbTransaction

FAKE_MODELS = ("8300i", "8200i-se", "8100-v2")


class JsonlSink:
    """Streams transactions (and bulk-in payloads) to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self._f = path.open("w", encoding="utf-8")

    def __call__(self, txn: UsbTransaction) -> None:
        self._f.write(json.dumps(txn.to_json()) + "\n")

    def write_data(self, data: bytes) -> None:
        self._f.write(json.dumps({"op": "bulk_data", "len": len(data), "hex": bytes(data).hex()}) + "\n")

    def close(self) -> None:
        self._f.close()


class PayloadRecorder(RecordingTransport):
    """:class:`RecordingTransport` that also logs bulk-in payloads."""

    def bulk_read(self, size: int, *, timeout_ms: int | None = None) -> bytes:
        data = super().bulk_read(size, timeout_ms=timeout_ms)
        if isinstance(self.sink, JsonlSink):
            self.sink.write_data(data)
        return data


def _fake_model(name: str) -> Any:
    if name == "8300i":
        from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE

        return MODEL_8300I_SE
    if name == "8200i-se":
        from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE

        return MODEL_8200I_SE
    from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2

    return MODEL_8100_V2


def parse_reg(text: str) -> tuple[int, int]:
    addr, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected ADDR=VALUE, got {text!r}")
    a, v = int(addr, 0), int(value, 0)
    if not (0 <= a <= 0x1FF and 0 <= v <= 0xFF):
        raise argparse.ArgumentTypeError(f"register out of range: {text!r}")
    return a, v


def with_overrides(model: Any, overrides: dict[int, int]) -> Any:
    if not overrides:
        return model
    regs = dict(getattr(model, "sensor_custom_regs", None) or {})
    regs.update(overrides)
    return dataclasses.replace(model, sensor_custom_regs=regs)


def build_info(model: Any, resolution: int, overrides: dict[int, int]) -> dict[str, Any]:
    sensor = dict(getattr(model, "sensor_custom_regs", None) or {})
    return {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pyopticfilm_version": pyopticfilm.__version__,
        "pyopticfilm_path": str(Path(pyopticfilm.__file__).parent),
        "python": sys.version,
        "platform": sys.platform,
        "model": getattr(model, "model", str(model)),
        "asic": getattr(model, "asic", None),
        "exposure_short": getattr(model, "exposure_short", None),
        "resolution": resolution,
        "calib_cache_version": getattr(calib_mod, "CACHE_VERSION", None),
        "calib_fingerprint": (calib_mod.calib_fingerprint(model) if hasattr(calib_mod, "calib_fingerprint") else None),
        "register_overrides": {f"0x{a:02X}": f"0x{v:02X}" for a, v in sorted(overrides.items())},
        "sensor_custom_regs": {f"0x{a:02X}": f"0x{v:02X}" for a, v in sorted(sensor.items())},
    }


def run(out_dir: Path, *, resolution: int, fake: str | None, overrides: dict[int, int]) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    enable_debug_logging()
    log = logging.getLogger("pyopticfilm")
    fh = logging.FileHandler(out_dir / "debug.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(fh)
    sink = JsonlSink(out_dir / "usb.jsonl")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "calib.json"
            if fake:
                from pyopticfilm.usb.fake import MockScannerTransport

                rec = PayloadRecorder(MockScannerTransport(), sink=sink, keep_in_memory=False)
                model = with_overrides(_fake_model(fake), overrides)
                scanner = Scanner.open_fake(model, rec, calib_cache=cache)
            else:
                from pyopticfilm.device.select import model_for_device
                from pyopticfilm.usb.device import UsbDeviceHandle

                handle = UsbDeviceHandle.open(None)
                rec = PayloadRecorder(handle, sink=sink, keep_in_memory=False)
                detected = model_for_device(handle.info.product_id, getattr(handle.info, "bcd_device", 0))
                model = with_overrides(detected, overrides)
                scanner = Scanner(handle, GenesysUsbProtocol(rec), model=model, calib_cache=cache)
            info = build_info(scanner.model, resolution, overrides)
            (out_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            print(json.dumps(info, indent=2))
            try:
                print(f"Calibrating {info['model']} at {resolution} dpi...")
                scanner.calibrate(resolution=resolution, force=True)
                print("Calibration accepted.")
            except Exception as exc:
                log.exception("diagnostic calibration failed")
                print(f"Calibration failed: {exc}")
            finally:
                scanner.close()
    finally:
        sink.close()
        log.removeHandler(fh)
        fh.close()

    archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
    print(f"Wrote {out_dir}")
    print(f"Attach {archive}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--resolution", type=int, default=1200, help="calibration dpi (default 1200)")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder (default ./pyopticfilm-diag-<timestamp>)",
    )
    parser.add_argument("--fake", choices=FAKE_MODELS, default=None, help="use mock USB for this model")
    parser.add_argument(
        "--reg",
        type=parse_reg,
        action="append",
        default=[],
        metavar="ADDR=VALUE",
        help="overlay a per-scan sensor register for this run (repeatable), e.g. 0x36=0x04",
    )
    args = parser.parse_args(argv)
    out = args.out or Path.cwd() / f"pyopticfilm-diag-{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    return run(out, resolution=args.resolution, fake=args.fake, overrides=dict(args.reg))


if __name__ == "__main__":
    raise SystemExit(main())
