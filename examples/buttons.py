# SPDX-License-Identifier: GPL-3.0-or-later
"""Listen for the scanner's QuickScan / IntelliScan buttons.

Opens the first scanner, polls its USB interrupt endpoint and prints which
button was pressed. No lamp, motor or calibration is touched. A host app would
map ``Button.QUICK_SCAN`` to a preview and ``Button.INTELLI_SCAN`` to a full
scan, and stop polling while a scan is running.

GL128 only (tested on the 8200i SE and 8300i SE)::

    uv run python examples/buttons.py
    uv run python examples/buttons.py --seconds 30
"""

from __future__ import annotations

import argparse
import sys
import time

from pyopticfilm import Scanner, UsbError
from pyopticfilm.asic.buttons import Button

ACTIONS = {
    Button.QUICK_SCAN: "preview scan",
    Button.INTELLI_SCAN: "full scan",
}


def poll_buttons(scanner: Scanner, timeout_ms: int = 250) -> set[Button]:
    """One poll; an idle scanner times out or returns no bytes, which is no press."""
    try:
        return scanner.button_state(timeout_ms=timeout_ms)
    except IndexError:
        return set()
    except UsbError as exc:
        if "time" in str(exc).lower():
            return set()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=None, help="Stop after this long (default: until Ctrl+C)")
    args = parser.parse_args()

    with Scanner.open() as scanner:
        if not hasattr(scanner.asic, "button_state"):
            print(f"{scanner.model.model} has no button support in pyopticfilm yet.", file=sys.stderr)
            return 1

        print(f"Opened {scanner.model.model}. Press QuickScan or IntelliScan (Ctrl+C to stop).")
        end = None if args.seconds is None else time.monotonic() + args.seconds
        try:
            while end is None or time.monotonic() < end:
                for button in poll_buttons(scanner):
                    print(f"{time.strftime('%H:%M:%S')}  {button.value:<12} -> {ACTIONS[button]}")
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
