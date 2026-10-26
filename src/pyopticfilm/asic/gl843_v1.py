# SPDX-License-Identifier: GPL-3.0-or-later
"""GL843 chip operations for OpticFilm 7600i v1 (bcdDevice 4.00): boot, lamp, homing, status.

Registers go through :class:`~pyopticfilm.usb.protocol.Gl843UsbProtocol` with every write
acknowledged (``0x8E``/``0x20``), as the vendor driver does. Boot and homing values are in
:mod:`pyopticfilm.device.tables_7600i_v1`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from pyopticfilm.asic.status import ScannerStatus
from pyopticfilm.device.protocol import ScanMethod
from pyopticfilm.device.tables_7600i_v1 import (
    AFE_RESET,
    BOOT_AFE,
    BOOT_CLOCK,
    BOOT_GPIO,
    FAST_RAMP,
    HOME_MOTOR_REGS,
    HOME_REG_01,
    POSITION_RAMP,
    POWER_ON_REGS,
    SCAN_REGS,
)
from pyopticfilm.exceptions import MotorTimeoutError, UsbError
from pyopticfilm.logging import get_logger
from pyopticfilm.usb.protocol import (
    INDEX,
    VALUE_BUFFER,
    VALUE_SET_REGISTER,
    GenesysUsbProtocol,
    Gl843UsbProtocol,
    UsbTransport,
)

logger = get_logger(__name__)

REG_STATUS = 0x41
STATUS_HOME = 0x08
STATUS_MOTORENB = 0x01
STATUS_FEEDFSH = 0x20
LAMP_BIT = 0x10  # 0x03: white LED
IR_GPIO_ON = 0x27  # 0xA8: GPIO27 drives the infrared LED
IR_GPIO_OFF = 0x20
MTRREV = 0x04  # 0x02: reverse
BULK_PACKET = 512


class Gl843V1Usb:
    """Status, AFE and motor-table helpers over a :class:`Gl843UsbProtocol` that acknowledges writes."""

    def __init__(self, usb: Gl843UsbProtocol | UsbTransport) -> None:
        self.protocol = Gl843UsbProtocol.wrap(usb, acknowledge_writes=True)
        self.transport = self.protocol.transport
        #: last acknowledged value of each register
        self.shadow = self.protocol.written

    def write_register(self, address: int, value: int) -> None:
        self.protocol.write_register(address, value)

    def write_registers(self, pairs: Sequence[tuple[int, int]]) -> None:
        self.protocol.write_registers_batched(list(pairs))

    def read_register(self, address: int) -> int:
        return self.protocol.read_register(address)

    def status(self) -> int:
        return self.read_register(REG_STATUS)

    def write_afe(self, addr: int, value: int) -> None:
        self.protocol.write_fe_register(addr, value)

    def upload_table(self, slot: int, table: Sequence[int]) -> None:
        """Motor table into RAM slot ``slot``."""
        addr = 0x40000 + 0x8000 * slot
        data = b"".join(bytes((v & 0xFF, v >> 8)) for v in table)
        self.write_register(0x5B, (addr >> 12) & 0xFF)
        self.write_register(0x5C, (addr >> 4) & 0xFF)
        self.protocol.control_out(VALUE_SET_REGISTER, INDEX, [0x28])
        n = len(data)
        self.protocol.control_out(VALUE_BUFFER, INDEX, [1, 0, 0, 0, n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF, 0])
        pad = data + bytes(-n % BULK_PACKET)
        if self.transport.bulk_write(pad) != len(pad):
            raise UsbError("short motor table upload")
        for _ in range(256):
            if not self.protocol.control_in(0x8E, 0x18, 1)[0] & 0x0C:
                break
        else:
            raise UsbError("motor table upload did not complete")
        self.write_register(0x5B, ((addr >> 12) & 0xFF) & ~0x40)


class Gl843V1Motion:
    """Homing from the vendor's motor primitives: fast reverse to the home sensor, 600 steps
    forward, then a slow reverse approach (the positioning ramp) stopped when the sensor trips."""

    def __init__(self, usb: Gl843V1Usb, *, sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.monotonic) -> None:
        self.usb = usb
        self.sleep = sleep
        self.now = now
        #: the table in slot 3, if this object uploaded it and nothing has overwritten it since
        self.loaded: tuple[int, ...] | None = None
        self._base01 = HOME_REG_01 & ~0x01

    @staticmethod
    def _feedl(n: int) -> list[tuple[int, int]]:
        return [(0x3D, (n >> 16) & 0x0F), (0x3E, (n >> 8) & 0xFF), (0x3F, n & 0xFF)]

    def wait_idle(self, timeout_s: float) -> int:
        t0 = self.now()
        s = self.usb.status()
        while s & STATUS_MOTORENB:
            if self.now() - t0 > timeout_s:
                raise MotorTimeoutError(f"motor still running after {timeout_s:.0f} s")
            self.sleep(0.01)
            s = self.usb.status()
        return s

    def stop(self) -> int:
        """Vendor stop: ``0x02=0x08`` then FEEDL=1 while the move runs."""
        self.usb.write_registers([(0x01, self._base01)])
        self.usb.write_registers([(0x02, 0x08)])
        self.usb.write_registers(self._feedl(1))
        return self.wait_idle(5.0)

    def move(self, steps: int, *, reverse: bool, table: tuple[int, ...], fmovno: int, timeout_s: float,
             until: Callable[[int], bool] | None = None) -> tuple[int, bool]:
        self.wait_idle(5.0)
        if self.loaded != table:
            self.usb.upload_table(3, table)
            self.loaded = table
        pairs = [(0x01, self._base01), *HOME_MOTOR_REGS.items()]
        pairs += [(r, 0) for r in range(0x10, 0x16)]
        pairs += [(0x6A, fmovno), (0x02, 0x18 | (MTRREV if reverse else 0))] + self._feedl(steps)
        self.usb.write_registers(pairs)
        self.usb.write_registers([(0x0F, 1)])
        t0 = self.now()
        while True:
            s = self.usb.status()
            if until is not None and until(s):
                self.stop()
                return self.usb.status(), True
            if not s & STATUS_MOTORENB and s & STATUS_FEEDFSH:
                return s, False
            if self.now() - t0 > timeout_s:
                self.stop()
                raise MotorTimeoutError(f"move did not finish in {timeout_s:.0f} s")
            self.sleep(0.002)

    def home(self, timeout_s: float = 30.0) -> int:
        s = self.usb.status()
        if s & STATUS_MOTORENB:
            s = self.stop()
        at_home = lambda x: bool(x & STATUS_HOME)
        if not s & STATUS_HOME:
            _, reached = self.move(80000, reverse=True, table=FAST_RAMP, fmovno=0xFF, until=at_home,
                                   timeout_s=timeout_s)
            if not reached:
                raise MotorTimeoutError("home sensor not found")
        for _ in range(5):
            s, _ = self.move(600, reverse=False, table=FAST_RAMP, fmovno=0x46, timeout_s=5.0)
            if not s & STATUS_HOME:
                break
        else:
            raise MotorTimeoutError("home sensor stays on after moving forward")
        s, reached = self.move(3000, reverse=True, table=POSITION_RAMP, fmovno=0xFF, until=at_home,
                               timeout_s=10.0)
        if not reached:
            raise MotorTimeoutError("home sensor not found on the final approach")
        return s


class Gl843V1:
    """High-level ASIC ops for the 7600i v1: boot, lamp, home/park, status."""

    def __init__(self, protocol: GenesysUsbProtocol | UsbTransport, model) -> None:
        self.protocol = Gl843UsbProtocol.wrap(protocol, acknowledge_writes=True)
        self.model = model
        self.usb = Gl843V1Usb(self.protocol)
        self._initialized = False
        self._scan_method: ScanMethod = "transparency"
        #: Set by homing and by a complete scan (the chip parks the carriage itself).
        self.position_known = False
        self.lamp_on_at: float | None = None
        self._motion: Gl843V1Motion | None = None

    @property
    def _reg_cache(self) -> dict[int, int]:
        return self.usb.shadow

    def read_status(self) -> ScannerStatus:
        return ScannerStatus.from_reg41(self.usb.status())

    def read_status_reliable(self) -> ScannerStatus:
        return self.read_status()

    def is_at_home(self) -> bool:
        return self.read_status().is_at_home

    def init(self, *, force: bool = False) -> None:
        """Vendor boot: the power-on and scan register files, GPIO, AFE."""
        if self._initialized and not force:
            return
        self.protocol.read_usb_speed_byte()
        self.usb.write_registers(sorted(POWER_ON_REGS.items()))
        self.protocol.write_0x8c(0x10, BOOT_CLOCK)
        self.usb.write_registers([(0x0B, 0x4A)])
        self._reset_afe()
        self.usb.write_registers([(0x03, 0xBF)])
        self.protocol.write_0x8c(0x10, BOOT_CLOCK)
        self.usb.write_registers(sorted(SCAN_REGS.items()))
        self._reset_afe()
        self.usb.write_registers(list(BOOT_GPIO))
        time.sleep(0.1)
        for addr, value in sorted(BOOT_AFE.items()):
            self.usb.write_afe(addr, value)
        self._initialized = True
        self.position_known = False
        self.forget_motor_tables()

    def _reset_afe(self) -> None:
        for addr, value in enumerate(AFE_RESET):
            self.usb.write_afe(addr, value)

    def set_scan_method(self, method: ScanMethod) -> None:
        self._scan_method = method

    def lamp_on(self) -> None:
        reg03 = self.usb.shadow.get(0x03, 0x9F)
        if self._scan_method == "infrared":
            self.usb.write_registers([(0x03, reg03 & ~LAMP_BIT), (0xA8, IR_GPIO_ON)])
        else:
            self.usb.write_registers([(0xA8, IR_GPIO_OFF), (0x03, reg03 | LAMP_BIT)])
        self.lamp_on_at = time.monotonic()

    def lamp_off(self) -> None:
        reg03 = self.usb.shadow.get(0x03, 0x9F)
        self.usb.write_registers([(0x03, reg03 & ~LAMP_BIT), (0xA8, IR_GPIO_OFF)])
        self.lamp_on_at = None

    def white_led(self, on: bool) -> None:
        reg03 = self.usb.shadow.get(0x03, 0x9F)
        self.usb.write_register(0x03, reg03 | LAMP_BIT if on else reg03 & ~LAMP_BIT)
        self.lamp_on_at = time.monotonic() if on else None

    def forget_motor_tables(self) -> None:
        """Boot and scan jobs overwrite the table slots homing uses."""
        if self._motion is not None:
            self._motion.loaded = None

    def motion(self) -> Gl843V1Motion:
        if self._motion is None:
            self._motion = Gl843V1Motion(self.usb)
        return self._motion

    def stop_motor(self) -> None:
        self.position_known = False
        self.motion().stop()

    def home(self, *, timeout_s: float = 30.0, wait: bool = True) -> None:
        self.position_known = False
        self.motion().home(timeout_s=max(5.0, timeout_s))
        self.position_known = True

    def park(self, *, timeout_s: float = 30.0) -> None:
        self.home(timeout_s=timeout_s)
