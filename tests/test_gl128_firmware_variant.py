# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-firmware GL128 register set (vendor ``PID_xxxx`` vs ``PID_xxxx-20241113``)."""

from __future__ import annotations

import dataclasses

import pytest

from pyopticfilm import Scanner
from pyopticfilm.device.gl128_common import (
    GL128_FIRMWARE_CURRENT,
    GL128_REG36_LEGACY_FIRMWARE,
    INIT_REGS,
    gl128_firmware_variant,
)
from pyopticfilm.device.model_7600i_v1 import Model7600iV1
from pyopticfilm.device.model_8100_v2 import Model8100V2
from pyopticfilm.device.model_8200i_se import Model8200iSE
from pyopticfilm.device.model_8300i_se import Model8300iSE
from pyopticfilm.scan.calibrate import calib_fingerprint
from pyopticfilm.usb.fake import FakeDeviceHandle, MockScannerTransport
from pyopticfilm.usb.protocol import GenesysUsbProtocol

GL128_MODELS = (Model8200iSE, Model8100V2, Model8300iSE)


def test_init_regs_are_the_current_firmware_set() -> None:
    assert INIT_REGS[0x36] == 0x1C


@pytest.mark.parametrize("model_cls", GL128_MODELS)
def test_current_firmware_keeps_model(model_cls) -> None:
    model = model_cls()
    assert gl128_firmware_variant(model, GL128_FIRMWARE_CURRENT) is model
    assert gl128_firmware_variant(model, f" {GL128_FIRMWARE_CURRENT} ") is model


@pytest.mark.parametrize("model_cls", GL128_MODELS)
def test_unreadable_serial_keeps_model(model_cls) -> None:
    model = model_cls()
    assert gl128_firmware_variant(model, None) is model


@pytest.mark.parametrize("model_cls", GL128_MODELS)
@pytest.mark.parametrize("serial", ["20201119", "20110214", ""])
def test_other_firmware_uses_plain_register_set(model_cls, serial: str) -> None:
    model = model_cls()
    variant = gl128_firmware_variant(model, serial)
    assert variant.init_regs[0x36] == GL128_REG36_LEGACY_FIRMWARE
    diff = {a for a in variant.init_regs if variant.init_regs[a] != model.init_regs[a]}
    assert diff == {0x36}
    assert model.init_regs[0x36] == 0x1C
    assert type(variant) is model_cls


def test_non_gl128_model_unchanged() -> None:
    model = Model7600iV1()
    assert gl128_firmware_variant(model, "20201119") is model


def test_fingerprint_separates_firmware_sets() -> None:
    model = Model8300iSE()
    assert calib_fingerprint(model) != calib_fingerprint(gl128_firmware_variant(model, "20201119"))


def _scanner_with_serial(serial: str | None) -> Scanner:
    model = Model8300iSE()
    handle = FakeDeviceHandle.for_model(model)
    handle.info = dataclasses.replace(handle.info, serial_number=serial)
    return Scanner(handle, GenesysUsbProtocol(MockScannerTransport()), model=model)


def test_scanner_applies_firmware_variant() -> None:
    legacy = _scanner_with_serial("20201119")
    assert legacy.model.init_regs[0x36] == GL128_REG36_LEGACY_FIRMWARE
    assert legacy._asic.model.init_regs[0x36] == GL128_REG36_LEGACY_FIRMWARE
    assert _scanner_with_serial(GL128_FIRMWARE_CURRENT).model.init_regs[0x36] == 0x1C
    assert _scanner_with_serial(None).model.init_regs[0x36] == 0x1C
