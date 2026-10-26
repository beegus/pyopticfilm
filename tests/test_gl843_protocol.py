# SPDX-License-Identifier: GPL-3.0-or-later
"""GL842/GL843 register framing: address with 0x83, read 0x84; optional write acknowledge."""

from __future__ import annotations

import pytest

from pyopticfilm.asic.gl842 import Gl842
from pyopticfilm.asic.gl843 import Gl843
from pyopticfilm.asic.gl843_v1 import Gl843V1
from pyopticfilm.device.model_7200 import MODEL_7200
from pyopticfilm.device.model_7200i import MODEL_7200I
from pyopticfilm.device.model_7600i_v1 import MODEL_7600I_V1
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.select import create_asic
from pyopticfilm.exceptions import UsbError
from pyopticfilm.usb.fake import FakeUsbTransport, MockScannerTransport
from pyopticfilm.usb.fake_gl843_v1 import SimulatedGl843V1Transport
from pyopticfilm.usb.protocol import GenesysUsbProtocol, Gl843UsbProtocol


def _ops(usb) -> list[tuple]:
    return [(t.operation, t.request, t.value, t.index, t.data or t.length) for t in usb.transactions]


def test_read_sets_the_address_then_reads_0x84():
    usb = FakeUsbTransport(registers={0x06: 0x5A, 0x07: 0xA5})
    proto = Gl843UsbProtocol(usb)
    assert proto.read_register(0x06) == 0x5A
    assert _ops(usb) == [
        ("control_write", 0x0C, 0x83, 0, bytes([0x06])),
        ("control_read", 0x0C, 0x84, 0, 1),
    ]
    assert usb.address == 0x07  # the chip increments the address after a read


def test_registers_are_8_bit():
    proto = Gl843UsbProtocol(FakeUsbTransport())
    with pytest.raises(ValueError, match="8-bit"):
        proto.read_register(0x101)
    with pytest.raises(ValueError, match="8-bit"):
        proto.write_register(0x101, 0)


def test_writes_are_pairs_without_acknowledge_by_default():
    usb = FakeUsbTransport()
    proto = Gl843UsbProtocol(usb)
    proto.write_register(0x03, 0x9F)
    proto.write_registers_batched([(0x01, 0x22), (0x02, 0x38)])
    proto.write_fe_register(0x02, 0x0123)
    assert _ops(usb) == [
        ("control_write", 0x04, 0x83, 0, bytes([0x03, 0x9F])),
        ("control_write", 0x04, 0x83, 0, bytes([0x01, 0x22, 0x02, 0x38])),
        ("control_write", 0x04, 0x83, 0, bytes([0x51, 0x02, 0x3A, 0x01, 0x3B, 0x23])),
    ]
    assert proto.written == {0x03: 0x9F, 0x01: 0x22, 0x02: 0x38, 0x51: 0x02, 0x3A: 0x01, 0x3B: 0x23}


def test_acknowledged_writes_poll_0x8e_0x20():
    sim = SimulatedGl843V1Transport()
    proto = Gl843UsbProtocol(sim, acknowledge_writes=True)
    proto.write_0x8c(0x10, 0xD4)
    proto.read_register(0x41)
    assert _ops(sim) == [
        ("control_write", 0x0C, 0x8C, 0x10, bytes([0xD4])),
        ("control_read", 0x0C, 0x8E, 0x20, 1),
        ("control_write", 0x0C, 0x83, 0, bytes([0x41])),
        ("control_read", 0x0C, 0x8E, 0x20, 1),
        ("control_read", 0x0C, 0x84, 0, 1),
    ]


def test_unacknowledged_write_is_sent_again_then_refused():
    class Deaf(SimulatedGl843V1Transport):
        def control_msg(self, rt, request, value, index, data_or_length, *, timeout_ms=None):
            if rt == 0xC0 and value == 0x8E and index == 0x20:
                return bytes([0])
            return super().control_msg(rt, request, value, index, data_or_length, timeout_ms=timeout_ms)

    sim = Deaf()
    with pytest.raises(UsbError, match="0x83 not acknowledged"):
        Gl843UsbProtocol(sim, acknowledge_writes=True).write_register(0x03, 0x9F)
    assert sum(t.operation == "control_write" for t in sim.transactions) == 10


def test_wrap_reframes_a_plain_protocol_over_the_same_transport():
    usb = FakeUsbTransport()
    proto = Gl843UsbProtocol.wrap(GenesysUsbProtocol(usb))
    assert isinstance(proto, Gl843UsbProtocol) and proto.transport is usb
    assert Gl843UsbProtocol.wrap(proto) is proto
    with pytest.raises(ValueError, match="acknowledge_writes"):
        Gl843UsbProtocol.wrap(proto, acknowledge_writes=True)


@pytest.mark.parametrize(("model", "cls"), [(MODEL_7200I, Gl843), (MODEL_7200, Gl842)])
def test_sane_gl84x_status_reads_by_address(model, cls):
    usb = MockScannerTransport()
    asic = create_asic(GenesysUsbProtocol(usb), model)
    assert isinstance(asic, cls) and isinstance(asic.protocol, Gl843UsbProtocol)
    assert not asic.protocol.acknowledge_writes  # SANE does not poll a write acknowledge
    assert asic.read_status().is_at_home
    assert [t.value for t in usb.transactions] == [0x83, 0x84]


@pytest.mark.parametrize(
    ("model", "transport", "framing", "ack"),
    [
        (MODEL_7600I_V1, SimulatedGl843V1Transport, Gl843UsbProtocol, True),
        (MODEL_7200I, MockScannerTransport, Gl843UsbProtocol, False),
        (MODEL_8200I_SE, MockScannerTransport, GenesysUsbProtocol, None),
    ],
)
def test_scanner_uses_the_asic_framing(model, transport, framing, ack, tmp_path):
    from pyopticfilm import Scanner

    with Scanner.open_fake(model, transport(), calib_cache=tmp_path / "c.json") as scanner:
        assert type(scanner.protocol) is framing
        assert scanner.protocol is scanner.asic.protocol
        assert getattr(scanner.protocol, "acknowledge_writes", None) == ack


def test_7600i_asic_and_usb_share_one_protocol():
    sim = SimulatedGl843V1Transport()
    asic = Gl843V1(GenesysUsbProtocol(sim), MODEL_7600I_V1)
    assert asic.usb.protocol is asic.protocol and asic.protocol.acknowledge_writes
    asic.protocol.write_register(0x03, 0x9F)
    assert asic.usb.shadow[0x03] == 0x9F
