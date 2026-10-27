# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 ``search_afe`` against a fake AD9826-style frontend.

The golden file freezes every ``write_fe_register_gl124`` call per scenario.
Regenerate it only for a deliberate behaviour change::

    GL128_AFE_REGEN=1 uv run pytest tests/test_gl128_afe_fields.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import pyopticfilm.asic.gl128 as gl128_mod
from pyopticfilm.asic.gl128 import Gl128
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.scan.calib_gl128 import (
    AFE_WHITE_TARGETS_IR,
    AFE_WIDE_BYTES,
    AFE_WIDE_PIXELS,
    channel_means_u16,
    coarse_gains_from_wide_means,
    decode_afe_offset,
    encode_afe_offset,
)

GOLDEN = Path(__file__).parent / "data" / "gl128_afe_fe_writes.json"

#: Every AFE strip starts with shielded optical-black pixels (columns 0-59 on
#: real units); they read the same with the lamp on.
OPTICAL_BLACK_PIXELS = 60

#: Per-pixel profile: mild vignette so lag statistics look like real strips.
_PROFILE = [0.85 + 0.15 * (1.0 - abs((x % 512) - 256) / 256.0) for x in range(16384)]


class FakeAd9826:
    """Per-channel 6-bit PGA (``0x02``-``0x04``) and sign-magnitude offset (``0x05``-``0x07``)."""

    def __init__(self, *, white: tuple[float, float, float], ir_white: tuple[float, float, float]) -> None:
        self.regs: dict[int, int] = {}
        self.writes: list[list[int]] = []
        self.white = white
        self.ir_white = ir_white
        self.lamp = False
        self.infrared = False

    def write_fe(self, index: int, value: int) -> None:
        self.regs[int(index)] = int(value)
        self.writes.append([int(index), int(value)])

    def channel_value(self, c: int, x: int) -> int:
        code = self.regs.get(0x05 + c, 0)
        offset = -(code & 0xFF) if code & 0x100 else code & 0xFF
        gain = 6.0 / (6.0 - 5.0 * (self.regs.get(0x02 + c, 0) & 0x3F) / 63.0)
        if self.lamp and x >= OPTICAL_BLACK_PIXELS:
            signal = (self.ir_white if self.infrared else self.white)[c] * _PROFILE[x]
        else:
            signal = 0.0
        return max(0, min(65535, int((signal + 450.0 + 20.0 * offset) * gain)))

    noise: random.Random | None = None

    def strip(self, size: int) -> bytes:
        pixels = size // 6
        out = bytearray()
        for x in range(pixels):
            for c in range(3):
                v = self.channel_value(c, x)
                if self.noise is not None:
                    v = max(0, min(65535, v + self.noise.randint(-1500, 1500)))
                out += v.to_bytes(2, "little")
        return bytes(out)


def _make_asic(monkeypatch, fake: FakeAd9826) -> Gl128:
    monkeypatch.setattr(gl128_mod.time, "sleep", lambda *_a, **_k: None)
    protocol = MagicMock()
    protocol.write_fe_register_gl124.side_effect = fake.write_fe
    asic = Gl128(protocol, MODEL_8200I_SE)
    asic._initialized = True

    def lamp_on() -> None:
        fake.lamp = True
        fake.infrared = asic._scan_method == "infrared"

    def lamp_off() -> None:
        fake.lamp = False

    asic.lamp_on = MagicMock(side_effect=lamp_on)
    asic.lamp_off = MagicMock(side_effect=lamp_off)
    asic._strike_lamp_on = MagicMock(side_effect=lamp_on)

    def acquire(size: int = 3072, **_k) -> bytes:
        return fake.strip(min(int(size), AFE_WIDE_BYTES))

    asic.acquire_afe_strip = MagicMock(side_effect=acquire)
    return asic


SCENARIOS: dict[str, dict] = {
    "colour": {"white": (25000.0, 34000.0, 30000.0), "ir_white": (0.0, 0.0, 0.0), "method": "transparency"},
    "colour_dim": {"white": (4000.0, 5000.0, 4500.0), "ir_white": (0.0, 0.0, 0.0), "method": "transparency"},
    "colour_saturated": {"white": (90000.0, 90000.0, 90000.0), "ir_white": (0.0, 0.0, 0.0), "method": "transparency"},
    "infrared": {"white": (25000.0, 34000.0, 30000.0), "ir_white": (9000.0, 6000.0, 7000.0), "method": "infrared"},
}


def _run(monkeypatch, name: str) -> dict:
    spec = SCENARIOS[name]
    fake = FakeAd9826(white=spec["white"], ir_white=spec["ir_white"])
    asic = _make_asic(monkeypatch, fake)
    result = asic.search_afe(method=spec["method"])
    return {
        "writes": fake.writes,
        "result": [[int(i), int(v)] for i, v in result.as_fe_writes()],
    }


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_search_afe_fe_writes_match_golden(monkeypatch, name: str) -> None:
    got = _run(monkeypatch, name)
    if os.environ.get("GL128_AFE_REGEN"):
        data = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
        data[name] = got
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip("golden regenerated")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[name]
    assert got["writes"] == golden["writes"]
    assert got["result"] == golden["result"]


@pytest.mark.parametrize("value", [0, 1, 16, 0x80, 0xFF, -1, -0x6A, -0xFF])
def test_afe_offset_sign_magnitude_round_trip(value: int) -> None:
    code = encode_afe_offset(value)
    assert 0 <= code <= 0x1FF
    assert decode_afe_offset(code) == value
    assert bool(code & 0x100) == (value < 0)


def test_afe_offset_capture_codes_decode_negative() -> None:
    # Session-03 SilverFast codes and the probe's minus-zero.
    assert decode_afe_offset(0x16A) == -106
    assert decode_afe_offset(0x142) == -66
    assert decode_afe_offset(0x100) == 0
    assert encode_afe_offset(300) == 0xFF
    assert encode_afe_offset(-300) == 0x1FF


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_search_afe_stays_inside_field_ranges(monkeypatch, name: str) -> None:
    """Gains stay inside the 6-bit PGA and offsets inside the 9-bit sign-magnitude field."""
    writes = _run(monkeypatch, name)["writes"]
    for index, value in writes:
        if 0x02 <= index <= 0x04:
            assert value <= 0x3F, (index, value)
        if 0x05 <= index <= 0x07:
            assert value <= 0x1FF, (index, value)


def test_ir_search_reads_every_strip_under_the_ir_led(monkeypatch) -> None:
    """IR gains come from a wide IR strip (SF session 05 / 8300i), never from a dark strip."""
    spec = SCENARIOS["infrared"]
    fake = FakeAd9826(white=spec["white"], ir_white=spec["ir_white"])
    asic = _make_asic(monkeypatch, fake)
    lighting: list[tuple[bool, bool, int]] = []
    wide_strips: list[bytes] = []
    acquire = asic.acquire_afe_strip.side_effect

    def recording_acquire(size: int = 3072, **kw) -> bytes:
        lighting.append((fake.lamp, fake.infrared, int(size)))
        data = acquire(size, **kw)
        if int(size) == AFE_WIDE_BYTES:
            wide_strips.append(data)
        return data

    asic.acquire_afe_strip.side_effect = recording_acquire
    result = asic.search_afe(method="infrared")

    assert lighting and all(lamp and ir for lamp, ir, _size in lighting)
    assert len(wide_strips) == 1
    asic.lamp_off.assert_not_called()
    assert result.gains == coarse_gains_from_wide_means(
        channel_means_u16(wide_strips[0], pixels=AFE_WIDE_PIXELS, planar=False),
        targets=AFE_WHITE_TARGETS_IR,
    )


def test_afe_strip_decode_is_chunky_even_when_lag_test_says_planar(monkeypatch) -> None:
    """The 8200i SE lag test picked planar on real strips; coarse gains must stay per channel."""
    monkeypatch.setattr(gl128_mod, "choose_usb_planar", lambda *_a, **_k: True, raising=False)
    fake = FakeAd9826(white=(15000.0, 38000.0, 26000.0), ir_white=(0.0, 0.0, 0.0))
    fake.noise = random.Random(7)
    asic = _make_asic(monkeypatch, fake)
    wide_strips: list[bytes] = []
    acquire = asic.acquire_afe_strip.side_effect

    def recording_acquire(size: int = 3072, **kw) -> bytes:
        data = acquire(size, **kw)
        if int(size) == AFE_WIDE_BYTES:
            wide_strips.append(data)
        return data

    asic.acquire_afe_strip.side_effect = recording_acquire
    result = asic.search_afe(method="transparency")

    wide = wide_strips[0]
    expected = coarse_gains_from_wide_means(channel_means_u16(wide, pixels=AFE_WIDE_PIXELS, planar=False))
    blended = coarse_gains_from_wide_means(channel_means_u16(wide, pixels=AFE_WIDE_PIXELS, planar=True))
    assert len(set(expected)) == 3
    assert result.gains == expected
    assert result.gains != blended


# --- SilverFast's coarse AFE algorithm (values from its own captures) ---------

#: 8200i SE Alignment_test: optical black (columns 0-59) at offsets 0x80 / 0xFF.
SF_SE_BLACK_GAIN0 = ((3526.0, 3380.0, 3425.0), (5491.0, 5427.0, 5482.0))
SF_SE_BLACK_GAINED = ((5426.0, 4388.0, 5023.0), (10100.0, 8145.0, 9366.0))


def test_offsets_for_black_level_reproduces_silverfast_codes() -> None:
    from pyopticfilm.scan.calib_gl128 import (
        AFE_BLACK_TARGET_COLOR,
        AFE_PRE_OFFSET_BLACK,
        offsets_for_black_level,
    )

    # Pre-offset at gain 0 (SF wrote -95/-82/-81) and black offsets at gains
    # 35/25/32 (SF wrote 15/23/19, putting optical black at ~0x500).
    pre = offsets_for_black_level(*SF_SE_BLACK_GAIN0, AFE_PRE_OFFSET_BLACK)
    assert all(abs(o - s) <= 3 for o, s in zip(pre, (-95, -82, -81), strict=True))
    assert offsets_for_black_level(*SF_SE_BLACK_GAINED, AFE_BLACK_TARGET_COLOR) == (15, 23, 19)


def test_offsets_for_black_level_flags_a_flat_probe() -> None:
    from pyopticfilm.scan.calib_gl128 import offsets_for_black_level

    low = (3000.0, 3000.0, 65535.0)
    high = (5000.0, 3001.0, 65535.0)
    assert offsets_for_black_level(low, high, 1280)[1:] == (None, None)


def test_optical_black_means_reads_only_the_shielded_columns() -> None:
    from pyopticfilm.scan.calib_gl128 import AFE_STRIP_PIXELS, optical_black_means

    pixels = [(1000, 2000, 3000) if 2 <= x < 56 else (60000, 60000, 60000) for x in range(AFE_STRIP_PIXELS)]
    strip = b"".join(v.to_bytes(2, "little") for px in pixels for v in px)
    assert optical_black_means(strip) == (1000.0, 2000.0, 3000.0)


@pytest.mark.parametrize(
    ("dpi", "narrow", "wide"), [(600, 10979, 13445), (2400, 10979, 13445), (3600, 11107, 16040), (7200, 11107, 16040)]
)
def test_afe_strip_timing_matches_silverfast(dpi: int, narrow: int, wide: int) -> None:
    from pyopticfilm.scan.calib_gl128 import afe_strip_timing

    pre = afe_strip_timing(dpi, wide=False, pre_offset=True)
    after = afe_strip_timing(dpi, wide=True)
    assert (pre.lperiod, pre.dummy) == (narrow, 0x02)
    assert (after.lperiod, after.dummy) == (wide, 0x17)


def test_search_afe_follows_silverfast_strip_sequence(monkeypatch) -> None:
    """Two gain-0 probes (pre timing), the wide strip, then two probes at the new gains."""
    spec = SCENARIOS["colour"]
    fake = FakeAd9826(white=spec["white"], ir_white=spec["ir_white"])
    asic = _make_asic(monkeypatch, fake)
    seen: list[tuple[int, int, int, tuple[int, ...]]] = []
    acquire = asic.acquire_afe_strip.side_effect

    def recording_acquire(size: int = 3072, **kw) -> bytes:
        t = kw["timing"]
        seen.append((int(size), t.lperiod, t.dummy, tuple(fake.regs.get(i, 0) for i in (0x02, 0x05))))
        return acquire(size, **kw)

    asic.acquire_afe_strip.side_effect = recording_acquire
    result = asic.search_afe(method="transparency", resolution=3600)

    gain = result.gains[0]
    assert seen == [
        (3072, 11107, 0x02, (0, 0x80)),
        (3072, 11107, 0x02, (0, 0xFF)),
        (AFE_WIDE_BYTES, 16040, 0x17, (0, seen[2][3][1])),
        (3072, 16040, 0x17, (gain, 0x80)),
        (3072, 16040, 0x17, (gain, 0xFF)),
    ]
    assert asic.last_afe is result


def test_search_afe_falls_back_per_channel_when_black_does_not_move(monkeypatch, caplog) -> None:
    from pyopticfilm.scan.calib_gl128 import AFE_FALLBACK_OFFSETS_COLOR, AFE_FALLBACK_PRE_OFFSETS

    spec = SCENARIOS["colour"]
    fake = FakeAd9826(white=spec["white"], ir_white=spec["ir_white"])
    asic = _make_asic(monkeypatch, fake)
    acquire = asic.acquire_afe_strip.side_effect

    def railed_red(size: int = 3072, **kw) -> bytes:
        data = bytearray(acquire(size, **kw))
        data[0::6] = b"\xff" * len(data[0::6])
        data[1::6] = b"\xff" * len(data[1::6])
        return bytes(data)

    asic.acquire_afe_strip.side_effect = railed_red
    result = asic.search_afe(method="transparency")

    assert decode_afe_offset(result.offsets[0]) == AFE_FALLBACK_OFFSETS_COLOR[0]
    # Writes 12-17 program the pre-offsets (gains 0x02-0x04, then offsets 0x05-0x07).
    assert fake.writes[15][0] == 0x05
    assert decode_afe_offset(fake.writes[15][1]) == AFE_FALLBACK_PRE_OFFSETS[0]
    assert "optical black barely moved" in caplog.text


def test_afe_strip_regs_program_silverfast_timing() -> None:
    from pyopticfilm.asic.registers import Gl128Registers
    from pyopticfilm.scan.calib_gl128 import AfeStripTiming

    r = Gl128Registers()
    protocol = MagicMock()
    asic = Gl128(protocol, MODEL_8200I_SE)
    asic._setup_afe_strip_regs(wide=True, timing=AfeStripTiming(lperiod=13445, dummy=0x17))

    u24 = {c.args[0]: c.args[1] for c in protocol.write_u24.call_args_list}
    assert u24[r.REG_LPERIOD] == 13445
    assert u24[r.REG_EXPOSURE] == MODEL_8200I_SE.exposure_lperiod
    regs = asic._reg_cache
    assert (regs[0x2B], regs[0xA3], regs[0xA6], regs[0xA7], regs[0xA8], regs[0xA9]) == (0x17, 0, 0, 1, 0, 1)
