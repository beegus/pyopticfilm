# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import pytest

from tools.probe_afe_fields import (
    SWEEP_A_CODES,
    SWEEP_B_CODES,
    classify_afe_field,
    field_channel,
    sweep_codes,
    wraps_at_0x40,
)


def test_field_channel_and_sweep_codes() -> None:
    assert [field_channel(i) for i in range(0x02, 0x08)] == [0, 1, 2, 0, 1, 2]
    assert sweep_codes(0x03) == SWEEP_A_CODES
    assert sweep_codes(0x07) == SWEEP_B_CODES
    with pytest.raises(ValueError):
        sweep_codes(0x01)


def _gain6(codes, *, base: float, per_code: float, pedestal: float) -> dict[int, float]:
    """6-bit PGA: only the low six bits count, so 0x40 reads like 0x00."""
    return {c: pedestal + base * (1.0 + per_code * (c & 0x3F)) for c in codes}


def _offset(codes, *, level: float, per_code: float) -> dict[int, float]:
    return {c: level + per_code * c for c in codes}


def test_six_bit_gain_wraps_and_classifies_as_gain() -> None:
    white = _gain6(SWEEP_A_CODES, base=8000.0, per_code=0.05, pedestal=2000.0)
    dark = _gain6(SWEEP_A_CODES, base=30.0, per_code=0.05, pedestal=2000.0)
    assert wraps_at_0x40(white)
    assert classify_afe_field(white, dark) == "gain"


def test_gain_without_wrap_classifies_from_ratio() -> None:
    white = {c: 4000.0 + 100.0 * c for c in SWEEP_B_CODES}
    dark = {c: 1500.0 + 2.0 * c for c in SWEEP_B_CODES}
    assert not wraps_at_0x40(white)
    assert classify_afe_field(white, dark) == "gain"


def test_offset_shifts_white_and_dark_together() -> None:
    white = _offset(SWEEP_A_CODES, level=20000.0, per_code=60.0)
    dark = _offset(SWEEP_A_CODES, level=1000.0, per_code=55.0)
    assert not wraps_at_0x40(white)
    assert classify_afe_field(white, dark) == "offset"


def test_offset_on_nine_bit_field() -> None:
    white = _offset(SWEEP_B_CODES, level=15000.0, per_code=40.0)
    dark = _offset(SWEEP_B_CODES, level=500.0, per_code=40.0)
    assert classify_afe_field(white, dark) == "offset"


def test_flat_response_is_unclear() -> None:
    white = {c: 30000.0 + (c % 3) for c in SWEEP_A_CODES}
    dark = {c: 1200.0 for c in SWEEP_A_CODES}
    assert classify_afe_field(white, dark) == "unclear"


def test_dark_clipped_at_zero_is_unclear() -> None:
    white = _offset(SWEEP_B_CODES, level=10000.0, per_code=50.0)
    dark = {c: 0.0 for c in SWEEP_B_CODES}
    assert classify_afe_field(white, dark) == "unclear"


def test_empty_input_is_unclear() -> None:
    assert classify_afe_field({}, {}) == "unclear"
