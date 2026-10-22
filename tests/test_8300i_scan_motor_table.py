# SPDX-License-Identifier: GPL-3.0-or-later
"""OpticFilm 8300i SE scan motor table (AHB 0x10000000/0x10004000/0x10008000)."""

from __future__ import annotations

import pytest

from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE
from pyopticfilm.device.tables_8200i_se import exposure_table
from pyopticfilm.device.tables_8300i_se import SCAN_RAMP, scan_motor_table
from pyopticfilm.scan.geometry import compute_geometry

#: SilverFast 1800 dpi image pass (capture_1800_full-frame_with-ir_no-me).
_SF_1800_HEAD = (40404, 4028, 3016, 2547, 2259, 2058, 1908, 1875, 1875)


def test_ramp_reproduces_silverfast_head_and_tail():
    assert SCAN_RAMP[:7] == _SF_1800_HEAD[:7]
    assert SCAN_RAMP[86:88] == (629, 626)
    assert len(SCAN_RAMP) == 256


@pytest.mark.parametrize(
    ("period", "head"),
    [
        (1875, _SF_1800_HEAD),
        (3750, (40404, 4028, 3750, 3750)),
        (15000, (40404, 15000, 15000)),
    ],
)
def test_table_clamps_ramp_at_period(period, head):
    table = scan_motor_table(period)
    assert table[: len(head)] == head
    assert table[-1] == period
    assert len(table) == 256


def test_table_rejects_period_above_16_bits():
    with pytest.raises(ValueError, match="16-bit"):
        scan_motor_table(0x10000)


def test_model_uses_ramp():
    assert MODEL_8300I_SE.scan_motor_table(1875)[:9] == _SF_1800_HEAD


@pytest.mark.parametrize("model", [MODEL_8200I_SE, MODEL_8100_V2])
def test_se_and_v2_keep_flat_table(model):
    assert model.scan_motor_table(7000) == exposure_table(7000)


@pytest.mark.parametrize(
    ("dpi", "lincnt", "nbytes"),
    [(1800, 1813, 28_195_776), (7200, 7253, 451_194_624)],
)
def test_full_frame_lincnt_and_bulk_size_match_silverfast(dpi, lincnt, nbytes):
    """At the SilverFast motor speed a 4x LINCNT ran the carriage into the end stop."""
    geometry = compute_geometry(dpi, model=MODEL_8300I_SE)
    assert geometry.lincnt_register == lincnt
    assert geometry.optical_line_count == lincnt
    assert geometry.total_bytes == nbytes
    assert geometry.travel_mm == pytest.approx(25.58, abs=0.02)
