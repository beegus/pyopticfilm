# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 image line count and window follow SilverFast, full frames and crops.

SilverFast programs ``frame + round(48 × dpi / 7200) + (5 at 7200 dpi, else 1)``
lines for a full frame on every model (8200i SE ``13_ppi_ladder`` / Alignment
test, 8100 V2 ``06_ppi_ladder`` / ``04_color_7200``, 8300i SE PPI ladder).
``LINCNT`` is that × 4 on the 8200i SE / 8100 V2 and × 1 on the 8300i SE.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyopticfilm.device.model_8100_v2 import MODEL_8100_V2
from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from pyopticfilm.device.model_8300i_se import MODEL_8300I_SE
from pyopticfilm.scan.geometry import compute_geometry
from pyopticfilm.scan.pipeline import ImagePipeline

MODELS = [MODEL_8200I_SE, MODEL_8100_V2, MODEL_8300I_SE]
IDS = ["se", "v2", "8300i"]

#: SilverFast full-frame lines per ASIC dpi (150/300 dpi run the ASIC at 600).
SF_LINES = {
    600: 605,
    720: 726,
    900: 907,
    1200: 1209,
    1440: 1451,
    1800: 1813,
    2400: 2417,
    3600: 3625,
    7200: 7253,
}
PPI = (150, 300, 600, 720, 900, 1200, 1440, 1800, 2400, 3600, 7200)


@pytest.mark.parametrize("model", MODELS, ids=IDS)
@pytest.mark.parametrize("ppi", PPI)
def test_full_frame_lincnt_matches_silverfast(model, ppi: int) -> None:
    g = compute_geometry(ppi, model=model)
    assert g.lincnt_register == SF_LINES[model.asic_dpi_for(ppi)] * g.lincnt_per_line
    feed2 = model.feed_to_scan_steps_for_dpi(ppi)
    assert g.lincnt_register <= model.max_safe_lincnt_for(feed2, ppi)


@pytest.mark.parametrize("model", MODELS, ids=IDS)
@pytest.mark.parametrize("ppi", (600, 1440, 2400, 3600, 7200))
def test_full_frame_output_is_still_one_frame(model, ppi: int) -> None:
    g = compute_geometry(ppi, model=model)
    pipe = ImagePipeline(model)
    rgb = np.zeros((g.optical_line_count, 4, 3), dtype=np.uint16)
    rgb = pipe.reduce_y_oversample(rgb, g)
    rgb = pipe.apply_line_shifts(rgb, g)
    rgb = pipe.apply_y_stagger(rgb, g)
    assert rgb.shape[0] >= g.frame_lines == round(model.full_frame_output_inches * model.asic_dpi_for(ppi))


#: SilverFast full-frame END per ASIC dpi (STR 242 everywhere).
SF_FULL_END = {
    600: 10610,
    720: 10602,
    900: 10610,
    1200: 10610,
    1440: 10612,
    1800: 10610,
    2400: 10610,
    3600: 10610,
    7200: 10610,
}

#: 8200i SE ``13_ppi_ladder`` crop (feed2 13560, 0.946667 in tall, native
#: 144..10368 wide): SilverFast's (STR, END, LINCNT) per dpi.
SF_LADDER = {
    600: (386, 10610, 2292),
    720: (382, 10602, 2748),
    900: (386, 10610, 3436),
    1200: (386, 10610, 4580),
    1440: (387, 10607, 5496),
    1800: (386, 10610, 6868),
    2400: (386, 10610, 9156),
    3600: (386, 10610, 13732),
    7200: (386, 10610, 27476),
}


@pytest.mark.parametrize("model", MODELS, ids=IDS)
@pytest.mark.parametrize("ppi", sorted(SF_FULL_END))
def test_full_frame_window_matches_silverfast(model, ppi: int) -> None:
    g = compute_geometry(ppi, model=model)
    assert (g.pixel_startx, g.pixel_endx) == (242, SF_FULL_END[ppi])


@pytest.mark.parametrize("ppi", sorted(SF_LADDER))
def test_ladder_crop_matches_silverfast(ppi: int) -> None:
    m = MODEL_8200I_SE
    x1 = 144 / 7200 * 25.4 / m.x_size_ta_mm
    x2 = (144 + 10224) / 7200 * 25.4 / m.x_size_ta_mm
    y1 = (13560 - m.feed_to_scan_top_steps) / (m.scan_window_end_steps - m.feed_to_scan_top_steps)
    y2 = y1 + 0.9466667 * 25.4 / m.y_size_ta_mm
    g = compute_geometry(ppi, model=m, area=(x1, y1, x2, y2))
    assert (g.pixel_startx, g.pixel_endx, g.lincnt_register) == SF_LADDER[ppi]


@pytest.mark.parametrize("model", MODELS, ids=IDS)
@pytest.mark.parametrize("ppi", (600, 1440, 2400, 3600, 7200))
@pytest.mark.parametrize("area", [(0, 0, 1, 1), (0, 0.5, 1, 1), (0.1, 0.2, 0.9, 0.8), (0.3, 0.9, 0.7, 1)])
def test_crop_stays_in_travel_and_window(model, ppi: int, area) -> None:
    g = compute_geometry(ppi, model=model, area=area)
    feed2 = model.feed_to_scan_steps_for_area(area)
    assert g.lincnt_register <= model.max_safe_lincnt_for(feed2, ppi)
    full = compute_geometry(ppi, model=model)
    assert full.pixel_startx <= g.pixel_startx < g.pixel_endx <= full.pixel_endx
    assert g.pixels % 2 == 0


@pytest.mark.parametrize("model", MODELS, ids=IDS)
@pytest.mark.parametrize("ppi", (600, 1440, 3600, 7200))
def test_crop_output_keeps_requested_height(model, ppi: int) -> None:
    area = (0.0, 0.2, 1.0, 0.6)
    g = compute_geometry(ppi, model=model, area=area)
    want = int(0.4 * model.y_size_ta_mm * model.asic_dpi_for(ppi) / 25.4)
    pipe = ImagePipeline(model)
    rgb = np.zeros((g.optical_line_count, 4, 3), dtype=np.uint16)
    rgb = pipe.apply_y_stagger(pipe.apply_line_shifts(pipe.reduce_y_oversample(rgb, g), g), g)
    assert g.frame_lines == want
    assert rgb.shape[0] >= want
