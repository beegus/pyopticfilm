# SPDX-License-Identifier: GPL-3.0-or-later
"""GL128 full-frame image line count follows SilverFast.

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


@pytest.mark.parametrize("model", MODELS, ids=IDS)
def test_crop_line_count_unchanged(model) -> None:
    g = compute_geometry(1800, model=model, area=(0.0, 0.25, 1.0, 0.75))
    lines = int(0.5 * model.y_size_ta_mm * 1800 / 25.4)
    assert g.lincnt_register == (lines + g.num_staggered_lines) * g.lincnt_per_line
