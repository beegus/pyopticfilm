# SPDX-License-Identifier: GPL-3.0-or-later
"""Native NumPy pass registration (ME, Multi-Pass, RGB↔IR)."""

from __future__ import annotations

from pyopticfilm.registration.engine import (
    apply_transform,
    estimate_banded,
    estimate_translation,
    register,
    register_and_align,
)
from pyopticfilm.registration.types import (
    LineWarpTransform,
    RegionalShift,
    RegistrationContext,
    RegistrationFailure,
    RegistrationMode,
    RegistrationOptions,
    RegistrationQuality,
    RegistrationResult,
    TranslationTransform,
)

__all__ = [
    "LineWarpTransform",
    "RegionalShift",
    "RegistrationContext",
    "RegistrationFailure",
    "RegistrationMode",
    "RegistrationOptions",
    "RegistrationQuality",
    "RegistrationResult",
    "TranslationTransform",
    "apply_transform",
    "estimate_banded",
    "estimate_translation",
    "register",
    "register_and_align",
]
