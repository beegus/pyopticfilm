# SPDX-License-Identifier: GPL-3.0-or-later
"""Mode-specific registration planes (NumPy only)."""

from __future__ import annotations

import numpy as np

from pyopticfilm.registration.types import RegistrationMode


def structural_luminance(image: np.ndarray) -> np.ndarray:
    """``(R + 2G + B) / 4`` as float32 — geometry, not display luma."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    return (r + 2.0 * g + b) * 0.25


def _gradient_magnitude(plane: np.ndarray) -> np.ndarray:
    """Centered differences; output is (H-2) x (W-2), then padded back."""
    p = plane.astype(np.float32, copy=False)
    h, w = p.shape
    if h < 3 or w < 3:
        return np.abs(p)
    gx = p[:, 2:] - p[:, :-2]
    gy = p[2:, :] - p[:-2, :]
    # Align to common (H-2, W-2) interior.
    gx_i = gx[1:-1, :]
    gy_i = gy[:, 1:-1]
    mag = np.hypot(gx_i, gy_i)
    out = np.zeros((h, w), dtype=np.float32)
    out[1:-1, 1:-1] = mag
    return out


def rgb_edge_energy(rgb: np.ndarray) -> np.ndarray:
    """Max channel gradient magnitude for RGB↔IR structural matching."""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        return _gradient_magnitude(arr)
    planes = [_gradient_magnitude(arr[:, :, c]) for c in range(arr.shape[2])]
    stacked = np.stack(planes, axis=0)
    return stacked.max(axis=0)


def ir_edge_energy(ir: np.ndarray) -> np.ndarray:
    return _gradient_magnitude(np.asarray(ir))


def clip_mask(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    high: float = 0.98 * 65535.0,
    low: float = 200.0,
) -> np.ndarray:
    """True where both frames have usable (non-clipped) content."""
    ref = structural_luminance(reference)
    mov = structural_luminance(moving)
    return (ref < high) & (mov < high) & (ref > low) & (mov > low)


def registration_plane(
    image: np.ndarray,
    mode: RegistrationMode,
    *,
    is_moving: bool = False,
    partner: np.ndarray | None = None,
) -> np.ndarray:
    """Build the float32 plane used for phase correlation."""
    if mode is RegistrationMode.RGB_IR:
        if is_moving or (partner is None and image.ndim == 2):
            # Moving IR, or already a mono plane.
            if image.ndim == 3:
                return rgb_edge_energy(image)
            return ir_edge_energy(image)
        # Reference RGB.
        if image.ndim == 2:
            return ir_edge_energy(image)
        return rgb_edge_energy(image)

    if mode is RegistrationMode.MULTI_EXPOSURE:
        plane = structural_luminance(image)
        if partner is not None:
            mask = clip_mask(partner if is_moving else image, image if is_moving else partner)
            # Downweight clipped structure by zeroing (phase correlation prefers edges).
            plane = plane * mask.astype(np.float32)
        return plane

    # MULTI_PASS and default: structural luminance.
    return structural_luminance(image)


def coarse_registration_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    mode: RegistrationMode,
    *,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Downsample first, then form registration planes (avoids full-res float).

    Returns ``(ref_coarse, mov_coarse, scale)`` with ``scale = full_w / coarse_w``.
    RGB↔IR: structural downsample then gradient on the coarse tiles.
    ME/MP: block-mean / luma on the reduced grid; ME zeros clipped structure.
    """
    from pyopticfilm.registration.pyramid import block_mean

    ref = np.asarray(reference)
    mov = np.asarray(moving)
    full_w = ref.shape[1]

    if mode is RegistrationMode.RGB_IR:
        # Block-mean first (uint16 → small float), then gradient — no full-res float plane.
        factor = max(1, round(full_w / target_w)) if full_w > target_w else 1
        while factor > 1 and (full_w // factor) < max(32, target_w // 4):
            factor -= 1
        if factor <= 1:
            ref_c = structural_luminance(ref) if ref.ndim == 3 else ref.astype(np.float32)
            mov_c = structural_luminance(mov) if mov.ndim == 3 else mov.astype(np.float32)
        else:
            ref_c = block_mean(ref, factor)
            mov_c = block_mean(mov, factor)
            if ref_c.ndim == 3:
                ref_c = ref_c.mean(axis=2)
            if mov_c.ndim == 3:
                mov_c = mov_c.mean(axis=2)
        h = min(ref_c.shape[0], mov_c.shape[0])
        w = min(ref_c.shape[1], mov_c.shape[1])
        ref_c = _gradient_magnitude(ref_c[:h, :w].astype(np.float32, copy=False))
        mov_c = _gradient_magnitude(mov_c[:h, :w].astype(np.float32, copy=False))
        scale = float(full_w) / float(w) if w else 1.0
        return ref_c, mov_c, scale

    # ME / MULTI_PASS: integer block-mean of RGB (or mono) → float plane.
    factor = max(1, round(full_w / target_w)) if full_w > target_w else 1
    while factor > 1 and (full_w // factor) < max(32, target_w // 4):
        factor -= 1
    if factor <= 1:
        ref_c = structural_luminance(ref)
        mov_c = structural_luminance(mov)
        scale = 1.0
    else:
        ref_c = block_mean(ref, factor)
        mov_c = block_mean(mov, factor)
        # block_mean already averages RGB channels to 2-D.
        if ref_c.ndim == 3:
            ref_c = ref_c.mean(axis=2)
        if mov_c.ndim == 3:
            mov_c = mov_c.mean(axis=2)
        scale = float(full_w) / float(ref_c.shape[1])

    h = min(ref_c.shape[0], mov_c.shape[0])
    w = min(ref_c.shape[1], mov_c.shape[1])
    ref_c = ref_c[:h, :w].astype(np.float32, copy=False)
    mov_c = mov_c[:h, :w].astype(np.float32, copy=False)

    if mode is RegistrationMode.MULTI_EXPOSURE:
        high = 0.98 * 65535.0
        low = 200.0
        mask = (ref_c < high) & (mov_c < high) & (ref_c > low) & (mov_c > low)
        ref_c = ref_c * mask.astype(np.float32)
        mov_c = mov_c * mask.astype(np.float32)

    return ref_c, mov_c, scale


def coarse_luma_registration_pair(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    target_w: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Block-mean structural luma pair at the same scale as RGB↔IR coarse.

    Used to cancel gradient-coarse bias when the shift is already small
    (OpenCV-aligned residual / fine lock). Gradients stay on the primary coarse
    stage for multimodal lock on large shifts.
    """
    from pyopticfilm.registration.pyramid import block_mean

    ref = np.asarray(reference)
    mov = np.asarray(moving)
    full_w = ref.shape[1]
    factor = max(1, round(full_w / target_w)) if full_w > target_w else 1
    while factor > 1 and (full_w // factor) < max(32, target_w // 4):
        factor -= 1
    if factor <= 1:
        ref_c = structural_luminance(ref)
        mov_c = structural_luminance(mov)
        scale = 1.0
    else:
        ref_c = block_mean(ref, factor)
        mov_c = block_mean(mov, factor)
        if ref_c.ndim == 3:
            ref_c = ref_c.mean(axis=2)
        if mov_c.ndim == 3:
            mov_c = mov_c.mean(axis=2)
        scale = float(full_w) / float(ref_c.shape[1])
    h = min(ref_c.shape[0], mov_c.shape[0])
    w = min(ref_c.shape[1], mov_c.shape[1])
    return (
        ref_c[:h, :w].astype(np.float32, copy=False),
        mov_c[:h, :w].astype(np.float32, copy=False),
        scale,
    )


def roi_registration_plane(
    tile: np.ndarray,
    mode: RegistrationMode,
    *,
    force: str | None = None,
) -> np.ndarray:
    """Plane for a small refine ROI.

    RGB↔IR refine uses structural luma / IR intensity (OpenCV-compatible), while
    the coarse stage keeps gradients for multimodal lock. Pass ``force`` as
    ``"luma"`` or ``"gradient"`` for diagnostics.
    """
    if force == "gradient":
        if tile.ndim == 3:
            return rgb_edge_energy(tile)
        return ir_edge_energy(tile)
    if force == "luma":
        return structural_luminance(tile)

    # Production: RGB↔IR refine = luma; ME/MP = luma; gradients are coarse-only.
    return structural_luminance(tile)
