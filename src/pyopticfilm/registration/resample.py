# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply registration transforms (integer-exact, separable linear/cubic)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Overlap:
    src_y: slice
    src_x: slice
    dst_y: slice
    dst_x: slice


def translation_overlap(h: int, w: int, dx: int, dy: int) -> Overlap | None:
    """Common overlap for integer translation of moving → reference frame."""
    if dx >= 0:
        dst_x0, src_x0 = dx, 0
        width = w - dx
    else:
        dst_x0, src_x0 = 0, -dx
        width = w + dx
    if dy >= 0:
        dst_y0, src_y0 = dy, 0
        height = h - dy
    else:
        dst_y0, src_y0 = 0, -dy
        height = h + dy
    if width <= 0 or height <= 0:
        return None
    return Overlap(
        src_y=slice(src_y0, src_y0 + height),
        src_x=slice(src_x0, src_x0 + width),
        dst_y=slice(dst_y0, dst_y0 + height),
        dst_x=slice(dst_x0, dst_x0 + width),
    )


def fill_shift_border(out: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Replace OOB strips with nearest interior column/row (IR-safe)."""
    h, w = out.shape[:2]
    bx = int(np.ceil(abs(float(dx))))
    by = int(np.ceil(abs(float(dy))))
    if bx <= 0 and by <= 0:
        return out
    result = np.array(out, copy=True)
    if bx > 0 and bx < w:
        if dx > 0:
            result[:, :bx] = result[:, bx : bx + 1]
        elif dx < 0:
            result[:, w - bx :] = result[:, w - bx - 1 : w - bx]
    if by > 0 and by < h:
        if dy > 0:
            result[:by] = result[by : by + 1]
        elif dy < 0:
            result[h - by :] = result[h - by - 1 : h - by]
    return result


def fill_row_shift_border(out: np.ndarray, dx: float, dy_per_row: np.ndarray) -> np.ndarray:
    """Per-row generalization of :func:`fill_shift_border`."""
    h, w = out.shape[:2]
    result = np.array(out, copy=True)
    bx = int(np.ceil(abs(float(dx))))
    if 0 < bx < w:
        if dx > 0:
            result[:, :bx] = result[:, bx : bx + 1]
        else:
            result[:, w - bx :] = result[:, w - bx - 1 : w - bx]
    src_y = np.arange(h, dtype=np.float64) - dy_per_row
    valid_rows = np.flatnonzero((src_y >= 0) & (src_y <= h - 1))
    if valid_rows.size and valid_rows.size < h:
        first_valid, last_valid = int(valid_rows[0]), int(valid_rows[-1])
        if first_valid > 0:
            result[:first_valid] = result[first_valid]
        if last_valid < h - 1:
            result[last_valid + 1 :] = result[last_valid]
    return result


def warp_integer(mov: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Index-gather translate; exact within overlap; border filled from interior."""
    if dx == 0 and dy == 0:
        return mov
    h, w = mov.shape[:2]
    x_idx = np.clip(np.arange(w) + dx, 0, w - 1)
    y_idx = np.clip(np.arange(h) + dy, 0, h - 1)
    if mov.ndim == 3:
        out = mov[y_idx][:, x_idx, :]
    else:
        out = mov[y_idx][:, x_idx]
    return fill_shift_border(out, float(dx), float(dy))


def _keys_cubic_weights(t: float) -> tuple[float, float, float, float]:
    """Keys cubic convolution weights for fractional offset ``t`` in [0, 1)."""
    a = -0.5
    t2 = t * t
    t3 = t2 * t
    w0 = a * (t3 - 2 * t2 + t)
    w1 = (a + 2) * t3 - (a + 3) * t2 + 1
    w2 = -(a + 2) * t3 + (2 * a + 3) * t2 - a * t
    w3 = a * (-t3 + t2)
    return w0, w1, w2, w3


def _linear_weights(shift: float, n: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Indices and weights for dst[i] = (1-f)*src[i+base] + f*src[i+base+1]."""
    base = int(np.floor(shift))
    frac = float(shift - base)
    idx = np.arange(n, dtype=np.int64)
    i0 = np.clip(idx + base, 0, n - 1)
    i1 = np.clip(idx + base + 1, 0, n - 1)
    return i0, i1, 1.0 - frac, frac


def _sample_1d_linear(plane: np.ndarray, axis: int, shift: float) -> np.ndarray:
    """Separable bilinear tap along one axis (float32)."""
    if abs(shift) < 1e-12:
        return plane.astype(np.float32, copy=False)
    n = plane.shape[axis]
    i0, i1, w0, w1 = _linear_weights(shift, n)
    if axis == 0:
        a = plane[i0].astype(np.float32, copy=False)
        b = plane[i1].astype(np.float32, copy=False)
    else:
        a = plane[:, i0].astype(np.float32, copy=False)
        b = plane[:, i1].astype(np.float32, copy=False)
    return w0 * a + w1 * b


def _sample_1d_cubic(plane: np.ndarray, axis: int, shift: float) -> np.ndarray:
    """Separable cubic translate along one axis (constant shift)."""
    if abs(shift) < 1e-12:
        return plane.astype(np.float32, copy=False)
    n = plane.shape[axis]
    base = int(np.floor(shift))
    frac = float(shift - base)
    kw0, kw1, kw2, kw3 = _keys_cubic_weights(frac)
    idx = np.arange(n)
    i0 = np.clip(idx + base - 1, 0, n - 1)
    i1 = np.clip(idx + base, 0, n - 1)
    i2 = np.clip(idx + base + 1, 0, n - 1)
    i3 = np.clip(idx + base + 2, 0, n - 1)

    def take(indices: np.ndarray) -> np.ndarray:
        if axis == 0:
            return plane[indices].astype(np.float32, copy=False)
        return plane[:, indices].astype(np.float32, copy=False)

    return kw0 * take(i0) + kw1 * take(i1) + kw2 * take(i2) + kw3 * take(i3)


def _finalize_warp(out: np.ndarray, dtype: np.dtype, dx: float, dy: float) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        out = np.clip(np.rint(out), 0, np.iinfo(dtype).max).astype(dtype)
    else:
        out = out.astype(dtype, copy=False)
    return fill_shift_border(out, dx, dy)


def warp_cubic(mov: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Separable Keys cubic translation."""
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return mov
    if abs(dx - round(dx)) < 1e-6 and abs(dy - round(dy)) < 1e-6:
        return warp_integer(mov, round(dx), round(dy))

    arr = np.asarray(mov)
    dtype = arr.dtype

    def warp_plane(p: np.ndarray) -> np.ndarray:
        tmp = _sample_1d_cubic(p, axis=1, shift=dx) if abs(dx) >= 1e-12 else p.astype(np.float32)
        return _sample_1d_cubic(tmp, axis=0, shift=dy) if abs(dy) >= 1e-12 else tmp

    if arr.ndim == 3:
        out = np.stack([warp_plane(arr[:, :, c]) for c in range(arr.shape[2])], axis=2)
    else:
        out = warp_plane(arr)
    return _finalize_warp(out, dtype, dx, dy)


def _affine_ishift_slices(arr: np.ndarray, sx: int, sy: int) -> np.ndarray:
    """``out[y, x] = arr[y + sy, x + sx]`` on the in-bounds overlap (zeros elsewhere)."""
    h, w = arr.shape[:2]
    out = np.zeros_like(arr)
    y0 = max(0, -sy)
    y1 = min(h, h - sy)
    x0 = max(0, -sx)
    x1 = min(w, w - sx)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1, ...] = arr[y0 + sy : y1 + sy, x0 + sx : x1 + sx, ...]
    return out


def warp_linear(
    mov: np.ndarray,
    dx: float,
    dy: float,
    *,
    chunk_rows: int = 512,
) -> np.ndarray:
    """Bilinear translate via slice copies + scalar weights (no mgrid / fancy index).

    ``chunk_rows`` is accepted for API compatibility and unused.
    """
    del chunk_rows
    if not np.isfinite(dx) or not np.isfinite(dy):
        return mov
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return mov
    if abs(dx - round(dx)) < 1e-6 and abs(dy - round(dy)) < 1e-6:
        return warp_integer(mov, round(dx), round(dy))

    arr = np.asarray(mov)
    dtype = arr.dtype
    ix = int(np.floor(dx))
    iy = int(np.floor(dy))
    fx = float(dx - ix)
    fy = float(dy - iy)

    c00 = _affine_ishift_slices(arr, ix, iy)
    if abs(fx) < 1e-6 and abs(fy) < 1e-6:
        out = c00.astype(np.float32, copy=False)
    elif abs(fx) < 1e-6:
        c01 = _affine_ishift_slices(arr, ix, iy + 1)
        out = (1.0 - fy) * c00.astype(np.float32) + fy * c01.astype(np.float32)
    elif abs(fy) < 1e-6:
        c10 = _affine_ishift_slices(arr, ix + 1, iy)
        out = (1.0 - fx) * c00.astype(np.float32) + fx * c10.astype(np.float32)
    else:
        c10 = _affine_ishift_slices(arr, ix + 1, iy)
        c01 = _affine_ishift_slices(arr, ix, iy + 1)
        c11 = _affine_ishift_slices(arr, ix + 1, iy + 1)
        c00f = c00.astype(np.float32)
        out = (
            (1.0 - fx) * (1.0 - fy) * c00f
            + fx * (1.0 - fy) * c10.astype(np.float32)
            + (1.0 - fx) * fy * c01.astype(np.float32)
            + fx * fy * c11.astype(np.float32)
        )
    return _finalize_warp(out, dtype, dx, dy)


def warp_shift(
    mov: np.ndarray,
    dx: float,
    dy: float,
    *,
    method: str = "linear",
    chunk_rows: int = 512,
) -> np.ndarray:
    """Translate ``mov`` by ``(dx, dy)`` — linear (OpenCV-like) or cubic."""
    if method == "cubic":
        return warp_cubic(mov, dx, dy)
    return warp_linear(mov, dx, dy, chunk_rows=chunk_rows)


def warp_row_shifts(
    mov: np.ndarray,
    dx: float,
    dy_per_row: np.ndarray,
    *,
    method: str = "linear",
    chunk_rows: int = 512,
) -> np.ndarray:
    """Per-row dy + constant dx without HxW map arrays (remap convention).

    Matches legacy OpenCV ``remap(x - dx, y - dy[y])`` used by banded Multi-Pass.
    """
    arr = np.asarray(mov)
    h, _w = arr.shape[:2]
    dy = np.asarray(dy_per_row, dtype=np.float64)
    if dy.shape[0] != h:
        raise ValueError("dy_per_row length must match image height")

    dtype = arr.dtype
    # Horizontal: constant -dx in remap convention ⇒ sample at x - dx = x + (-dx)
    # using separable linear with shift = -dx for warpAffine-style helper...
    # Remap: dst(x) = src(x - dx). Separable helper uses dst(x)=src(x+s) ⇒ s = -dx.
    shift_x = -float(dx)

    def horiz(p: np.ndarray) -> np.ndarray:
        if abs(shift_x) < 1e-12:
            return p.astype(np.float32, copy=False)
        return _sample_1d_linear(p, axis=1, shift=shift_x)

    if arr.ndim == 3:
        planes = [horiz(arr[:, :, c]) for c in range(arr.shape[2])]
        tmp = np.stack(planes, axis=2)
    else:
        tmp = horiz(arr)

    # Vertical: dst(y) = src(y - dy[y]) ⇒ sample index sy = y - dy[y]
    out = np.empty_like(tmp, dtype=np.float32)
    src_y = np.arange(h, dtype=np.float64) - dy
    y0 = np.floor(src_y).astype(np.int64)
    frac = (src_y - y0).astype(np.float32)
    y1 = y0 + 1
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    chunk = max(32, int(chunk_rows))
    for r0 in range(0, h, chunk):
        r1 = min(h, r0 + chunk)
        f = frac[r0:r1]
        a0 = y0c[r0:r1]
        a1 = y1c[r0:r1]
        if tmp.ndim == 3:
            # Broadcast weights over width/channels.
            w0 = (1.0 - f)[:, None, None]
            w1 = f[:, None, None]
            out[r0:r1] = w0 * tmp[a0] + w1 * tmp[a1]
        else:
            w0 = (1.0 - f)[:, None]
            w1 = f[:, None]
            out[r0:r1] = w0 * tmp[a0] + w1 * tmp[a1]

    if np.issubdtype(dtype, np.integer):
        out_i = np.clip(np.rint(out), 0, np.iinfo(dtype).max).astype(dtype)
    else:
        out_i = out.astype(dtype, copy=False)
    return fill_row_shift_border(out_i, dx, dy)


def valid_mask_translation(h: int, w: int, dx: float, dy: float) -> np.ndarray:
    """Boolean mask of pixels that sample inside the moving image."""
    yy = np.arange(h, dtype=np.float64)[:, None]
    xx = np.arange(w, dtype=np.float64)[None, :]
    sx = xx + dx
    sy = yy + dy
    return (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
