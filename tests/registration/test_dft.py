# SPDX-License-Identifier: GPL-3.0-or-later
"""Guizar localized DFT vs brute-force upsample on small arrays."""

from __future__ import annotations

import numpy as np

from pyopticfilm.registration.dft import brute_force_upsample_peak, refine_peak_guizar


def test_guizar_matches_brute_force_on_small_array():
    rng = np.random.default_rng(2)
    h, w = 32, 32
    a = rng.standard_normal((h, w))
    # Fractional shift via Fourier phase (test-only independent of production).
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    true_dx, true_dy = 1.35, -0.72
    phase = np.exp(-2j * np.pi * (fx * true_dx + fy * true_dy))
    b = np.real(np.fft.ifft2(np.fft.fft2(a) * phase))
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    # Integer peak of inverse.
    corr = np.real(np.fft.ifft2(cross / (np.abs(cross) + 1e-12)))
    peak = int(np.argmax(corr))
    py, px = divmod(peak, w)
    ix = px if px < w / 2 else px - w
    iy = py if py < h / 2 else py - h
    gdx, gdy = refine_peak_guizar(cross, iy, ix, upsample_factor=16)
    bdx, bdy = brute_force_upsample_peak(cross, upsample_factor=16, search_radius=3.0)
    # Guizar and brute should agree closely; both near the true shift of b vs a.
    # cross = Fa * conj(Fb): peak location is shift of b relative to a? 
    # We mainly check Guizar ≈ brute.
    assert abs(gdx - bdx) < 0.15
    assert abs(gdy - bdy) < 0.15
