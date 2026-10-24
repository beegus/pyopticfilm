# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark native registration accuracy and speed.

Synthetic:
  - known translations on textured RGB
  - Multi-Pass progressive Y drift (banded)
  - ME gain-changed brackets

Real RGB↔IR (optional ``--real DIR``):
  - residual shift on already-aligned pairs (expect ~0)
  - recover an artificial misalignment applied to the IR plane

Usage::

    uv run python -m tools.bench_registration
    uv run python -m tools.bench_registration --real D:/test3
    uv run python -m tools.bench_registration --real D:/test3 --skip-7200
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyopticfilm.pass_align import (
    align_pass_to_reference_banded,
    estimate_pass_shift,
)
from pyopticfilm.registration import RegistrationMode, register, register_and_align
from pyopticfilm.registration.engine import _refine_translation, estimate_translation
from pyopticfilm.registration.phase import phase_correlate_shift
from pyopticfilm.registration.preprocess import coarse_registration_pair
from pyopticfilm.registration.resample import warp_shift
from pyopticfilm.registration.types import RegistrationOptions


@dataclass
class Row:
    name: str
    dx_err: float
    dy_err: float
    ms: float
    extra: str = ""


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _print_table(title: str, rows: list[Row]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'case':<42} {'dx_err':>8} {'dy_err':>8} {'ms':>9}  notes")
    print("-" * 90)
    for r in rows:
        print(
            f"{r.name:<42} {r.dx_err:8.3f} {r.dy_err:8.3f} {r.ms:9.1f}  {r.extra}"
        )


def timed_estimate(
    reference: np.ndarray,
    moving: np.ndarray,
    mode: RegistrationMode,
) -> tuple[tuple[float, float], dict[str, float], object]:
    """Run estimate with staged timers (plane / coarse / refine)."""
    opt = RegistrationOptions()
    stages: dict[str, float] = {}
    t0 = time.perf_counter()
    coarse_ref, coarse_mov, scale = coarse_registration_pair(
        reference, moving, mode, target_w=opt.coarse_target_px
    )
    stages["plane_ms"] = _ms(t0)

    t1 = time.perf_counter()
    dx, dy, _resp = phase_correlate_shift(coarse_ref, coarse_mov, scale=scale)
    stages["coarse_ms"] = _ms(t1)

    t2 = time.perf_counter()
    dx, dy = _refine_translation(reference, moving, (dx, dy), mode, opt)
    stages["refine_ms"] = _ms(t2)

    t3 = time.perf_counter()
    result = estimate_translation(reference, moving, mode, options=opt)
    stages["total_ms"] = _ms(t3)
    # Prefer the full estimate result (includes accept/reject); stages are informative.
    return result.shift, stages, result


def bench_synthetic_translation() -> list[Row]:
    rng = np.random.default_rng(42)
    rows: list[Row] = []
    cases = [
        ("synth 256² Δ(3,-2)", 256, (3.0, -2.0)),
        ("synth 512² Δ(7.3,-4.1)", 512, (7.3, -4.1)),
        ("synth 1024² Δ(-12.25, 5.5)", 1024, (-12.25, 5.5)),
        ("synth 2048² Δ(20.0, -15.0)", 2048, (20.0, -15.0)),
    ]
    for name, side, (tdx, tdy) in cases:
        base = rng.integers(1000, 40000, (side, side, 3), dtype=np.uint16)
        mov = warp_shift(base, tdx, tdy)
        t0 = time.perf_counter()
        dx, dy = estimate_pass_shift(base, mov)
        ms = _ms(t0)
        _, stages, _ = timed_estimate(base, mov, RegistrationMode.MULTI_EXPOSURE)
        rows.append(
            Row(
                name,
                abs(dx - tdx),
                abs(dy - tdy),
                ms,
                f"got=({dx:.2f},{dy:.2f}) "
                f"plane={stages['plane_ms']:.0f} coarse={stages['coarse_ms']:.0f} "
                f"refine={stages['refine_ms']:.0f}",
            )
        )
    return rows


def bench_synthetic_me_gain() -> list[Row]:
    rng = np.random.default_rng(7)
    rows: list[Row] = []
    for name, side, (tdx, tdy), gain in [
        ("ME 512² gain3× Δ(2,1)", 512, (2.0, 1.0), 3.0),
        ("ME 1024² gain3× Δ(-5.5, 3.2)", 1024, (-5.5, 3.2), 3.0),
    ]:
        short = rng.integers(2000, 12000, (side, side, 3), dtype=np.uint16)
        long = np.clip(short.astype(np.float64) * gain, 0, 65535).astype(np.uint16)
        long = warp_shift(long, tdx, tdy)
        t0 = time.perf_counter()
        result = register(short, long, RegistrationMode.MULTI_EXPOSURE)
        ms = _ms(t0)
        dx, dy = result.shift
        rows.append(
            Row(
                name,
                abs(dx - tdx),
                abs(dy - tdy),
                ms,
                f"accepted={result.accepted} resp={result.quality.response:.3f}",
            )
        )
    return rows


def bench_synthetic_banded() -> list[Row]:
    rng = np.random.default_rng(5)
    h, w = 2400, 300
    base = rng.integers(1000, 40000, (h, w, 3), dtype=np.uint16).astype(np.float64)
    true_dy = np.linspace(0.0, 40.0, h)
    row_idx = np.clip(np.arange(h) - np.round(true_dy).astype(int), 0, h - 1)
    moving = base[row_idx].astype(np.uint16)
    reference = base.astype(np.uint16)

    t0 = time.perf_counter()
    warped, (dx, dy_c) = align_pass_to_reference_banded(reference, moving)
    ms = _ms(t0)

    def resid(a, b, y0, y1):
        return float(
            np.mean(np.abs(a[y0:y1].astype(np.float64) - b[y0:y1].astype(np.float64)))
        )

    band = 200
    top = resid(reference, warped, 0, band)
    bot = resid(reference, warped, h - band, h)
    return [
        Row(
            "banded 2400×300 progressive 0→40px",
            abs(dx),
            abs(abs(dy_c) - 20.0),
            ms,
            f"top_resid={top:.1f} bot_resid={bot:.1f} dy_c={dy_c:.2f}",
        )
    ]


def bench_apply_throughput() -> list[Row]:
    rng = np.random.default_rng(9)
    rows: list[Row] = []
    for side in (1024, 2048, 4096):
        arr = rng.integers(0, 65535, (side, side, 3), dtype=np.uint16)
        # Warm-up
        _ = warp_shift(arr, 3.25, -2.5)
        t0 = time.perf_counter()
        _ = warp_shift(arr, 3.25, -2.5)
        ms = _ms(t0)
        mp = (side * side) / 1e6
        rows.append(
            Row(
                f"apply bilinear {side}² RGB16",
                0.0,
                0.0,
                ms,
                f"{mp / (ms / 1000):.1f} MP/s",
            )
        )
    return rows


def _load_pair(rgb_path: Path, ir_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    from pyopticfilm.image import load_rgb16_tiff

    rgb, dpi = load_rgb16_tiff(rgb_path)
    ir_rgb, _ = load_rgb16_tiff(ir_path)
    ir = np.ascontiguousarray(ir_rgb[:, :, 0])
    return rgb, ir, dpi


def _diag_refine_variants(
    rgb: np.ndarray, ir: np.ndarray, label: str
) -> list[Row]:
    """Compare coarse-only vs refine plane/estimator choices (RGB↔IR residual)."""
    opt = RegistrationOptions()
    mode = RegistrationMode.RGB_IR
    coarse_ref, coarse_mov, scale = coarse_registration_pair(
        rgb, ir, mode, target_w=opt.coarse_target_px
    )
    cdx, cdy, _ = phase_correlate_shift(coarse_ref, coarse_mov, scale=scale)
    rows = [
        Row(
            f"{label} coarse-only",
            abs(cdx),
            abs(cdy),
            0.0,
            f"got=({cdx:.3f},{cdy:.3f}) scale={scale:.2f}",
        )
    ]
    for plane, guizar, tag in (
        ("luma", False, "luma+centroid"),
        ("gradient", False, "grad+centroid"),
        ("luma", True, "luma+Guizar"),
        ("gradient", True, "grad+Guizar"),
    ):
        t0 = time.perf_counter()
        dx, dy = _refine_translation(
            rgb,
            ir,
            (cdx, cdy),
            mode,
            opt,
            plane_force=plane,
            force_guizar=guizar,
            second_pass=False,
        )
        ms = _ms(t0)
        rows.append(
            Row(
                f"{label} {tag}",
                abs(dx),
                abs(dy),
                ms,
                f"got=({dx:.3f},{dy:.3f})",
            )
        )
    # Production path: small coarse → luma coarse re-lock; else multi-ROI refine.
    from pyopticfilm.registration.preprocess import coarse_luma_registration_pair

    t0 = time.perf_counter()
    if max(abs(cdx), abs(cdy)) <= opt.refine_max_residual + 0.5:
        luma_ref, luma_mov, luma_scale = coarse_luma_registration_pair(
            rgb, ir, target_w=opt.coarse_target_px
        )
        dx, dy, _ = phase_correlate_shift(
            luma_ref, luma_mov, scale=luma_scale, upsample=1
        )
        tag = "production (luma coarse)"
    else:
        dx, dy = _refine_translation(
            rgb, ir, (cdx, cdy), mode, opt, second_pass=False
        )
        tag = "production (ROI refine)"
    rows.append(
        Row(
            f"{label} {tag}",
            abs(dx),
            abs(dy),
            _ms(t0),
            f"got=({dx:.3f},{dy:.3f})",
        )
    )
    return rows


def bench_real_ir(
    root: Path, *, skip_7200: bool, diag: bool = False
) -> list[Row]:
    rows: list[Row] = []
    pairs = sorted(root.glob("*.tif"))
    rgb_files = [p for p in pairs if not p.name.endswith("_IR.tif")]
    for rgb_path in rgb_files:
        ir_path = rgb_path.with_name(rgb_path.stem + "_IR.tif")
        if not ir_path.exists():
            continue
        rgb, ir, dpi = _load_pair(rgb_path, ir_path)
        if skip_7200 and dpi >= 7200:
            rows.append(Row(f"{rgb_path.name} (skip 7200)", 0, 0, 0, "skipped"))
            continue
        h, w = rgb.shape[:2]
        label = f"{rgb_path.stem} {w}×{h} @{dpi}"

        if diag:
            rows.extend(_diag_refine_variants(rgb, ir, label))

        shift0, stages0, result0 = timed_estimate(rgb, ir, RegistrationMode.RGB_IR)
        dx0, dy0 = shift0
        rows.append(
            Row(
                f"{label} residual",
                abs(dx0),
                abs(dy0),
                stages0["total_ms"],
                f"acc={result0.accepted} resp={result0.quality.response:.3f} "
                f"plane={stages0['plane_ms']:.0f} coarse={stages0['coarse_ms']:.0f} "
                f"refine={stages0['refine_ms']:.0f}",
            )
        )

        true = (5.0, -3.0) if dpi < 3600 else (12.0, -8.0)
        mis = warp_shift(ir, true[0], true[1])
        t0 = time.perf_counter()
        result = register(rgb, mis, RegistrationMode.RGB_IR)
        ms = _ms(t0)
        dx, dy = result.shift
        t1 = time.perf_counter()
        _aligned, _ = register_and_align(rgb, mis, RegistrationMode.RGB_IR)
        ms_apply = _ms(t1)
        rows.append(
            Row(
                f"{label} recover Δ{true}",
                abs(dx - true[0]),
                abs(dy - true[1]),
                ms,
                f"got=({dx:.2f},{dy:.2f}) align+apply={ms_apply:.0f}ms "
                f"acc={result.accepted}",
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        type=Path,
        default=None,
        help="Directory of RGB + *_IR.tif pairs (already OpenCV-aligned OK)",
    )
    parser.add_argument(
        "--skip-7200",
        action="store_true",
        help="Skip 7200 dpi real pairs (large / slow)",
    )
    parser.add_argument(
        "--no-synth",
        action="store_true",
        help="Skip synthetic benches",
    )
    parser.add_argument(
        "--diag",
        action="store_true",
        help="With --real: print coarse vs luma/gradient/Guizar refine residuals",
    )
    args = parser.parse_args(argv)

    print("Native registration benchmark")
    print(f"numpy {np.__version__}")

    if not args.no_synth:
        _print_table("Synthetic translation (estimate_pass_shift)", bench_synthetic_translation())
        _print_table("Synthetic ME gain", bench_synthetic_me_gain())
        _print_table("Synthetic Multi-Pass banded", bench_synthetic_banded())
        _print_table("Apply throughput (bilinear warp)", bench_apply_throughput())

    if args.real is not None:
        if not args.real.is_dir():
            print(f"not a directory: {args.real}")
            return 1
        _print_table(
            f"Real RGB↔IR under {args.real}",
            bench_real_ir(args.real, skip_7200=args.skip_7200, diag=args.diag),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
