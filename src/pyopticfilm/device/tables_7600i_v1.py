# SPDX-License-Identifier: GPL-3.0-or-later
"""OpticFilm 7600i v1 (GL843) tables, decoded from SilverFast's recorded jobs.

Every value here was checked against the four recorded jobs (1440 / 3600 / 7200 dpi colour,
3600 dpi infrared) by ``tools/decode_7600i_v1_jobs.py``; :mod:`pyopticfilm.scan.job_gl843_v1`
builds the jobs from them. Values given per resolution where no formula explains them are
SilverFast's choices; formulas are used where they hold for every recorded frame.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Register file SilverFast writes first at the start of every job (= the boot program).
POWER_ON_REGS: dict[int, int] = {
    0x01: 0x22, 0x02: 0x38, 0x03: 0x9F, 0x04: 0x22, 0x05: 0x48, 0x06: 0xF0, 0x08: 0x00, 0x09: 0x00,
    0x0A: 0x00, 0x0B: 0x4A, 0x0D: 0x00, 0x10: 0x00, 0x11: 0x00, 0x12: 0x00, 0x13: 0x00, 0x14: 0x00,
    0x15: 0x00, 0x16: 0x3B, 0x17: 0x0C, 0x18: 0x10, 0x19: 0x2A, 0x1A: 0x30, 0x1B: 0x00, 0x1C: 0x00,
    0x1D: 0x9A, 0x1E: 0x10, 0x1F: 0x01, 0x20: 0x10, 0x21: 0x04, 0x22: 0x01, 0x23: 0x01, 0x24: 0x04,
    0x25: 0x00, 0x26: 0x00, 0x27: 0x00, 0x2C: 0x02, 0x2D: 0x58, 0x2E: 0x80, 0x2F: 0x80, 0x30: 0x00,
    0x31: 0x14, 0x32: 0x27, 0x33: 0xEC, 0x34: 0x3C, 0x35: 0x00, 0x36: 0x40, 0x37: 0x00, 0x38: 0x2A,
    0x39: 0xF8, 0x3D: 0x00, 0x3E: 0x00, 0x3F: 0x01, 0x52: 0x01, 0x53: 0x04, 0x54: 0x06, 0x55: 0x09,
    0x56: 0x0C, 0x57: 0x0F, 0x58: 0x1B, 0x59: 0x00, 0x5A: 0x40, 0x5E: 0x01, 0x5F: 0x01, 0x67: 0x80,
    0x68: 0x80, 0x69: 0x01, 0x6A: 0x04, 0x6B: 0x31, 0x6C: 0x4C, 0x6D: 0x80, 0x6E: 0x4C, 0x6F: 0x80,
    0x70: 0x07, 0x71: 0x08, 0x72: 0x08, 0x73: 0x09, 0x74: 0x00, 0x75: 0x00, 0x76: 0x00, 0x77: 0x00,
    0x78: 0x00, 0x79: 0xFF, 0x7A: 0x00, 0x7B: 0x00, 0x7C: 0x00, 0x7D: 0x00, 0x7E: 0x00, 0x7F: 0x00,
    0x80: 0x00, 0x81: 0x00, 0x82: 0x00, 0x83: 0x00, 0x84: 0x00, 0x85: 0x00, 0x86: 0x00, 0x87: 0x00,
    0x9D: 0x00, 0xA6: 0x00, 0xA7: 0x07, 0xA8: 0x20, 0xA9: 0x01,
}

#: Register file written after the clock switch; every job step starts from it.
SCAN_REGS: dict[int, int] = {
    0x01: 0x22, 0x02: 0x38, 0x03: 0x9F, 0x04: 0x22, 0x05: 0x48, 0x06: 0xF0, 0x08: 0x00, 0x09: 0x00,
    0x0A: 0x00, 0x0B: 0x4A, 0x0D: 0x00, 0x10: 0x00, 0x11: 0x00, 0x12: 0x00, 0x13: 0x00, 0x14: 0x00,
    0x15: 0x00, 0x16: 0x27, 0x17: 0x0C, 0x18: 0x10, 0x19: 0x2A, 0x1A: 0x00, 0x1B: 0x00, 0x1C: 0x20,
    0x1D: 0x84, 0x1E: 0x10, 0x1F: 0x01, 0x20: 0x10, 0x21: 0x04, 0x22: 0x01, 0x23: 0x01, 0x24: 0x04,
    0x25: 0x00, 0x26: 0x00, 0x27: 0x00, 0x2C: 0x02, 0x2D: 0x58, 0x2E: 0x80, 0x2F: 0x80, 0x30: 0x00,
    0x31: 0x14, 0x32: 0x27, 0x33: 0xEC, 0x34: 0x3C, 0x35: 0x00, 0x36: 0x40, 0x37: 0x00, 0x38: 0x36,
    0x39: 0xB0, 0x3D: 0x00, 0x3E: 0x00, 0x3F: 0x01, 0x52: 0x0A, 0x53: 0x0D, 0x54: 0x0F, 0x55: 0x01,
    0x56: 0x04, 0x57: 0x07, 0x58: 0x31, 0x59: 0x79, 0x5A: 0xC0, 0x5E: 0x01, 0x5F: 0x01, 0x67: 0x80,
    0x68: 0x80, 0x69: 0x01, 0x6A: 0x04, 0x6B: 0x31, 0x6C: 0x4C, 0x6D: 0x00, 0x6E: 0x4C, 0x6F: 0x80,
    0x70: 0x0C, 0x71: 0x0D, 0x72: 0x0E, 0x73: 0x0F, 0x74: 0x00, 0x75: 0x00, 0x76: 0x00, 0x77: 0x00,
    0x78: 0x00, 0x79: 0x00, 0x7A: 0x00, 0x7B: 0x00, 0x7C: 0x00, 0x7D: 0x00, 0x7E: 0x00, 0x7F: 0x00,
    0x80: 0xFF, 0x81: 0x00, 0x82: 0x00, 0x83: 0x00, 0x84: 0x00, 0x85: 0x00, 0x86: 0x00, 0x87: 0x00,
    0x9D: 0x00, 0xA6: 0x00, 0xA7: 0x07, 0xA8: 0x20, 0xA9: 0x01,
}

#: Slot-3 table of fast moves (wake-up move, second positioning feed, homing).
FAST_RAMP: tuple[int, ...] = (
    64102, 6297, 4716, 3980, 3531, 3217, 2983, 2796, 2645, 2520, 2410, 2316,
    2234, 2160, 2095, 2035, 1982, 1931, 1886, 1845, 1806, 1769, 1734, 1704,
    1673, 1645, 1619, 1593, 1569, 1547, 1525, 1504, 1484, 1465, 1447, 1430,
    1413, 1397, 1381, 1366, 1352, 1339, 1325, 1312, 1300, 1287, 1276, 1264,
    1253, 1243, 1232, 1222, 1212, 1203, 1193, 1184, 1175, 1167, 1158, 1150,
    1142, 1134, 1126, 1119, 1112, 1104, 1097, 1091, 1084, 1077, 1071, 1065,
    1058, 1052, 1046, 1040, 1035, 1029, 1024, 1018, 1013, 1008, 1002, 998,
    992, 988, 983, 978, 973, 969, 964, 960, 956, 951, 947, 943,
    939, 935, 931, 927, 923, 919, 915, 912, 908, 904, 901, 897,
    894, 890, 887, 884, 880, 877, 874, 871, 868, 864, 861, 858,
    855, 852, 849, 846, 844, 841, 838, 835, 833, 830, 827, 825,
    822, 819, 817, 814, 812, 809, 807, 804, 802, 800, 797, 795,
    793, 790, 788, 786, 784, 781, 779, 777, 775, 773, 771, 769,
    767, 764, 763, 761, 759, 757, 755, 753, 751, 749, 747, 745,
    743, 742, 740, 738, 736, 734, 732, 731, 729, 727, 726, 724,
    722, 721, 719, 717, 716, 714, 712, 711, 709, 708, 706, 705,
    703, 702, 700, 699, 697, 695, 694, 693, 691, 690, 688, 687,
    686, 684, 683, 681, 680, 679, 677, 676, 675, 673, 672, 671,
    669, 668, 667, 665, 664, 663, 662, 661, 659, 658, 657, 656,
    654, 653, 652, 651, 650, 649, 647, 646, 645, 644, 643, 642,
    641, 640, 638, 637, 636, 635, 634, 633, 632, 631, 630, 629,
    628, 627, 626, 625,
)

#: Slot-3 table of the first positioning move (stopped on a timer).
POSITION_RAMP: tuple[int, ...] = (
    20325, 12195, 10040, 8960, 8278, 7763, 7396, 7082, 6830, 6596, 6410, 6250,
    6097, 5966, 5841, 5733, 5630, 5530, 5446, 5364, 5285, 5219, 5144, 5081,
    5030, 4970, 4911, 4863, 4816, 4770, 4725, 4681, 4638, 4604, 4562, 4528,
    4488, 4456, 4424, 4393, 4363, 4332, 4302, 4273, 4244, 4222, 4194, 4166,
    4145, 4118, 4098, 4078, 4051, 4032, 4012, 3993, 3968, 3949, 3930, 3912,
    3894, 3875, 3858, 3840, 3828, 3810, 3793, 3776, 3765, 3748, 3731, 3714,
    3703, 3687, 3676, 3660, 3649, 3633, 3623, 3607, 3597, 3581, 3571, 3561,
    3546, 3536, 3526, 3511, 3501, 3491, 3481, 3467, 3457, 3448, 3438, 3429,
    3419, 3410, 3396, 3387, 3378, 3369, 3360, 3351, 3342, 3333, 3324, 3315,
    3306, 3298, 3289, 3285, 3276, 3267, 3259, 3250, 3242, 3234, 3225, 3221,
    3213, 3205, 3196, 3188, 3184, 3176, 3168, 3160, 3156, 3148, 3140, 3136,
    3128, 3121, 3117, 3109, 3101, 3097, 3090, 3082, 3078, 3071, 3067, 3059,
    3052, 3048, 3041, 3037, 3030, 3026, 3019, 3015, 3008, 3001, 2997, 2990,
    2986, 2983, 2976, 2972, 2965, 2962, 2955, 2951, 2944, 2941, 2934, 2930,
    2927, 2920, 2917, 2910, 2906, 2903, 2896, 2893, 2886, 2883, 2880, 2873,
    2870, 2866, 2860, 2857, 2853, 2847, 2844, 2840, 2834, 2831, 2828, 2824,
    2818, 2815, 2812, 2805, 2802, 2799, 2796, 2790, 2787, 2783, 2780, 2774,
    2771, 2768, 2765, 2762, 2756, 2753, 2750, 2747, 2744, 2738, 2735, 2732,
    2729, 2726, 2720, 2717, 2714, 2711, 2708, 2705, 2699, 2696, 2693, 2691,
    2688, 2685, 2682, 2679, 2673, 2670, 2668, 2665, 2662, 2659, 2656, 2653,
    2651, 2645, 2642, 2639, 2637, 2634, 2631, 2628, 2626, 2623, 2620, 2617,
    2615, 2612, 2609, 2604,
)


#: Motor table of the stationary AFE probes: two entries (the move is one step).
STATIONARY_TABLE: tuple[int, ...] = (14000, 14000)
#: First entry of every scan table: the start speed before the constant cruise.
SCAN_RAMP_START = 25252

LPERIOD = 14000
#: Motor steps per inch of carriage travel.
MOTOR_STEPS_PER_INCH = 14400
#: Sensor pixels per inch at DPISET 1200 (STRPIXEL / ENDPIXEL are counted in these).
SENSOR_DPI = 1200
#: AFE probes and the gain line run at full sensor resolution.
PROBE_DPISET = 1200
PROBE_START = 20
PROBE_PIXELS = 512
GAIN_LINE_PIXELS = 10378
#: Dark and white frames read this many lines.
CALIBRATION_LINES = 128
#: FEEDL of the two positioning moves before the main scan: the first is stopped on a timer,
#: the second runs to the end.
POSITION_FEEDS = (19490, 13228)
#: When the first positioning move is stopped (ms after its start), as in each recorded job.
#: The vendor stops it on an event the captures do not show; between captures it varies by
#: 12 ms. The 7200 dpi infrared job keeps the 7200 dpi colour job's.
POSITION_STOP_MS: dict[tuple[str, int], float] = {
    ("color", 1440): 2571.9,
    ("color", 3600): 2559.7,
    ("color", 7200): 2571.8,
    ("infrared", 3600): 2571.8,
    ("infrared", 7200): 2571.8,
}
#: FMOVNO (0x6A) of the calibration steps and of the positioning moves and main scan.
FMOVNO_CALIBRATION = 0x46
FMOVNO_SCAN = 0xFF
#: Pixel clock of the line period (LPERIOD 14000 at 2.5 MHz = 5.6 ms per sensor line).
PIXEL_CLOCK_HZ = 2_500_000

#: Boot: ``0x8C`` index 0x10 (clock) between the register files, and the AFE reset after each.
BOOT_CLOCK = 0xD4
AFE_RESET = (0xF8, 0x80, 0, 0, 0, 0, 0, 0)
#: GPIO after the boot program; ``0xA8`` = 0x20 keeps the infrared LED (GPIO27) off.
BOOT_GPIO: tuple[tuple[int, int], ...] = (
    (0x6B, 0x31), (0x6C, 0x4C), (0x6D, 0x00), (0x6E, 0x4C), (0x6F, 0x80),
    (0xA6, 0x00), (0xA7, 0x07), (0xA8, 0x20), (0xA9, 0x01),
)
#: AFE (AD9826) after boot: configuration 0/1, gains 2-4, offsets 5-7.
BOOT_AFE: dict[int, int] = {0: 0xF8, 1: 0x80, 2: 0x23, 3: 0x1B, 4: 0x24, 5: 0x1E, 6: 0x30, 7: 0x18}

#: Motor registers of homing moves: those of the vendor's second positioning feed at 3600 dpi.
HOME_MOTOR_REGS: dict[int, int] = {
    0x1C: 0x20, 0x1F: 0x00, 0x21: 0x02, 0x22: 0x01, 0x23: 0x01, 0x24: 0x02, 0x38: 0x36, 0x39: 0xB0,
    0x5E: 0x01, 0x5F: 0x01, 0x67: 0x80, 0x68: 0x80, 0x69: 0x01, 0x6B: 0x31, 0x6C: 0x4C, 0x6D: 0x00,
    0x6E: 0x4C, 0x6F: 0x80, 0x80: 0xFF,
}
#: ``0x01`` during homing (SCAN off).
HOME_REG_01 = 0x22


@dataclass(frozen=True)
class Resolution:
    """SilverFast's per-resolution choices for the dark, white and main frames."""

    #: sensor window (STRPIXEL, ENDPIXEL) at DPISET 1200
    window: tuple[int, int]
    #: main-scan lines at twice the nominal vertical resolution
    main_lines: int
    #: LINESEL (0x1E low nibble) of the dark and white frames, and of the main scan
    linesel_calibration: int
    linesel_main: int
    #: BUFSEL (0x20) of the dark, white and main frames
    bufsel: tuple[int, int, int]
    #: 0x21 / 0x24 (motor step counts) of the white frame and main scan
    steps: int
    #: MAXWD of the probes, the gain line, and the dark / white / main frames. At 3600 and
    #: 7200 dpi it is pixels + 5450; the 1440 dpi job (SilverFast's prescan) does not follow it.
    maxwd: tuple[int, int, int]


RESOLUTIONS: dict[int, Resolution] = {
    1440: Resolution((213, 10463), 2824, 1, 0, (0x05, 0x07, 0x05), 2, (5706, 10639, 6475)),
    3600: Resolution((210, 10458), 7058, 2, 1, (0x0C, 0x13, 0x10), 2, (5962, 15828, 10574)),
    7200: Resolution((210, 10458), 14122, 5, 2, (0x17, 0x35, 0x35), 1, (5962, 15828, 15698)),
}


def dpiset(dpi: int) -> int:
    """DPISET of the dark, white and main frames: a sixth of the resolution."""
    return dpi // 6


def scan_window(dpi: int, first_pixel: int | None = None) -> tuple[int, int]:
    """(STRPIXEL, ENDPIXEL) of the dark, white and main frames.

    SilverFast's window, or one that starts at sensor pixel ``first_pixel`` (or the next one that
    keeps the pixel count even: with an odd count the scanner sends a pixel less per line).
    ENDPIXEL stays SilverFast's.
    """
    start, end = RESOLUTIONS[dpi].window
    if first_pixel is not None:
        step = SENSOR_DPI // dpiset(dpi)
        start -= max(0, (start - first_pixel) // step // 2 * 2) * step
    return start, end


def window_pixels(dpi: int, first_pixel: int | None = None) -> int:
    start, end = scan_window(dpi, first_pixel)
    return (end - start) * dpiset(dpi) // SENSOR_DPI


def scan_table(cruise: int, *, first: int | None = None) -> tuple[int, ...]:
    """A scan motor table: ``first`` (the ramp start, or the cruise if slower), then the cruise period."""
    return (max(SCAN_RAMP_START, cruise) if first is None else first,) + (cruise,) * 255


def cruise(linesel: int, steps_per_line: int, *, lperiod: int = LPERIOD) -> int:
    """Motor period that moves ``steps_per_line`` steps per sensor line of LPERIOD × (LINESEL+1)."""
    period, rest = divmod((linesel + 1) * lperiod, steps_per_line)
    if rest or not 0 < period <= 0xFFFF:
        raise ValueError(f"no whole motor period for {steps_per_line} steps per line at LINESEL {linesel}")
    return period


def main_steps_per_line(yres: int) -> int:
    """Motor steps per sensor line when the main scan samples ``yres`` lines per inch."""
    return MOTOR_STEPS_PER_INCH // yres


#: The white frame always moves two steps per line.
WHITE_STEPS_PER_LINE = 2
