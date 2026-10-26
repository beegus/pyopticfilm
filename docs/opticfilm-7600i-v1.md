# OpticFilm 7600i v1 (GL843)

`07b3:0c3b`, bcdDevice `0x0400`. Capture-derived: scans run SilverFast's job, built from tables
decoded from USB captures of a real 7600i v1 (captures and findings from
[OpenOptic](https://github.com/bazfp/OpenOptic), where they run on this hardware). The 7600i v2
(bcdDevice `0x0605`, GL845) is unaffected.

| | |
|---|---|
| Colour | 1440, 3600, 7200 dpi full frame, 16-bit linear RGB |
| Sampling | Nominal vertical resolution, one dummy CCD line at 7200 dpi (zero at lower resolutions), two-line buffer threshold and backtracking disabled |
| Area | 36.61 × 24.72 mm: sensor pixels 80–10458 (3600 dpi: 82–10458; 1440 dpi: 83–10463), the full area. SilverFast's window is 210–10458 (36.15 mm). The pixel count must stay even: with an odd count the scanner sends a pixel less per line and the read stalls |
| Infrared | 3600, 7200 dpi (`mode="infrared"`, or `infrared=True` with colour) |
| Crop | `area=`, on the host; acquisition time and peak memory still reflect the full frame |
| Not yet | multi-exposure, multi-pass, manual exposure, other resolutions, grey |

Differences from the SANE GL843 path:

- Register reads set the address (`0x83`, one byte) then read `0x84`; the address auto-increments
  after each read. Every write is acknowledged by polling `0x8E`/`0x20` bit 0.
- Boot, homing and scans use the decoded register files and motor tables. Scan timing, window
  and calibration differ from SilverFast's as described above and below. The first positioning move stops
  after the captured interval (2.56–2.57 s), rather than a live position event; host scheduling
  delays can affect its endpoint. Repeated positioning across devices needs hardware validation.
- Raw frames arrive mirrored with fractional R/G/B row delays. Captured jobs sample at twice
  the nominal vertical resolution; defaults sample once. At 7200 dpi the column stagger is
  eight raw lines in captured jobs, four with single sampling. Infrared uses the red row.
- Infrared at 7200 dpi was not captured: it is the 7200 dpi job with the changes SilverFast
  makes for infrared at 3600 dpi (lamp off, infrared LED, AFE), and shading computed from the job's
  white frame as SilverFast does (`0x13000 × 0x2000 / white`).
- With `apply_calib=True` (default), live probes calibrate each scan: four-pixel peak means select AFE gains;
  32 black pixels determine offsets; 128-line references discard eight extremes at each end.
  Dark correction uses a forward 100-value mean (separate column parities at 7200 dpi).
  Both shading uploads cover the actual window, including added pixels and infrared hot columns.

Calibration runs inside each scan; `scanner.calibrate()` is unsupported and no cached calibration
is used. `apply_calib=False` keeps captured AFE values and recorded shading where available;
widened windows and 7200 dpi IR still recompute final gains from the white frame.
For recorded sampling and timing, use a `Model7600iV1` with `single_sample=False` and `dummy_lines="recorded"`; the
window and live calibration remain independent settings. These are model/session options,
not additional `Scanner.scan()` arguments.

Validation: without options, the built jobs give the same host program as the four captured
jobs (1440/3600/7200 dpi colour, 3600 dpi IR) and the same register state at every frame.
7200 dpi IR is checked against the 3600 dpi IR changes, not against a capture. The default jobs
are locked to the jobs validated on hardware (`tests/model_lock/opticfilm_7600i_v1/`).
Separate tests cover calibration substitution, assembly, homing, upload failure and cancellation,
with simulated end-to-end scans. None of this establishes optical accuracy of the defaults.

Live calibration was derived from eight captured passes. The dark outlier gate uses inclusive
64; captures do not uniquely identify the threshold. Final colour gain increments remain
0/+1/+2 as captured. Optical verification on hardware is still needed. Full-frame colour plus
IR retains both three-channel frames during processing, increasing peak memory use.

Job model: every recorded job runs the same steps. Two AFE offset probes, the gain line, two more probes, the dark and white frames, two positioning moves, the main scan and the park. `device/tables_7600i_v1.py` holds what the steps write. That is the power-on and scan register files, per-resolution window, line select, buffer and step settings, the motor ramps, and the boot and homing values. DPISET (resolution ÷ 6), the window, frame size, LINCNT and the scan motor cruise come from formulas. `scan/job_gl843_v1.build_job` builds a job from these and the calibration results. Its options set the window start, the main scan's dummy lines, single sampling and exposure. The values pyopticfilm changes in SilverFast's main scan are written with the scan start. `job_for` is the job a scan runs, with the model's options. `host_program` reduces a job to the host's writes, payloads, reads and waits, without poll counts or untimed delays. The recorded jobs stay in the data file: the reference for the built jobs, and SilverFast's calibration results (`apply_calib=False`) and lamp levels. Not explained yet: MAXWD at 1440 dpi (pixels + 5450 at 3600 and 7200 dpi), and what ends the first positioning move.

Data: `device/data/opticfilm_7600i_v1.json.gz`, rebuilt with
`python tools/import_7600i_v1_profiles.py /path/to/OpenOptic`.
