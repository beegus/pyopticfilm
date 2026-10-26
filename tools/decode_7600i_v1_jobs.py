# SPDX-License-Identifier: GPL-3.0-or-later
"""Decode the recorded OpticFilm 7600i v1 jobs into named tables, and check what they explain.

Usage::

    python tools/decode_7600i_v1_jobs.py [--json decoded.json]

Three levels, each checked against all four recorded jobs:

A. Register state at every frame read = base map + per-step map + per-resolution map + per-mode
   map. Candidate formulas for the per-resolution registers are tested on every frame.
B. The operation sequence of each step (register batches, AFE writes, uploads, set-ups, waits,
   delays, reads) against one template job, with every deviation listed.
C. Motor tables: the distinct tables, and the scan cruise formula.

Status polls become ``wait`` items (a run of identical expected values is one wait); the
write acknowledge after every control write is checked, then left out.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "pyopticfilm" / "device" / "data" / "opticfilm_7600i_v1.json.gz"

TEMPLATE = "color_3600"
STEPS = ("probe_1a", "probe_1b", "gain_line", "probe_2a", "probe_2b", "dark", "white", "main", "park")
STATUS = 0x41
#: written by every AFE access (0x51 address, 0x3A/0x3B data): the snapshot keeps the last one
AFE_PORT = {0x51, 0x3A, 0x3B}
#: 16/24-bit fields as (first register, bytes)
FIELDS = {"LINCNT": (0x25, 3), "DPISET": (0x2C, 2), "STRPIXEL": (0x30, 2), "ENDPIXEL": (0x32, 2),
          "MAXWD": (0x35, 3), "LPERIOD": (0x38, 2), "FEEDL": (0x3D, 3)}
FIELD_OF = {a + i: name for name, (a, n) in FIELDS.items() for i in range(n)}


def field(regs: dict[int, int], name: str) -> int:
    a, n = FIELDS[name]
    return int.from_bytes(bytes(regs.get(a + i, 0) for i in range(n)), "big")


def table_words(data: str) -> tuple[int, ...]:
    b = base64.b64decode(data)
    return tuple(b[i] | b[i + 1] << 8 for i in range(0, len(b) - 1, 2))


# --- segmentation -------------------------------------------------------------------------------

def segment(profile: dict) -> tuple[list[list[tuple]], list[dict[int, int]], dict]:
    """Per-step operation items, the register state at each frame's last read, and checks."""
    steps: list[list[tuple]] = [[]]
    states: list[dict[int, int]] = []
    regs: dict[int, int] = {}
    got: Counter = Counter()
    checks = Counter()
    unacked = False
    ops = profile["ops"]
    for i, o in enumerate(ops):
        cur = steps[-1]
        if o["kind"] == "control" and o["rt"] == 0xC0:
            if o["value"] == 0x8E and o["index"] == 0x20:
                unacked = False
                checks["acks"] += 1
            elif o["value"] == 0x8E and o["index"] == 0x18:
                cur.append(("bulk_done",))
            elif o["value"] == 0x84 and o.get("register") == STATUS:
                e = o["expected"][0]
                if not (cur and cur[-1][0] == "wait" and cur[-1][1] == e):
                    cur.append(("wait", e))
            else:
                cur.append(("read_reg", o["value"], o["index"], o.get("register")))
            continue
        if o["kind"] == "control":
            if unacked:
                checks["writes_without_ack"] += 1
            unacked = True
            d = o["data"]
            if o["value"] == 0x83 and d == [STATUS]:
                continue  # status address before a poll
            if o["value"] == 0x83 and len(d) > 1 and d[0] == 0x51:
                cur.append(("afe", d[1], d[3] << 8 | d[5]))
                regs.update(zip(d[::2], d[1::2], strict=True))
            elif o["value"] == 0x83 and len(d) > 1:
                pairs = tuple(zip(d[::2], d[1::2], strict=True))
                regs.update(pairs)
                cur.append(("regs", pairs))
            elif o["value"] == 0x83:
                cur.append(("addr", d[0]))
            elif o["value"] == 0x82:
                cur.append(("setup", int.from_bytes(bytes(d[4:8]), "little")))
            elif o["value"] == 0x8C:
                cur.append(("8c", o["index"], d[0]))
            else:
                cur.append(("control", o["value"], tuple(d)))
        elif o["kind"] == "write":
            n = o.get("length") or len(base64.b64decode(o["data"]))
            hi, lo = regs.get(0x5B, 0), regs.get(0x5C, 0)
            if hi & 0x40:
                cur.append(("motor_table", (hi >> 3) & 7, table_words(o["data"])))
            elif (hi, lo) == (0x10, 0x00):
                cur.append(("shading", n))
            else:
                cur.append(("upload", hi, lo, n))
        elif o["kind"] == "delay":
            cur.append(("delay", o["ms"], bool(o.get("timedStop"))))
        elif o["kind"] == "read":
            f = o["frame"]
            got[f] += o["length"]
            cur.append(("read", f, o["length"]))
            if got[f] >= profile["frames"][f]["bytes"]:
                states.append(dict(regs))
                steps.append([])
    return steps, states, dict(checks)


def shape(item: tuple) -> tuple:
    """An item without its values: what a driver must do, not with which numbers."""
    k = item[0]
    if k == "regs":
        return ("regs", tuple(r for r, _ in item[1]))
    if k == "afe":
        return ("afe", item[1])
    if k == "motor_table":
        return ("motor_table", item[1])
    if k in ("setup", "shading", "read", "wait", "delay"):
        return (k,) + ((item[2],) if k == "delay" else ())
    return item


# --- level A: register state --------------------------------------------------------------------

def decode_state(jobs: dict[str, dict], states: dict[str, list[dict[int, int]]]) -> dict:
    keys = list(jobs)
    roles = STEPS[:-1]
    regs = sorted({r for s in states.values() for st in s for r in st} - AFE_PORT)
    snap = {(k, i): states[k][i] for k in keys for i in range(len(roles))}
    base = {r: snap[(keys[0], 0)].get(r) for r in regs
            if len({v.get(r) for v in snap.values()}) == 1}
    step, by_dpi, by_mode, by_job = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for i, role in enumerate(roles):
        for r in regs:
            if r in base:
                continue
            vals = {k: snap[(k, i)].get(r) for k in keys}
            if len(set(vals.values())) == 1:
                step[role][r] = vals[keys[0]]
                continue
            mode = {k: k.split("_")[0] for k in keys}
            dpi = {k: jobs[k]["dpi"] for k in keys}
            # explained by resolution if jobs at the same dpi agree; by mode if same-mode jobs agree
            if all(vals[a] == vals[b] for a in keys for b in keys if dpi[a] == dpi[b]):
                for k in keys:
                    by_dpi[role].setdefault(r, {})[dpi[k]] = vals[k]
            elif all(vals[a] == vals[b] for a in keys for b in keys if mode[a] == mode[b]):
                for k in keys:
                    by_mode[role].setdefault(r, {})[mode[k]] = vals[k]
            else:
                by_job[role][r] = vals
    # rebuild every snapshot from the tables
    mismatches = []
    for (k, i), st in snap.items():
        role = roles[i]
        mode = k.split("_")[0]
        rebuilt = dict(base)
        rebuilt.update(step[role])
        rebuilt.update({r: v[jobs[k]["dpi"]] for r, v in by_dpi[role].items()})
        rebuilt.update({r: v[mode] for r, v in by_mode[role].items()})
        rebuilt.update({r: v[k] for r, v in by_job[role].items()})
        bad = {r: (rebuilt.get(r), st.get(r)) for r in regs if rebuilt.get(r) != st.get(r)}
        if bad:
            mismatches.append((k, role, bad))
    return {"base": base, "step": dict(step), "by_dpi": dict(by_dpi), "by_mode": dict(by_mode),
            "by_job": dict(by_job), "mismatches": mismatches, "regs": regs}


def formulas(jobs: dict[str, dict], states: dict[str, list[dict[int, int]]]) -> list[tuple[str, int, list]]:
    """Candidate formulas for the per-resolution registers: (name, frames that hold, failures)."""
    probes = {"probe_1a", "probe_1b", "gain_line", "probe_2a", "probe_2b"}
    tests = {
        "DPISET = 1200 (probes), dpi / 6 (dark, white, main)":
            lambda p, f, role, s: field(s, "DPISET") == (1200 if role in probes else p["dpi"] // 6),
        "pixels = (ENDPIXEL - STRPIXEL) * DPISET / 1200":
            lambda p, f, role, s: f["pixels"] == (field(s, "ENDPIXEL") - field(s, "STRPIXEL")) * field(s, "DPISET") // 1200,
        "STRPIXEL = 20 (probes)":
            lambda p, f, role, s: role not in probes or field(s, "STRPIXEL") == 20,
        "LINCNT = lines (calibration), lines * yres / dpi (main)":
            lambda p, f, role, s: field(s, "LINCNT") == f["lines"] * (p["yres"] // p["dpi"] if role == "main" else 1),
        "bytes = pixels * lines * 6 (RGB16)":
            lambda p, f, role, s: f["bytes"] == f["pixels"] * f["lines"] * 6,
        "LPERIOD = 14000":
            lambda p, f, role, s: field(s, "LPERIOD") == 14000,
        "FEEDL = 1 at every frame read":
            lambda p, f, role, s: field(s, "FEEDL") == 1,
        # the second probe pair does not rewrite MAXWD: it keeps the gain line's
        "MAXWD = pixels + 5450 (hypothesis; probe_2a/2b keep the gain line's)":
            lambda p, f, role, s: field(s, "MAXWD") == p["frames"][2 if role in ("probe_2a", "probe_2b") else
                                                                    STEPS.index(role)]["pixels"] + 5450,
        "0x1E low nibble = SilverFast lineSel (main)":
            lambda p, f, role, s: role != "main" or s[0x1E] & 0x0F == p["scan"]["lineSel"],
    }
    out = []
    for name, test in tests.items():
        ok, fail = 0, []
        for k, p in jobs.items():
            for i, role in enumerate(STEPS[:-1]):
                f, s = p["frames"][i], states[k][i]
                if test(p, f, role, s):
                    ok += 1
                else:
                    fail.append(f"{k}/{role}")
        out.append((name, ok, fail))
    return out


# --- level B: operation sequence ----------------------------------------------------------------

def sequence_variants(steps: dict[str, list[list[tuple]]]) -> dict[str, list[str]]:
    ref = steps[TEMPLATE]
    out = {}
    for k, job in steps.items():
        notes = []
        for i, name in enumerate(STEPS):
            a, b = [shape(x) for x in ref[i]], [shape(x) for x in job[i]]
            for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                if tag != "equal":
                    notes.append(f"{name}: {tag} {_brief(a[a1:a2])} -> {_brief(b[b1:b2])}")
        out[k] = notes
    return out


def _brief(items: list[tuple]) -> str:
    def one(x: tuple) -> str:
        if x[0] == "regs":
            return "regs[" + ",".join(f"{r:02x}" for r in x[1]) + "]"
        if x[0] == "wait":
            return "wait"
        return x[0] + ("" if len(x) == 1 else "(" + ",".join(str(v) for v in x[1:]) + ")")
    return "[" + " ".join(one(x) for x in items) + "]" if items else "[]"


# --- level C: motor tables ----------------------------------------------------------------------

def motor_tables(jobs: dict[str, dict], steps: dict[str, list[list[tuple]]], states) -> dict:
    names: dict[tuple[int, ...], str] = {}
    uses = defaultdict(set)
    cruise = []
    for k, job in steps.items():
        p = jobs[k]
        for i, name in enumerate(STEPS):
            for x in job[i]:
                if x[0] != "motor_table":
                    continue
                t = x[2]
                n = names.setdefault(t, f"T{len(names)}")
                uses[n].add(f"{name}/slot{x[1]}")
                if name in ("white", "main") and x[1] <= 2:
                    s = states[k][i]
                    # motor steps per line: 14400 / yres in the main scan, always 2 for the white frame
                    per_line = 14400 // p["yres"] if name == "main" else 2
                    expect = ((s[0x1E] & 0x0F) + 1) * field(s, "LPERIOD") // per_line
                    cruise.append((k, name, t[-1], expect))
    desc = {n: {"first": t[0], "last": t[-1], "entries": len(t),
                "ramp": next(i for i, v in enumerate(t) if v == t[-1]), "used": sorted(uses[n])}
            for t, n in names.items()}
    return {"tables": desc, "cruise": cruise}


# --- report -------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--json", type=Path, help="write the decoded tables here")
    args = parser.parse_args(argv)

    jobs = json.loads(gzip.decompress(args.data.read_bytes()))["profiles"]
    steps, states, checks = {}, {}, {}
    for k, p in jobs.items():
        steps[k], states[k], checks[k] = segment(p)
        if len(steps[k]) != len(STEPS) or len(p["frames"]) != len(STEPS) - 1:
            print(f"{k}: {len(steps[k])} steps, expected {len(STEPS)}")
            return 1
        snap_bad = [i for i, f in enumerate(p["frames"])
                    if any(states[k][i].get(int(r)) != v for r, v in f["regs"].items())]
        print(f"{k}: {len(p['ops'])} ops -> {sum(map(len, steps[k]))} items; "
              f"write acks {checks[k].get('acks', 0)}, writes without ack {checks[k].get('writes_without_ack', 0)}; "
              f"rebuilt state differs from the recorded snapshot at frames {snap_bad or 'none'}")

    st = decode_state(jobs, states)
    n = len(st["regs"])
    print(f"\nA. register state ({n} registers per frame, AFE port 0x51/0x3A/0x3B left out)")
    print(f"   base (same in all 32 frames): {len(st['base'])}")
    for role in STEPS[:-1]:
        print(f"   {role:9s} step-constant {len(st['step'].get(role, {})):3d}  by dpi "
              f"{_regs(st['by_dpi'].get(role, {}))}"
              + (f"  colour/infrared {_modes(st['by_mode'][role])}" if st["by_mode"].get(role) else "")
              + (f"  UNEXPLAINED {_regs(st['by_job'][role])}" if st["by_job"].get(role) else ""))
    print(f"   rebuilt from tables: {32 - len(st['mismatches'])}/32 frames exact")
    for k, role, bad in st["mismatches"]:
        print(f"     {k}/{role}: {bad}")

    print("\n   formulas:")
    for name, ok, fail in formulas(jobs, states):
        print(f"   {'ok  ' if not fail else 'FAIL'} {ok:2d}/32  {name}" + (f"   fails: {', '.join(fail)}" if fail else ""))

    print(f"\nB. operation sequence vs {TEMPLATE}")
    for k, notes in sequence_variants(steps).items():
        print(f"   {k}: {len(notes)} deviation(s)")
        for line in notes:
            print(f"     {line}")

    mt = motor_tables(jobs, steps, states)
    print("\nC. motor tables")
    for n, t in mt["tables"].items():
        print(f"   {n}: {t['first']} -> {t['last']} (ramp {t['ramp']} of {t['entries']}), used in {', '.join(t['used'])}")
    bad = [c for c in mt["cruise"] if c[2] != c[3]]
    print(f"   cruise = (LINESEL + 1) * LPERIOD / steps per line (main 14400 / yres, white 2): {len(mt['cruise']) - len(bad)}/{len(mt['cruise'])} tables"
          + (f"; fails {bad}" if bad else ""))

    if args.json:
        out = {
            "base": _hex(st["base"]),
            "step": {r: _hex(v) for r, v in st["step"].items()},
            "by_dpi": {r: {f"0x{a:02x}": v for a, v in t.items()} for r, t in st["by_dpi"].items()},
            "by_mode": {r: {f"0x{a:02x}": v for a, v in t.items()} for r, t in st["by_mode"].items()},
            "unexplained": {r: {f"0x{a:02x}": v for a, v in t.items()} for r, t in st["by_job"].items()},
            "template": {name: [list(map(_jsonable, shape(x))) for x in steps[TEMPLATE][i]]
                         for i, name in enumerate(STEPS)},
            "motor_tables": mt["tables"],
        }
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}")
    return 0 if not st["mismatches"] else 1


def _regs(t: dict) -> str:
    return "[" + " ".join(f"{r:02x}" for r in sorted(t)) + "]"


def _modes(t: dict) -> str:
    return "[" + " ".join(f"{r:02x}={v['color']:02x}/{v['infrared']:02x}" for r, v in sorted(t.items())) + "]"


def _hex(t: dict) -> dict:
    return {f"0x{r:02x}": v for r, v in sorted(t.items())}


def _jsonable(v):
    return list(v) if isinstance(v, tuple) else v


if __name__ == "__main__":
    sys.exit(main())
