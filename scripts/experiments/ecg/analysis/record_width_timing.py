#!/usr/bin/env python3
"""Read the record-width timing matrix and answer the pre-registered question.

The matrix reports execution time, off-chip bytes and DRAM bus utilisation
together, because K2's trade only resolves when all three are read at once: it
can spend more bandwidth while exposing far fewer demand misses to full DRAM
latency, and which side binds depends on saturation.

Pre-registered before the run (see research/ecg-hpca/RESULTS.md):
  low utilisation  -> the measured reductions in exposed demand misses should
                      appear as speedup at BOTH record widths
  high utilisation -> the 8-byte width should lose in proportion to its traffic

This deliberately reports the utilisation FIRST, then time against traffic, so
the timing numbers are interpreted rather than merely ranked.

Usage:
  python3 scripts/experiments/ecg/analysis/record_width_timing.py [RUN_DIR]
"""
from __future__ import annotations

import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "results/ecg_experiments/final_paper_runs"

# Utilisation below this is treated as "the memory system has headroom".
SATURATION_LOW = 20.0
SATURATION_HIGH = 70.0


def newest_run() -> Path | None:
    candidates = sorted(RUNS.glob("ecg_record_width_timing_*"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def num(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load(run_dir: Path):
    rows = []
    for csv_path in run_dir.rglob("roi_matrix.csv"):
        stage = "?"
        for part in csv_path.parts:
            if part.startswith("31_gem5_record_width"):
                stage = part.rsplit("_", 1)[-1]
        for row in csv.DictReader(csv_path.open()):
            if row.get("status") != "ok":
                continue
            row["_stage"] = stage
            row["_graph"] = csv_path.parent.parent.name
            row["_kernel"] = csv_path.parent.name
            label = row.get("policy_label", "")
            matches = sorted((csv_path.parent / "gem5").glob(
                f"gem5_{row['_kernel']}_{label}_L3*/stats.txt"))
            row["_stats"] = matches[0] if matches else None
            rows.append(row)
    return rows


def geomean(values):
    vals = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def roi_insts(row):
    """ROI-scoped committed instructions for a cell.

    Prefer the captured column, but fall back to the archived gem5 stats so
    cells produced before the column existed can still be attributed. Note this
    is deliberately NOT simInsts: that counter is not cleared by
    m5_reset_stats, so it includes graph loading and metadata construction and
    reports an impossible IPC above 1 on an in-order core.
    """
    direct = num(row.get("roi_insts"))
    if direct:
        return direct
    stats = row.get("_stats")
    if not stats or not stats.exists():
        return None
    first_dump = stats.read_text().split(
        "---------- Begin Simulation Statistics ----------")[1:2]
    if not first_dump:
        return None
    m = re.search(r"^system\.cpu\.commitStats0\.numInsts\s+(\d+)",
                  first_dump[0], re.M)
    return float(m.group(1)) if m else None


def main(argv):
    run_dir = Path(argv[0]) if argv else newest_run()
    if not run_dir or not run_dir.exists():
        raise SystemExit("no record-width timing run found")
    rows = load(run_dir)
    if not rows:
        raise SystemExit(f"no completed cells yet in {run_dir}")

    print(f"run: {run_dir.name}")
    print(f"completed cells: {len(rows)}")

    # ---- 1. Saturation, read first -------------------------------------
    utils = [num(r.get("dram_bus_util_pct")) for r in rows]
    utils = [u for u in utils if u is not None]
    print()
    print("=" * 72)
    print("1. DRAM BUS UTILISATION  (read first: it decides how to read the rest)")
    print("=" * 72)
    if utils:
        print(f"  min {min(utils):.2f}%   median {statistics.median(utils):.2f}%"
              f"   max {max(utils):.2f}%")
        peak = num(rows[0].get("dram_peak_bw_mibs"))
        if peak:
            print(f"  peak bandwidth modelled: {peak:.0f} MiB/s")
        if max(utils) < SATURATION_LOW:
            verdict = ("LOW -- the memory system has headroom. Extra metadata "
                       "traffic is close to free; exposed latency is what "
                       "costs. Pre-registered prediction: the exposed-miss "
                       "reduction should appear as speedup at BOTH widths.")
        elif min(utils) > SATURATION_HIGH:
            verdict = ("HIGH -- bandwidth binds. The wider record should lose "
                       "in proportion to its traffic increase.")
        else:
            verdict = ("MIXED -- neither regime dominates; report per-cell and "
                       "do not generalise.")
        print(f"  verdict: {verdict}")

    # ---- 2. Time against traffic, per stage ----------------------------
    print()
    print("=" * 72)
    print("2. TIME AND TRAFFIC versus LRU, by record width")
    print("=" * 72)
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["_stage"]].append(r)

    for stage in sorted(by_stage):
        cells = defaultdict(dict)
        for r in by_stage[stage]:
            cells[(r["_graph"], r["_kernel"])][r.get("policy_label", "?")] = r
        ratios = defaultdict(lambda: {"time": [], "traffic": [], "insts": []})
        for _, per_policy in cells.items():
            base = per_policy.get("LRU")
            if not base:
                continue
            bt, bb = num(base.get("sim_ticks")), num(base.get("dram_offchip_bytes"))
            bi = roi_insts(base)
            for label, r in per_policy.items():
                t, b = num(r.get("sim_ticks")), num(r.get("dram_offchip_bytes"))
                i = roi_insts(r)
                if bt and t:
                    ratios[label]["time"].append(t / bt)
                if bb and b:
                    ratios[label]["traffic"].append(b / bb)
                if bi and i:
                    ratios[label]["insts"].append(i / bi)
        print(f"\n  stage: {stage}   ({len(cells)} cell(s))")
        print(f"    {'policy':<24}{'time':>9}{'traffic':>10}{'ROI insts':>11}"
              f"{'cells':>7}")
        for label in sorted(ratios, key=lambda k: geomean(ratios[k]["time"]) or 9):
            gt = geomean(ratios[label]["time"])
            gb = geomean(ratios[label]["traffic"])
            gi = geomean(ratios[label]["insts"])
            print(f"    {label:<24}{gt if gt else float('nan'):>9.3f}"
                  f"{gb if gb else float('nan'):>10.3f}"
                  f"{gi if gi else float('nan'):>11.3f}"
                  f"{len(ratios[label]['time']):>7}")
        print("    (ROI insts versus LRU: a policy that executes more "
              "instructions per edge\n     can lose time while winning "
              "traffic, and at low utilisation it usually does.)")

    # ---- 3. The width contrast, matched --------------------------------
    print()
    print("=" * 72)
    print("3. THE WIDTH CONTRAST  (4b versus 8b, identical in all else)")
    print("=" * 72)
    paired = defaultdict(dict)
    for r in rows:
        key = (r["_graph"], r["_kernel"], r.get("policy_label", "?"))
        paired[key][r["_stage"]] = r
    deltas_t, deltas_b, deltas_i = [], [], []
    for (graph, kernel, policy), stages in sorted(paired.items()):
        if "4b" in stages and "8b" in stages:
            t4, t8 = num(stages["4b"].get("sim_ticks")), num(stages["8b"].get("sim_ticks"))
            b4, b8 = (num(stages["4b"].get("dram_offchip_bytes")),
                      num(stages["8b"].get("dram_offchip_bytes")))
            i4, i8 = roi_insts(stages["4b"]), roi_insts(stages["8b"])
            if t4 and t8 and b4 and b8:
                deltas_t.append(t8 / t4)
                deltas_b.append(b8 / b4)
                extra = ""
                if i4 and i8:
                    deltas_i.append(i8 / i4)
                    extra = f"  ROI insts x{i8/i4:.3f}"
                print(f"  {graph}/{kernel}/{policy:<22} "
                      f"time x{t8/t4:.3f}  traffic x{b8/b4:.3f}{extra}")
    if deltas_t:
        gt, gb = geomean(deltas_t), geomean(deltas_b)
        gi = geomean(deltas_i)
        print(f"\n  geomean: widening the record costs x{gt:.3f} time "
              f"for x{gb:.3f} traffic")
        if gi:
            print(f"  the 8-byte arm executes x{gi:.3f} the ROI instructions "
                  "of the compact arm")
            if gi < 0.995:
                print("  -> the arms are NOT matched on work: the compact "
                      "record is decoded in\n     software, so this contrast "
                      "is width PLUS decode, not width alone. Report the\n"
                      "     decode cost explicitly or move it into the ISA.")
        if gb > 1.0:
            print(f"  -> time cost is {(gt-1)/(gb-1)*100:.0f}% of the traffic "
                  "cost, so the extra bytes are "
                  f"{'largely absorbed' if (gt-1) < (gb-1)*0.5 else 'not absorbed'}")
    else:
        print("  (no matched 4b/8b pairs completed yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
