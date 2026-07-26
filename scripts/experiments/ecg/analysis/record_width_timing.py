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
            if part.startswith("31_gem5_record_width_timing"):
                stage = part.replace("31_gem5_record_width_timing_", "")
        for row in csv.DictReader(csv_path.open()):
            if row.get("status") != "ok":
                continue
            row["_stage"] = stage
            row["_graph"] = csv_path.parent.parent.name
            row["_kernel"] = csv_path.parent.name
            rows.append(row)
    return rows


def geomean(values):
    vals = [v for v in values if v and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


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
        ratios = defaultdict(lambda: {"time": [], "traffic": []})
        for _, per_policy in cells.items():
            base = per_policy.get("LRU")
            if not base:
                continue
            bt, bb = num(base.get("sim_ticks")), num(base.get("dram_offchip_bytes"))
            for label, r in per_policy.items():
                t, b = num(r.get("sim_ticks")), num(r.get("dram_offchip_bytes"))
                if bt and t:
                    ratios[label]["time"].append(t / bt)
                if bb and b:
                    ratios[label]["traffic"].append(b / bb)
        print(f"\n  stage: {stage}   ({len(cells)} cell(s))")
        print(f"    {'policy':<24}{'time':>9}{'traffic':>10}{'cells':>7}")
        for label in sorted(ratios, key=lambda k: geomean(ratios[k]["time"]) or 9):
            gt = geomean(ratios[label]["time"])
            gb = geomean(ratios[label]["traffic"])
            print(f"    {label:<24}{gt if gt else float('nan'):>9.3f}"
                  f"{gb if gb else float('nan'):>10.3f}"
                  f"{len(ratios[label]['time']):>7}")

    # ---- 3. The width contrast, matched --------------------------------
    print()
    print("=" * 72)
    print("3. THE WIDTH CONTRAST  (4b versus 8b, identical in all else)")
    print("=" * 72)
    paired = defaultdict(dict)
    for r in rows:
        key = (r["_graph"], r["_kernel"], r.get("policy_label", "?"))
        paired[key][r["_stage"]] = r
    deltas_t, deltas_b = [], []
    for (graph, kernel, policy), stages in sorted(paired.items()):
        if "4b" in stages and "8b" in stages:
            t4, t8 = num(stages["4b"].get("sim_ticks")), num(stages["8b"].get("sim_ticks"))
            b4, b8 = (num(stages["4b"].get("dram_offchip_bytes")),
                      num(stages["8b"].get("dram_offchip_bytes")))
            if t4 and t8 and b4 and b8:
                deltas_t.append(t8 / t4)
                deltas_b.append(b8 / b4)
                print(f"  {graph}/{kernel}/{policy:<22} "
                      f"time x{t8/t4:.3f}  traffic x{b8/b4:.3f}")
    if deltas_t:
        gt, gb = geomean(deltas_t), geomean(deltas_b)
        print(f"\n  geomean: widening the record costs x{gt:.3f} time "
              f"for x{gb:.3f} traffic")
        if gb > 1.0:
            print(f"  -> time cost is {(gt-1)/(gb-1)*100:.0f}% of the traffic "
                  "cost, so the extra bytes are "
                  f"{'largely absorbed' if (gt-1) < (gb-1)*0.5 else 'not absorbed'}")
    else:
        print("  (no matched 4b/8b pairs completed yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
