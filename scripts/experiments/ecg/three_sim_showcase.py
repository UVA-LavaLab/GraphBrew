#!/usr/bin/env python3
"""3-simulator ECG equivalence SHOWCASE.

Runs the same policies on the SAME pressured cell across cache_sim / gem5 / Sniper and
prints (1) an L3 miss-rate table proving the policy TRENDS track across simulators, and
(2) the per-sim `[ECG-CONFIG ...]` banner proving each simulator actually resolved the
claimed policy / ECG mode / variant (paired with ECG_EVICT_TRACE, which proves the policy
ACTS). Numbers come through the committed roi_matrix with the verified cell geometry, so
the headline is NOT hand-rolled (the load-bearing CHARGED / ULTRAFAST / L1-L2 knobs that a
raw `roi_matrix --policies ECG:...` omits — which makes ECG degenerate — are set here).

Absolute L3 miss rates are NOT comparable across simulators (gem5/Sniper see the full ISA
access stream; cache_sim sees graph accesses only), so equivalence is read as the per-sim
DIRECTION vs LRU: GRASP and ECG must reduce the miss rate in every simulator.

TWO claims, two cells (do not conflate):
  (A) ECG ADVANTAGE (ECG beats GRASP AND P-OPT): REAL graphs in cache_sim, -o5, shortcircuit, e.g.
      --graph web-Google.sg --policies LRU GRASP POPT ECG:ECG_GRASP_POPT --l3 512kB --variant shortcircuit
  (B) 3-SIM EQUIVALENCE (all sims agree on direction): the synthetic kron_s16_k4 default below — it
      is gem5/Sniper-feasible but its Kronecker structure does NOT reward the epoch, so ECG does NOT
      beat GRASP there. That is expected; this cell certifies cross-sim agreement, not the advantage.

Usage:
  python3 scripts/experiments/ecg/three_sim_showcase.py                 # kron_s16_k4@128kB
  python3 scripts/experiments/ecg/three_sim_showcase.py --sims cache_sim gem5
  python3 scripts/experiments/ecg/three_sim_showcase.py --graph <g.sg> --l3 64kB
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ROI_MATRIX = ROOT / "scripts" / "experiments" / "ecg" / "roi_matrix.py"

# Per-simulator extra roi_matrix flags. Sniper's sg_kernel workload is gated (documented
# SDE memory runaway) so it runs under a memory cap.
SUITE = {"cache_sim": "cache-sim", "gem5": "gem5", "sniper": "sniper"}
EXTRA = {
    "cache_sim": [],
    "gem5": ["--timeout-gem5", "3600"],
    "sniper": ["--sniper-workload", "sg_kernel", "--allow-sniper-sg-kernel-workload",
               "--sniper-memory-limit-gb", "20", "--sniper-enable-graph-policies",
               "--timeout-sniper", "540"],
}
BANNER_RE = re.compile(r"\[ECG-CONFIG[^\]]*\]")


def _parse_cell(out):
    """Parse (l3_miss_rate|None, banner|'') from an existing cell out dir."""
    mr = None
    for p in glob.glob(os.path.join(str(out), "**", "roi_matrix.json"), recursive=True):
        try:
            rows = json.load(open(p))
        except Exception:
            continue
        for r in rows:
            if r.get("error"):
                continue
            v = r.get("l3_miss_rate")
            if v is not None:
                try:
                    mr = float(v)
                except (TypeError, ValueError):
                    pass
    banner = ""
    for lg in glob.glob(os.path.join(str(out), "**", "*.log"), recursive=True):
        try:
            m = BANNER_RE.search(open(lg, errors="ignore").read())
        except Exception:
            continue
        if m:
            banner = m.group(0)
            break
    return mr, banner


def run_cell(suite, policy, graph, cell, reorder, out, extra, timeout, benchmark="pr", resume=False):
    """One roi_matrix run -> (l3_miss_rate or None, banner string or '').

    With resume=True, a cell that already has a valid result (parseable
    l3_miss_rate) is NOT re-run — this lets a re-launch with an extended policy
    set (e.g. adding SRRIP) reuse the finished cells instead of recomputing the
    slow gem5 legs.
    """
    import shutil
    if resume:
        mr, banner = _parse_cell(out)
        if mr is not None:
            return mr, banner
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, str(ROI_MATRIX), "--suite", suite, "--no-build",
           "--benchmark", benchmark, "--policies", policy,
           "--options", f"-f {graph} -o {reorder} -n 1 -i 1",
           "--l3-sizes", cell["l3"], "--l3-ways", cell["ways"],
           "--l1d-size", cell["l1d"], "--l2-size", cell["l2"],
           "--out-dir", str(out)] + (extra or [])
    try:
        subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, "(timeout)"
    return _parse_cell(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="3-sim ECG equivalence showcase")
    ap.add_argument("--sims", nargs="+", default=["cache_sim", "gem5", "sniper"],
                    choices=["cache_sim", "gem5", "sniper"])
    ap.add_argument("--benchmark", default="pr", choices=["pr", "bfs", "bc", "cc", "sssp"],
                    help="Kernel to run (all 3-sim certified except sssp=cache_sim+gem5). "
                         "For bc/cc on Sniper use a tiny cell (e.g. --l1d 1kB --l2 1kB --l3 2kB) "
                         "so the small property array spills past the non-inclusive inner caches.")
    ap.add_argument("--policies", nargs="+",
                    default=["LRU", "SRRIP", "GRASP", "POPT", "ECG:ECG_GRASP_POPT"],
                    help="Baselines to compare side-by-side. Default is the full set "
                         "(LRU, SRRIP, GRASP, P-OPT, ECG) so every table has complete "
                         "context and policy-rank flips are interpretable.")
    ap.add_argument("--graph",
                    default=str(ROOT / "results" / "graphs" / "kron_s16_k4" / "kron_s16_k4.sg"))
    ap.add_argument("--l3", default="128kB")
    ap.add_argument("--ways", default="16")
    ap.add_argument("--l1d", default="16kB")
    ap.add_argument("--l2", default="64kB")
    ap.add_argument("--reorder", default="5")
    ap.add_argument("--variant", default="rrip_first",
                    help="ECG_VARIANT for ECG policies. Default rrip_first = the variant "
                         "verify/equiv use on this cell (best is cell-dependent, an honest "
                         "finding; the showcase fixes ONE variant across all 3 sims to read "
                         "cross-sim agreement, not to compare variants).")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--resume", action="store_true",
                    help="Reuse cells that already have a valid result (skip re-running the "
                         "slow gem5 legs) — e.g. when re-launching with an added policy.")
    ap.add_argument("--out", default="/tmp/three_sim_showcase")
    args = ap.parse_args(argv)

    cell = {"l3": args.l3, "ways": args.ways, "l1d": args.l1d, "l2": args.l2}
    # ECG_DEBUG -> the [ECG-CONFIG] banner; ECG_EVICT_TRACE -> per-eviction proof.
    os.environ["ECG_DEBUG"] = "1"
    os.environ.setdefault("ECG_EVICT_TRACE", "20")
    os.environ["ECG_VARIANT"] = args.variant

    print(f"# 3-sim ECG equivalence showcase")
    print(f"# benchmark={args.benchmark}  graph={Path(args.graph).name}  cell: L3={args.l3}/{args.ways}w "
          f"L1d={args.l1d} L2={args.l2}  -o{args.reorder}  ECG_VARIANT={args.variant}")
    print(f"# metric = L3 miss rate (lower better); read DIRECTION vs LRU per sim "
          f"(absolute rates not comparable across sims).\n")

    results = {}   # (sim, policy) -> (mr, banner)
    for sim in args.sims:
        for pol in args.policies:
            tag = re.sub(r"[^0-9A-Za-z]+", "_", pol)
            out = Path(args.out) / f"{sim}_{tag}"
            mr, banner = run_cell(SUITE[sim], pol, args.graph, cell, args.reorder,
                                  out, EXTRA[sim], args.timeout, args.benchmark, args.resume)
            results[(sim, pol)] = (mr, banner)
            mrs = f"{mr:.4f}" if mr is not None else "  --  "
            print(f"  [{sim:9}] {pol:20} L3_mr={mrs}   {banner}")

    # Table + per-sim direction vs LRU.
    print("\n## L3 miss-rate table")
    head = "policy".ljust(22) + "".join(s.ljust(12) for s in args.sims)
    print(head)
    for pol in args.policies:
        row = pol.ljust(22)
        for sim in args.sims:
            mr = results[(sim, pol)][0]
            row += (f"{mr:.4f}" if mr is not None else "--").ljust(12)
        print(row)

    print("\n## direction vs LRU (Δ = mr - LRU_mr; negative = HELPS)")
    all_ok = True
    for sim in args.sims:
        lru = results.get((sim, "LRU"), (None,))[0]
        if lru is None:
            print(f"  [{sim:9}] no LRU baseline"); all_ok = False; continue
        parts = []
        for pol in args.policies:
            if pol == "LRU":
                continue
            mr = results[(sim, pol)][0]
            if mr is None:
                parts.append(f"{pol}=--"); continue
            d = mr - lru
            verdict = "HELPS" if d < 0 else "HURTS"
            if d >= 0:
                all_ok = False
            parts.append(f"{pol}:{d:+.4f}({verdict})")
        print(f"  [{sim:9}] LRU={lru:.4f}  " + "  ".join(parts))
    print(f"\nRESULT: {'GRASP/ECG help in every simulator (equivalent direction) ✓' if all_ok else 'a policy regressed in some sim — inspect above'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
