#!/usr/bin/env python3
"""ECG prefetch-comparison matrix: rows=graph x prefetcher, one table per simulator.

Fixed eviction (LRU by default), vary ONLY the prefetcher so the comparison
isolates the prefetch lever (mirrors ecg_variant_matrix.py for eviction):

  none    - no prefetch (baseline)
  DROPLET - sequential next-K prefetch (Basak HPCA'19 baseline; cache_sim mode 3)
  ECG_PFX - lookahead POPT-best target (the shared ecg_mode6::selectPrefetchTarget;
            cache_sim mode 6) -- the SAME target function all 3 simulators call.

Reported metrics (the two axes that matter for a prefetcher + its efficiency):
  l3_mr       L3 miss rate (NB: misleading under prefetch -- pulls hits up).
  demand2mem  demand misses reaching memory = LATENCY proxy (lower=better).
  bandwidth   total DRAM traffic = demand + prefetch fills (lower=better).
  fills       prefetches issued.
  useful%     fraction of fills later demand-hit (prefetch ACCURACY).

The honest story this surfaces: a prefetcher CONSERVES total bandwidth (it only
relocates demand->prefetch); DROPLET cuts demand2mem aggressively but leaves
bandwidth ~flat; ECG_PFX is selective (far fewer fills at the same useful-rate).
Only the EVICTION policy cuts total bandwidth -- see ecg_variant_matrix.py.

Usage:
  python3 scripts/experiments/ecg/ecg_prefetch_matrix.py --suite cache-sim \
      --cells "web-Google:512kB cit-Patents:1MB soc-pokec:1MB"
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRAPHS = ROOT / "results" / "graphs"

PREFETCHERS = ["none", "DROPLET", "ECG_PFX"]


def run_cell(suite, graph, l3, order, eviction, prefetcher, lookahead, opts):
    gpath = GRAPHS / graph / f"{graph}.sg"
    outdir = Path("/tmp") / f"pfm_{suite}_{graph}_{l3}_o{order}_{eviction}_{prefetcher}"
    cmd = [sys.executable, str(ROOT / "scripts/experiments/ecg/roi_matrix.py"),
           "--suite", suite, "--no-build", "--benchmark", "pr",
           "--policies", eviction,
           "--options", f"-f {gpath} -o {order} {opts}",
           "--l3-sizes", l3, "--l3-ways", "16",
           "--l1d-size", "32kB", "--l2-size", "256kB",
           "--prefetcher", prefetcher, "--ecg-pfx-lookahead", str(lookahead),
           "--out-dir", str(outdir)]
    try:
        subprocess.run(cmd, env=dict(os.environ), cwd=str(ROOT),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=3600, check=False)
        rows = json.load(open(outdir / "roi_matrix.json"))
        rows = rows if isinstance(rows, list) else [rows]
        r = rows[0]
        fills = r.get("prefetch_fills") or 0
        return {
            "l3_mr": r.get("l3_miss_rate"),
            "demand2mem": r.get("memory_accesses"),
            "bandwidth": r.get("total_memory_traffic"),
            "fills": fills,
            "useful": (r.get("prefetch_fill_useful_rate") or 0.0) if fills else None,
        }
    except Exception:
        return None


def fmt_m(v):
    if v is None:
        return "  --  "
    return f"{v/1e6:8.2f}M" if v >= 1e6 else f"{v:9d}"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="cache-sim", choices=["cache-sim", "gem5", "sniper"])
    ap.add_argument("--cells", default="web-Google:512kB",
                    help="space-separated graph:l3size cells")
    ap.add_argument("--order", default="0", help="reorder code (0=ORIGINAL)")
    ap.add_argument("--eviction", default="LRU",
                    help="fixed eviction policy (vary only the prefetcher)")
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--options", default="-n 1 -i 1")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cells = [c.split(":") for c in args.cells.split()]
    hdr = ["graph/L3", "prefetcher", "l3_mr", "demand2mem", "bandwidth", "fills", "useful%"]
    print(f"\n=== ECG prefetch matrix [{args.suite}]  eviction={args.eviction} "
          f"-o{args.order} lookahead={args.lookahead} ===")
    print("demand2mem=latency proxy, bandwidth=total DRAM traffic (both lower=better)")
    print("".join(h.ljust(18) if i < 2 else h.rjust(12) for i, h in enumerate(hdr)))
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for graph, l3 in cells:
        for pf in PREFETCHERS:
            m = run_cell(args.suite, graph, l3, args.order, args.eviction, pf,
                         args.lookahead, args.options)
            tag = f"{graph}/{l3}"
            if m is None:
                cells_txt = ["  --  "] * 5
            else:
                useful = f"{m['useful']*100:6.1f}" if m["useful"] is not None else "   -  "
                l3mr = f"{m['l3_mr']:.4f}" if isinstance(m["l3_mr"], float) else "  --  "
                cells_txt = [l3mr, fmt_m(m["demand2mem"]), fmt_m(m["bandwidth"]),
                             fmt_m(m["fills"]), useful]
            print(tag.ljust(18) + pf.ljust(12) +
                  "".join(c.rjust(12) for c in cells_txt))
            md.append("| " + tag + " | " + pf + " | " + " | ".join(c.strip() for c in cells_txt) + " |")
        md.append("|" + "   |" * len(hdr))
    if args.out:
        Path(args.out).write_text("\n".join(md) + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
