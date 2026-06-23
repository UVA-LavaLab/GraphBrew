#!/usr/bin/env python3
"""Combined-stack matrix: eviction + prefetch, reported on the CORRECT metric.

The eviction headline (ecg_variant_matrix.py) reports l3_miss_rate. Once a
prefetcher is on, l3_miss_rate is CONFOUNDED (prefetch fills inflate L3 accesses),
so the prefetch + combined story must be read on demand-to-memory (latency proxy)
and total DRAM traffic (bandwidth). This driver produces those two metrics for the
canonical configs, at BOTH reorderings, with CONTROLLED per-run env so the
prefetcher is never leaked across arms.

Canonical config (one config, no variant sprawl):
  - eviction = ECG:ECG_GRASP_POPT, ECG_VARIANT=shortcircuit (the headline variant)
  - prefetch = Path A (ECG_EDGE_MASK_PREFETCH=K, epoch-filtered CSR read-ahead)
  - both reorderings: -o0 (ORIGINAL) and -o5 (DBG; GRASP/ECG-tier need degree-grouping)

Arms per (graph, L3, order):
  | arm            | eviction            | prefetch        | why |
  | LRU            | LRU                 | none            | floor baseline |
  | LRU+DROPLET    | LRU                 | DROPLET         | prefetch baseline (Basak HPCA'19) |
  | ECG            | ECG:ECG_GRASP_POPT  | none            | eviction-only headline |
  | ECG+PathA      | ECG:ECG_GRASP_POPT  | Path A (K)      | the combined stack |

Each arm is a separate roi_matrix subprocess with its OWN env (no global leak): the
eviction arms set NO prefetch env; the prefetch arms set it explicitly. This is the
single source of truth for the prefetch/combined tables (eviction table stays in
ecg_variant_matrix.py).

Usage:
  python3 scripts/experiments/ecg/combined_stack_matrix.py \
      --cells "web-Google:512kB cit-Patents:1MB soc-pokec:1MB" --orders "0 5"
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRAPHS = ROOT / "results" / "graphs"
ROI = ROOT / "scripts" / "experiments" / "ecg" / "roi_matrix.py"

# (label, eviction policy spec, prefetcher flag, Path-A K). K>0 sets ECG_EDGE_MASK_PREFETCH.
ARMS = [
    ("LRU",         "LRU",                "none",    0),
    ("LRU+DROPLET", "LRU",                "DROPLET", 0),
    ("ECG",         "ECG:ECG_GRASP_POPT", "none",    0),
    ("ECG+PathA",   "ECG:ECG_GRASP_POPT", "DROPLET", 8),  # prefetcher flag unused; Path A via K
]


def run_arm(graph, l3, order, label, policy, prefetcher, kpfx, opts):
    gpath = GRAPHS / graph / f"{graph}.sg"
    outdir = Path("/tmp") / f"csm_{graph}_{l3}_o{order}_{label.replace('+', '_')}"
    cmd = [
        sys.executable, str(ROI),
        "--suite", "cache-sim", "--no-build", "--benchmark", "pr",
        "--policies", policy,
        "--options", f"-f {gpath} -o {order} {opts}",
        "--l3-sizes", l3, "--l3-ways", "16",
        "--l1d-size", "32kB", "--l2-size", "256kB", "--line-size", "64",
        "--cache-sim-omp-threads", "1",
        "--out-dir", str(outdir),
    ]
    # Controlled env: start from a clean copy, strip any inherited prefetch knobs,
    # then set ONLY what this arm needs.
    env = dict(os.environ)
    for k in ("ECG_PREFETCH", "ECG_EDGE_MASK_PREFETCH", "ECG_PREFETCH_MODE"):
        env.pop(k, None)
    if policy.startswith("ECG"):
        env["ECG_VARIANT"] = "shortcircuit"  # the canonical headline variant
    if kpfx > 0:
        # Path A: epoch-filtered CSR read-ahead (the combined stack).
        env["ECG_PREFETCH"] = "1"
        env["ECG_EDGE_MASK_PREFETCH"] = str(kpfx)
        cmd += ["--prefetcher", "ECG_PFX", "--ecg-pfx-mode", "per_edge"]
    elif prefetcher == "DROPLET":
        env["ECG_PREFETCH"] = "1"
        cmd += ["--prefetcher", "DROPLET"]
    else:
        cmd += ["--prefetcher", "none"]
    try:
        subprocess.run(cmd, env=env, cwd=str(ROOT),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=7200, check=False)
        r = json.load(open(outdir / "roi_matrix.json"))
        r = (r if isinstance(r, list) else [r])[0]
        bw = r.get("total_memory_traffic") or r.get("memory_accesses")
        return {
            "l3_mr": r.get("l3_miss_rate"),
            "demand": r.get("memory_accesses"),
            "bw": bw,
            "fills": r.get("prefetch_fills", 0),
        }
    except Exception as e:
        return {"error": str(e)}


def fmt_m(v):
    if v is None:
        return "  --  "
    return f"{v/1e6:7.2f}M" if v >= 1e6 else f"{v:8.0f}"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="web-Google:512kB",
                    help="space-separated graph:l3 cells")
    ap.add_argument("--orders", default="0 5", help="reorder codes (0=ORIGINAL, 5=DBG)")
    ap.add_argument("--options", default="-n 1 -i 3")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cells = [c.split(":") for c in args.cells.split()]
    orders = args.orders.split()
    md = ["| graph/L3 | order | arm | l3_mr | demand2mem | bandwidth | fills |",
          "|---|---|---|---|---|---|---|"]
    print("\n=== Combined-stack matrix (demand2mem + bandwidth; lower=better) ===")
    print("(l3_mr shown for reference but is CONFOUNDED once prefetch is on)")
    hdr = f"{'graph/L3':<20}{'order':<7}{'arm':<13}{'l3_mr':>8}{'demand':>10}{'bw':>10}{'fills':>10}"
    print(hdr)
    for graph, l3 in cells:
        for order in orders:
            for (label, policy, pf, kpfx) in ARMS:
                m = run_arm(graph, l3, order, label, policy, pf, kpfx, args.options)
                if "error" in m:
                    print(f"{graph+'/'+l3:<20}o{order:<6}{label:<13}  ERROR")
                    md.append(f"| {graph}/{l3} | o{order} | {label} | ERR | ERR | ERR | ERR |")
                    continue
                l3s = f"{m['l3_mr']:.4f}" if m['l3_mr'] is not None else "--"
                print(f"{graph+'/'+l3:<20}o{order:<6}{label:<13}{l3s:>8}"
                      f"{fmt_m(m['demand']):>10}{fmt_m(m['bw']):>10}{fmt_m(m['fills']):>10}")
                md.append(f"| {graph}/{l3} | o{order} | {label} | {l3s} | "
                          f"{fmt_m(m['demand']).strip()} | {fmt_m(m['bw']).strip()} | "
                          f"{fmt_m(m['fills']).strip()} |")
    if args.out:
        Path(args.out).write_text("\n".join(md) + "\n")
        print(f"\n[write] {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
