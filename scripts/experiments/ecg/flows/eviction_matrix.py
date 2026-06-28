#!/usr/bin/env python3
"""ECG factorial-ablation matrix: rows=graphs, cols=variants, one table per simulator.

Columns (Stage-1 eviction): LRU | GRASP | POPT | ECG:grasp_only | ECG:epoch_only |
                            ECG:rrip_first | ECG:epoch_first  (+ legacy shortcircuit)
Each ECG variant is the SAME policy (ECG:ECG_GRASP_POPT) selected by env ECG_VARIANT,
so the comparison isolates exactly one lever. See files/ecg_hardened_context.md sec 14.

Usage:
  python3 scripts/experiments/ecg/ecg_variant_matrix.py --suite cache-sim \
      --cells "kron_s16_k4:128kB web-Google:512kB cit-Patents:1MB"
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GRAPHS = ROOT / "results" / "graphs"

# (column label, policy spec, ECG_VARIANT or None, extra env)
COLUMNS = [
    ("LRU",            "LRU",                None),
    ("GRASP",          "GRASP",              None),
    ("POPT",           "POPT",               None),
    ("ECG:grasp_only", "ECG:ECG_GRASP_POPT", "grasp_only"),
    ("ECG:epoch_only", "ECG:ECG_GRASP_POPT", "epoch_only"),
    ("ECG:rrip_first", "ECG:ECG_GRASP_POPT", "rrip_first"),
    ("ECG:epoch_first","ECG:ECG_GRASP_POPT", "epoch_first"),
    ("ECG:shortcirc",  "ECG:ECG_GRASP_POPT", "shortcircuit"),
]


def run_cell(suite, graph, l3, order, label, policy, variant, opts, gem5_env,
             timeout_cell=3600, run_id="run", popt_reserve_model="fixed_one",
             ecg_epoch_pack_bits=32, ecg_epochs=65535, stream_prefetch_degree=0,
             benchmark="pr"):
    gpath = GRAPHS / graph / f"{graph}.sg"
    outdir = Path("/tmp") / f"evm_{run_id}_{suite}_{benchmark}_{graph}_{l3}_o{order}_{label.replace(':','_')}"
    cmd = [sys.executable, str(ROOT / "scripts/experiments/ecg/roi_matrix.py"),
           "--suite", suite, "--no-build", "--benchmark", benchmark,
           "--policies", policy,
           "--options", f"-f {gpath} -o {order} {opts}",
           "--l3-sizes", l3, "--l3-ways", "16",
           "--l1d-size", "32kB", "--l2-size", "256kB",
           "--popt-reserve-model", popt_reserve_model,
           "--ecg-epoch-pack-bits", str(ecg_epoch_pack_bits),
           "--ecg-epochs", str(ecg_epochs),
           "--cache-stream-prefetch-degree", str(stream_prefetch_degree),
           "--out-dir", str(outdir)]
    env = dict(os.environ)
    if variant:
        env["ECG_VARIANT"] = variant
    if suite == "gem5":
        env.update(gem5_env)
    try:
        subprocess.run(cmd, env=env, cwd=str(ROOT),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout_cell, check=False)
        rows = json.load(open(outdir / "roi_matrix.json"))
        rows = rows if isinstance(rows, list) else [rows]
        x = rows[0]
        # Guard empty cells: if the L3 saw no accesses (e.g. a kernel whose ROI
        # never reached the LLC, or a sink source), l3_miss_rate degenerates to
        # 1.0 with 0 hits/0 misses -> report None so the matrix shows "--" not a
        # fake 100% miss.
        if x.get("l3_exercised") is False:
            return None
        hits = x.get("l3_hits") or 0
        misses = x.get("l3_misses") or 0
        if (x.get("total_accesses") == 0) or (hits == 0 and misses == 0):
            return None
        return x.get("l3_miss_rate")
    except Exception as e:
        return None


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="cache-sim", choices=["cache-sim", "gem5", "sniper"])
    ap.add_argument("--benchmark", default="pr",
                    help="Kernel to run (pr/bfs/sssp/bc/cc). Per-kernel flags go in --options "
                         "(e.g. pr '-i 2', bfs '-r 0', sssp '-r 0 -d 1').")
    ap.add_argument("--cells", default="kron_s16_k4:128kB",
                    help="space-separated graph:l3size cells")
    ap.add_argument("--orders", default="0 5",
                    help="space-separated reorder codes; 0=ORIGINAL, 5=DBG (GRASP's intended)")
    ap.add_argument("--options", default="-n 1 -i 1")
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout-cell", type=int, default=3600,
                    help="per-policy-run subprocess timeout in seconds (raise for big graphs)")
    ap.add_argument("--run-id", default="run",
                    help="tag for the /tmp out-dirs so concurrent runs don't collide")
    ap.add_argument("--popt-reserve-model", default="fixed_one",
                    choices=["fixed_one", "size_correct"],
                    help="P-OPT reserved-LLC-way charge model forwarded to roi_matrix. "
                         "'fixed_one' (default) keeps the legacy 1-way charge (existing "
                         "matrices reproduce unchanged); 'size_correct' charges the paper-"
                         "faithful resident-column reservation (scales with |V|) for the "
                         "iso-area POPT comparison.")
    ap.add_argument("--ecg-epoch-pack-bits", type=int, default=32, choices=[32, 64],
                    help="ECG epoch packed-record width forwarded to roi_matrix. 32 (default) = "
                         "committed reproductions; 64 = ISA-faithful full epoch resolution at "
                         "scale (honest 8B record charged under CHARGED=1).")
    ap.add_argument("--cache-stream-prefetch-degree", type=int, default=0,
                    help="Uniform structure-stream prefetcher degree (0=off) forwarded to roi_matrix; applied to all policies. Hides the read-once structure stream so total mr reflects property accesses.")
    ap.add_argument("--ecg-epochs", type=int, default=65535,
                    help="ECG epoch count (eviction-epoch resolution) forwarded to roi_matrix. "
                         "Default 65535. Eviction quality is non-monotonic in ne; pair a "
                         "sweet-spot ne with --ecg-epoch-pack-bits 64 to MAINTAIN it at scale.")
    args = ap.parse_args(argv)

    gem5_env = {"GEM5_OPT": str(ROOT/"bench/include/gem5_sim/gem5/build/RISCV/gem5.opt"),
                "GEM5_KERNEL_SUFFIX": "_riscv_m5ops", "GEM5_FORCE_ECG_EXTRACT": "1",
                "GEM5_ECG_PFX_MODE": "6", "ECG_PREFETCH_MODE": "6"}
    cells = [c.split(":") for c in args.cells.split()]
    orders = args.orders.split()
    labels = [c[0] for c in COLUMNS]

    print(f"\n=== ECG variant matrix [{args.suite}/{args.benchmark}]  (l3_miss_rate; lower=better) ===")
    print("(-o0 = original order, -o5 = DBG reorder which GRASP/ECG-insertion need)")
    print("graph/L3/order".ljust(24) + "".join(l.rjust(15) for l in labels))
    md = ["| graph/L3/order | " + " | ".join(labels) + " |",
          "|" + "---|" * (len(labels)+1)]
    for graph, l3 in cells:
        for order in orders:
            vals = []
            for (label, policy, variant) in COLUMNS:
                r = run_cell(args.suite, graph, l3, order, label, policy, variant,
                             args.options, gem5_env,
                             timeout_cell=args.timeout_cell, run_id=args.run_id,
                             popt_reserve_model=args.popt_reserve_model,
                             ecg_epoch_pack_bits=args.ecg_epoch_pack_bits,
                             ecg_epochs=args.ecg_epochs,
                             stream_prefetch_degree=args.cache_stream_prefetch_degree,
                             benchmark=args.benchmark)
                vals.append(r)
            tag = f"{graph}/{l3}/o{order}"
            print(tag.ljust(24) + "".join(
                (f"{v:.4f}" if isinstance(v, float) else "  --  ").rjust(15) for v in vals),
                flush=True)
            md.append("| " + tag + " | " +
                      " | ".join(f"{v:.4f}" if isinstance(v, float) else "--" for v in vals) + " |")
            # Incremental write: a long matrix's partial rows survive an interrupt/timeout.
            if args.out:
                Path(args.out).write_text("\n".join(md) + "\n")
    if args.out:
        Path(args.out).write_text("\n".join(md) + "\n")
        print(f"\n[written] {args.out}")



if __name__ == "__main__":
    main(sys.argv[1:])
