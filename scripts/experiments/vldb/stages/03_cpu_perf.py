#!/usr/bin/env python3
"""Stage 03 — CPU wall-clock benchmarks (real hardware).

Runs the selected experiment's kernel sweep on real CPU and writes
JSON results under results/vldb_paper/exp<N>_*/.

This is the stage you want to scale on real CPU partitions. It does
NOT do cache simulation (that's stage 04, independent and can run on
a separate machine where CPU speed doesn't matter).

Reads .lo mappings from stage 02 to avoid recomputing reorders.

Examples:
    # exp2 (kernel speedup) on the local 6-graph set:
    python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --local

    # exp4 end-to-end on one graph:
    python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 4 --graphs com-Orkut

Supported experiments: 2, 3, 4, 5, 6, 7, 8 (exp 1 is cache-sim only;
use stage 04 for it).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, resolve_config, banner, V


# exp_id -> callable in vldb_paper_experiments
_EXPS = {
    2: V.exp2_kernel_speedup,
    3: V.exp3_reorder_overhead,
    4: V.exp4_end_to_end,
    5: V.exp5_ablation,
    6: V.exp6_sensitivity,
    7: V.exp7_chained,
    8: V.exp8_scalability,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument(
        "--tune-sssp-delta",
        action="store_true",
        help="Tune weighted SSSP delta on the SHUFFLED baseline and exit.",
    )
    p.add_argument(
        "--freeze-sssp-policy",
        action="store_true",
        help="Freeze reviewed tuner recommendations into policy SSOT.",
    )
    p.add_argument(
        "--verify-gate",
        action="store_true",
        help="Run the untimed semantic verification gate and exit.",
    )
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Use the already-reviewed binaries without invoking make.",
    )
    args = p.parse_args()
    if args.freeze_sssp_policy and not args.tune_sssp_delta:
        p.error("--freeze-sssp-policy requires --tune-sssp-delta")
    cfg = resolve_config(args)
    banner("03_cpu_perf", cfg)

    if cfg["exp"] == 1:
        print("ERROR: experiment 1 is cache-sim only. Use 04_cache_sim.py instead.",
              file=sys.stderr)
        sys.exit(2)
    fn = _EXPS.get(cfg["exp"])
    if fn is None:
        print(f"ERROR: experiment {cfg['exp']} not supported by 03_cpu_perf.",
              file=sys.stderr)
        sys.exit(2)

    if not args.freeze_sssp_policy and not args.skip_build:
        V._setup_build_binaries(
            benchmarks=(
                ["sssp"] if args.tune_sssp_delta
                else cfg["benchmarks"]
            ),
            include_standard=True,
            include_sim=args.verify_gate and "pr" in cfg["benchmarks"],
            include_converter=not args.tune_sssp_delta,
            include_work=args.verify_gate,
            sim_benchmarks=["pr"],
        )
    if args.tune_sssp_delta:
        V.tune_sssp_deltas(
            graphs=cfg["graphs"],
            graph_dir=cfg["graph_dir"],
            timeout=cfg["timeout"],
            dry_run=cfg["dry_run"],
            trials=1 if args.preview else V.SSSP_TUNING_TRIALS,
            freeze_policy=args.freeze_sssp_policy,
        )
        print("SSSP DELTA TUNING COMPLETE.")
        return
    if args.verify_gate:
        V.run_verification_gate(
            graphs=cfg["graphs"],
            benchmarks=cfg["benchmarks"],
            graph_dir=cfg["graph_dir"],
            timeout=cfg["timeout"],
            dry_run=cfg["dry_run"],
        )
        print("VERIFICATION GATE COMPLETE.")
        return
    V.preflight_benchmark_policies(
        cfg["graphs"], cfg["benchmarks"], cfg["graph_dir"],
    )
    V.require_timing_machine_policy(preview=args.preview)
    V.require_verification_gate(
        cfg["graphs"], cfg["benchmarks"], cfg["graph_dir"],
    )
    experiment_timeout = (
        cfg["reorder_timeout"]
        if cfg["exp"] in {3, 8}
        else cfg["timeout"]
    )
    fn(
        graphs=cfg["graphs"],
        benchmarks=cfg["benchmarks"],
        trials=cfg["trials"],
        timeout=experiment_timeout,
        dry_run=cfg["dry_run"],
        graph_dir=cfg["graph_dir"],
    )
    print(f"STAGE 03 COMPLETE — results under results/vldb_paper/exp{cfg['exp']}_*/")


if __name__ == "__main__":
    main()
