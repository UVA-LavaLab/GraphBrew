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
    args = p.parse_args()
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

    fn(
        graphs=cfg["graphs"],
        benchmarks=cfg["benchmarks"],
        trials=cfg["trials"],
        timeout=cfg["timeout"],
        dry_run=cfg["dry_run"],
        graph_dir=cfg["graph_dir"],
    )
    print(f"STAGE 03 COMPLETE — results under results/vldb_paper/exp{cfg['exp']}_*/")


if __name__ == "__main__":
    main()
