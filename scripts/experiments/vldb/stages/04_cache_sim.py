#!/usr/bin/env python3
"""Stage 04 — Cache simulation (CPU speed independent).

Runs the cache-simulator binaries (bench/bin_sim/*) on the same algorithm
matrix as stage 03 but only collects cache statistics (L1/L2/L3 hit rates,
miss counts, total accesses). CPU performance of the runner host is
irrelevant — fine to run on a slow / shared / different machine than 03.

Currently only experiment 1 (cache-performance analysis) drives this
stage in the canonical paper. You can SKIP this stage entirely if you
only care about wall-clock.

Examples:
    # Canonical exp1 cache sweep:
    python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --local

    # Single-graph quick check:
    python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --graphs cit-Patents
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, resolve_config, banner, V


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args()
    cfg = resolve_config(args)
    banner("04_cache_sim", cfg)

    if cfg["exp"] != 1:
        print(f"WARNING: experiment {cfg['exp']} is not a cache-sim experiment; "
              "only exp 1 uses bench/bin_sim. Proceeding anyway and routing "
              "through exp1_cache_performance.", file=sys.stderr)

    V.exp1_cache_performance(
        graphs=cfg["graphs"],
        benchmarks=cfg["benchmarks"],
        trials=cfg["trials"],
        timeout=cfg["timeout"],
        dry_run=cfg["dry_run"],
        graph_dir=cfg["graph_dir"],
    )
    print("STAGE 04 COMPLETE — results under results/vldb_paper/exp1_cache/")


if __name__ == "__main__":
    main()
