#!/usr/bin/env python3
"""Stage 02 — Pre-generate reorder mappings (reorder-once-reuse-everywhere).

Runs every algorithm + COMPOSE_VARIANT spec once per graph, dumps the
permutation to results/vldb_mappings/<graph>/<algo_key>.lo plus a
.json sidecar with the measured reorder overhead and provenance.

Downstream stages (03_cpu_perf, 04_cache_sim) automatically pick up
these cached mappings via algo_flags_or_map() and swap the original
'-o <spec>' for '-o 13:<path>.lo' so each kernel run is reorder-free.

CPU-bound, no network. Single-node OK. Safe to run on cluster compute.

Examples:
    python3 scripts/experiments/vldb/stages/02_reorder.py --exp 2 --preview
    python3 scripts/experiments/vldb/stages/02_reorder.py --exp 2 --graphs com-Orkut
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.experiments.vldb.stages._common import (
    add_common_args, resolve_config, banner, V,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = p.parse_args()
    cfg = resolve_config(args)
    banner("02_reorder", cfg)

    V._setup_build_binaries(
        benchmarks=[],
        include_standard=False,
        include_sim=False,
        include_converter=True,
    )
    V._pregenerate_mappings(
        cfg["graphs"],
        cfg["graph_dir"],
        dry_run=cfg["dry_run"],
        timeout=cfg["reorder_timeout"],
    )
    print("STAGE 02 COMPLETE — mappings under results/vldb_mappings/")


if __name__ == "__main__":
    main()
