#!/usr/bin/env python3
"""Stage 05 — Aggregate JSON results into LaTeX tables + PNG figures.

Reads everything under results/vldb_paper/ and emits paper-ready
artefacts via vldb_generate_figures.py. Pure I/O, no benchmarks.

Examples:
    python3 scripts/experiments/vldb/stages/05_aggregate.py --exp 0
    # (--exp is required by the common parser but ignored here; use any value.)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, resolve_config, V  # noqa: F401


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # We accept --exp for argparse uniformity but this stage aggregates ALL exps.
    p.add_argument("--exp", type=int, default=0,
                   help="Ignored — figure generator aggregates all exps under results/vldb_paper/.")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation; only rebuild INDEX.json.")
    args = p.parse_args()
    print("STAGE 05 — rebuilding results/INDEX.json")
    try:
        from lib.analysis.results_index import write_index
        out = write_index()
        print(f"  wrote {out}")
    except Exception as e:
        print(f"  WARN: could not build INDEX.json ({e})")

    if not args.no_figures:
        print("STAGE 05 — generating figures + LaTeX tables from results/vldb_paper/")
        V._generate_figures()
    print("STAGE 05 COMPLETE.")


if __name__ == "__main__":
    main()
