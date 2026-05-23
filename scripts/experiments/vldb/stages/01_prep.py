#!/usr/bin/env python3
"""Stage 01 — Data preparation.

Downloads graphs listed for the selected experiment from VLDB_GRAPH_SOURCES
(auto-catalog: email-Eu-core, cit-Patents, com-Orkut, ...) and converts
.mtx -> .sg using bench/bin/converter.

Network-only. No CPU benchmarks, no cache-sim. Safe to run on a login node.

Examples:
    # Preview (downloads ~2 MB of tiny graphs):
    python3 scripts/experiments/vldb/stages/01_prep.py --exp 2 --preview

    # Local 6-graph eval set:
    python3 scripts/experiments/vldb/stages/01_prep.py --exp 2 --local

    # Single graph:
    python3 scripts/experiments/vldb/stages/01_prep.py --exp 2 --graphs com-Orkut
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_common_args, resolve_config, banner, V


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--skip-download", action="store_true",
                   help="Skip the download step (only run .mtx -> .sg conversion).")
    args = p.parse_args()
    cfg = resolve_config(args)
    banner("01_prep", cfg)

    graphs_path = Path(cfg["graph_dir"])
    graphs_path.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        V._setup_download_graphs(cfg["graphs"], graphs_path)
    V._setup_convert_graphs(cfg["graphs"], graphs_path)
    print("STAGE 01 COMPLETE.")


if __name__ == "__main__":
    main()
