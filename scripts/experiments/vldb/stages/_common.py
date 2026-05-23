"""Shared helpers for stage scripts.

Each stage script (01_prep .. 05_aggregate) is independently runnable.
They all share the same graph/benchmark selection logic, lifted out here
so the per-stage scripts stay small.

Usage from a stage script:
    from _common import add_common_args, resolve_config
    args = parser.parse_args()
    cfg = resolve_config(args)   # -> dict with graphs, benchmarks, trials, timeout, graph_dir
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make sibling package importable
_THIS = Path(__file__).resolve()
_VLDB_DIR = _THIS.parent.parent          # scripts/experiments/vldb
_EXP_DIR = _VLDB_DIR.parent              # scripts/experiments
_SCRIPTS = _EXP_DIR.parent               # scripts
_ROOT = _SCRIPTS.parent                  # repo root
for p in (str(_SCRIPTS), str(_EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.vldb import runner as V                   # noqa: E402
from experiments.vldb.config import (                      # noqa: E402
    EVAL_GRAPHS, EVAL_GRAPHS_64GB, EVAL_GRAPHS_LOCAL, PREVIEW_GRAPHS,
    BENCHMARKS, BENCHMARKS_PREVIEW,
    TRIALS_FULL, TRIALS_PREVIEW,
    TIMEOUT_FULL, TIMEOUT_PREVIEW,
)

PROJECT_ROOT = _ROOT


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add the graph-set / preview / graph-dir options shared by every stage."""
    p.add_argument("--exp", type=int, required=True, choices=range(1, 9),
                   help="Experiment ID (1..8)")
    p.add_argument("--graphs", nargs="+",
                   help="Override graph list (by name). Otherwise picks from --64gb/--local/--preview/full.")
    p.add_argument("--graph-dir", type=str, default=str(PROJECT_ROOT / "results" / "graphs"),
                   help="Directory containing graph files (.sg/.el/.mtx).")
    p.add_argument("--preview", action="store_true",
                   help="Preview mode: 2 tiny graphs, 1 trial, 2 benchmarks (fast smoke test).")
    p.add_argument("--64gb", action="store_true", dest="use_64gb",
                   help="Use 64 GB graph set (11 auto-downloadable graphs).")
    p.add_argument("--local", action="store_true", dest="use_local",
                   help="Use local graph set (6 graphs <= 117M edges).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands instead of executing.")


def resolve_config(args: argparse.Namespace) -> dict:
    """Return a dict with the selected graph set + benchmark/trial/timeout knobs."""
    if args.preview:
        graphs, benchmarks, trials, timeout = (
            PREVIEW_GRAPHS, BENCHMARKS_PREVIEW, TRIALS_PREVIEW, TIMEOUT_PREVIEW,
        )
    elif getattr(args, "use_64gb", False):
        graphs, benchmarks, trials, timeout = (
            EVAL_GRAPHS_64GB, BENCHMARKS, TRIALS_FULL, TIMEOUT_FULL,
        )
    elif getattr(args, "use_local", False):
        graphs, benchmarks, trials, timeout = (
            EVAL_GRAPHS_LOCAL, BENCHMARKS, TRIALS_FULL, TIMEOUT_FULL,
        )
    else:
        graphs, benchmarks, trials, timeout = (
            EVAL_GRAPHS, BENCHMARKS, TRIALS_FULL, TIMEOUT_FULL,
        )

    if args.graphs:
        pool = EVAL_GRAPHS + EVAL_GRAPHS_64GB + EVAL_GRAPHS_LOCAL + PREVIEW_GRAPHS
        seen = set()
        picked = []
        for name in args.graphs:
            for g in pool:
                if g["name"] == name and name not in seen:
                    picked.append(g)
                    seen.add(name)
                    break
            else:
                # unknown name: synthesize a minimal entry
                if name not in seen:
                    picked.append({"name": name, "short": name, "type": "unknown",
                                   "vertices_m": 0, "edges_m": 0})
                    seen.add(name)
        graphs = picked

    return {
        "exp": args.exp,
        "graphs": graphs,
        "benchmarks": benchmarks,
        "trials": trials,
        "timeout": timeout,
        "graph_dir": args.graph_dir,
        "dry_run": args.dry_run,
    }


def banner(stage: str, cfg: dict) -> None:
    print("=" * 60)
    print(f"STAGE: {stage}  |  exp{cfg['exp']}  |  "
          f"{len(cfg['graphs'])} graph(s)  |  "
          f"trials={cfg['trials']}  timeout={cfg['timeout']}s")
    print(f"  graph_dir = {cfg['graph_dir']}")
    print(f"  graphs    = {[g['name'] for g in cfg['graphs']]}")
    print("=" * 60)
