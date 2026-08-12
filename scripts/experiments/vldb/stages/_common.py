"""Shared helpers for stage scripts.

Each stage script (01_prep .. 05_aggregate) is independently runnable.
They all share the same graph/benchmark selection logic, lifted out here
so the per-stage scripts stay small.

Usage from a stage script:
    from scripts.experiments.vldb.stages._common import (
        add_common_args, resolve_config,
    )
    args = parser.parse_args()
    cfg = resolve_config(args)   # -> dict with graphs, benchmarks, trials, timeout, graph_dir
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make the repository package importable under one canonical identity.
_THIS = Path(__file__).resolve()
_VLDB_DIR = _THIS.parent.parent          # scripts/experiments/vldb
_EXP_DIR = _VLDB_DIR.parent              # scripts/experiments
_SCRIPTS = _EXP_DIR.parent               # scripts
_ROOT = _SCRIPTS.parent                  # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.experiments.vldb import runner as V           # noqa: E402
from scripts.experiments.vldb.config import (              # noqa: E402
    EVAL_GRAPHS, EVAL_GRAPHS_64GB, EVAL_GRAPHS_LOCAL, PREVIEW_GRAPHS,
    CACHE_GRAPH_NAMES, SCALABILITY_GRAPH_NAMES,
    BENCHMARKS, BENCHMARKS_PREVIEW,
    TRIALS_FULL, TRIALS_PREVIEW,
    REORDER_TIMEOUT_FULL, REORDER_TIMEOUT_PREVIEW,
    TIMEOUT_FULL, TIMEOUT_PREVIEW,
    PAPER_ARTIFACT_ROOT, PAPER_GRAPH_ROOT,
)

PROJECT_ROOT = _ROOT


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add the graph-set / preview / graph-dir options shared by every stage."""
    p.add_argument("--exp", type=int, required=True, choices=range(1, 9),
                   help="Experiment ID (1..8)")
    p.add_argument("--graphs", nargs="+",
                   help="Override graph list (by name). Otherwise picks from --64gb/--local/--preview/full.")
    p.add_argument("--algorithms", nargs="+",
                   help="Restrict to exact canonical algorithm keys.")
    p.add_argument("--graph-dir", type=str, default=str(PAPER_GRAPH_ROOT),
                   help="Directory containing graph files (.sg/.el/.mtx).")
    p.add_argument("--artifact-root", type=str, default=str(PAPER_ARTIFACT_ROOT),
                   help="Root for vldb_paper/, vldb_mappings/, and vldb_runs/.")
    p.add_argument(
        "--measurement-generation",
        help="Explicit shared generation ID for distributed/fan-out runs.",
    )
    p.add_argument("--threads", type=int,
                   help="Base OpenMP threads (default: 4 preview, 16 otherwise).")
    p.add_argument("--cpu-list", type=str,
                   help="taskset CPU list for benchmark isolation (e.g. 0-15).")
    p.add_argument("--cache-mode",
                   choices=["accurate", "fast", "ultrafast", "sampled"],
                   help="Cache mode (default: ultrafast).")
    p.add_argument("--cache-sample-rate", type=int, default=64,
                   help="Sampling interval when --cache-mode sampled.")
    p.add_argument("--cache-sizes-kib", nargs="+", type=int,
                   help="Override cache-capacity sweep in KiB.")
    p.add_argument("--cache-all-algorithms", action="store_true",
                   help="Use the full paper algorithm matrix for cache sweeps.")
    p.add_argument("--publish-paper-figures", action="store_true",
                   help="Ignored before stage 05; accepted for SLURM argument forwarding.")
    p.add_argument("--preview", action="store_true",
                   help="Preview mode: 2 tiny graphs, 1 trial, 2 benchmarks (fast smoke test).")
    p.add_argument("--64gb", action="store_true", dest="use_64gb",
                   help="Use 64 GB graph set (11 auto-downloadable graphs).")
    p.add_argument("--local", action="store_true", dest="use_local",
                   help="Use local graph set (6 graphs <= 117M edges).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands instead of executing.")
    p.add_argument("--timeout", type=int,
                   help="Override per-command timeout in seconds.")
    p.add_argument(
        "--reorder-timeout",
        type=int,
        help="Override the mapping-generation timeout in seconds.",
    )


def resolve_config(args: argparse.Namespace) -> dict:
    """Return a dict with the selected graph set + benchmark/trial/timeout knobs."""
    reorder_timeout = (
        REORDER_TIMEOUT_PREVIEW
        if args.preview
        else REORDER_TIMEOUT_FULL
    )
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
    elif args.exp == 1 and not args.preview:
        graph_by_name = {graph["name"]: graph for graph in graphs}
        graphs = [
            graph_by_name[name] for name in CACHE_GRAPH_NAMES
            if name in graph_by_name
        ]
    elif args.exp == 8 and not args.preview:
        graph_by_name = {graph["name"]: graph for graph in graphs}
        graphs = [
            graph_by_name[name] for name in SCALABILITY_GRAPH_NAMES
            if name in graph_by_name
        ]
    if args.timeout is not None:
        timeout = args.timeout
        if args.reorder_timeout is None:
            reorder_timeout = args.timeout
    if args.reorder_timeout is not None:
        reorder_timeout = args.reorder_timeout

    threads = args.threads or (4 if args.preview else 16)
    cache_mode = args.cache_mode or "ultrafast"
    V.configure_artifact_root(args.artifact_root)
    V.configure_measurement_generation(args.measurement_generation)
    V.configure_runtime_policy(threads, args.cpu_list)
    V.configure_cache_policy(
        preview=args.preview,
        mode=cache_mode,
        sample_rate=args.cache_sample_rate,
        all_algorithms=args.cache_all_algorithms,
        sizes_kib=args.cache_sizes_kib,
    )
    V.configure_algorithm_filter(args.algorithms)
    V.configure_execution_mode(dry_run=args.dry_run)

    return {
        "exp": args.exp,
        "graphs": graphs,
        "benchmarks": benchmarks,
        "trials": trials,
        "timeout": timeout,
        "reorder_timeout": reorder_timeout,
        "graph_dir": args.graph_dir,
        "artifact_root": args.artifact_root,
        "measurement_generation": args.measurement_generation,
        "threads": threads,
        "cpu_list": args.cpu_list,
        "cache_mode": cache_mode,
        "algorithms": args.algorithms,
        "dry_run": args.dry_run,
    }


def banner(stage: str, cfg: dict) -> None:
    print("=" * 60)
    print(f"STAGE: {stage}  |  exp{cfg['exp']}  |  "
          f"{len(cfg['graphs'])} graph(s)  |  "
          f"trials={cfg['trials']}  timeout={cfg['timeout']}s  "
          f"reorder_timeout={cfg['reorder_timeout']}s")
    print(f"  graph_dir = {cfg['graph_dir']}")
    print(f"  artifact_root = {cfg['artifact_root']}")
    print(f"  threads   = {cfg['threads']}  cpu_list = {cfg['cpu_list'] or 'scheduler-managed'}")
    print(f"  cache_mode = {cfg['cache_mode']}")
    print(f"  algorithms = {cfg['algorithms'] or 'full matrix'}")
    print(f"  graphs    = {[g['name'] for g in cfg['graphs']]}")
    print("=" * 60)
