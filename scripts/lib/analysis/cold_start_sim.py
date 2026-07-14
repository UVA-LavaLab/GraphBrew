#!/usr/bin/env python3
"""
Cold-Start → Warm Learning Curve Simulation.

Simulates the streaming-database learning process:
  - Start with an empty knowledge base.
  - Process graphs one by one (random order).
  - After each graph, retrain the perceptron on the accumulated data.
  - Evaluate selection accuracy (top-1, top-3, regret) on the remaining
    unseen graphs.
  - Repeat for multiple random permutations to reduce variance.

Produces data for a learning-curve plot:
  x-axis = number of graphs seen
  y-axis = selection accuracy / regret on unseen graphs

Usage:
    python -m scripts.lib.analysis.cold_start_sim \\
        --benchmark-db results/data/benchmarks.json \\
        --output results/vldb_experiments/exp5_cold_start/learning_curve.json \\
        --permutations 10

Requires existing benchmark data for ALL candidate algorithms × graphs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Resolve project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from lib.ml.weights import (  # noqa: E402
    compute_weights_from_results,
    cross_validate_logo,
    get_perceptron_candidates,
)
from lib.ml.eval_weights import load_benchmark_entries  # noqa: E402
from lib.core.datastore import get_benchmark_store, get_props_store  # noqa: E402

log = logging.getLogger("cold_start_sim")


def _graph_names_from_records(records: List[dict]) -> List[str]:
    """Extract unique graph names from benchmark records."""
    return sorted({r["graph"] for r in records})


def _filter_records(records: List[dict], graphs: set) -> List[dict]:
    """Keep only records whose graph is in the given set."""
    return [r for r in records if r["graph"] in graphs]


def simulate_cold_start(
    all_records: List[dict],
    graph_order: List[str],
    benchmarks: List[str],
    weights_dir: Optional[str] = None,
) -> List[dict]:
    """
    Run one cold-start simulation for `graph_order`.

    Returns a list of dicts, one per step:
      {
        "n_seen": int,
        "graph_added": str,
        "n_unseen": int,
        "top1_accuracy": float,
        "top3_accuracy": float,
        "mean_regret": float,
        "predictions": {graph: {predicted: str, oracle: str, correct: bool}}
      }
    """
    candidates = get_perceptron_candidates()
    all_graphs = set(graph_order)
    seen = set()
    curve = []

    tmp_weights = tempfile.mkdtemp(prefix="cold_start_")

    for step, graph_name in enumerate(graph_order):
        seen.add(graph_name)
        unseen = all_graphs - seen

        if len(seen) < 2:
            # Need at least 2 graphs to train; skip
            curve.append({
                "n_seen": len(seen),
                "graph_added": graph_name,
                "n_unseen": len(unseen),
                "top1_accuracy": None,
                "top3_accuracy": None,
                "mean_regret": None,
            })
            continue

        if not unseen:
            break

        # Train on seen graphs only
        seen_records = _filter_records(all_records, seen)
        try:
            compute_weights_from_results(
                benchmark_results=seen_records,
                weights_dir=tmp_weights,
                candidate_algos=candidates,
            )
        except Exception as e:
            log.warning(f"  Step {step}: training failed — {e}")
            curve.append({
                "n_seen": len(seen),
                "graph_added": graph_name,
                "n_unseen": len(unseen),
                "top1_accuracy": None,
                "top3_accuracy": None,
                "mean_regret": None,
            })
            continue

        # Evaluate on unseen graphs via LOGO (each unseen graph is test)
        unseen_records = _filter_records(all_records, unseen)
        try:
            logo_result = cross_validate_logo(
                benchmark_results=unseen_records,
                weights_dir=tmp_weights,
            )
            top1 = logo_result.get("accuracy", 0.0)
            top3 = logo_result.get("top3_accuracy", 0.0)
            regret = logo_result.get("regret", {}).get("mean", 0.0)
        except Exception as e:
            log.warning(f"  Step {step}: LOGO evaluation failed — {e}")
            top1, top3, regret = None, None, None

        curve.append({
            "n_seen": len(seen),
            "graph_added": graph_name,
            "n_unseen": len(unseen),
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "mean_regret": regret,
        })

        log.info(
            f"  Step {step + 1}/{len(graph_order)}: "
            f"seen={len(seen)}, unseen={len(unseen)}, "
            f"top1={top1}, top3={top3}, regret={regret}"
        )

    return curve


def run_cold_start_experiment(
    records: List[dict],
    n_permutations: int = 10,
    benchmarks: Optional[List[str]] = None,
    seed: int = 42,
) -> dict:
    """
    Run multiple cold-start permutations and aggregate.

    Returns {
        "n_graphs": int,
        "n_permutations": int,
        "curves": [[step_dict, ...], ...],
        "aggregate": {n_seen: {top1_mean, top3_mean, regret_mean}}
    }
    """
    graphs = _graph_names_from_records(records)
    log.info(f"Cold-start experiment: {len(graphs)} graphs, {n_permutations} permutations")

    rng = random.Random(seed)
    all_curves = []

    for perm_idx in range(n_permutations):
        order = list(graphs)
        rng.shuffle(order)
        log.info(f"Permutation {perm_idx + 1}/{n_permutations}")
        curve = simulate_cold_start(records, order, benchmarks or ["pr", "bfs", "sssp", "cc"])
        all_curves.append(curve)

    # Aggregate across permutations
    aggregate = {}
    max_len = max(len(c) for c in all_curves)
    for step_idx in range(max_len):
        vals = {"top1": [], "top3": [], "regret": []}
        n_seen = None
        for curve in all_curves:
            if step_idx < len(curve):
                entry = curve[step_idx]
                n_seen = entry["n_seen"]
                if entry["top1_accuracy"] is not None:
                    vals["top1"].append(entry["top1_accuracy"])
                if entry["top3_accuracy"] is not None:
                    vals["top3"].append(entry["top3_accuracy"])
                if entry["mean_regret"] is not None:
                    vals["regret"].append(entry["mean_regret"])

        if n_seen is not None and any(vals[k] for k in vals):
            aggregate[n_seen] = {
                "top1_mean": sum(vals["top1"]) / len(vals["top1"]) if vals["top1"] else None,
                "top3_mean": sum(vals["top3"]) / len(vals["top3"]) if vals["top3"] else None,
                "regret_mean": sum(vals["regret"]) / len(vals["regret"]) if vals["regret"] else None,
                "n_samples": len(vals["top1"]),
            }

    return {
        "n_graphs": len(graphs),
        "n_permutations": n_permutations,
        "curves": all_curves,
        "aggregate": aggregate,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cold-Start Learning Curve Simulation",
    )
    parser.add_argument(
        "--benchmark-db",
        default="results/data/benchmarks.json",
        help="Path to benchmarks.json (default: results/data/benchmarks.json)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/vldb_experiments/exp5_cold_start/learning_curve.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10,
        help="Number of random permutations (default: 10)",
    )
    parser.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        default=["pr", "bfs", "sssp", "cc"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load benchmark data
    records = load_benchmark_entries(benchmark_file=args.benchmark_db)
    if not records:
        log.error(f"No benchmark data found in {args.benchmark_db}")
        sys.exit(1)
    log.info(f"Loaded {len(records)} benchmark records")

    # Run simulation
    result = run_cold_start_experiment(
        records,
        n_permutations=args.permutations,
        benchmarks=args.benchmarks,
        seed=args.seed,
    )

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved learning curve data → {args.output}")

    # Print summary
    agg = result["aggregate"]
    print("\n=== Cold-Start Learning Curve Summary ===")
    print(f"{'Seen':>6} {'Top-1':>8} {'Top-3':>8} {'Regret':>8} {'Samples':>8}")
    for n_seen in sorted(int(k) for k in agg):
        entry = agg[str(n_seen)] if str(n_seen) in agg else agg[n_seen]
        print(
            f"{n_seen:>6} "
            + (f"{entry['top1_mean']:>7.1%} " if entry['top1_mean'] is not None else f"{'N/A':>8} ")
            + (f"{entry['top3_mean']:>7.1%} " if entry['top3_mean'] is not None else f"{'N/A':>8} ")
            + (f"{entry['regret_mean']:>7.3f} " if entry['regret_mean'] is not None else f"{'N/A':>8} ")
            + f"{entry['n_samples']:>8}"
        )


if __name__ == "__main__":
    main()
