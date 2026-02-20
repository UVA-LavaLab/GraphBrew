#!/usr/bin/env python3
"""
Weight training and evaluation: train perceptron weights, simulate C++ selection,
report accuracy and regret.

Replaces the legacy eval_weights.py top-level script with a proper lib module
that can be called from graphbrew_experiment.py or standalone.

Usage (standalone):
    python -m scripts.lib.eval_weights
    python -m scripts.lib.eval_weights --sg-only
    python -m scripts.lib.eval_weights --benchmark-file results/benchmark_fresh.json

Usage (library):
    from scripts.lib.eval_weights import train_and_evaluate
    report = train_and_evaluate(sg_only=True)
    print(f"Accuracy: {report['accuracy']:.1%}")
"""

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .utils import (
    BENCHMARKS, BenchmarkResult, RESULTS_DIR, WEIGHTS_DIR, Logger,
    weights_type_path, weights_bench_path,
    VARIANT_PREFIXES, DISPLAY_TO_CANONICAL,
)
from .weights import compute_weights_from_results, cross_validate_logo
from .features import (
    load_graph_properties_cache,
    update_graph_properties,
    save_graph_properties_cache,
)

log = Logger()

# ============================================================================
# Constants
# ============================================================================

# Algorithm name resolution uses DISPLAY_TO_CANONICAL from utils.py (SSOT).
# The old ADAPTIVE_ALGO_MAP is kept as a backward-compat alias.
ADAPTIVE_ALGO_MAP = DISPLAY_TO_CANONICAL

# Variant-prefixed names auto-pass unchanged — imported from SSOT.
_AUTO_PASS_PREFIXES = VARIANT_PREFIXES

# Per-benchmark weight file names to load (from SSOT)
_PER_BENCH_NAMES = list(BENCHMARKS)


# ============================================================================
# Helpers
# ============================================================================

# No longer collapsing variants — comparison at canonical variant level.


def _simulate_score(algo_data: dict, feats: dict) -> float:
    """Mimic C++ scoreBase() — must match reorder_types.h PerceptronWeights::scoreBase() exactly.

    Feature list (17 terms + 3 quadratic + reorder penalty):
      1. bias
      2. w_modularity * modularity
      3. w_log_nodes * log10(nodes+1)
      4. w_log_edges * log10(edges+1)
      5. w_density * density
      6. w_avg_degree * avg_degree/100
      7. w_degree_variance * degree_variance
      8. w_hub_concentration * hub_concentration
      9. w_clustering_coeff * clustering_coefficient
     10. w_avg_path_length * avg_path_length/10   (computed via sampled BFS)
     11. w_diameter * diameter/50                  (computed via sampled BFS)
     12. w_community_count * log10(community_count+1)  (connected components)
     13. w_packing_factor * packing_factor
     14. w_forward_edge_fraction * forward_edge_fraction
     15. w_working_set_ratio * log2(working_set_ratio+1)
     16. w_dv_x_hub * degree_variance * hub_concentration
     17. w_mod_x_logn * modularity * log_nodes
     18. w_pf_x_wsr * packing_factor * log2(wsr+1)
     19. w_reorder_time * reorder_time  (unused in eval — no reorder_time in features)
    """
    s = algo_data.get("bias", 0.5)

    # Core features
    s += algo_data.get("w_modularity", 0) * feats.get("modularity", 0.0)
    log_nodes = feats.get("log_nodes", 5.0)
    log_edges = feats.get("log_edges", 6.0)
    s += algo_data.get("w_log_nodes", 0) * log_nodes
    s += algo_data.get("w_log_edges", 0) * log_edges
    s += algo_data.get("w_density", 0) * feats.get("density", 0.0)
    s += algo_data.get("w_avg_degree", 0) * feats.get("avg_degree", 10.0) / 100.0
    dv = feats.get("degree_variance", 1.0)
    hc = feats.get("hub_concentration", 0.3)
    s += algo_data.get("w_degree_variance", 0) * dv
    s += algo_data.get("w_hub_concentration", 0) * hc

    # Extended features
    s += algo_data.get("w_clustering_coeff", 0) * feats.get("clustering_coefficient", 0.0)
    s += algo_data.get("w_avg_path_length", 0) * feats.get("avg_path_length", 0.0) / 10.0
    s += algo_data.get("w_diameter", 0) * feats.get("diameter", 0.0) / 50.0
    cc_val = feats.get("community_count", 0.0)
    s += algo_data.get("w_community_count", 0) * (math.log10(cc_val + 1) if cc_val > 0 else 0)

    # Locality features (IISWC'18 / GoGraph / P-OPT)
    pf = feats.get("packing_factor", 0.0)
    s += algo_data.get("w_packing_factor", 0) * pf
    s += algo_data.get("w_forward_edge_fraction", 0) * feats.get("forward_edge_fraction", 0.5)
    wsr = feats.get("working_set_ratio", 0.0)
    log_wsr = math.log2(wsr + 1.0)
    s += algo_data.get("w_working_set_ratio", 0) * log_wsr

    # Quadratic interaction terms
    modularity = feats.get("modularity", 0.0)
    s += algo_data.get("w_dv_x_hub", 0) * dv * hc
    s += algo_data.get("w_mod_x_logn", 0) * modularity * log_nodes
    s += algo_data.get("w_pf_x_wsr", 0) * pf * log_wsr

    return s


def _build_features(props: dict) -> dict:
    """Build C++-aligned feature dict from graph properties.

    Includes all features used by C++ scoreBase():
    - Core: modularity, degree_variance, hub_concentration, avg_degree, log_nodes, log_edges, density
    - Extended: clustering_coefficient, avg_path_length, diameter, community_count
      (now computed at runtime via ComputeExtendedFeatures; falls back to 0 if absent)
    - Locality: packing_factor, forward_edge_fraction, working_set_ratio
    """
    nodes = props.get("nodes", 1000)
    edges = props.get("edges", 5000)
    cc = props.get("clustering_coefficient", 0.0)
    avg_degree = props.get("avg_degree", 10.0)
    return {
        "modularity": min(0.9, cc * 1.5),  # C++ estimated_modularity
        "degree_variance": props.get("degree_variance", 1.0),
        "hub_concentration": props.get("hub_concentration", 0.3),
        "avg_degree": avg_degree,
        "log_nodes": math.log10(nodes + 1) if nodes > 0 else 0,
        "log_edges": math.log10(edges + 1) if edges > 0 else 0,
        "density": avg_degree / (nodes - 1) if nodes > 1 else 0,
        "clustering_coefficient": cc,
        # Extended — now computed at C++ runtime via ComputeExtendedFeatures()
        "avg_path_length": props.get("avg_path_length", 0.0),
        "diameter": props.get("diameter_estimate", props.get("diameter", 0.0)),
        "community_count": props.get("community_count", 0.0),
        # Locality features from graph_properties_cache
        "packing_factor": props.get("packing_factor", 0.0),
        "forward_edge_fraction": props.get("forward_edge_fraction", 0.5),
        "working_set_ratio": props.get("working_set_ratio", 0.0),
    }


# ============================================================================
# Core Pipeline
# ============================================================================

@dataclass
class EvalReport:
    """Results of a train-and-evaluate cycle."""
    # Training
    num_entries: int = 0
    num_algorithms: int = 0
    # Evaluation
    correct: int = 0
    total: int = 0
    accuracy: float = 0.0
    unique_predicted: List[str] = field(default_factory=list)
    per_bench_accuracy: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    avg_regret: float = 0.0
    median_regret: float = 0.0
    top2_accuracy: float = 0.0
    predictions: Dict[Tuple[str, str], dict] = field(default_factory=dict)


def load_benchmark_entries(
    results_dir: str = None,
    sg_only: bool = False,
    benchmark_file: str = None,
) -> List[dict]:
    """
    Load raw benchmark entries from JSON files in results_dir.

    Args:
        results_dir: Directory containing benchmark_*.json files
        sg_only: If True, only load .sg benchmark data
        benchmark_file: If set, load only this file

    Returns:
        List of raw entry dicts.
    """
    results_dir = str(results_dir or RESULTS_DIR)

    if benchmark_file:
        with open(benchmark_file) as f:
            entries = json.load(f)
        print(f"Loaded {len(entries)} entries from {benchmark_file}")
        return entries

    bench_files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith("benchmark_") and f.endswith(".json")
    )

    # Identify .sg benchmark files and graphs
    graphs_dir = os.path.join(results_dir, "graphs")
    sg_bench_files = set()
    sg_bench_graphs = set()
    for bf in bench_files:
        path = os.path.join(results_dir, bf)
        try:
            with open(path) as f:
                entries = json.load(f)
            if entries and any(
                e.get("extra") in ("sg", "sg_benchmark") for e in entries
            ):
                sg_bench_files.add(bf)
                sg_bench_graphs.update(
                    e["graph"] for e in entries
                    if e.get("extra") in ("sg", "sg_benchmark")
                )
        except Exception:
            pass
    if sg_bench_graphs:
        print(f"Found .sg benchmark data for {len(sg_bench_graphs)} graphs")

    raw: List[dict] = []
    for bf in bench_files:
        path = os.path.join(results_dir, bf)
        try:
            with open(path) as f:
                entries = json.load(f)
            if bf in sg_bench_files:
                raw.extend(entries)
                print(f"Loaded {len(entries):>6} entries from {bf} [.sg data]")
            elif sg_only:
                print(f"  SKIP {bf} (sg_only mode)")
            else:
                kept = [
                    e for e in entries
                    if e.get("graph", "") not in sg_bench_graphs
                ]
                skipped = len(entries) - len(kept)
                raw.extend(kept)
                msg = f"Loaded {len(kept):>6} entries from {bf}"
                if skipped:
                    msg += f" (skipped {skipped} .sg-graph entries)"
                print(msg)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  SKIP {bf}: {e}")

    print(f"Total: {len(raw)} raw entries from {len(bench_files)} files")
    return raw


def filter_to_benchmark_results(raw: List[dict]) -> List[BenchmarkResult]:
    """Filter raw entries to canonical AdaptiveOrder-selectable BenchmarkResults.

    Names in ADAPTIVE_ALGO_MAP are normalized explicitly.
    Names matching _AUTO_PASS_PREFIXES pass through unchanged (extensible
    for new variants like GraphBrewOrder_newpreset or RABBITORDER_newimpl).
    """
    results = []
    skipped = set()
    for e in raw:
        algo = e["algorithm"]
        canonical = ADAPTIVE_ALGO_MAP.get(algo)
        if canonical is None:
            # Auto-pass variant-prefixed names
            if any(algo.startswith(p) for p in _AUTO_PASS_PREFIXES):
                canonical = algo
            else:
                skipped.add(algo)
                continue
        results.append(BenchmarkResult(
            graph=e["graph"],
            algorithm=canonical,
            algorithm_id=e.get("algorithm_id", 0),
            benchmark=e["benchmark"],
            time_seconds=e["time_seconds"],
            reorder_time=e.get("reorder_time", 0.0),
            trials=e.get("trials", 1),
            success=e.get("success", True),
            error=e.get("error", ""),
            extra=e.get("extra", {}),
        ))
    print(f"After filtering: {len(results)} entries "
          f"({len(skipped)} non-AdaptiveOrder algos skipped)")
    if skipped:
        print(f"  Skipped: {sorted(skipped)}")
    return results


def merge_per_graph_features(results_dir: str = None):
    """Merge per-graph features.json into central cache."""
    results_dir = str(results_dir or RESULTS_DIR)
    graph_props = load_graph_properties_cache(results_dir)
    graphs_dir = os.path.join(results_dir, "graphs")
    merged = 0
    if os.path.isdir(graphs_dir):
        for gname in os.listdir(graphs_dir):
            feat_path = os.path.join(graphs_dir, gname, "features.json")
            if not os.path.isfile(feat_path):
                continue
            try:
                with open(feat_path) as f:
                    gfeats = json.load(f)
                existing = graph_props.get(gname, {})
                for k, v in gfeats.items():
                    if isinstance(v, (int, float)) and v == 0 and existing.get(k, 0) != 0:
                        continue
                    existing[k] = v
                update_graph_properties(gname, existing, results_dir)
                merged += 1
            except Exception:
                pass
        if merged:
            save_graph_properties_cache(results_dir)
            print(f"Merged features from {merged} per-graph features.json "
                  f"({len(graph_props)} total)")


def load_reorder_results(results_dir: str = None) -> List[BenchmarkResult]:
    """Load the latest reorder_*.json file."""
    results_dir = str(results_dir or RESULTS_DIR)
    reorder_files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith("reorder_") and f.endswith(".json")
    )
    if not reorder_files:
        return []
    with open(os.path.join(results_dir, reorder_files[-1])) as f:
        raw_r = json.load(f)
    results = []
    for e in raw_r:
        results.append(BenchmarkResult(
            graph=e.get("graph", ""),
            algorithm=e.get("algorithm", e.get("algorithm_name", "")),
            algorithm_id=0,
            benchmark="reorder",
            time_seconds=e.get("reorder_time", e.get("time_seconds", 0.0)),
            reorder_time=e.get("reorder_time", e.get("time_seconds", 0.0)),
        ))
    print(f"Loaded {len(results)} reorder results")
    return results


def evaluate_predictions(
    bench_results: List[BenchmarkResult],
    weights_dir: str,
    results_dir: str = None,
) -> EvalReport:
    """
    Simulate C++ scoring on benchmark data and compare to oracle.

    Uses per-benchmark weight files (type_0/{bench}.json) when available,
    falling back to type_0/weights.json.

    Returns:
        EvalReport with accuracy, regret, and per-prediction details.
    """
    results_dir = str(results_dir or RESULTS_DIR)
    report = EvalReport()

    # Load type_0/weights.json
    type0_file = weights_type_path('type_0', weights_dir)
    if not os.path.isfile(type0_file):
        log.error(f"type_0 weights not found in {weights_dir}")
        return report
    with open(type0_file) as f:
        saved = json.load(f)
    saved_algos = {k: v for k, v in saved.items() if not k.startswith("_")}

    # Load per-benchmark weight files
    per_bench_weights: Dict[str, dict] = {}
    for bn in _PER_BENCH_NAMES:
        bench_file = weights_bench_path('type_0', bn, weights_dir)
        if os.path.isfile(bench_file):
            with open(bench_file) as f:
                per_bench_weights[bn] = json.load(f)

    # Load graph properties
    graph_props = load_graph_properties_cache(results_dir)

    # Build per-(graph, bench) result lists using total time
    gb_results: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
    for r in bench_results:
        if r.success and r.time_seconds > 0:
            total_time = r.time_seconds + r.reorder_time
            gb_results[(r.graph, r.benchmark)].append((r.algorithm, total_time))

    correct = 0
    total = 0
    predictions: Dict[Tuple[str, str], dict] = {}
    bench_stats: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    for (graph_name, bench), algo_times in gb_results.items():
        if graph_name not in graph_props:
            continue

        feats = _build_features(graph_props[graph_name])

        # Predict: score each algorithm using per-bench weights if available
        best_score = float("-inf")
        predicted_algo = None
        scoring_algos = per_bench_weights.get(bench, saved_algos)
        for algo, data in scoring_algos.items():
            if algo.startswith("_"):
                continue
            score = _simulate_score(data, feats)
            if score > best_score:
                best_score = score
                predicted_algo = algo

        # Ground truth: fastest algorithm by total time
        algo_times.sort(key=lambda x: x[1])
        actual_algo = algo_times[0][0]
        best_time = algo_times[0][1]

        # Find predicted algo's actual time
        pred_time = None
        for a, t in algo_times:
            if a == predicted_algo:
                pred_time = t
                break
        if pred_time is None:
            pred_time = algo_times[-1][1]

        is_correct = predicted_algo == actual_algo
        if is_correct:
            correct += 1
        total += 1

        bench_stats[bench][1] += 1
        if is_correct:
            bench_stats[bench][0] += 1

        predictions[(graph_name, bench)] = {
            "predicted": predicted_algo,
            "actual": actual_algo,
            "pred_time": pred_time,
            "best_time": best_time,
            "correct": is_correct,
        }

    # Compute metrics
    report.correct = correct
    report.total = total
    report.accuracy = correct / total if total else 0
    report.unique_predicted = sorted(
        set(p["predicted"] for p in predictions.values())
    )
    report.per_bench_accuracy = {
        b: tuple(v) for b, v in bench_stats.items()
    }
    report.predictions = predictions

    # Regret analysis
    regrets = []
    for p in predictions.values():
        if p["best_time"] > 0:
            regrets.append((p["pred_time"] - p["best_time"]) / p["best_time"] * 100)
    if regrets:
        report.avg_regret = sum(regrets) / len(regrets)
        regrets_sorted = sorted(regrets)
        report.median_regret = regrets_sorted[len(regrets_sorted) // 2]

    # Top-2 accuracy
    top2 = 0
    for (g, b), p in predictions.items():
        if p["correct"]:
            top2 += 1
        else:
            at = gb_results.get((g, b), [])
            at.sort(key=lambda x: x[1])
            top2_names = [a for a, _ in at[:2]]
            if p["predicted"] in top2_names:
                top2 += 1
    report.top2_accuracy = top2 / total if total else 0

    return report


def print_report(report: EvalReport, weights_dir: str):
    """Print a human-readable evaluation report."""
    type0_file = weights_type_path('type_0', weights_dir)
    if os.path.isfile(type0_file):
        with open(type0_file) as f:
            saved = json.load(f)
        saved_algos = {k: v for k, v in saved.items() if not k.startswith("_")}
        print(f"\nAlgorithms in type_0 weights: {len(saved_algos)}")

        # Weight summary
        print("\n=== Weight Summary (type_0) ===")
        hdr = (f"{'Algorithm':<35} {'Bias':>7} {'w_mod':>7} {'w_logN':>7} "
               f"{'w_logE':>7} {'w_dens':>7} {'w_dv':>7} {'w_hub':>7} {'w_cc':>7}")
        print(hdr)
        for algo in sorted(saved_algos):
            d = saved_algos[algo]
            print(
                f"{algo:<35} {d.get('bias',0):>7.3f} "
                f"{d.get('w_modularity',0):>7.3f} "
                f"{d.get('w_log_nodes',0):>7.3f} "
                f"{d.get('w_log_edges',0):>7.3f} "
                f"{d.get('w_density',0):>7.3f} "
                f"{d.get('w_degree_variance',0):>7.3f} "
                f"{d.get('w_hub_concentration',0):>7.3f} "
                f"{d.get('w_clustering_coeff',0):>7.3f}"
            )

    # Accuracy
    print("\n=== Evaluation ===")
    print(f"Overall accuracy: {report.correct}/{report.total} = {report.accuracy:.1%}")
    print(f"Unique predicted algorithms: {len(report.unique_predicted)}: "
          f"{report.unique_predicted}")

    # Per-benchmark
    print("\nPer-benchmark accuracy:")
    for b in sorted(report.per_bench_accuracy):
        c, t = report.per_bench_accuracy[b]
        print(f"  {b}: {c}/{t} = {c/t:.1%}" if t else f"  {b}: 0/0")

    # Regret
    print(f"\nAverage regret: {report.avg_regret:.1f}%")
    print(f"Median regret: {report.median_regret:.1f}%")
    print(f"Top-2 accuracy: {report.top2_accuracy:.1%}")

    # Worst predictions
    worst = sorted(
        report.predictions.items(),
        key=lambda x: (x[1]["pred_time"] - x[1]["best_time"])
        / max(x[1]["best_time"], 0.0001),
        reverse=True,
    )[:10]
    if worst:
        print("\nWorst predictions (highest regret):")
        for (g, b), p in worst:
            regret = (
                (p["pred_time"] - p["best_time"]) / p["best_time"] * 100
                if p["best_time"] > 0 else 0
            )
            print(f"  {g}/{b}: predicted={p['predicted']}, "
                  f"actual={p['actual']}, regret={regret:.0f}%")


def train_and_evaluate(
    results_dir: str = None,
    weights_dir: str = None,
    sg_only: bool = False,
    benchmark_file: str = None,
    logo: bool = False,
) -> EvalReport:
    """
    Complete pipeline: load data → merge features → train weights → evaluate.

    This is the single-call replacement for the old eval_weights.py script.

    Args:
        results_dir: Directory with benchmark_*.json and graphs/
        weights_dir: Directory for weight files (type_0/weights.json, type_0/{bench}.json)
        sg_only: Only use .sg benchmark data
        benchmark_file: Load a specific benchmark JSON file instead of auto-discover
        logo: Run Leave-One-Graph-Out cross-validation to measure generalization

    Returns:
        EvalReport with all accuracy/regret metrics.
    """
    results_dir = str(results_dir or RESULTS_DIR)
    if weights_dir is None:
        weights_dir = str(WEIGHTS_DIR)

    # 1. Load benchmark entries
    raw = load_benchmark_entries(results_dir, sg_only=sg_only,
                                 benchmark_file=benchmark_file)
    bench_results = filter_to_benchmark_results(raw)

    # 2. Merge per-graph features
    merge_per_graph_features(results_dir)

    # 3. Load reorder results
    reorder_results = load_reorder_results(results_dir)

    # 4. Train weights
    print("\n=== Training weights ===")
    weights = compute_weights_from_results(
        benchmark_results=bench_results,
        reorder_results=reorder_results,
        weights_dir=weights_dir,
    )
    algo_weights = {k: v for k, v in weights.items() if not k.startswith("_")}
    print(f"\nAlgorithms in weights: {len(algo_weights)}")

    # 5. Evaluate (in-sample)
    print("\n=== Simulating C++ adaptive selection ===")
    report = evaluate_predictions(bench_results, weights_dir, results_dir)
    report.num_entries = len(bench_results)
    report.num_algorithms = len(algo_weights)

    # 6. Print report
    print_report(report, weights_dir)

    # 7. LOGO cross-validation (generalization accuracy)
    if logo:
        print("\n" + "=" * 60)
        print("LEAVE-ONE-GRAPH-OUT CROSS-VALIDATION")
        print("=" * 60)
        print("Training N separate models (one per graph), each time")
        print("predicting on the held-out graph it has never seen.\n")

        logo_result = cross_validate_logo(
            bench_results,
            reorder_results=reorder_results,
            weights_dir=weights_dir,
        )

        print("\n=== LOGO Results (Generalization Accuracy) ===")
        print(f"Graphs:             {logo_result['num_graphs']}")
        print(f"Predictions:        {logo_result['correct']}/{logo_result['total']}")
        print(f"LOGO Accuracy:      {logo_result['accuracy']:.1%}  "
              "(on UNSEEN graphs)")
        print(f"In-sample Accuracy: {logo_result['full_training_accuracy']:.1%}  "
              "(on training graphs)")
        gap = logo_result['overfitting_score']
        print(f"Overfit gap:        {gap:.1%}  "
              f"({'⚠ possible overfitting' if gap > 0.2 else '✓ OK'})")
        print(f"Avg regret:         {logo_result['avg_regret']:.1f}%")
        print(f"Median regret:      {logo_result['median_regret']:.1f}%")

        # Per-graph breakdown
        if logo_result.get('per_graph'):
            print("\nPer-graph breakdown:")
            for g in sorted(logo_result['per_graph']):
                pg = logo_result['per_graph'][g]
                mark = "✓" if pg['accuracy'] == 1.0 else "✗" if pg['accuracy'] == 0.0 else "~"
                print(f"  {mark} {g:<30} {pg['correct']}/{pg['total']}  "
                      f"regret={pg['avg_regret']:.1f}%")

    return report


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train perceptron weights and evaluate prediction accuracy",
    )
    parser.add_argument("--results-dir", default=None,
                        help="Directory with benchmark results")
    parser.add_argument("--weights-dir", default=None,
                        help="Directory for weight files")
    parser.add_argument("--sg-only", action="store_true",
                        help="Only use .sg benchmark data for training")
    parser.add_argument("--benchmark-file", default=None,
                        help="Load a specific benchmark JSON file")
    parser.add_argument("--logo", action="store_true",
                        help="Run Leave-One-Graph-Out cross-validation (generalization accuracy)")
    args = parser.parse_args()

    train_and_evaluate(
        results_dir=args.results_dir,
        weights_dir=args.weights_dir,
        sg_only=args.sg_only,
        benchmark_file=args.benchmark_file,
        logo=args.logo,
    )


if __name__ == "__main__":
    main()
