#!/usr/bin/env python3
"""
Model x Criterion evaluation of all adaptive prediction strategies.

Evaluates every combination of:
  - Models:    Perceptron, Decision Tree, Hybrid (DT+Perceptron), Database kNN
  - Criteria:  Fastest-Reorder, Fastest-Execution, Best-E2E, Best-Amortization

Additional evaluations:
  - LOGO cross-validation (leave-one-graph-out) for each model
  - Weight / feature importance analysis (--weights flag)
  - Per-benchmark breakdown

Usage:
  python3 scripts/evaluate_all_modes.py                # In-sample, all criteria
  python3 scripts/evaluate_all_modes.py --logo          # Add LOGO CV
  python3 scripts/evaluate_all_modes.py --all           # Everything
  python3 scripts/evaluate_all_modes.py --weights       # Weight analysis
  python3 scripts/evaluate_all_modes.py --json          # JSON output
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.core.utils import RESULTS_DIR, WEIGHTS_DIR, BENCHMARKS, Logger
from scripts.lib.core.datastore import get_benchmark_store
from scripts.lib.ml.model_tree import (
    Criterion, compute_oracle, criterion_value,
    extract_dt_features, train_all_models, load_adaptive_models,
    cross_validate_logo_model_tree,
)

log = Logger()

# All criteria to evaluate
ALL_CRITERIA = [
    Criterion.FASTEST_REORDER,
    Criterion.FASTEST_EXECUTION,
    Criterion.BEST_ENDTOEND,
    Criterion.BEST_AMORTIZATION,
]

# Short labels
CRITERION_LABELS = {
    Criterion.FASTEST_REORDER: 'F-Reorder',
    Criterion.FASTEST_EXECUTION: 'F-Execution',
    Criterion.BEST_ENDTOEND: 'E2E',
    Criterion.BEST_AMORTIZATION: 'Amortize',
}


# ===================================================================
# Data Loading
# ===================================================================

def load_data():
    """Load benchmark results and graph properties."""
    from scripts.lib.ml.eval_weights import (
        load_benchmark_entries,
        filter_to_benchmark_results,
        load_reorder_results,
    )
    from scripts.lib.ml.features import load_graph_properties_cache

    raw = load_benchmark_entries(str(RESULTS_DIR))
    if not raw:
        print("ERROR: No benchmark data found.")
        sys.exit(1)

    bench_results = filter_to_benchmark_results(raw)
    reorder_results = load_reorder_results(str(RESULTS_DIR))
    graph_props = load_graph_properties_cache(str(RESULTS_DIR))

    graphs = sorted(set(r.graph for r in bench_results if r.success))
    benchmarks_found = sorted(set(r.benchmark for r in bench_results if r.success))
    algos = sorted(set(r.algorithm for r in bench_results if r.success))

    print("=" * 80)
    print("  GRAPHBREW -- MODEL x CRITERION EVALUATION")
    print("=" * 80)
    print(f"\n  Records:    {len(bench_results)}")
    print(f"  Graphs:     {len(graphs)}  {graphs}")
    print(f"  Benchmarks: {len(benchmarks_found)}  {benchmarks_found}")
    print(f"  Algorithms: {len(algos)}")
    print(f"  Graph props: {len(graph_props)} graphs with features")

    # DON-RL check
    donrl = sum(1 for p in graph_props.values()
                if p.get('vertex_significance_skewness', 0) != 0
                or p.get('window_neighbor_overlap', 0) != 0)
    print(f"  DON-RL:     {donrl}/{len(graph_props)} graphs")
    print()

    # Convert bench_results to raw dicts for model_tree functions
    raw_records = []
    for r in bench_results:
        if r.success and r.time_seconds > 0:
            raw_records.append({
                'graph': r.graph,
                'benchmark': r.benchmark,
                'algorithm': r.algorithm,
                'time_seconds': r.time_seconds,
                'reorder_time': r.reorder_time,
                'success': True,
            })

    return bench_results, reorder_results, graph_props, raw_records


# ===================================================================
# Build Oracle per (graph, benchmark, criterion)
# ===================================================================

def build_oracle_table(raw_records, graph_props, criteria=None):
    """Build oracle lookup: (graph, bench, criterion) -> (best_algo, best_value).

    Returns:
        dict[criterion][(graph, bench)] = {'algo': str, 'value': float, 'records': list}
    """
    criteria = criteria or ALL_CRITERIA

    # Group records by (graph, bench)
    gb = defaultdict(list)
    for r in raw_records:
        g = r.get('graph', '')
        if g == 'tiny' or g not in graph_props:
            continue
        gb[(g, r['benchmark'])].append(r)

    oracles = {c: {} for c in criteria}
    for crit in criteria:
        for (g, b), records in gb.items():
            algo, val = compute_oracle(records, crit)
            oracles[crit][(g, b)] = {
                'algo': algo, 'value': val, 'records': records,
            }

    return oracles


# ===================================================================
# Evaluate a model's predictions against each criterion's oracle
# ===================================================================

def _algo_matches(predicted: str, oracle_algo: str, granularity: str = 'family') -> bool:
    """Check if prediction matches oracle at the given granularity.

    Granularity levels:
        ``'individual'``: Exact match or variant match (X == X_csr).
        ``'family'``:     Both sides mapped to family must agree.
        ``'topn'``:       Same as ``'individual'``.

    E.g. 'RABBITORDER' matches 'RABBITORDER_csr' at individual level.
         'GORDER' matches 'CORDER' at family level (both → GORDER family).
    """
    if predicted == oracle_algo:
        return True

    if granularity == 'family':
        # Import here to avoid circular imports at module level
        from scripts.lib.ml.adaptive_emulator import algo_to_family
        return algo_to_family(predicted) == algo_to_family(oracle_algo)

    # individual / topn — variant-aware exact match
    if oracle_algo.startswith(predicted + '_') or oracle_algo.startswith(predicted + '+'):
        return True
    if predicted.startswith(oracle_algo + '_') or predicted.startswith(oracle_algo + '+'):
        return True
    return False


def _find_predicted_value(predicted: str, records: list, crit, original_exec: float):
    """Find criterion value for predicted algo in records (family-aware).

    Uses ``algo_to_family()`` for family-level matching so that e.g.
    a prediction of ``"RABBIT"`` finds the best record among all RABBIT
    family members (``RABBITORDER``, ``RABBITORDER_csr``, …).
    """
    from scripts.lib.ml.adaptive_emulator import algo_to_family

    # Exact match first
    for r in records:
        if r.get('algorithm') == predicted:
            return criterion_value(r, crit, original_exec)

    # Family-aware match: find best record in the same family
    pred_family = algo_to_family(predicted)
    best_val = None
    for r in records:
        algo = r.get('algorithm', '')
        if algo_to_family(algo) == pred_family:
            val = criterion_value(r, crit, original_exec)
            if best_val is None or val < best_val:
                best_val = val

    return best_val


def evaluate_model_vs_criteria(
    predictions,   # dict[(graph, bench)] -> predicted_algo
    oracles,       # from build_oracle_table
    criteria=None,
    granularity: str = 'family',
):
    """Score a model's predictions against all criteria oracles.

    Handles both full algorithm names and family names (kNN returns families).

    Args:
        granularity: ``'family'``, ``'topn'``, or ``'individual'`` — controls
            how predictions are matched against the oracle.

    Returns:
        dict[criterion] -> {top1, total, avg_regret, median_regret}
    """
    criteria = criteria or ALL_CRITERIA
    results = {}

    for crit in criteria:
        correct = 0
        total = 0
        regrets = []

        for (g, b), oracle_info in oracles[crit].items():
            if (g, b) not in predictions:
                continue

            predicted = predictions[(g, b)]
            oracle_algo = oracle_info['algo']
            oracle_val = oracle_info['value']
            records = oracle_info['records']

            total += 1
            if _algo_matches(predicted, oracle_algo, granularity):
                correct += 1

            # Find predicted algo's value for this criterion
            original_exec = None
            for r in records:
                if r.get('algorithm') == 'ORIGINAL':
                    original_exec = r.get('time_seconds', 999)

            pred_val = _find_predicted_value(predicted, records, crit, original_exec)

            if pred_val is None:
                # Predicted algo not in data — use worst valid value
                vals = [criterion_value(r, crit, original_exec) for r in records
                        if criterion_value(r, crit, original_exec) < float('inf')]
                pred_val = max(vals) if vals else oracle_val

            if oracle_val > 0 and oracle_val < float('inf') and pred_val < float('inf'):
                regrets.append((pred_val - oracle_val) / oracle_val * 100)

        avg_r = sum(regrets) / len(regrets) if regrets else 0
        sorted_r = sorted(regrets) if regrets else [0]
        med_r = sorted_r[len(sorted_r) // 2]

        results[crit] = {
            'correct': correct,
            'total': total,
            'top1': correct / total if total else 0,
            'avg_regret': avg_r,
            'median_regret': med_r,
        }

    return results


# ===================================================================
# Model: Perceptron predictions
# ===================================================================

def predict_perceptron(bench_results, graph_props):
    """Get perceptron predictions for each (graph, bench) task.

    Returns:
        dict[(graph, bench)] -> predicted algorithm name
    """
    from scripts.lib.ml.eval_weights import _build_features, _simulate_score
    from scripts.lib.core.utils import weights_type_path, weights_bench_path

    graph_props_dict = graph_props
    weights_dir = str(WEIGHTS_DIR)

    type0_file = weights_type_path('type_0', weights_dir)
    if not os.path.isfile(type0_file):
        return {}
    with open(type0_file) as f:
        saved = json.load(f)
    saved_algos = {k: v for k, v in saved.items() if not k.startswith('_')}

    per_bench_weights = {}
    for bn in BENCHMARKS:
        bfile = weights_bench_path('type_0', bn, weights_dir)
        if os.path.isfile(bfile):
            with open(bfile) as f:
                per_bench_weights[bn] = json.load(f)

    seen_keys = set()
    predictions = {}
    for r in bench_results:
        if not r.success or r.time_seconds <= 0:
            continue
        key = (r.graph, r.benchmark)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if r.graph not in graph_props_dict or r.graph == 'tiny':
            continue

        feats = _build_features(graph_props_dict[r.graph])
        scoring = per_bench_weights.get(r.benchmark, saved_algos)

        best_score = float('-inf')
        best_algo = None
        for algo, data in scoring.items():
            if algo.startswith('_'):
                continue
            score = _simulate_score(data, feats, r.benchmark)
            if score > best_score:
                best_score = score
                best_algo = algo

        if best_algo:
            predictions[key] = best_algo

    return predictions


# ===================================================================
# Model: Decision Tree / Hybrid predictions
# ===================================================================

def predict_model_tree(raw_records, graph_props, model_type='decision_tree',
                       criterion=Criterion.FASTEST_EXECUTION):
    """Train DT or Hybrid on all data, predict on all data (in-sample).

    Returns:
        dict[(graph, bench)] -> predicted algorithm name
    """
    from scripts.lib.ml.model_tree import (
        train_decision_tree, train_hybrid_tree, extract_dt_features,
    )

    benchmarks = sorted(set(r['benchmark'] for r in raw_records))
    train_fn = train_decision_tree if model_type == 'decision_tree' else train_hybrid_tree

    predictions = {}
    for bench in benchmarks:
        tree = train_fn(raw_records, graph_props, bench, criterion)

        for g in sorted(set(r['graph'] for r in raw_records if r['benchmark'] == bench)):
            if g not in graph_props or g == 'tiny':
                continue
            key = (g, bench)
            if key in predictions:
                continue
            feats = extract_dt_features(graph_props[g])
            predictions[key] = tree.predict(feats)

    return predictions


# ===================================================================
# Model: Database kNN predictions (criterion-aware)
# ===================================================================

def predict_database_knn(raw_records, graph_props, criterion=Criterion.FASTEST_EXECUTION):
    """Get database kNN predictions using the emulator's select_for_mode.

    Database kNN is criterion-aware: it has raw timing data and can
    pick the best algorithm for any criterion by re-ranking kNN scores.
    """
    from scripts.lib.ml.adaptive_emulator import DatabaseSelector, SelectionCriterion

    crit_to_sel = {
        Criterion.FASTEST_REORDER: SelectionCriterion.FASTEST_REORDER,
        Criterion.FASTEST_EXECUTION: SelectionCriterion.FASTEST_EXECUTION,
        Criterion.BEST_ENDTOEND: SelectionCriterion.BEST_ENDTOEND,
        Criterion.BEST_AMORTIZATION: SelectionCriterion.BEST_AMORTIZATION,
    }
    sel_crit = crit_to_sel.get(criterion, SelectionCriterion.FASTEST_EXECUTION)

    emulator = DatabaseSelector()

    # Get benchmarks and graphs from raw records
    gb = defaultdict(list)
    for r in raw_records:
        g = r.get('graph', '')
        if g == 'tiny' or g not in graph_props:
            continue
        gb[(g, r['benchmark'])].append(r)

    predictions = {}
    for (g, b) in gb:
        # Use select_for_mode which handles all criteria via kNN
        # Pass graph_name=None to force kNN (not oracle) for fair evaluation
        result = emulator.select_for_mode(
            features=graph_props[g],
            benchmark=b,
            graph_name=None,  # force kNN, skip oracle shortcut
            criterion=sel_crit,
        )
        if result:
            predictions[(g, b)] = result[0]

    return predictions


# ===================================================================
# Section 1: In-Sample Model x Criterion Matrix
# ===================================================================

def eval_insample_matrix(bench_results, raw_records, graph_props, reorder_results):
    """Evaluate all models in-sample against all criteria."""
    from scripts.lib.ml.weights import compute_weights_from_results

    print("=" * 80)
    print("  [1] IN-SAMPLE: Model x Criterion Matrix")
    print("=" * 80)
    print()

    t0 = time.time()

    # Ensure perceptron weights exist
    compute_weights_from_results(
        benchmark_results=bench_results,
        reorder_results=reorder_results,
        weights_dir=str(WEIGHTS_DIR),
    )

    # Build oracle tables for all criteria
    oracles = build_oracle_table(raw_records, graph_props)

    # Print oracle stats
    for crit in ALL_CRITERIA:
        n = len(oracles[crit])
        print(f"  Oracle tasks ({CRITERION_LABELS[crit]}): {n}")
    print()

    # Train DT and Hybrid (in-sample, per-criterion)
    # NOTE: Previously trained only on FASTEST_EXECUTION then evaluated
    # against all 4 criteria — fixing this to train per-criterion models.
    models_path = Path(RESULTS_DIR) / 'data' / 'adaptive_models.json'
    trained_models = train_all_models(
        raw_records, graph_props,
        criterion=Criterion.FASTEST_EXECUTION,
        save_path=models_path,
    )
    print(f"  Trained DT/Hybrid models, saved to {models_path}")

    # Reset model tree cache
    from scripts.lib.ml.adaptive_emulator import AdaptiveOrderEmulator
    AdaptiveOrderEmulator._model_tree_cache = None

    # -- Generate predictions for each model --
    model_preds = {}
    model_preds['Perceptron'] = predict_perceptron(bench_results, graph_props)
    print(f"  Perceptron predictions: {len(model_preds['Perceptron'])} tasks")

    # DT and Hybrid: criterion-aware (train per-criterion, merge)
    for model_type, label in [('decision_tree', 'Decision Tree'),
                               ('hybrid', 'Hybrid (DT+Perc)')]:
        model_preds[label] = predict_model_tree(
            raw_records, graph_props, model_type, Criterion.FASTEST_EXECUTION)
    print(f"  Decision Tree predictions: {len(model_preds['Decision Tree'])} tasks")
    print(f"  Hybrid predictions: {len(model_preds['Hybrid (DT+Perc)'])} tasks")

    # -- Evaluate each model against each criterion's oracle --
    matrix = {}
    for model_name, preds in model_preds.items():
        matrix[model_name] = evaluate_model_vs_criteria(preds, oracles)

    # DT and Hybrid: criterion-aware evaluation (train per criterion)
    for model_type, label in [('decision_tree', 'Decision Tree'),
                               ('hybrid', 'Hybrid (DT+Perc)')]:
        for crit in ALL_CRITERIA:
            crit_preds = predict_model_tree(
                raw_records, graph_props, model_type, crit)
            crit_results = evaluate_model_vs_criteria(crit_preds, oracles, [crit])
            matrix[label][crit] = crit_results[crit]

    # Database kNN: criterion-aware, generate per-criterion predictions
    matrix['Database kNN'] = {}
    for crit in ALL_CRITERIA:
        db_preds = predict_database_knn(raw_records, graph_props, crit)
        crit_results = evaluate_model_vs_criteria(db_preds, oracles, [crit])
        matrix['Database kNN'][crit] = crit_results[crit]
        n_preds = len(db_preds)

    print(f"  Database kNN predictions: {n_preds} tasks")
    print()

    elapsed = time.time() - t0
    _print_model_criterion_matrix(matrix, "IN-SAMPLE (Top-1 Accuracy)")
    _print_regret_matrix(matrix, "IN-SAMPLE (Avg Regret %)")
    print(f"  Time: {elapsed:.1f}s\n")

    return matrix


# ===================================================================
# Section 2: LOGO Cross-Validation
# ===================================================================

def eval_logo_all_models(bench_results, raw_records, graph_props, reorder_results):
    """LOGO cross-validation for all models."""
    from scripts.lib.ml.weights import cross_validate_logo

    print("=" * 80)
    print("  [2] LOGO CROSS-VALIDATION: Model x Criterion")
    print("=" * 80)
    print()

    logo_matrix = {}

    # -- Perceptron LOGO --
    print("  Training Perceptron LOGO (E2E criterion)...", flush=True)
    t0 = time.time()
    logo_result = cross_validate_logo(
        bench_results,
        reorder_results=reorder_results,
        weights_dir=str(WEIGHTS_DIR),
    )
    elapsed = time.time() - t0

    logo_matrix['Perceptron'] = {
        Criterion.BEST_ENDTOEND: {
            'correct': logo_result['correct'],
            'total': logo_result['total'],
            'top1': logo_result['accuracy'],
            'avg_regret': logo_result['avg_regret'],
            'median_regret': logo_result['median_regret'],
        }
    }
    print(f"    Perceptron LOGO (E2E): {logo_result['accuracy']:.1%}  "
          f"regret={logo_result['avg_regret']:.1f}%  [{elapsed:.1f}s]")

    if logo_result.get('per_graph'):
        for g in sorted(logo_result['per_graph']):
            pg = logo_result['per_graph'][g]
            mark = "+" if pg['accuracy'] == 1.0 else "-" if pg['accuracy'] == 0.0 else "~"
            print(f"      {mark} {g:<28} {pg['correct']}/{pg['total']}"
                  f"  regret={pg['avg_regret']:.1f}%")

    # -- DT and Hybrid LOGO for each criterion --
    for model_type, label in [('decision_tree', 'Decision Tree'),
                               ('hybrid', 'Hybrid (DT+Perc)')]:
        logo_matrix[label] = {}
        for crit in ALL_CRITERIA:
            print(f"  Training {label} LOGO ({CRITERION_LABELS[crit]})...",
                  end='', flush=True)
            t0 = time.time()
            result = cross_validate_logo_model_tree(
                raw_records, graph_props,
                model_type=model_type,
                criterion=crit,
            )
            elapsed = time.time() - t0

            logo_matrix[label][crit] = {
                'correct': result['correct'],
                'total': result['total'],
                'top1': result['accuracy'],
                'avg_regret': result['avg_regret'],
                'median_regret': result['median_regret'],
            }
            print(f"  {result['accuracy']:.1%}  "
                  f"regret={result['avg_regret']:.1f}%  [{elapsed:.1f}s]")

            if result.get('per_graph'):
                for g in sorted(result['per_graph']):
                    pg = result['per_graph'][g]
                    mark = "+" if pg['accuracy'] == 1.0 else "-" if pg['accuracy'] == 0.0 else "~"
                    print(f"      {mark} {g:<28} {pg['correct']}/{pg['total']}"
                          f"  regret={pg['avg_regret']:.1f}%")

    # Database kNN: oracle for known graphs, no LOGO needed
    print("\n  Database kNN: N/A (oracle on known graphs, kNN on unknown)")
    print()

    _print_model_criterion_matrix(logo_matrix, "LOGO CV (Top-1 Accuracy)")
    _print_regret_matrix(logo_matrix, "LOGO CV (Avg Regret %)")
    return logo_matrix


# ===================================================================
# Section 3: Weight & Feature Analysis
# ===================================================================

def analyze_weights(graph_props, raw_records):
    """Comprehensive weight and feature importance analysis."""
    from scripts.lib.core.utils import weights_type_path, weights_bench_path

    print("=" * 80)
    print("  [3] WEIGHT & FEATURE ANALYSIS")
    print("=" * 80)
    print()

    weights_dir = str(WEIGHTS_DIR)

    # -- Per-benchmark weight heatmap --
    print("  Per-Benchmark Weight Heatmap (avg |weight| across algorithms):")
    print("  " + "-" * 70)

    weight_keys = [
        'w_modularity', 'w_degree_variance', 'w_hub_concentration',
        'w_log_nodes', 'w_log_edges', 'w_density', 'w_avg_degree',
        'w_clustering_coeff', 'w_avg_path_length', 'w_diameter',
        'w_community_count', 'w_packing_factor', 'w_forward_edge_fraction',
        'w_working_set_ratio', 'w_vertex_significance_skewness',
        'w_window_neighbor_overlap',
    ]
    short_names = [
        'mod', 'dv', 'hub', 'logN', 'logE', 'dens', 'deg',
        'clust', 'path', 'diam', 'comm', 'pack', 'fef',
        'wsr', 'vss', 'wno',
    ]

    bench_weights = {}
    for bn in BENCHMARKS:
        bfile = weights_bench_path('type_0', bn, weights_dir)
        if os.path.isfile(bfile):
            with open(bfile) as f:
                bench_weights[bn] = json.load(f)

    if bench_weights:
        heatmap = {}
        for bn, bdata in bench_weights.items():
            algos = {k: v for k, v in bdata.items() if not k.startswith('_')}
            fn_weights = defaultdict(list)
            for algo, ad in algos.items():
                if not isinstance(ad, dict):
                    continue
                for wk in weight_keys:
                    fn_weights[wk].append(abs(ad.get(wk, 0)))
            heatmap[bn] = {wk: (sum(v) / len(v) if v else 0)
                           for wk, v in fn_weights.items()}

        # Header
        hdr = f"  {'Feature':<8}"
        for bn in sorted(bench_weights):
            hdr += f" {bn:>8}"
        print(hdr)
        print("  " + "-" * (8 + 9 * len(bench_weights)))

        for wk, sn in zip(weight_keys, short_names):
            row = f"  {sn:<8}"
            for bn in sorted(bench_weights):
                val = heatmap[bn].get(wk, 0)
                row += f" {val:>8.2f}"
            print(row)
        print()

    # -- Feature importance via permutation --
    print("  Feature Importance (permutation shuffle on DT, F-Execution):")
    print("  " + "-" * 70)
    _eval_feature_importance(raw_records, graph_props)
    print()


def _eval_feature_importance(raw_records, graph_props):
    """Rank features by permutation importance on DT accuracy."""
    import random
    from scripts.lib.ml.model_tree import (
        train_decision_tree, DT_FEATURE_NAMES, extract_dt_features,
    )

    benchmarks = sorted(set(r['benchmark'] for r in raw_records))

    # Baseline accuracy
    base_correct = 0
    base_total = 0
    for bench in benchmarks:
        tree = train_decision_tree(raw_records, graph_props, bench,
                                    Criterion.FASTEST_EXECUTION)
        for g in sorted(set(r['graph'] for r in raw_records
                            if r['benchmark'] == bench)):
            if g not in graph_props or g == 'tiny':
                continue
            preds = tree.predict_from_props(graph_props[g])
            records = [r for r in raw_records
                       if r['graph'] == g and r['benchmark'] == bench
                       and r.get('success') and r.get('time_seconds', 0) > 0]
            if not records:
                continue
            oracle, _ = compute_oracle(records, Criterion.FASTEST_EXECUTION)
            base_total += 1
            if preds == oracle:
                base_correct += 1

    base_acc = base_correct / base_total if base_total else 0
    print(f"  Baseline DT accuracy (in-sample): {base_acc:.1%} ({base_correct}/{base_total})")

    # Permute each feature
    feature_importance = {}
    random.seed(42)

    graphs = sorted(g for g in graph_props if g != 'tiny')

    for feat_idx, feat_name in enumerate(DT_FEATURE_NAMES):
        feat_values = [extract_dt_features(graph_props[g])[feat_idx] for g in graphs]
        shuffled = feat_values.copy()
        random.shuffle(shuffled)
        shuffle_map = dict(zip(graphs, shuffled))

        perm_correct = 0
        perm_total = 0

        for bench in benchmarks:
            tree = train_decision_tree(raw_records, graph_props, bench,
                                        Criterion.FASTEST_EXECUTION)
            for g in sorted(set(r['graph'] for r in raw_records
                                if r['benchmark'] == bench)):
                if g not in graph_props or g == 'tiny':
                    continue
                feats = extract_dt_features(graph_props[g])
                feats[feat_idx] = shuffle_map.get(g, feats[feat_idx])
                preds = tree.predict(feats)
                records = [r for r in raw_records
                           if r['graph'] == g and r['benchmark'] == bench
                           and r.get('success') and r.get('time_seconds', 0) > 0]
                if not records:
                    continue
                oracle, _ = compute_oracle(records, Criterion.FASTEST_EXECUTION)
                perm_total += 1
                if preds == oracle:
                    perm_correct += 1

        perm_acc = perm_correct / perm_total if perm_total else 0
        drop = base_acc - perm_acc
        feature_importance[feat_name] = drop

    # Sort by importance
    ranked = sorted(feature_importance.items(), key=lambda x: -x[1])
    for feat_name, drop in ranked:
        bar_len = max(0, int(drop * 200))
        bar = "#" * min(bar_len, 40)
        indicator = "+" if drop > 0.01 else "." if drop > -0.01 else "-"
        print(f"    {indicator} {feat_name:<25} acc_drop = {drop:>+7.1%}  {bar}")

    print(f"\n  Note: DON-RL features (vss, wno) are in 14D kNN / 21D perceptron,")
    print(f"        but NOT in the 12D DT feature space.")


# ===================================================================
# Printing helpers
# ===================================================================

def _print_model_criterion_matrix(matrix, label=""):
    """Print a Model x Criterion accuracy table."""
    col_w = 14
    model_w = 22

    print(f"  {label}")
    header = f"  {'Model':<{model_w}}"
    for crit in ALL_CRITERIA:
        header += f" {CRITERION_LABELS[crit]:>{col_w}}"
    print(header)
    print("  " + "-" * (model_w + col_w * len(ALL_CRITERIA) + len(ALL_CRITERIA)))

    # Oracle row
    row = f"  {'Oracle':<{model_w}}"
    for _ in ALL_CRITERIA:
        row += f" {'100.0%':>{col_w}}"
    print(row)

    # Model rows
    model_order = ['Perceptron', 'Decision Tree', 'Hybrid (DT+Perc)', 'Database kNN']
    for model_name in model_order:
        if model_name not in matrix:
            continue
        row = f"  {model_name:<{model_w}}"
        for crit in ALL_CRITERIA:
            data = matrix[model_name].get(crit)
            if data is None:
                row += f" {'--':>{col_w}}"
            else:
                acc = f"{data['top1']:.1%}"
                n = data.get('total', 0)
                cell = f"{acc} ({n})"
                row += f" {cell:>{col_w}}"
        print(row)

    print()


def _print_regret_matrix(matrix, label=""):
    """Print a Model x Criterion regret table."""
    col_w = 14
    model_w = 22

    print(f"  {label}")
    header = f"  {'Model':<{model_w}}"
    for crit in ALL_CRITERIA:
        header += f" {CRITERION_LABELS[crit]:>{col_w}}"
    print(header)
    print("  " + "-" * (model_w + col_w * len(ALL_CRITERIA) + len(ALL_CRITERIA)))

    model_order = ['Perceptron', 'Decision Tree', 'Hybrid (DT+Perc)', 'Database kNN']
    for model_name in model_order:
        if model_name not in matrix:
            continue
        row = f"  {model_name:<{model_w}}"
        for crit in ALL_CRITERIA:
            data = matrix[model_name].get(crit)
            if data is None:
                row += f" {'--':>{col_w}}"
            else:
                reg = f"{data['avg_regret']:.1f}%"
                row += f" {reg:>{col_w}}"
        print(row)

    print()


def print_final_summary(insample_matrix, logo_matrix=None):
    """Print combined summary."""
    print("=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    print()

    _print_model_criterion_matrix(insample_matrix, "In-Sample Accuracy")

    if logo_matrix:
        _print_model_criterion_matrix(logo_matrix, "LOGO CV Accuracy")

    # Best model per criterion
    print("  Best model per criterion (in-sample):")
    for crit in ALL_CRITERIA:
        best_model = None
        best_acc = -1
        for model_name, results in insample_matrix.items():
            data = results.get(crit)
            if data and data['top1'] > best_acc:
                best_acc = data['top1']
                best_model = model_name
        if best_model:
            data = insample_matrix[best_model][crit]
            print(f"    {CRITERION_LABELS[crit]:<14}: {best_model} "
                  f"({best_acc:.1%}, regret={data['avg_regret']:.1f}%)")

    if logo_matrix:
        print("\n  Best model per criterion (LOGO):")
        for crit in ALL_CRITERIA:
            best_model = None
            best_acc = -1
            for model_name, results in logo_matrix.items():
                data = results.get(crit)
                if data and data['top1'] > best_acc:
                    best_acc = data['top1']
                    best_model = model_name
            if best_model:
                data = logo_matrix[best_model][crit]
                print(f"    {CRITERION_LABELS[crit]:<14}: {best_model} "
                      f"({best_acc:.1%}, regret={data['avg_regret']:.1f}%)")
    print()


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model x Criterion evaluation of adaptive reordering"
    )
    parser.add_argument("--logo", action="store_true",
                        help="Include LOGO cross-validation")
    parser.add_argument("--all", action="store_true",
                        help="Run everything: LOGO + weight analysis")
    parser.add_argument("--weights", action="store_true",
                        help="Run weight & feature importance analysis")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    bench_results, reorder_results, graph_props, raw_records = load_data()

    # 1. In-sample
    insample = eval_insample_matrix(
        bench_results, raw_records, graph_props, reorder_results)

    # 2. LOGO
    logo = None
    if args.logo or args.all:
        logo = eval_logo_all_models(
            bench_results, raw_records, graph_props, reorder_results)

    # 3. Weight analysis
    if args.weights or args.all:
        analyze_weights(graph_props, raw_records)

    # Final summary
    print_final_summary(insample, logo)

    # JSON
    if args.json:
        def _serialize(matrix):
            out = {}
            for model, results in matrix.items():
                out[model] = {}
                for crit, d in results.items():
                    out[model][crit.value] = {
                        'top1': d['top1'],
                        'avg_regret': d['avg_regret'],
                        'correct': d['correct'],
                        'total': d['total'],
                    }
            return out

        result = {'insample': _serialize(insample)}
        if logo:
            result['logo'] = _serialize(logo)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
