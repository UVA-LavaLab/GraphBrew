#!/usr/bin/env python3
"""Quick evaluation script: train weights, simulate selection, report accuracy."""
import sys, json, os, math, logging
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from lib.utils import BenchmarkResult
from lib.weights import compute_weights_from_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights', 'active')

# ---------- Load benchmark results ----------
bench_files = sorted(f for f in os.listdir(RESULTS_DIR)
                     if f.startswith('benchmark_') and f.endswith('.json'))
bench_file = os.path.join(RESULTS_DIR, bench_files[-1])
print(f"Loading: {bench_file}")

with open(bench_file) as f:
    raw = json.load(f)
print(f"Loaded {len(raw)} raw entries")

bench_results = []
for e in raw:
    r = BenchmarkResult(
        graph=e['graph'], algorithm=e['algorithm'],
        algorithm_id=e.get('algorithm_id', 0), benchmark=e['benchmark'],
        time_seconds=e['time_seconds'], reorder_time=e.get('reorder_time', 0.0),
        trials=e.get('trials', 1), success=e.get('success', True),
        error=e.get('error', ''), extra=e.get('extra', {}),
    )
    bench_results.append(r)

# ---------- Load reorder results ----------
reorder_files = sorted(f for f in os.listdir(RESULTS_DIR)
                       if f.startswith('reorder_') and f.endswith('.json'))
reorder_results = []
if reorder_files:
    with open(os.path.join(RESULTS_DIR, reorder_files[-1])) as f:
        raw_r = json.load(f)
    for e in raw_r:
        r = BenchmarkResult(
            graph=e.get('graph', ''), algorithm=e.get('algorithm', e.get('algorithm_name', '')),
            algorithm_id=0, benchmark='reorder',
            time_seconds=e.get('reorder_time', e.get('time_seconds', 0.0)),
            reorder_time=e.get('reorder_time', e.get('time_seconds', 0.0)),
        )
        reorder_results.append(r)
    print(f"Loaded {len(reorder_results)} reorder results")

# ---------- Compute weights ----------
print("\n=== Training weights ===")
weights = compute_weights_from_results(
    benchmark_results=bench_results,
    reorder_results=reorder_results,
    weights_dir=WEIGHTS_DIR,
)

algo_weights = {k: v for k, v in weights.items() if not k.startswith('_')}
print(f"\nAlgorithms in weights: {len(algo_weights)}")

# ---------- Load saved type_0.json ----------
type0_file = os.path.join(WEIGHTS_DIR, 'type_0.json')
with open(type0_file) as f:
    saved = json.load(f)
saved_algos = {k: v for k, v in saved.items() if not k.startswith('_')}
print(f"Algorithms in type_0.json: {len(saved_algos)}")

# ---------- Show weights summary ----------
print("\n=== Weight Summary (type_0.json) ===")
print(f"{'Algorithm':<35} {'Bias':>7} {'w_mod':>7} {'w_logN':>7} {'w_logE':>7} {'w_dens':>7} {'w_dv':>7} {'w_hub':>7} {'w_cc':>7}")
for algo in sorted(saved_algos):
    d = saved_algos[algo]
    print(f"{algo:<35} {d.get('bias',0):>7.3f} {d.get('w_modularity',0):>7.3f} "
          f"{d.get('w_log_nodes',0):>7.3f} {d.get('w_log_edges',0):>7.3f} "
          f"{d.get('w_density',0):>7.3f} {d.get('w_degree_variance',0):>7.3f} "
          f"{d.get('w_hub_concentration',0):>7.3f} {d.get('w_clustering_coeff',0):>7.3f}")

# Show benchmark_weights
print(f"\n{'Algorithm':<35} {'b_pr':>7} {'b_bfs':>7} {'b_cc':>7} {'b_sssp':>7}")
for algo in sorted(saved_algos):
    bw = saved_algos[algo].get('benchmark_weights', {})
    print(f"{algo:<35} {bw.get('pr',1.0):>7.3f} {bw.get('bfs',1.0):>7.3f} "
          f"{bw.get('cc',1.0):>7.3f} {bw.get('sssp',1.0):>7.3f}")

# ---------- Simulate C++ scoring ----------
print("\n=== Simulating C++ adaptive selection ===")
from lib.features import load_graph_properties_cache

graph_props = load_graph_properties_cache(RESULTS_DIR)

def simulate_score(algo_data, feats, bench_type):
    """Mimic C++ scoreBase() * benchmarkMultiplier()"""
    s = algo_data.get('bias', 0.5)
    s += algo_data.get('w_modularity', 0) * feats.get('modularity', 0.5)
    s += algo_data.get('w_log_nodes', 0) * feats.get('log_nodes', 5.0)
    s += algo_data.get('w_log_edges', 0) * feats.get('log_edges', 6.0)
    s += algo_data.get('w_density', 0) * feats.get('density', 0.001)
    s += algo_data.get('w_avg_degree', 0) * feats.get('avg_degree', 10.0) / 100.0
    s += algo_data.get('w_degree_variance', 0) * feats.get('degree_variance', 1.0)
    s += algo_data.get('w_hub_concentration', 0) * feats.get('hub_concentration', 0.3)
    s += algo_data.get('w_clustering_coeff', 0) * feats.get('clustering_coefficient', 0.0)
    s += algo_data.get('w_avg_path_length', 0) * feats.get('avg_path_length', 0.0) / 10.0
    s += algo_data.get('w_diameter', 0) * feats.get('diameter', 0.0) / 50.0
    cc = feats.get('community_count', 0.0)
    s += algo_data.get('w_community_count', 0) * (math.log10(cc + 1) if cc > 0 else 0)
    
    # Benchmark multiplier
    bw = algo_data.get('benchmark_weights', {})
    mult = bw.get(bench_type, 1.0)
    return s * mult

# Build ground truth and predictions
correct = 0
total = 0
predictions = {}  # (graph, bench) -> (predicted, actual, predicted_time, best_time)

_VARIANT_PREFIXES = ['GraphBrewOrder_', 'LeidenCSR_', 'LeidenDendrogram_', 'RABBITORDER_']
def get_base(name):
    for prefix in _VARIANT_PREFIXES:
        if name.startswith(prefix):
            return prefix.rstrip('_')
    return name

# Build per-graph-bench results from raw data
from collections import defaultdict
graph_bench_results = defaultdict(list)  # (graph, bench) -> [(algo, time)]
for e in raw:
    if e.get('success', False):
        graph_bench_results[(e['graph'], e['benchmark'])].append((e['algorithm'], e['time_seconds']))

for (graph_name, bench), algo_times in graph_bench_results.items():
    if graph_name not in graph_props:
        continue
    
    props = graph_props[graph_name]
    nodes = props.get('nodes', 1000)
    edges = props.get('edges', 5000)
    feats = {
        'modularity': props.get('modularity', 0.5),
        'degree_variance': props.get('degree_variance', 1.0),
        'hub_concentration': props.get('hub_concentration', 0.3),
        'avg_degree': props.get('avg_degree', 10.0),
        'log_nodes': math.log10(nodes + 1) if nodes > 0 else 0,
        'log_edges': math.log10(edges + 1) if edges > 0 else 0,
        'density': 2 * edges / (nodes * (nodes - 1)) if nodes > 1 else 0,
        'clustering_coefficient': props.get('clustering_coefficient', 0.0),
        'avg_path_length': props.get('avg_path_length', 0.0),
        'diameter': props.get('diameter', 0.0),
        'community_count': props.get('community_count', 0.0),
    }
    
    # C++ picks best among type_0.json entries
    best_score = float('-inf')
    predicted_algo = None
    for algo, data in saved_algos.items():
        score = simulate_score(data, feats, bench)
        if score > best_score:
            best_score = score
            predicted_algo = algo
    
    # Ground truth: fastest algorithm
    algo_times.sort(key=lambda x: x[1])
    actual_algo = algo_times[0][0]
    best_time = algo_times[0][1]
    
    # Find predicted algo's actual time
    # Match exact name, or any variant of the same base
    pred_time = None
    pred_base = get_base(predicted_algo)
    for a, t in algo_times:
        if a == predicted_algo:
            pred_time = t
            break
    if pred_time is None:
        # Find best time among same-base variants
        for a, t in algo_times:
            if get_base(a) == pred_base:
                pred_time = t
                break
    if pred_time is None:
        pred_time = algo_times[-1][1]  # worst case
    
    # Check if any variant of predicted base matches any variant of actual base
    actual_base = get_base(actual_algo)
    
    is_correct = (pred_base == actual_base)
    if is_correct:
        correct += 1
    total += 1
    
    predictions[(graph_name, bench)] = (predicted_algo, actual_algo, pred_time, best_time, is_correct)

accuracy = correct / total if total else 0
print(f"\nOverall accuracy: {correct}/{total} = {accuracy:.1%}")

# Unique predicted algorithms
pred_algos = set(p[0] for p in predictions.values())
print(f"Unique predicted algorithms: {len(pred_algos)}: {sorted(pred_algos)}")

# Per-benchmark accuracy
bench_stats = defaultdict(lambda: [0, 0])
for (g, b), (pred, actual, pt, bt, ok) in predictions.items():
    bench_stats[b][1] += 1
    if ok:
        bench_stats[b][0] += 1

print(f"\nPer-benchmark accuracy:")
for b in sorted(bench_stats):
    c, t = bench_stats[b]
    print(f"  {b}: {c}/{t} = {c/t:.1%}")

# Regret analysis
total_regret = 0
count_regret = 0
for (g, b), (pred, actual, pt, bt, ok) in predictions.items():
    if bt > 0:
        regret = (pt - bt) / bt * 100
        total_regret += regret
        count_regret += 1

avg_regret = total_regret / count_regret if count_regret else 0
print(f"\nAverage regret: {avg_regret:.1f}% (lower is better)")

# Top-2 accuracy (if predicted is the 2nd best, still useful)
top2_correct = 0
for (g, b), (pred, actual, pt, bt, ok) in predictions.items():
    if ok:
        top2_correct += 1
    else:
        # Check if predicted algo is in top-2 for this graph/bench
        algo_times_gb = graph_bench_results.get((g, b), [])
        algo_times_gb.sort(key=lambda x: x[1])
        top2_bases = [get_base(a) for a, _ in algo_times_gb[:2]]
        if get_base(pred) in top2_bases:
            top2_correct += 1
print(f"Top-2 accuracy: {top2_correct}/{total} = {top2_correct/total:.1%}")

# Median regret (more robust than mean)
regrets_list = []
base_regrets_list = []
for (g, b), (pred, actual, pt, bt, ok) in predictions.items():
    if bt > 0:
        regret = (pt - bt) / bt * 100
        regrets_list.append(regret)
        # Base-aware regret: if we got the base right, use best variant of that base
        if get_base(pred) == get_base(actual):
            base_regrets_list.append(0.0)
        else:
            base_regrets_list.append(regret)
regrets_list.sort()
base_regrets_list.sort()
median_regret = regrets_list[len(regrets_list)//2] if regrets_list else 0
median_base_regret = base_regrets_list[len(base_regrets_list)//2] if base_regrets_list else 0
avg_base_regret = sum(base_regrets_list) / len(base_regrets_list) if base_regrets_list else 0
print(f"Median regret: {median_regret:.1f}%")
print(f"Base-aware avg regret: {avg_base_regret:.1f}% (variant mismatches = 0%)")
print(f"Base-aware median regret: {median_base_regret:.1f}%")

# Show worst predictions
print(f"\nWorst predictions (highest regret):")
worst = sorted(predictions.items(), key=lambda x: (x[1][2] - x[1][3]) / max(x[1][3], 0.0001), reverse=True)[:10]
for (g, b), (pred, actual, pt, bt, ok) in worst:
    regret = (pt - bt) / bt * 100 if bt > 0 else 0
    print(f"  {g}/{b}: predicted={pred}, actual={actual}, regret={regret:.0f}%")
