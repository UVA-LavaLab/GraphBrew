#!/usr/bin/env python3
"""
Decision Tree Algorithm Selector for GraphBrew.

Replaces the linear perceptron with a Decision Tree classifier that captures
non-linear feature interactions (e.g., "if graph fits in cache AND is modular,
use Leiden").

The trained tree is exported as:
  1. A C++ inline function (if/else chain) for reorder_types.h
  2. A JSON model for Python evaluation
  3. A human-readable text dump

Key advantages over the perceptron:
  - Captures non-linear interactions (AND/OR/XOR) naturally
  - Depth-5 tree = 5 comparisons at runtime (same speed as perceptron dot product)
  - Interpretable: you can read each decision path
  - Redundant features are automatically ignored (zero importance)

Usage:
    python3 -m scripts.lib.decision_tree --train
    python3 -m scripts.lib.decision_tree --train --export-cpp
    python3 -m scripts.lib.decision_tree --evaluate
    python3 -m scripts.lib.decision_tree --show-tree

Author: GraphBrew Team
"""

import sys
import json
import math
import argparse
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import LabelEncoder

from .utils import Logger, BENCHMARKS, RESULTS_DIR, MODELS_DIR, WEIGHTS_DIR
from .datastore import get_benchmark_store, get_props_store

log = Logger()

# =============================================================================
# Algorithm Coarsening
# =============================================================================

# Map fine-grained algorithm names to coarse families.
# Rationale: with only 44 training graphs, 16+ classes is too fragmented
# for any ML model. Coarsening to 5-6 families gives ~8 samples/class.
#
# The C++ runtime only has access to these families anyway:
#   - It can pick GORDER, SORT, RCM, RABBIT, LEIDEN, etc.
#   - Compound algorithms like "SORT+RABBITORDER_boost" are not available
#     as runtime choices (they were manual experiment constructions).

ALGO_FAMILY = {
    # Basic reorderings
    'ORIGINAL': 'ORIGINAL',
    'RANDOM': 'SORT',       # random is ~sort in practice
    'SORT': 'SORT',
    
    # RCM family
    'RCM_default': 'RCM',
    'RCM_bnf': 'RCM',
    
    # Hub-based
    'HUBSORT': 'HUBSORT',
    'HUBCLUSTER': 'HUBSORT',
    'HUBSORTDBG': 'HUBSORT',
    'HUBCLUSTERDBG': 'HUBSORT',
    'DBG': 'HUBSORT',
    
    # Gorder
    'GORDER': 'GORDER',
    'GORDER_csr': 'GORDER',
    'CORDER': 'GORDER',    # CORDER is similar to GORDER in the benchmark
    
    # Rabbit family
    'RABBITORDER_csr': 'RABBIT',
    'RABBITORDER_boost': 'RABBIT',
    
    # Leiden / GraphBrew family
    'LeidenOrder': 'LEIDEN',
    'GraphBrewOrder_leiden': 'LEIDEN',
    'GraphBrewOrder_rabbit': 'LEIDEN',
    'GraphBrewOrder_hubcluster': 'LEIDEN',
    
    # Compound algorithms → map to the dominant component
    'SORT+RABBITORDER_csr': 'RABBIT',
    'SORT+RABBITORDER_boost': 'RABBIT',
    'SORT+GraphBrewOrder_leiden': 'LEIDEN',
    'DBG+GraphBrewOrder_leiden': 'LEIDEN',
    'HUBCLUSTERDBG+RABBITORDER_csr': 'RABBIT',
}

# Reverse: coarse family → list of fine-grained algos that belong to it
FAMILY_MEMBERS = {}
for algo, family in ALGO_FAMILY.items():
    FAMILY_MEMBERS.setdefault(family, []).append(algo)


def coarsen_algo(algo_name: str) -> str:
    """Map a fine-grained algorithm name to its coarse family."""
    return ALGO_FAMILY.get(algo_name, algo_name)


# =============================================================================
# Feature Configuration
# =============================================================================

# Features used for prediction.
# Deliberately excludes redundant features:
#   - degree_variance: highly correlated with hub_concentration
#   - avg_path_length: derived from same BFS as diameter, nearly perfectly correlated
#
# Feature order matters: indices must match the C++ export.
FEATURE_NAMES = [
    'modularity',                # 0: Community structure quality [0, 1]
    'hub_concentration',         # 1: Edge fraction from top-10% nodes [0, 1]
    'log_nodes',                 # 2: log10(num_nodes + 1)
    'log_edges',                 # 3: log10(num_edges + 1)
    'density',                   # 4: edges / possible_edges
    'avg_degree',                # 5: avg_degree / 100 (scaled)
    'clustering_coefficient',    # 6: Local clustering coefficient [0, 1]
    'packing_factor',            # 7: Hub neighbor co-location [0, 1]
    'forward_edge_fraction',     # 8: Fraction edges u<v [0, 1]
    'working_set_ratio',         # 9: log2(graph_bytes/LLC + 1)
    'community_count',           # 10: log10(community_count + 1)
    'diameter',                  # 11: diameter_estimate / 50 (scaled)
]

# C++ feature extraction: maps FEATURE_NAMES index to C++ expression
CPP_FEATURE_EXPR = [
    'feat.modularity',                                      # 0
    'feat.hub_concentration',                                # 1
    'std::log10(static_cast<double>(feat.num_nodes) + 1.0)', # 2
    'std::log10(static_cast<double>(feat.num_edges) + 1.0)', # 3
    'feat.internal_density',                                 # 4
    'feat.avg_degree / 100.0',                               # 5
    'feat.clustering_coeff',                                 # 6
    'feat.packing_factor',                                   # 7
    'feat.forward_edge_fraction',                            # 8
    'std::log2(feat.working_set_ratio + 1.0)',               # 9
    'std::log10(feat.community_count + 1.0)',                # 10
    'feat.diameter_estimate / 50.0',                         # 11
]


# =============================================================================
# Data Loading
# =============================================================================

def load_benchmark_data() -> Tuple[Dict, Dict]:
    """
    Load benchmark results and graph properties from the centralized data store.
    
    Returns:
        (perf_matrix, graph_props)
        perf_matrix: {graph: {algo: {bench: time}}}
        graph_props: {graph: {feature: value}}
    """
    store = get_benchmark_store()
    perf_matrix = store.perf_matrix()
    
    props_store = get_props_store()
    graph_props = props_store.all()
    
    log.info(f"Loaded benchmark data for {len(perf_matrix)} graphs, "
             f"{sum(len(v) for v in perf_matrix.values())} graph-algo pairs")
    log.info(f"Graph properties available for {len(graph_props)} graphs")
    
    return dict(perf_matrix), graph_props


def extract_features(graph_name: str, graph_props: Dict) -> Optional[np.ndarray]:
    """
    Extract feature vector for a graph.
    
    Returns None if the graph has no properties available.
    Applies the same transforms as the C++ scoring code.
    """
    props = graph_props.get(graph_name)
    if not props:
        return None
    
    nodes = props.get('nodes', 1000)
    edges = props.get('edges', 5000)
    
    features = np.array([
        props.get('modularity', 0.5),                                   # 0
        props.get('hub_concentration', 0.3),                            # 1
        math.log10(nodes + 1),                                          # 2
        math.log10(edges + 1),                                          # 3
        props.get('density', 0.01),                                     # 4
        props.get('avg_degree', 10.0) / 100.0,                          # 5
        props.get('clustering_coefficient', 0.1),                       # 6
        props.get('packing_factor', 0.0),                               # 7
        props.get('forward_edge_fraction', 0.5),                        # 8
        math.log2(props.get('working_set_ratio', 0.0) + 1.0),          # 9
        math.log10(props.get('community_count', 10) + 1),               # 10
        props.get('diameter', props.get('diameter_estimate', 5.0)) / 50.0,  # 11
    ], dtype=np.float64)
    
    return features


def extract_features_with_interactions(
    graph_name: str, graph_props: Dict
) -> Optional[np.ndarray]:
    """
    Extract feature vector with key cross-term interactions.
    
    Adds interaction features that capture important non-linear
    relationships identified from domain knowledge:
      - modularity × log_nodes: modular structure matters more in large graphs
      - density × avg_degree: distinguishes dense-small from sparse-large
      - packing_factor × working_set_ratio: cache pressure × hub locality
      - hub_concentration × clustering: hub-dominated vs community vs hybrid
    """
    base = extract_features(graph_name, graph_props)
    if base is None:
        return None
    
    interactions = np.array([
        base[0] * base[2],   # modularity × log_nodes
        base[4] * base[5],   # density × avg_degree
        base[7] * base[9],   # packing_factor × working_set_ratio
        base[1] * base[6],   # hub_concentration × clustering_coefficient
    ], dtype=np.float64)
    
    return np.concatenate([base, interactions])


FEATURE_NAMES_INTERACTIONS = FEATURE_NAMES + [
    'modularity_x_log_nodes',         # 12
    'density_x_avg_degree',           # 13
    'packing_x_working_set',          # 14
    'hub_x_clustering',               # 15
]

CPP_FEATURE_EXPR_INTERACTIONS = CPP_FEATURE_EXPR + [
    'feat.modularity * std::log10(static_cast<double>(feat.num_nodes) + 1.0)',  # 12
    'feat.internal_density * (feat.avg_degree / 100.0)',                        # 13
    'feat.packing_factor * std::log2(feat.working_set_ratio + 1.0)',           # 14
    'feat.hub_concentration * feat.clustering_coeff',                          # 15
]


def find_best_algorithm(
    perf_matrix: Dict, graph: str, benchmark: str
) -> Tuple[str, float]:
    """Find the best algorithm for a graph/benchmark pair."""
    best_algo = 'ORIGINAL'
    best_time = float('inf')
    
    for algo, benches in perf_matrix.get(graph, {}).items():
        t = benches.get(benchmark, float('inf'))
        if t < best_time:
            best_time = t
            best_algo = algo
    
    return best_algo, best_time


def find_best_family(
    perf_matrix: Dict, graph: str, benchmark: str
) -> Tuple[str, float]:
    """
    Find the best algorithm FAMILY for a graph/benchmark pair.
    
    For each family, picks the best-performing member on this graph,
    then returns the family whose best member is fastest overall.
    """
    family_best: Dict[str, float] = {}
    
    for algo, benches in perf_matrix.get(graph, {}).items():
        t = benches.get(benchmark, float('inf'))
        if t == float('inf'):
            continue
        family = coarsen_algo(algo)
        if family not in family_best or t < family_best[family]:
            family_best[family] = t
    
    if not family_best:
        return 'ORIGINAL', float('inf')
    
    best_family = min(family_best, key=family_best.get)
    return best_family, family_best[best_family]


# =============================================================================
# Training
# =============================================================================

def build_training_data(
    perf_matrix: Dict,
    graph_props: Dict,
    benchmark: str = 'pr',
    min_speedup: float = 1.01,
    use_families: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build (X, y) training data for a specific benchmark.
    
    Args:
        perf_matrix: {graph: {algo: {bench: time}}}
        graph_props: {graph: {feature: value}}
        benchmark: Which benchmark to train for
        min_speedup: Minimum speedup over ORIGINAL to be considered a valid winner
        use_families: If True, coarsen algorithm names to families (recommended)
    
    Returns:
        X: feature matrix (n_graphs, n_features)
        y: label vector (n_graphs,) — index into algo_classes
        graphs: list of graph names corresponding to rows
        algo_classes: list of algorithm names corresponding to labels
    """
    X_rows = []
    y_labels = []
    graph_names = []
    
    for graph in sorted(perf_matrix.keys()):
        feat = extract_features(graph, graph_props)
        if feat is None:
            log.warning(f"No properties for {graph}, skipping")
            continue
        
        if use_families:
            best_label, best_time = find_best_family(perf_matrix, graph, benchmark)
        else:
            best_label, best_time = find_best_algorithm(perf_matrix, graph, benchmark)
        
        if best_time == float('inf'):
            continue
        
        X_rows.append(feat)
        y_labels.append(best_label)
        graph_names.append(graph)
    
    if not X_rows:
        raise ValueError(f"No training data for benchmark '{benchmark}'")
    
    X = np.array(X_rows)
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    algo_classes = list(le.classes_)
    
    log.info(f"Built training data: {len(X)} graphs, {len(algo_classes)} classes "
             f"({'families' if use_families else 'exact'})")
    counts = defaultdict(int)
    for label in y_labels:
        counts[label] += 1
    for algo, count in sorted(counts.items(), key=lambda x: -x[1]):
        log.info(f"    {algo:35s}: {count:3d} graphs ({100*count/len(y_labels):.0f}%)")
    
    return X, y, graph_names, algo_classes


def build_multi_benchmark_training_data(
    perf_matrix: Dict,
    graph_props: Dict,
    benchmarks: List[str] = None,
    use_families: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    """
    Build training data across multiple benchmarks.
    
    Adds a one-hot benchmark indicator as extra features so the tree
    can learn benchmark-specific splits.
    
    Returns:
        X: feature matrix (n_samples, n_features + n_benchmarks)
        y: label vector (n_samples,)
        sample_info: list of "graph:benchmark" for each row
        algo_classes: label encoder classes
        bench_list: list of benchmark names used
    """
    if benchmarks is None:
        benchmarks = ['pr', 'bfs', 'cc', 'sssp', 'bc']
    
    X_rows = []
    y_labels = []
    sample_info = []
    
    for graph in sorted(perf_matrix.keys()):
        feat = extract_features(graph, graph_props)
        if feat is None:
            continue
        
        for bench in benchmarks:
            if use_families:
                best_label, best_time = find_best_family(
                    perf_matrix, graph, bench
                )
            else:
                best_label, best_time = find_best_algorithm(
                    perf_matrix, graph, bench
                )
            if best_time == float('inf'):
                continue
            
            # Add one-hot benchmark encoding
            bench_onehot = [1.0 if b == bench else 0.0 for b in benchmarks]
            combined = np.concatenate([feat, bench_onehot])
            
            X_rows.append(combined)
            y_labels.append(best_label)
            sample_info.append(f"{graph}:{bench}")
    
    if not X_rows:
        raise ValueError("No training data found")
    
    X = np.array(X_rows)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    algo_classes = list(le.classes_)
    
    log.info(f"Multi-benchmark training data: {len(X)} samples, "
             f"{len(algo_classes)} classes, {len(benchmarks)} benchmarks")
    
    return X, y, sample_info, algo_classes, benchmarks


def train_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    algo_classes: List[str],
    max_depth: int = 6,
    min_samples_leaf: int = 2,
    class_weight: str = 'balanced',
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.
    
    Args:
        max_depth: Maximum tree depth. 5-7 gives good accuracy without overfitting.
        min_samples_leaf: Minimum samples per leaf. Higher = more conservative.
        class_weight: 'balanced' upweights rare classes.
    """
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=42,
        criterion='gini',
    )
    clf.fit(X, y)
    
    train_acc = clf.score(X, y)
    log.info(f"Decision Tree: depth={clf.get_depth()}, "
             f"leaves={clf.get_n_leaves()}, train_acc={train_acc:.1%}")
    
    return clf


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    algo_classes: List[str],
    n_estimators: int = 50,
    max_depth: int = 6,
    min_samples_leaf: int = 2,
) -> RandomForestClassifier:
    """
    Train a Random Forest for comparison / feature importance analysis.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    
    train_acc = clf.score(X, y)
    log.info(f"Random Forest: trees={n_estimators}, train_acc={train_acc:.1%}")
    
    return clf


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    algo_classes: List[str],
    graph_names: List[str],
    perf_matrix: Dict,
    benchmark: str,
    label: str = "Model",
    use_families: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained model: accuracy, regret, per-graph details.
    
    When use_families=True, regret is computed as the slowdown compared to
    the best member of the best family, meaning we pick the actual fastest
    algorithm within the predicted family.
    """
    y_pred = clf.predict(X)
    
    correct = 0
    total = len(y)
    details = []
    regrets = []
    
    for i in range(total):
        actual_class = algo_classes[y[i]]
        predicted_class = algo_classes[y_pred[i]]
        graph = graph_names[i] if i < len(graph_names) else f"sample_{i}"
        
        is_correct = (y[i] == y_pred[i])
        if is_correct:
            correct += 1
        
        # Compute regret: best time overall vs best time in predicted family
        if use_families:
            _, actual_time = find_best_family(perf_matrix, graph, benchmark)
            # Find best time among algorithms in the predicted family
            predicted_time = float('inf')
            for algo, benches in perf_matrix.get(graph, {}).items():
                if coarsen_algo(algo) == predicted_class:
                    t = benches.get(benchmark, float('inf'))
                    if t < predicted_time:
                        predicted_time = t
        else:
            actual_time = perf_matrix.get(graph, {}).get(
                actual_class, {}).get(benchmark, float('inf'))
            predicted_time = perf_matrix.get(graph, {}).get(
                predicted_class, {}).get(benchmark, float('inf'))
        
        regret = 0.0
        if actual_time > 0 and actual_time != float('inf'):
            if predicted_time == float('inf'):
                regret = 100.0  # cap at 100%
            else:
                regret = (predicted_time / actual_time - 1.0) * 100
        regrets.append(regret)
        
        details.append({
            'graph': graph,
            'actual': actual_class,
            'predicted': predicted_class,
            'correct': is_correct,
            'regret_pct': round(regret, 2),
        })
    
    accuracy = correct / total * 100 if total > 0 else 0
    avg_regret = sum(regrets) / len(regrets) if regrets else 0
    within_5 = sum(1 for r in regrets if r <= 5.0)
    within_10 = sum(1 for r in regrets if r <= 10.0)
    
    result = {
        'label': label,
        'accuracy': round(accuracy, 2),
        'accuracy_5pct': round(within_5 / total * 100, 2) if total > 0 else 0,
        'accuracy_10pct': round(within_10 / total * 100, 2) if total > 0 else 0,
        'correct': correct,
        'total': total,
        'avg_regret_pct': round(avg_regret, 2),
        'details': details,
    }
    
    return result


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    algo_classes: List[str],
    max_depth: int = 6,
    min_samples_leaf: int = 1,
) -> Dict[str, float]:
    """
    Run Leave-One-Out cross-validation (since we have few graphs).
    """
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42,
    )
    
    loo = LeaveOneOut()
    scores = cross_val_score(clf, X, y, cv=loo)
    
    return {
        'loo_accuracy': round(scores.mean() * 100, 2),
        'loo_std': round(scores.std() * 100, 2),
        'n_splits': len(scores),
    }


def cross_validate_regret(
    X: np.ndarray,
    y: np.ndarray,
    algo_classes: List[str],
    graph_names: List[str],
    perf_matrix: Dict,
    benchmark: str,
    max_depth: int = 6,
    min_samples_leaf: int = 1,
    use_forest: bool = False,
    n_estimators: int = 50,
) -> Dict[str, float]:
    """
    Leave-One-Out cross-validation with REGRET-AWARE metrics.
    
    Instead of just counting exact matches, measures:
      - LOO accuracy (exact family match)
      - LOO accuracy@5% (predicted family within 5% of optimal)
      - LOO accuracy@10% (within 10%)
      - LOO avg regret (average % slowdown vs optimal)
      - LOO median regret
    """
    loo = LeaveOneOut()
    exact = 0
    within_5 = 0
    within_10 = 0
    regrets = []
    n = len(y)
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if use_forest:
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )
        else:
            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced',
                random_state=42,
            )
        
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]
        
        # Exact match
        if pred == y_test[0]:
            exact += 1
        
        # Compute regret
        idx = test_idx[0]
        graph = graph_names[idx]
        predicted_family = algo_classes[pred]
        _, best_time = find_best_family(perf_matrix, graph, benchmark)
        
        predicted_time = float('inf')
        for algo, benches in perf_matrix.get(graph, {}).items():
            if coarsen_algo(algo) == predicted_family:
                t = benches.get(benchmark, float('inf'))
                if t < predicted_time:
                    predicted_time = t
        
        if best_time > 0 and best_time != float('inf') and predicted_time != float('inf'):
            regret = (predicted_time / best_time - 1.0) * 100
        else:
            regret = 100.0
        
        regrets.append(regret)
        if regret <= 5.0:
            within_5 += 1
        if regret <= 10.0:
            within_10 += 1
    
    return {
        'loo_accuracy': round(exact / n * 100, 2),
        'loo_accuracy_5pct': round(within_5 / n * 100, 2),
        'loo_accuracy_10pct': round(within_10 / n * 100, 2),
        'loo_avg_regret': round(np.mean(regrets), 2) if regrets else 0,
        'loo_median_regret': round(np.median(regrets), 2) if regrets else 0,
        'loo_max_regret': round(max(regrets), 2) if regrets else 0,
        'n_splits': n,
    }


def grid_search_depth(
    X: np.ndarray,
    y: np.ndarray,
    algo_classes: List[str],
    graph_names: List[str] = None,
    perf_matrix: Dict = None,
    benchmark: str = None,
) -> List[Dict]:
    """
    Grid search over tree depths to find optimal complexity.
    
    If graph_names/perf_matrix/benchmark are provided, also computes
    regret-aware LOO metrics for better model selection.
    """
    results = []
    has_regret = (graph_names is not None and perf_matrix is not None 
                  and benchmark is not None)
    
    for depth in range(2, 12):
        for min_leaf in [1, 2, 3, 5]:
            clf = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=min_leaf,
                class_weight='balanced',
                random_state=42,
            )
            clf.fit(X, y)
            train_acc = clf.score(X, y)
            
            # LOO cross-validation
            loo = LeaveOneOut()
            cv_scores = cross_val_score(clf, X, y, cv=loo)
            
            entry = {
                'depth': depth,
                'min_leaf': min_leaf,
                'train_acc': round(train_acc * 100, 1),
                'loo_acc': round(cv_scores.mean() * 100, 1),
                'loo_std': round(cv_scores.std() * 100, 1),
                'n_leaves': clf.get_n_leaves(),
            }
            
            if has_regret:
                cv_regret = cross_validate_regret(
                    X, y, algo_classes, graph_names, perf_matrix, benchmark,
                    max_depth=depth, min_samples_leaf=min_leaf,
                )
                entry['loo_within_5'] = cv_regret['loo_accuracy_5pct']
                entry['loo_within_10'] = cv_regret['loo_accuracy_10pct']
                entry['loo_avg_regret'] = cv_regret['loo_avg_regret']
                entry['loo_median_regret'] = cv_regret['loo_median_regret']
            
            results.append(entry)
    
    # Sort by: lowest average regret (if available), else highest LOO accuracy
    if has_regret:
        results.sort(key=lambda x: (x.get('loo_avg_regret', 999), -x['loo_acc'], x['depth']))
    else:
        results.sort(key=lambda x: (-x['loo_acc'], x['depth']))
    return results


def grid_search_full(
    perf_matrix: Dict,
    graph_props: Dict,
    benchmark: str = 'pr',
) -> Dict:
    """
    Full grid search comparing DecisionTree vs RandomForest,
    with and without feature interactions, optimizing for LOO regret.
    
    Returns the best configuration found.
    """
    results = []
    
    for use_interactions in [False, True]:
        # Build training data with appropriate features
        if use_interactions:
            feat_fn = extract_features_with_interactions
            feat_names = FEATURE_NAMES_INTERACTIONS
            feat_label = 'interactions'
        else:
            feat_fn = extract_features
            feat_names = FEATURE_NAMES
            feat_label = 'base'
        
        # Build X, y manually with the chosen feature extractor
        X_rows, y_labels, graph_names = [], [], []
        for graph in sorted(perf_matrix.keys()):
            feat = feat_fn(graph, graph_props)
            if feat is None:
                continue
            best_label, best_time = find_best_family(perf_matrix, graph, benchmark)
            if best_time == float('inf'):
                continue
            X_rows.append(feat)
            y_labels.append(best_label)
            graph_names.append(graph)
        
        if not X_rows:
            continue
        X = np.array(X_rows)
        le = LabelEncoder()
        y = le.fit_transform(y_labels)
        algo_classes = list(le.classes_)
        
        for model_type in ['tree', 'forest']:
            depths = range(2, 10)
            leaves = [1, 2, 3, 5]
            
            for depth in depths:
                for min_leaf in leaves:
                    cv = cross_validate_regret(
                        X, y, algo_classes, graph_names, perf_matrix, benchmark,
                        max_depth=depth, min_samples_leaf=min_leaf,
                        use_forest=(model_type == 'forest'),
                        n_estimators=20,  # fewer trees for speed
                    )
                    
                    results.append({
                        'model': model_type,
                        'features': feat_label,
                        'depth': depth,
                        'min_leaf': min_leaf,
                        'loo_acc': cv['loo_accuracy'],
                        'loo_5pct': cv['loo_accuracy_5pct'],
                        'loo_10pct': cv['loo_accuracy_10pct'],
                        'loo_regret': cv['loo_avg_regret'],
                        'loo_median_regret': cv['loo_median_regret'],
                    })
    
    results.sort(key=lambda x: (x['loo_regret'], -x['loo_5pct'], x['depth']))
    
    return {
        'best': results[0] if results else None,
        'top_10': results[:10],
        'all': results,
    }


# =============================================================================
# Feature Importance
# =============================================================================

def feature_importance_report(
    clf,
    feature_names: List[str],
) -> List[Tuple[str, float]]:
    """
    Report feature importances from a trained tree/forest.
    """
    importances = clf.feature_importances_
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: -x[1])
    
    log.info("Feature importances:")
    for name, imp in pairs:
        bar = '#' * int(imp * 50)
        log.info(f"  {name:30s}: {imp:.4f} {bar}")
    
    return pairs


# =============================================================================
# C++ Code Generation
# =============================================================================

def _tree_to_cpp_recursive(
    tree,
    node: int,
    feature_names: List[str],
    cpp_feature_exprs: List[str],
    class_names: List[str],
    indent: int = 1,
) -> str:
    """
    Recursively convert a sklearn decision tree node to C++ if/else code.
    """
    tab = '    ' * indent
    
    left = tree.children_left[node]
    right = tree.children_right[node]
    
    # Leaf node
    if left == right:
        class_idx = int(np.argmax(tree.value[node]))
        algo_name = class_names[class_idx]
        samples = int(tree.n_node_samples[node])
        # Return the ReorderingAlgo enum value
        return f'{tab}return "{algo_name}"; // {samples} training samples\n'
    
    # Decision node
    feat_idx = tree.feature[node]
    threshold = tree.threshold[node]
    
    feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat[{feat_idx}]"
    cpp_expr = cpp_feature_exprs[feat_idx] if feat_idx < len(cpp_feature_exprs) else f"feat[{feat_idx}]"
    
    code = ''
    code += f'{tab}// Split on {feat_name}\n'
    code += f'{tab}if ({cpp_expr} <= {threshold:.6f}) {{\n'
    code += _tree_to_cpp_recursive(
        tree, left, feature_names, cpp_feature_exprs, class_names, indent + 1
    )
    code += f'{tab}}} else {{\n'
    code += _tree_to_cpp_recursive(
        tree, right, feature_names, cpp_feature_exprs, class_names, indent + 1
    )
    code += f'{tab}}}\n'
    
    return code


def export_tree_to_cpp(
    clf: DecisionTreeClassifier,
    algo_classes: List[str],
    feature_names: List[str] = None,
    cpp_feature_exprs: List[str] = None,
    benchmark: str = None,
) -> str:
    """
    Export a trained Decision Tree as a C++ function.
    
    Returns a string containing a complete C++ inline function
    that can be pasted into reorder_types.h.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    if cpp_feature_exprs is None:
        cpp_feature_exprs = CPP_FEATURE_EXPR
    
    bench_suffix = f"_{benchmark}" if benchmark else ""
    func_name = f"SelectAlgorithmDecisionTree{bench_suffix}"
    
    tree = clf.tree_
    
    code = f'''/**
 * @brief Select reordering algorithm using a trained Decision Tree.
 *
 * Auto-generated by scripts/lib/decision_tree.py on {datetime.now().strftime("%Y-%m-%d %H:%M")}.
 * Trained on {tree.n_node_samples[0]} graphs, {len(algo_classes)} algorithm classes.
 * Tree depth: {clf.get_depth()}, leaves: {clf.get_n_leaves()}.
 * Benchmark: {benchmark or "all"}.
 *
 * Features used ({len(feature_names)}):
'''
    for i, name in enumerate(feature_names):
        code += f' *   {i:2d}. {name}\n'
    code += f''' *
 * @param feat  Graph community features (computed at runtime)
 * @return Algorithm name string
 */
inline std::string {func_name}(const CommunityFeatures& feat) {{
'''
    
    code += _tree_to_cpp_recursive(
        tree, 0, feature_names, cpp_feature_exprs, algo_classes
    )
    
    code += '}\n'
    return code


def export_per_benchmark_trees_to_cpp(
    trees: Dict[str, Tuple[DecisionTreeClassifier, List[str]]],
    use_interactions: bool = False,
) -> str:
    """
    Export per-benchmark trees as a dispatcher function.
    
    Args:
        trees: {benchmark: (clf, algo_classes)}
        use_interactions: Whether these trees use interaction features
    """
    feat_names = FEATURE_NAMES_INTERACTIONS if use_interactions else FEATURE_NAMES
    feat_exprs = CPP_FEATURE_EXPR_INTERACTIONS if use_interactions else CPP_FEATURE_EXPR
    code = ''
    
    # Generate individual tree functions
    for bench, (clf, algo_classes) in trees.items():
        code += export_tree_to_cpp(
            clf, algo_classes, 
            feature_names=feat_names, cpp_feature_exprs=feat_exprs,
            benchmark=bench,
        )
        code += '\n'
    
    # Generate dispatcher
    code += '''/**
 * @brief Select algorithm using the benchmark-specific Decision Tree.
 *
 * Dispatches to the appropriate per-benchmark tree based on the
 * benchmark being run. Falls back to the generic tree if available.
 */
inline std::string SelectAlgorithmDecisionTreeDispatch(
    const CommunityFeatures& feat,
    const std::string& benchmark)
{
'''
    for i, bench in enumerate(trees.keys()):
        prefix = 'if' if i == 0 else '} else if'
        code += f'    {prefix} (benchmark == "{bench}") {{\n'
        code += f'        return SelectAlgorithmDecisionTree_{bench}(feat);\n'
    
    # Default fallback
    first_bench = list(trees.keys())[0] if trees else 'pr'
    code += f'    }} else {{\n'
    code += f'        return SelectAlgorithmDecisionTree_{first_bench}(feat);\n'
    code += f'    }}\n'
    code += '}\n'
    
    return code


# =============================================================================
# JSON Export
# =============================================================================

def export_tree_to_json(
    clf: DecisionTreeClassifier,
    algo_classes: List[str],
    feature_names: List[str],
    metadata: Dict = None,
) -> Dict:
    """Export tree model as JSON-serializable dict."""
    tree = clf.tree_
    
    def _node_to_dict(node_id):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        
        if left == right:  # Leaf
            class_idx = int(np.argmax(tree.value[node_id]))
            return {
                'type': 'leaf',
                'class': algo_classes[class_idx],
                'class_idx': class_idx,
                'samples': int(tree.n_node_samples[node_id]),
                'distribution': [int(v) for v in tree.value[node_id][0]],
            }
        
        return {
            'type': 'split',
            'feature': feature_names[tree.feature[node_id]],
            'feature_idx': int(tree.feature[node_id]),
            'threshold': float(tree.threshold[node_id]),
            'samples': int(tree.n_node_samples[node_id]),
            'left': _node_to_dict(left),
            'right': _node_to_dict(right),
        }
    
    model = {
        'model_type': 'decision_tree',
        'tree': _node_to_dict(0),
        'classes': algo_classes,
        'features': feature_names,
        'depth': clf.get_depth(),
        'n_leaves': clf.get_n_leaves(),
        'metadata': metadata or {},
        'created': datetime.now().isoformat(),
    }
    
    return model


def export_model_to_flat_json(
    dt_clf: DecisionTreeClassifier,
    families: List[str],
    benchmark: str,
    model_type: str = 'decision_tree',
    leaf_perceptrons: Dict[int, Dict[str, np.ndarray]] = None,
    metadata: Dict = None,
) -> Dict:
    """
    Export a DT or hybrid model as a flat-node-array JSON dict.

    Uses a flat array of nodes (indexed by node ID) instead of a recursive
    tree, making it trivial to parse in C++ without recursion.

    For model_type='decision_tree', leaves store only ``leaf_class``.
    For model_type='hybrid', leaves also store per-family perceptron weights.

    Format::

        {
          "model_type": "decision_tree" | "hybrid",
          "benchmark": "pr",
          "families": ["GORDER", ...],
          "features": ["modularity", ...],
          "n_nodes": 7,
          "nodes": [
            {"feature_idx": 0, "threshold": 0.617, "left": 1, "right": 4},
            {"leaf_class": "ORIGINAL", "weights": {...}}
          ]
        }
    """
    tree = dt_clf.tree_
    n_nodes = tree.node_count
    nodes = []

    for i in range(n_nodes):
        left = int(tree.children_left[i])
        right = int(tree.children_right[i])
        is_leaf = (left == right)

        if is_leaf:
            class_idx = int(np.argmax(tree.value[i]))
            leaf_class = families[class_idx] if class_idx < len(families) else 'ORIGINAL'
            node = {
                'leaf_class': leaf_class,
                'samples': int(tree.n_node_samples[i]),
            }
            # Hybrid: attach perceptron weights at this leaf
            if model_type == 'hybrid' and leaf_perceptrons and i in leaf_perceptrons:
                wts = {}
                for fam, warr in leaf_perceptrons[i].items():
                    wts[fam] = [float(v) for v in warr]
                node['weights'] = wts
            nodes.append(node)
        else:
            nodes.append({
                'feature_idx': int(tree.feature[i]),
                'threshold': float(tree.threshold[i]),
                'left': left,
                'right': right,
                'samples': int(tree.n_node_samples[i]),
            })

    return {
        'model_type': model_type,
        'benchmark': benchmark,
        'families': list(families),
        'features': list(FEATURE_NAMES),
        'depth': int(dt_clf.get_depth()),
        'n_leaves': int(dt_clf.get_n_leaves()),
        'n_nodes': int(n_nodes),
        'nodes': nodes,
        'metadata': metadata or {},
        'created': datetime.now().isoformat(),
    }


def export_all_models_to_json(
    dt_trees: Dict[str, Tuple] = None,
    hybrid_models: Dict[str, Tuple] = None,
    out_dir: Path = None,
) -> List[Path]:
    """
    Export all trained models (DT and/or hybrid) to JSON files.

    Writes one file per benchmark per model type:
        results/models/decision_tree/{bench}.json
        results/models/hybrid/{bench}.json

    Args:
        dt_trees: {bench: (clf, algo_classes)} from per-benchmark DT training
        hybrid_models: {bench: (dt_clf, leaf_perceptrons, families)}
        out_dir: base directory (default: results/models/)

    Returns:
        List of paths written.
    """
    if out_dir is None:
        out_dir = MODELS_DIR

    written = []

    if dt_trees:
        dt_dir = out_dir / 'decision_tree'
        dt_dir.mkdir(parents=True, exist_ok=True)
        for bench, (clf, algo_classes) in dt_trees.items():
            model = export_model_to_flat_json(
                clf, algo_classes, bench, model_type='decision_tree',
            )
            p = dt_dir / f'{bench}.json'
            with open(p, 'w') as f:
                json.dump(model, f, indent=2)
            written.append(p)
            log.info(f"  DT model → {p}")

    if hybrid_models:
        h_dir = out_dir / 'hybrid'
        h_dir.mkdir(parents=True, exist_ok=True)
        for bench, (dt_clf, leaf_perceptrons, families) in hybrid_models.items():
            model = export_model_to_flat_json(
                dt_clf, families, bench, model_type='hybrid',
                leaf_perceptrons=leaf_perceptrons,
            )
            p = h_dir / f'{bench}.json'
            with open(p, 'w') as f:
                json.dump(model, f, indent=2)
            written.append(p)
            log.info(f"  Hybrid model → {p}")

    return written


# =============================================================================
# Hybrid DT + Perceptron (Model Tree)
# =============================================================================

# Perceptron weight keys in scoring order (maps to FEATURE_NAMES indices)
PERCEPTRON_FEATURE_KEYS = [
    'w_modularity',               # FEATURE_NAMES[0]
    'w_hub_concentration',         # FEATURE_NAMES[1]
    'w_log_nodes',                 # FEATURE_NAMES[2]
    'w_log_edges',                 # FEATURE_NAMES[3]
    'w_density',                   # FEATURE_NAMES[4]
    'w_avg_degree',                # FEATURE_NAMES[5]
    'w_clustering_coeff',          # FEATURE_NAMES[6]
    'w_packing_factor',            # FEATURE_NAMES[7]
    'w_forward_edge_fraction',     # FEATURE_NAMES[8]
    'w_working_set_ratio',         # FEATURE_NAMES[9]
    'w_community_count',           # FEATURE_NAMES[10]
    'w_diameter',                  # FEATURE_NAMES[11]
]


def _train_leaf_perceptron(
    X_leaf: np.ndarray,
    graph_names_leaf: List[str],
    perf_matrix: Dict,
    benchmark: str,
    families: List[str],
    n_epochs: int = 200,
    learning_rate: float = 0.01,
    weight_decay: float = 0.001,
    min_samples_for_perceptron: int = 4,
) -> Tuple[Dict[str, np.ndarray], bool]:
    """
    Train a linear scoring model (perceptron) for a DT leaf.

    For each family, learns weights w such that score = w · x + bias.
    The family with highest score is predicted.

    Training uses pairwise ranking loss with L2 regularization.
    If the leaf has fewer than min_samples_for_perceptron graphs,
    falls back to majority-vote (returns is_fallback=True).

    Args:
        X_leaf: Feature matrix for graphs in this leaf (n_leaf, n_features)
        graph_names_leaf: Graph names for the leaf samples
        perf_matrix: Full performance matrix
        benchmark: Benchmark name
        families: List of all family names to score
        n_epochs: Training epochs
        learning_rate: SGD learning rate
        weight_decay: L2 regularization strength
        min_samples_for_perceptron: Minimum leaf samples for perceptron training

    Returns:
        ({family: np.array of shape (n_features+1,)}, is_fallback)
        is_fallback=True means majority-vote weights (no real perceptron)
    """
    n_features = X_leaf.shape[1]

    # Build per-graph ground truth: best family timing
    best_families = []
    for g in graph_names_leaf:
        ft = {}
        for fam in families:
            best_t = float('inf')
            for algo, benches in perf_matrix.get(g, {}).items():
                if coarsen_algo(algo) == fam:
                    t = benches.get(benchmark, float('inf'))
                    if t < best_t:
                        best_t = t
            ft[fam] = best_t
        best_fam = min(ft, key=ft.get) if ft else families[0]
        best_families.append(best_fam)

    # Fallback: too few samples for meaningful perceptron learning
    if len(X_leaf) < min_samples_for_perceptron:
        # Use regret-weighted majority vote
        from collections import Counter
        vote_counts = Counter(best_families)
        majority_fam = vote_counts.most_common(1)[0][0]
        weights = {fam: np.zeros(n_features + 1) for fam in families}
        weights[majority_fam][-1] = 100.0  # high bias for majority
        return weights, True

    # Initialize weights: small random values
    rng = np.random.RandomState(42)
    weights = {fam: rng.randn(n_features + 1) * 0.01 for fam in families}

    # Pairwise ranking SGD with L2 regularization
    for epoch in range(n_epochs):
        for i in range(len(X_leaf)):
            x = X_leaf[i]
            x_aug = np.append(x, 1.0)  # add bias term
            best = best_families[i]

            # Compute scores for all families
            scores = {fam: np.dot(weights[fam], x_aug) for fam in families}

            # For each non-best family, enforce margin
            margin = 1.0
            for fam in families:
                if fam == best:
                    continue
                # Hinge loss: max(0, margin - (score_best - score_fam))
                gap = scores[best] - scores[fam]
                if gap < margin:
                    # Update: push best up, push fam down
                    weights[best] += learning_rate * x_aug
                    weights[fam] -= learning_rate * x_aug

            # L2 weight decay (don't decay bias)
            for fam in families:
                weights[fam][:-1] *= (1.0 - learning_rate * weight_decay)

    return weights, False


def _perceptron_predict_family(
    x: np.ndarray,
    leaf_weights: Dict[str, np.ndarray],
) -> str:
    """Score all families and return the one with highest score."""
    x_aug = np.append(x, 1.0)
    best_fam = None
    best_score = -float('inf')
    for fam, w in leaf_weights.items():
        s = np.dot(w, x_aug)
        if s > best_score:
            best_score = s
            best_fam = fam
    return best_fam


def train_hybrid_model(
    X: np.ndarray,
    y: np.ndarray,
    algo_classes: List[str],
    graph_names: List[str],
    perf_matrix: Dict,
    benchmark: str,
    max_depth: int = 4,
    min_samples_leaf: int = 2,
    n_epochs: int = 200,
    learning_rate: float = 0.01,
) -> Tuple[DecisionTreeClassifier, Dict[int, Dict[str, np.ndarray]], List[str]]:
    """
    Train a Hybrid DT + Perceptron model (Model Tree / HME).

    1. Train a Decision Tree to partition the feature space
    2. At each leaf, train a specialized perceptron on the local data

    Args:
        X, y, algo_classes, graph_names: standard training data
        perf_matrix: performance matrix for regret-aware perceptron training
        benchmark: benchmark name
        max_depth, min_samples_leaf: DT hyperparameters
        n_epochs, learning_rate: perceptron hyperparameters

    Returns:
        (dt_clf, leaf_perceptrons, families)
        dt_clf: trained DecisionTreeClassifier (for routing)
        leaf_perceptrons: {leaf_id: {family: weight_vector}}
        families: sorted list of all family names
    """
    # Get all families present in training data
    families = sorted(set(algo_classes))

    # Train the routing DT
    dt_clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42,
    )
    dt_clf.fit(X, y)

    # Identify which leaf each sample lands on
    leaf_ids = dt_clf.apply(X)
    unique_leaves = sorted(set(leaf_ids))

    log.info(f"Hybrid: DT depth={dt_clf.get_depth()}, "
             f"leaves={dt_clf.get_n_leaves()}, training {len(unique_leaves)} leaf perceptrons")

    # Train a perceptron at each leaf
    leaf_perceptrons = {}
    for leaf_id in unique_leaves:
        mask = (leaf_ids == leaf_id)
        X_leaf = X[mask]
        g_leaf = [graph_names[i] for i in range(len(graph_names)) if mask[i]]

        log.info(f"  Leaf {leaf_id}: {len(g_leaf)} graphs → "
                 f"training perceptron ({n_epochs} epochs)")

        leaf_weights, is_fallback = _train_leaf_perceptron(
            X_leaf, g_leaf, perf_matrix, benchmark,
            families, n_epochs, learning_rate,
        )
        leaf_perceptrons[leaf_id] = leaf_weights
        if is_fallback:
            log.info(f"    → fallback to majority vote (too few samples)")

    return dt_clf, leaf_perceptrons, families


def predict_hybrid(
    dt_clf: DecisionTreeClassifier,
    leaf_perceptrons: Dict[int, Dict[str, np.ndarray]],
    X: np.ndarray,
    families: List[str],
) -> List[str]:
    """
    Predict using the hybrid model: DT routes → leaf perceptron scores.

    Returns list of predicted family names.
    """
    leaf_ids = dt_clf.apply(X)
    predictions = []
    for i, leaf_id in enumerate(leaf_ids):
        if leaf_id in leaf_perceptrons:
            pred = _perceptron_predict_family(X[i], leaf_perceptrons[leaf_id])
        else:
            # Shouldn't happen, but fallback to DT prediction
            pred_idx = dt_clf.predict(X[i:i+1])[0]
            pred = families[pred_idx] if pred_idx < len(families) else families[0]
        predictions.append(pred)
    return predictions


def cross_validate_hybrid_regret(
    X: np.ndarray,
    y: np.ndarray,
    algo_classes: List[str],
    graph_names: List[str],
    perf_matrix: Dict,
    benchmark: str,
    max_depth: int = 4,
    min_samples_leaf: int = 2,
    n_epochs: int = 200,
    learning_rate: float = 0.01,
) -> Dict[str, float]:
    """
    Leave-One-Out cross-validation for the hybrid DT+Perceptron model.

    Same metrics as cross_validate_regret but using hybrid prediction.
    """
    loo = LeaveOneOut()
    exact = 0
    within_5 = 0
    within_10 = 0
    regrets = []
    n = len(y)
    families = sorted(set(algo_classes))

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        g_train = [graph_names[i] for i in train_idx]

        # Train hybrid on n-1 graphs
        dt_clf, leaf_perceptrons, _ = train_hybrid_model(
            X_train, y_train, algo_classes, g_train,
            perf_matrix, benchmark,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
        )

        # Predict on held-out graph
        preds = predict_hybrid(dt_clf, leaf_perceptrons, X_test, families)
        predicted_family = preds[0]

        # Check exact match
        idx = test_idx[0]
        true_family = algo_classes[y[idx]]
        if predicted_family == true_family:
            exact += 1

        # Compute regret
        graph = graph_names[idx]
        _, best_time = find_best_family(perf_matrix, graph, benchmark)

        predicted_time = float('inf')
        for algo, benches in perf_matrix.get(graph, {}).items():
            if coarsen_algo(algo) == predicted_family:
                t = benches.get(benchmark, float('inf'))
                if t < predicted_time:
                    predicted_time = t

        if best_time > 0 and best_time != float('inf') and predicted_time != float('inf'):
            regret = (predicted_time / best_time - 1.0) * 100
        else:
            regret = 100.0

        regrets.append(regret)
        if regret <= 5.0:
            within_5 += 1
        if regret <= 10.0:
            within_10 += 1

    return {
        'loo_accuracy': round(exact / n * 100, 2),
        'loo_accuracy_5pct': round(within_5 / n * 100, 2),
        'loo_accuracy_10pct': round(within_10 / n * 100, 2),
        'loo_avg_regret': round(np.mean(regrets), 2) if regrets else 0,
        'loo_median_regret': round(np.median(regrets), 2) if regrets else 0,
        'loo_max_regret': round(max(regrets), 2) if regrets else 0,
        'n_splits': n,
    }


def _hybrid_tree_to_cpp_recursive(
    tree,
    node: int,
    feature_names: List[str],
    cpp_feature_exprs: List[str],
    leaf_perceptrons: Dict[int, Dict[str, np.ndarray]],
    families: List[str],
    indent: int = 1,
) -> str:
    """
    Recursively convert a hybrid model node to C++ if/else code.

    At leaf nodes, emits perceptron dot-product scoring instead of
    a fixed return value.
    """
    tab = '    ' * indent

    left = tree.children_left[node]
    right = tree.children_right[node]

    # Leaf node → emit perceptron scoring
    if left == right:
        # Find the leaf_id for this node
        # In sklearn, leaf node IDs are just the node indices
        leaf_id = node
        samples = int(tree.n_node_samples[node])

        if leaf_id not in leaf_perceptrons:
            # Fallback: return majority class
            class_idx = int(np.argmax(tree.value[node]))
            algo_name = families[class_idx] if class_idx < len(families) else 'ORIGINAL'
            return f'{tab}return "{algo_name}"; // {samples} samples (no perceptron)\n'

        weights = leaf_perceptrons[leaf_id]

        code = f'{tab}// Leaf: {samples} training samples — perceptron scoring\n'
        code += f'{tab}{{\n'
        tab2 = '    ' * (indent + 1)

        # Emit scoring for each family
        code += f'{tab2}double best_score = -1e30;\n'
        code += f'{tab2}std::string best_family = "ORIGINAL";\n'
        code += f'{tab2}double score;\n'

        for fam in families:
            w = weights.get(fam)
            if w is None:
                continue
            # Build dot product: w[0]*feat0 + w[1]*feat1 + ... + w[-1] (bias)
            terms = []
            for j, cpp_expr in enumerate(cpp_feature_exprs):
                if j < len(w) - 1 and abs(w[j]) > 1e-10:
                    terms.append(f'{w[j]:.8f} * {cpp_expr}')
            bias = w[-1] if len(w) > 0 else 0.0
            if abs(bias) > 1e-10:
                terms.append(f'{bias:.8f}')
            if not terms:
                terms = ['0.0']

            score_expr = '\n{0}         + '.format(tab2).join(terms)
            code += f'{tab2}score = {score_expr};\n'
            code += f'{tab2}if (score > best_score) {{ best_score = score; best_family = "{fam}"; }}\n'

        code += f'{tab2}return best_family;\n'
        code += f'{tab}}}\n'
        return code

    # Decision node
    feat_idx = tree.feature[node]
    threshold = tree.threshold[node]

    feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feat[{feat_idx}]"
    cpp_expr = cpp_feature_exprs[feat_idx] if feat_idx < len(cpp_feature_exprs) else f"feat[{feat_idx}]"

    code = ''
    code += f'{tab}// Split on {feat_name}\n'
    code += f'{tab}if ({cpp_expr} <= {threshold:.6f}) {{\n'
    code += _hybrid_tree_to_cpp_recursive(
        tree, left, feature_names, cpp_feature_exprs,
        leaf_perceptrons, families, indent + 1,
    )
    code += f'{tab}}} else {{\n'
    code += _hybrid_tree_to_cpp_recursive(
        tree, right, feature_names, cpp_feature_exprs,
        leaf_perceptrons, families, indent + 1,
    )
    code += f'{tab}}}\n'

    return code


def export_hybrid_to_cpp(
    dt_clf: DecisionTreeClassifier,
    leaf_perceptrons: Dict[int, Dict[str, np.ndarray]],
    families: List[str],
    benchmark: str,
    feature_names: List[str] = None,
    cpp_feature_exprs: List[str] = None,
) -> str:
    """
    Export a hybrid DT+Perceptron model as a C++ function.

    The tree structure routes via if/else, and each leaf contains
    a perceptron dot-product scoring over all families.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
    if cpp_feature_exprs is None:
        cpp_feature_exprs = CPP_FEATURE_EXPR

    tree = dt_clf.tree_
    func_name = f"SelectAlgorithmHybrid_{benchmark}"

    code = f'''/**
 * @brief Hybrid DT+Perceptron selector for {benchmark.upper()} benchmark.
 *
 * Auto-generated by scripts/lib/decision_tree.py on {datetime.now().strftime("%Y-%m-%d %H:%M")}.
 * Architecture: Decision Tree routes to leaf, per-leaf perceptron scores families.
 * Scientific basis: Logistic Model Trees (Landwehr et al. 2005, Machine Learning),
 *   Hierarchical Mixtures of Experts (Jordan & Jacobs 1994, Neural Computation).
 * Tree depth: {dt_clf.get_depth()}, leaves: {dt_clf.get_n_leaves()}.
 * Families: {", ".join(families)}.
 * Features ({len(feature_names)}): {", ".join(feature_names)}.
 *
 * @param feat  Graph community features (computed at runtime)
 * @return Algorithm family name
 */
inline std::string {func_name}(const CommunityFeatures& feat) {{
'''

    code += _hybrid_tree_to_cpp_recursive(
        tree, 0, feature_names, cpp_feature_exprs,
        leaf_perceptrons, families,
    )

    code += '}\n'
    return code


def export_hybrid_per_benchmark_to_cpp(
    models: Dict[str, Tuple[DecisionTreeClassifier, Dict[int, Dict[str, np.ndarray]], List[str]]],
) -> str:
    """
    Export per-benchmark hybrid models + dispatcher as C++ code.

    Args:
        models: {benchmark: (dt_clf, leaf_perceptrons, families)}
    """
    code = '''// ============================================================================
// HYBRID DT + PERCEPTRON MODEL (Model Tree / Hierarchical Mixture of Experts)
//
// Architecture: Decision Tree routes input to a leaf node, where a specialized
// perceptron scores all algorithm families. The family with the highest score wins.
//
// Scientific basis:
//   - Logistic Model Trees: Landwehr, Hall & Frank (2005), Machine Learning
//   - Hierarchical Mixtures of Experts: Jordan & Jacobs (1994), Neural Computation (4737 citations)
//   - Model Trees: Frank et al. (1998), Machine Learning (558 citations)
//   - Hybrid DT Learners: Seewald et al. (2001), FLAIRS/AAAI
//
// Features (12): modularity, hub_concentration, log_nodes, log_edges,
//   density, avg_degree/100, clustering_coeff, packing_factor,
//   forward_edge_fraction, log2(wsr+1), log10(community_count+1), diameter/50
// ============================================================================

'''

    for bench, (dt_clf, leaf_perceptrons, families) in models.items():
        code += export_hybrid_to_cpp(
            dt_clf, leaf_perceptrons, families, bench,
        )
        code += '\n'

    # Dispatcher
    code += '''/**
 * @brief Dispatch to the benchmark-specific Hybrid DT+Perceptron model.
 *
 * @param feat  Graph community features (computed at runtime)
 * @param bench Benchmark type enum
 * @return Algorithm family name
 */
inline std::string SelectAlgorithmHybridDispatch(
    const CommunityFeatures& feat, BenchmarkType bench) {
    switch (bench) {
'''
    bench_list = list(models.keys())
    bench_to_enum = {
        'pr': ['BENCH_PR', 'BENCH_PR_SPMV'],
        'bfs': ['BENCH_BFS'],
        'cc': ['BENCH_CC', 'BENCH_CC_SV'],
        'sssp': ['BENCH_SSSP'],
        'bc': ['BENCH_BC'],
        'tc': ['BENCH_TC'],
    }
    for bench in bench_list:
        for enum_val in bench_to_enum.get(bench, []):
            code += f'        case {enum_val}:\n'
        code += f'            return SelectAlgorithmHybrid_{bench}(feat);\n'

    first_bench = bench_list[0] if bench_list else 'pr'
    code += f'        default:\n'
    code += f'            return SelectAlgorithmHybrid_{first_bench}(feat);\n'
    code += f'    }}\n'
    code += '}\n'

    return code


# =============================================================================
# Comparison with Perceptron
# =============================================================================

def compare_with_perceptron(
    perf_matrix: Dict,
    graph_props: Dict,
    benchmark: str = 'pr',
) -> Dict:
    """
    Compare Decision Tree accuracy against the current perceptron.
    """
    from .perceptron import (
        load_all_results, build_performance_matrix,
        compute_graph_features, evaluate_weights,
    )
    from .weights import load_type_weights
    
    # Build tree training data
    X, y, graphs, algo_classes = build_training_data(
        perf_matrix, graph_props, benchmark
    )
    
    # Train tree
    tree_clf = train_decision_tree(X, y, algo_classes, max_depth=6)
    tree_result = evaluate_model(
        tree_clf, X, y, algo_classes, graphs, perf_matrix, benchmark,
        label="Decision Tree (depth=6)"
    )
    
    # LOO cross-validation for tree
    tree_cv = cross_validate(X, y, algo_classes, max_depth=6)
    
    # Load perceptron weights
    try:
        perceptron_weights = load_type_weights('type_0', str(WEIGHTS_DIR))
        all_results = load_all_results()
        perceptron_result = evaluate_weights(
            perceptron_weights,
            dict(perf_matrix),
            graphs,
            all_results,
            benchmark,
        )
        has_perceptron = True
    except Exception as e:
        log.warning(f"Could not load perceptron weights: {e}")
        perceptron_result = {'accuracy': 0, 'avg_regret_pct': 100}
        has_perceptron = False
    
    comparison = {
        'benchmark': benchmark,
        'decision_tree': {
            'train_accuracy': tree_result['accuracy'],
            'loo_accuracy': tree_cv['loo_accuracy'],
            'avg_regret': tree_result['avg_regret_pct'],
        },
        'perceptron': {
            'accuracy': perceptron_result.get('accuracy', 0),
            'avg_regret': perceptron_result.get('avg_regret_pct', 0),
        },
        'has_perceptron': has_perceptron,
    }
    
    return comparison


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Decision Tree Algorithm Selector for GraphBrew",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Train and show results
          python3 -m scripts.lib.decision_tree --train
          
          # Train with grid search over tree depths
          python3 -m scripts.lib.decision_tree --grid-search
          
          # Export tree as C++ code
          python3 -m scripts.lib.decision_tree --train --export-cpp
          
          # Compare with perceptron
          python3 -m scripts.lib.decision_tree --compare
          
          # Train per-benchmark trees
          python3 -m scripts.lib.decision_tree --train --per-benchmark
        """),
    )
    
    parser.add_argument('--train', action='store_true',
                       help='Train a Decision Tree and show results')
    parser.add_argument('--grid-search', action='store_true',
                       help='Grid search over tree depths')
    parser.add_argument('--compare', action='store_true',
                       help='Compare Decision Tree vs Perceptron')
    parser.add_argument('--export-cpp', action='store_true',
                       help='Export tree as C++ function')
    parser.add_argument('--export-json', action='store_true',
                       help='Export tree as JSON model')
    parser.add_argument('--per-benchmark', action='store_true',
                       help='Train separate trees per benchmark')
    parser.add_argument('--benchmark', default='pr',
                       help='Benchmark to train/evaluate (default: pr)')
    parser.add_argument('--max-depth', type=int, default=6,
                       help='Maximum tree depth (default: 6)')
    parser.add_argument('--min-leaf', type=int, default=2,
                       help='Min samples per leaf (default: 2)')
    parser.add_argument('--show-tree', action='store_true',
                       help='Print the tree in text format')
    parser.add_argument('--feature-importance', action='store_true',
                       help='Show feature importance ranking')
    parser.add_argument('--random-forest', action='store_true',
                       help='Also train a Random Forest for comparison')
    parser.add_argument('--use-interactions', action='store_true',
                       help='Use feature interaction terms (16 features)')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Full grid search: DT vs RF, base vs interactions, '
                            'optimize for LOO regret')
    parser.add_argument('--auto-depth', action='store_true',
                       help='Auto-select optimal depth per benchmark via LOO regret '
                            '(for --per-benchmark mode)')
    parser.add_argument('--hybrid', action='store_true',
                       help='Train Hybrid DT+Perceptron model (Model Tree). '
                            'Uses --per-benchmark automatically.')
    
    args = parser.parse_args()
    
    if not any([args.train, args.grid_search, args.compare,
                args.export_cpp, args.feature_importance, args.fine_tune,
                args.hybrid]):
        parser.print_help()
        return
    
    # Load data
    perf_matrix, graph_props = load_benchmark_data()
    
    if not perf_matrix:
        log.error("No benchmark data found. Run benchmarks first.")
        sys.exit(1)
    
    if args.hybrid:
        print("\n" + "=" * 70)
        print("HYBRID DT + PERCEPTRON (MODEL TREE)")
        print("=" * 70)
        print(f"Architecture: Decision Tree routes → per-leaf Perceptron scores")
        print(f"Based on: LMT (Landwehr 2005), HME (Jordan & Jacobs 1994)")
        print()

        benchmarks = ['pr', 'bfs', 'cc', 'sssp', 'bc']
        hybrid_models = {}  # {bench: (dt_clf, leaf_perceptrons, families)}
        all_results = {}

        for bench in benchmarks:
            try:
                X, y, graphs, algo_classes = build_training_data(
                    perf_matrix, graph_props, bench
                )
            except ValueError:
                log.warning(f"No data for {bench}")
                continue

            # Auto-depth search for the DT routing component
            best_depth = args.max_depth
            best_leaf = args.min_leaf
            best_regret = float('inf')

            if args.auto_depth:
                print(f"\n--- {bench.upper()}: Auto-tuning hybrid depth ---")
                for d in range(2, 8):
                    for ml in [1, 2, 3, 5]:
                        cv = cross_validate_hybrid_regret(
                            X, y, algo_classes, graphs,
                            perf_matrix, bench,
                            max_depth=d, min_samples_leaf=ml,
                            n_epochs=200, learning_rate=0.01,
                        )
                        if cv['loo_avg_regret'] < best_regret:
                            best_regret = cv['loo_avg_regret']
                            best_depth = d
                            best_leaf = ml
                print(f"  Best: depth={best_depth}, min_leaf={best_leaf}, "
                      f"LOO_regret={best_regret:.1f}%")

            # Train final model
            dt_clf, leaf_perceptrons, families = train_hybrid_model(
                X, y, algo_classes, graphs,
                perf_matrix, bench,
                max_depth=best_depth,
                min_samples_leaf=best_leaf,
                n_epochs=200,
                learning_rate=0.01,
            )
            hybrid_models[bench] = (dt_clf, leaf_perceptrons, families)

            # Evaluate with LOO
            cv = cross_validate_hybrid_regret(
                X, y, algo_classes, graphs,
                perf_matrix, bench,
                max_depth=best_depth,
                min_samples_leaf=best_leaf,
            )

            # Also get pure DT LOO for comparison
            cv_dt = cross_validate_regret(
                X, y, algo_classes, graphs,
                perf_matrix, bench,
                max_depth=best_depth,
                min_samples_leaf=best_leaf,
            )

            all_results[bench] = {
                'hybrid': cv,
                'dt': cv_dt,
                'depth': best_depth,
                'leaf': best_leaf,
                'n_leaves': dt_clf.get_n_leaves(),
            }

            print(f"\n{bench.upper()} (depth={best_depth}, min_leaf={best_leaf}, "
                  f"leaves={dt_clf.get_n_leaves()}):")
            print(f"  {'Metric':<20} {'Hybrid':>10} {'Pure DT':>10}")
            print(f"  {'-'*42}")
            print(f"  {'LOO Accuracy':<20} {cv['loo_accuracy']:>9.1f}% {cv_dt['loo_accuracy']:>9.1f}%")
            print(f"  {'Within 5%':<20} {cv['loo_accuracy_5pct']:>9.1f}% {cv_dt['loo_accuracy_5pct']:>9.1f}%")
            print(f"  {'Within 10%':<20} {cv['loo_accuracy_10pct']:>9.1f}% {cv_dt['loo_accuracy_10pct']:>9.1f}%")
            print(f"  {'Avg Regret':<20} {cv['loo_avg_regret']:>9.1f}% {cv_dt['loo_avg_regret']:>9.1f}%")
            print(f"  {'Median Regret':<20} {cv['loo_median_regret']:>9.1f}% {cv_dt['loo_median_regret']:>9.1f}%")
            print(f"  {'Max Regret':<20} {cv['loo_max_regret']:>9.1f}% {cv_dt['loo_max_regret']:>9.1f}%")

        # Summary
        if all_results:
            print(f"\n{'='*70}")
            print("SUMMARY: Hybrid vs Pure DT")
            print(f"{'='*70}")
            print(f"{'Bench':>6} {'Hybrid Regret':>14} {'DT Regret':>10} {'Winner':>8} {'Depth':>6}")
            print(f"{'-'*48}")
            h_total, d_total = 0, 0
            for bench in benchmarks:
                if bench not in all_results:
                    continue
                r = all_results[bench]
                h_reg = r['hybrid']['loo_avg_regret']
                d_reg = r['dt']['loo_avg_regret']
                winner = 'Hybrid' if h_reg < d_reg else ('Tie' if h_reg == d_reg else 'DT')
                print(f"{bench.upper():>6} {h_reg:>13.1f}% {d_reg:>9.1f}% {winner:>8} {r['depth']:>6}")
                h_total += h_reg
                d_total += d_reg
            n_bench = len(all_results)
            print(f"{'AVG':>6} {h_total/n_bench:>13.1f}% {d_total/n_bench:>9.1f}%")

        if args.export_cpp and hybrid_models:
            cpp_code = export_hybrid_per_benchmark_to_cpp(hybrid_models)
            out_file = WEIGHTS_DIR / "hybrid_per_benchmark.cpp"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                f.write(cpp_code)
            print(f"\nC++ code written to {out_file}")
            print(f"Total lines: {cpp_code.count(chr(10))}")

        # Always export JSON models (runtime-loadable by C++)
        if hybrid_models:
            written = export_all_models_to_json(hybrid_models=hybrid_models)
            print(f"\nJSON models written ({len(written)} files):")
            for p in written:
                print(f"  {p}")

        return

    if args.grid_search:
        print("\n" + "=" * 70)
        print("GRID SEARCH: Decision Tree Depth vs Accuracy (Regret-Aware)")
        print("=" * 70)
        
        benchmarks = ['pr', 'bfs', 'cc', 'sssp', 'bc']
        
        for bench in benchmarks:
            try:
                X, y, graphs, algo_classes = build_training_data(
                    perf_matrix, graph_props, bench
                )
            except ValueError:
                continue
            
            results = grid_search_depth(
                X, y, algo_classes,
                graph_names=graphs, perf_matrix=perf_matrix, benchmark=bench,
            )
            
            has_regret = 'loo_avg_regret' in results[0]
            
            print(f"\n--- {bench.upper()} ({len(X)} graphs, {len(algo_classes)} classes) ---")
            if has_regret:
                print(f"{'Depth':>5} {'MinLf':>5} {'Train%':>7} {'LOO%':>6} "
                      f"{'@5%':>5} {'@10%':>5} {'Regret':>7} {'MedReg':>7}")
                for r in results[:8]:
                    marker = " <--" if r == results[0] else ""
                    print(f"{r['depth']:5d} {r['min_leaf']:5d} {r['train_acc']:6.1f}% "
                          f"{r['loo_acc']:5.1f}% {r['loo_within_5']:4.0f}% "
                          f"{r['loo_within_10']:4.0f}% {r['loo_avg_regret']:6.1f}% "
                          f"{r['loo_median_regret']:6.1f}%{marker}")
            else:
                print(f"{'Depth':>5} {'MinLeaf':>7} {'Train%':>7} {'LOO%':>6} {'Leaves':>6}")
                for r in results[:8]:
                    marker = " <--" if r == results[0] else ""
                    print(f"{r['depth']:5d} {r['min_leaf']:7d} {r['train_acc']:6.1f}% "
                          f"{r['loo_acc']:5.1f}% {r['n_leaves']:6d}{marker}")
        
        return
    
    if args.fine_tune:
        print("\n" + "=" * 70)
        print("FINE-TUNE: Full Grid Search (DT vs RF × base vs interactions)")
        print("=" * 70)
        
        benchmarks = ['pr', 'bfs', 'cc', 'sssp', 'bc']
        best_configs = {}
        
        for bench in benchmarks:
            print(f"\n{'='*50}")
            print(f"  {bench.upper()}")
            print(f"{'='*50}")
            
            search = grid_search_full(perf_matrix, graph_props, bench)
            
            if not search['best']:
                print("  No results")
                continue
            
            best_configs[bench] = search['best']
            
            print(f"{'Model':>8} {'Feats':>12} {'Depth':>5} {'MinLf':>5} "
                  f"{'LOO%':>6} {'@5%':>5} {'@10%':>5} {'Regret':>7} {'MedReg':>7}")
            for r in search['top_10']:
                marker = " <--" if r == search['best'] else ""
                print(f"{r['model']:>8} {r['features']:>12} {r['depth']:5d} "
                      f"{r['min_leaf']:5d} {r['loo_acc']:5.1f}% {r['loo_5pct']:4.0f}% "
                      f"{r['loo_10pct']:4.0f}% {r['loo_regret']:6.1f}% "
                      f"{r['loo_median_regret']:6.1f}%{marker}")
        
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION PER BENCHMARK")
        print(f"{'='*70}")
        for bench, cfg in best_configs.items():
            print(f"  {bench.upper():>5}: model={cfg['model']}, features={cfg['features']}, "
                  f"depth={cfg['depth']}, min_leaf={cfg['min_leaf']} → "
                  f"LOO@5%={cfg['loo_5pct']:.0f}%, regret={cfg['loo_regret']:.1f}%")
        
        return
    
    if args.per_benchmark:
        print("\n" + "=" * 70)
        print("PER-BENCHMARK DECISION TREES")
        print("=" * 70)
        
        use_inter = args.use_interactions
        feat_fn = extract_features_with_interactions if use_inter else extract_features
        feat_names = FEATURE_NAMES_INTERACTIONS if use_inter else FEATURE_NAMES
        
        benchmarks = ['pr', 'bfs', 'cc', 'sssp', 'bc']
        trees = {}
        all_cpp = ''
        
        for bench in benchmarks:
            try:
                # Build training data with chosen feature set
                X_rows, y_labels, graph_names_list = [], [], []
                for graph in sorted(perf_matrix.keys()):
                    feat = feat_fn(graph, graph_props)
                    if feat is None:
                        continue
                    best_label, best_time = find_best_family(perf_matrix, graph, bench)
                    if best_time == float('inf'):
                        continue
                    X_rows.append(feat)
                    y_labels.append(best_label)
                    graph_names_list.append(graph)
                
                if not X_rows:
                    log.warning(f"No data for {bench}")
                    continue
                
                X = np.array(X_rows)
                le = LabelEncoder()
                y = le.fit_transform(y_labels)
                algo_classes = list(le.classes_)
                graphs = graph_names_list
                
                log.info(f"Built training data: {len(X)} graphs, {len(algo_classes)} classes (families)")
                for ac in algo_classes:
                    cnt = sum(1 for l in y_labels if l == ac)
                    log.info(f"    {ac:35s}: {cnt:3d} graphs ({100*cnt/len(y_labels):.0f}%)")
                    
            except ValueError:
                log.warning(f"No data for {bench}")
                continue
            
            if args.random_forest:
                clf = train_random_forest(
                    X, y, algo_classes,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_leaf,
                )
            else:
                # Auto-select best depth if requested
                if args.auto_depth:
                    best_depth = args.max_depth
                    best_leaf = args.min_leaf
                    best_regret = float('inf')
                    for d in range(2, 10):
                        for ml in [1, 2, 3, 5]:
                            cv_r = cross_validate_regret(
                                X, y, algo_classes, graphs,
                                perf_matrix, bench,
                                max_depth=d, min_samples_leaf=ml,
                            )
                            if cv_r['loo_avg_regret'] < best_regret:
                                best_regret = cv_r['loo_avg_regret']
                                best_depth = d
                                best_leaf = ml
                    log.info(f"Auto-depth: {bench} → depth={best_depth}, "
                             f"min_leaf={best_leaf}, LOO_regret={best_regret:.1f}%")
                    clf = train_decision_tree(
                        X, y, algo_classes,
                        max_depth=best_depth,
                        min_samples_leaf=best_leaf,
                    )
                else:
                    clf = train_decision_tree(
                        X, y, algo_classes,
                        max_depth=args.max_depth,
                        min_samples_leaf=args.min_leaf,
                    )
            
            # Evaluate
            actual_depth = clf.get_depth() if hasattr(clf, 'get_depth') else args.max_depth
            actual_leaf = args.min_leaf
            if args.auto_depth and hasattr(clf, 'tree_'):
                # Use the auto-selected params for CV consistency
                actual_depth = best_depth  # noqa: F821 — set in auto_depth block
                actual_leaf = best_leaf    # noqa: F821
            
            result = evaluate_model(
                clf, X, y, algo_classes, graphs, perf_matrix, bench,
                label=f"DT-{bench}"
            )
            
            # Cross-validate with regret using SAME params as the trained tree
            cv_regret = cross_validate_regret(
                X, y, algo_classes, graphs, perf_matrix, bench,
                max_depth=actual_depth, min_samples_leaf=actual_leaf,
                use_forest=args.random_forest,
            )
            
            print(f"\n{bench.upper()}: train_acc={result['accuracy']:.1f}%, "
                  f"LOO_acc={cv_regret['loo_accuracy']:.1f}%, "
                  f"LOO@5%={cv_regret['loo_accuracy_5pct']:.1f}%, "
                  f"LOO@10%={cv_regret['loo_accuracy_10pct']:.1f}%, "
                  f"LOO_regret={cv_regret['loo_avg_regret']:.1f}%")
            
            # For trees (not forests), we can export to C++
            if hasattr(clf, 'tree_'):
                trees[bench] = (clf, algo_classes)
            else:
                # For Random Forest, extract the best single tree by accuracy
                best_tree = None
                best_tree_acc = -1
                for estimator in clf.estimators_:
                    acc = estimator.score(X, y)
                    if acc > best_tree_acc:
                        best_tree_acc = acc
                        best_tree = estimator
                if best_tree is not None:
                    trees[bench] = (best_tree, algo_classes)
            
            if args.show_tree and hasattr(clf, 'tree_'):
                print(export_text(clf, feature_names=feat_names, max_depth=10))
            
            if args.feature_importance:
                feature_importance_report(clf, feat_names)
        
        if args.export_cpp and trees:
            cpp_code = export_per_benchmark_trees_to_cpp(
                trees, use_interactions=use_inter
            )
            out_file = WEIGHTS_DIR / "decision_tree_per_benchmark.cpp"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                f.write(cpp_code)
            print(f"\nC++ code written to {out_file}")
            print(f"Total lines: {cpp_code.count(chr(10))}")
        
        if args.export_json and trees:
            for bench, (clf, algo_classes) in trees.items():
                model = export_tree_to_json(clf, algo_classes, FEATURE_NAMES)
                out_file = WEIGHTS_DIR / f"decision_tree_{bench}.json"
                with open(out_file, 'w') as f:
                    json.dump(model, f, indent=2)
                print(f"JSON model written to {out_file}")
        
        # Always export flat JSON models (runtime-loadable by C++)
        if trees:
            written = export_all_models_to_json(dt_trees=trees)
            print(f"\nJSON models written ({len(written)} files):")
            for p in written:
                print(f"  {p}")

        return
    
    # Single-benchmark mode
    X, y, graphs, algo_classes = build_training_data(
        perf_matrix, graph_props, args.benchmark
    )
    
    if args.train or args.export_cpp or args.export_json:
        clf = train_decision_tree(
            X, y, algo_classes,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_leaf,
        )
        
        result = evaluate_model(
            clf, X, y, algo_classes, graphs, perf_matrix, args.benchmark,
            label="Decision Tree"
        )
        cv = cross_validate(X, y, algo_classes, max_depth=args.max_depth)
        
        print(f"\n{'='*70}")
        print(f"DECISION TREE RESULTS — {args.benchmark.upper()}")
        print(f"{'='*70}")
        print(f"Training accuracy:  {result['accuracy']:.1f}%")
        print(f"LOO accuracy:       {cv['loo_accuracy']:.1f}% (±{cv['loo_std']:.1f}%)")
        print(f"Average regret:     {result['avg_regret_pct']:.1f}%")
        print(f"Within 5% of best:  {result['accuracy_5pct']:.1f}%")
        print(f"Within 10% of best: {result['accuracy_10pct']:.1f}%")
        print(f"Tree depth:         {clf.get_depth()}")
        print(f"Tree leaves:        {clf.get_n_leaves()}")
        
        if args.show_tree:
            print(f"\n--- Tree Structure ---")
            print(export_text(clf, feature_names=FEATURE_NAMES, max_depth=10))
        
        if args.feature_importance:
            feature_importance_report(clf, FEATURE_NAMES)
        
        # Show misclassifications
        wrong = [d for d in result['details'] if not d['correct']]
        if wrong:
            print(f"\n--- Misclassifications ({len(wrong)}) ---")
            for d in wrong:
                print(f"  {d['graph']:25s}: predicted={d['predicted']:25s} "
                      f"actual={d['actual']:25s} regret={d['regret_pct']:.1f}%")
        
        if args.export_cpp:
            cpp_code = export_tree_to_cpp(clf, algo_classes, benchmark=args.benchmark)
            out_file = WEIGHTS_DIR / f"decision_tree_{args.benchmark}.cpp"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w') as f:
                f.write(cpp_code)
            print(f"\nC++ code written to {out_file}")
            # Also print to stdout
            print(cpp_code)
        
        if args.export_json:
            model = export_tree_to_json(clf, algo_classes, FEATURE_NAMES, {
                'benchmark': args.benchmark,
                'train_accuracy': result['accuracy'],
                'loo_accuracy': cv['loo_accuracy'],
                'avg_regret': result['avg_regret_pct'],
            })
            out_file = WEIGHTS_DIR / f"decision_tree_{args.benchmark}.json"
            with open(out_file, 'w') as f:
                json.dump(model, f, indent=2)
            print(f"JSON model written to {out_file}")
        
        if args.random_forest:
            print(f"\n--- Random Forest Comparison ---")
            rf = train_random_forest(X, y, algo_classes)
            rf_result = evaluate_model(
                rf, X, y, algo_classes, graphs, perf_matrix, args.benchmark,
                label="Random Forest"
            )
            print(f"RF accuracy:  {rf_result['accuracy']:.1f}%")
            print(f"RF regret:    {rf_result['avg_regret_pct']:.1f}%")
            feature_importance_report(rf, FEATURE_NAMES)
    
    if args.compare:
        comparison = compare_with_perceptron(
            perf_matrix, graph_props, args.benchmark
        )
        
        print(f"\n{'='*70}")
        print(f"COMPARISON: Decision Tree vs Perceptron — {args.benchmark.upper()}")
        print(f"{'='*70}")
        dt = comparison['decision_tree']
        pt = comparison['perceptron']
        print(f"{'Metric':<25} {'Decision Tree':>15} {'Perceptron':>15}")
        print(f"{'-'*55}")
        print(f"{'Train Accuracy':<25} {dt['train_accuracy']:>14.1f}% {'N/A':>15}")
        print(f"{'LOO Accuracy':<25} {dt['loo_accuracy']:>14.1f}% {pt['accuracy']:>14.1f}%")
        print(f"{'Avg Regret':<25} {dt['avg_regret']:>14.1f}% {pt['avg_regret']:>14.1f}%")


if __name__ == '__main__':
    main()
