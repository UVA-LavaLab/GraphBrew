#!/usr/bin/env python3
"""
Model Tree (Decision Tree / Hybrid) for algorithm selection.

Mirrors the C++ ModelTree / ModelTreeNode structs from reorder_types.h.
Supports:
  - Pure Decision Tree: leaf_class prediction
  - Hybrid (Model Tree): per-leaf perceptron weights for scoring

The 12-feature vector matches C++ ModelTree::extract_features():
  [0] modularity
  [1] hub_concentration
  [2] log10(N+1)
  [3] log10(E+1)
  [4] density
  [5] avg_degree/100
  [6] clustering_coeff
  [7] packing_factor
  [8] forward_edge_fraction
  [9] log2(working_set_ratio+1)
  [10] log10(community_count+1)
  [11] diameter/50

Training uses scikit-learn and exports to C++-compatible JSON format.
"""

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core.utils import RESULTS_DIR


# ============================================================================
# Criterion enum — the optimization objective for "best" algorithm
# ============================================================================

class Criterion(Enum):
    """What metric defines the 'best' reordering algorithm."""
    FASTEST_REORDER = "fastest-reorder"       # Minimize reorder time only
    FASTEST_EXECUTION = "fastest-execution"   # Minimize kernel execution time
    BEST_ENDTOEND = "best-endtoend"           # Minimize (reorder + execution)
    BEST_AMORTIZATION = "best-amortization"   # Minimize iterations to amortize


def criterion_value(record: dict, criterion: Criterion,
                    original_exec: float = None) -> float:
    """Compute the criterion-specific metric for a benchmark record.

    Args:
        record: dict with 'time_seconds' (kernel), 'reorder_time', 'algorithm'
        criterion: which objective
        original_exec: execution time of ORIGINAL (for amortization)

    Returns:
        float value to MINIMIZE (lower is better).
    """
    exec_t = record.get('time_seconds', 999.0)
    reorder_t = record.get('reorder_time', 0.0)
    algo = record.get('algorithm', '')

    if criterion == Criterion.FASTEST_REORDER:
        if algo == 'ORIGINAL':
            return float('inf')  # ORIGINAL doesn't reorder
        return reorder_t

    elif criterion == Criterion.FASTEST_EXECUTION:
        return exec_t

    elif criterion == Criterion.BEST_ENDTOEND:
        return exec_t + reorder_t

    elif criterion == Criterion.BEST_AMORTIZATION:
        if original_exec is None:
            original_exec = 999.0
        saved = original_exec - exec_t
        if algo == 'ORIGINAL' or saved <= 0:
            return float('inf')  # No benefit
        return reorder_t / saved  # iterations needed to amortize

    return exec_t  # fallback


def compute_oracle(records: List[dict], criterion: Criterion) -> Tuple[str, float]:
    """Find the oracle (best) algorithm for given criterion.

    Args:
        records: list of dicts for one (graph, benchmark) with
                 keys: algorithm, time_seconds, reorder_time, success

    Returns:
        (best_algorithm, best_value) tuple
    """
    if not records:
        return 'ORIGINAL', float('inf')

    # Get ORIGINAL time for amortization
    original_exec = None
    for r in records:
        if r.get('algorithm') == 'ORIGINAL' and r.get('success', True):
            original_exec = r.get('time_seconds', 999.0)
            break

    best_algo = 'ORIGINAL'
    best_val = float('inf')

    for r in records:
        if not r.get('success', True) or r.get('time_seconds', 0) <= 0:
            continue
        val = criterion_value(r, criterion, original_exec)
        if val < best_val:
            best_val = val
            best_algo = r.get('algorithm', 'ORIGINAL')

    return best_algo, best_val


# ============================================================================
# Feature extraction (12D — matches C++ ModelTree::extract_features)
# ============================================================================

MODEL_TREE_N_FEATURES = 12

DT_FEATURE_NAMES = [
    'modularity', 'hub_concentration', 'log_nodes', 'log_edges',
    'density', 'avg_degree_100', 'clustering_coeff', 'packing_factor',
    'forward_edge_fraction', 'log2_wsr', 'log10_community_count',
    'diameter_50',
]


def extract_dt_features(props: dict) -> List[float]:
    """Extract the 12-element feature vector for DT splits.

    Matches C++ ModelTree::extract_features() exactly.
    """
    nodes = props.get('nodes', 1000)
    edges = props.get('edges', 5000)
    avg_degree = props.get('avg_degree', 10.0)

    return [
        props.get('modularity', 0.5),                                    # 0
        props.get('hub_concentration', 0.3),                             # 1
        math.log10(nodes + 1) if nodes > 0 else 0,                      # 2
        math.log10(edges + 1) if edges > 0 else 0,                      # 3
        avg_degree / (nodes - 1) if nodes > 1 else 0,                   # 4: density
        avg_degree / 100.0,                                              # 5
        props.get('clustering_coefficient', 0.0),                        # 6
        props.get('packing_factor', 0.0),                                # 7
        props.get('forward_edge_fraction', 0.5),                         # 8
        math.log2(props.get('working_set_ratio', 0.0) + 1.0),           # 9
        math.log10(props.get('community_count', 0.0) + 1.0),            # 10
        props.get('diameter_estimate', props.get('diameter', 0.0)) / 50, # 11
    ]


# ============================================================================
# ModelTreeNode / ModelTree — mirrors C++ structs
# ============================================================================

@dataclass
class ModelTreeNode:
    """A single node in a model tree (matches C++ ModelTreeNode)."""
    feature_idx: int = -1           # -1 = leaf
    threshold: float = 0.0
    left: int = -1
    right: int = -1
    leaf_class: str = ''
    samples: int = 0
    leaf_weights: Dict[str, List[float]] = field(default_factory=dict)

    def is_leaf(self) -> bool:
        return self.feature_idx < 0

    def to_dict(self) -> dict:
        """Serialize to C++-compatible JSON dict."""
        d = {}
        if self.is_leaf():
            d['leaf_class'] = self.leaf_class
            d['samples'] = self.samples
            if self.leaf_weights:
                d['weights'] = {fam: list(w) for fam, w in self.leaf_weights.items()}
        else:
            d['feature_idx'] = self.feature_idx
            d['threshold'] = self.threshold
            d['left'] = self.left
            d['right'] = self.right
            d['samples'] = self.samples
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelTreeNode':
        """Deserialize from C++ JSON dict."""
        node = cls()
        node.feature_idx = d.get('feature_idx', -1)
        node.threshold = d.get('threshold', 0.0)
        node.left = d.get('left', -1)
        node.right = d.get('right', -1)
        node.leaf_class = d.get('leaf_class', '')
        node.samples = d.get('samples', 0)
        weights = d.get('weights', {})
        node.leaf_weights = {k: list(v) for k, v in weights.items()}
        return node


class ModelTree:
    """A complete model tree for one benchmark (matches C++ ModelTree)."""

    def __init__(self, model_type: str = 'decision_tree', benchmark: str = '',
                 families: List[str] = None, nodes: List[ModelTreeNode] = None):
        self.model_type = model_type
        self.benchmark = benchmark
        self.families = families or []
        self.nodes = nodes or []

    def predict(self, features_12d: List[float]) -> str:
        """Walk the tree and return predicted algorithm family.

        For pure DT: returns leaf_class directly.
        For hybrid: scores families via dot product at leaf.

        Matches C++ ModelTree::predict() exactly.
        """
        if not self.nodes:
            return 'ORIGINAL'

        idx = 0
        max_iters = len(self.nodes) * 2
        for _ in range(max_iters):
            if idx < 0 or idx >= len(self.nodes):
                break
            node = self.nodes[idx]

            if node.is_leaf():
                # Hybrid: score families via perceptron dot products
                if node.leaf_weights:
                    best_score = -1e30
                    best_family = node.leaf_class
                    for fam, weights in node.leaf_weights.items():
                        score = 0.0
                        n_w = len(weights)
                        # weights = [w0, w1, ..., w_{n-2}, bias]
                        for i in range(min(n_w - 1, MODEL_TREE_N_FEATURES)):
                            score += weights[i] * features_12d[i]
                        if n_w > 0:
                            score += weights[-1]  # bias
                        if score > best_score:
                            best_score = score
                            best_family = fam
                    return best_family
                # Pure DT: return class directly
                return node.leaf_class

            # Split node
            feat_val = (features_12d[node.feature_idx]
                        if 0 <= node.feature_idx < len(features_12d) else 0.0)
            if feat_val <= node.threshold:
                idx = node.left
            else:
                idx = node.right

        return 'ORIGINAL'  # safety fallback

    def predict_from_props(self, props: dict) -> str:
        """Convenience: extract features, then predict."""
        feats = extract_dt_features(props)
        return self.predict(feats)

    def to_dict(self) -> dict:
        """Serialize to C++-compatible JSON."""
        return {
            'model_type': self.model_type,
            'benchmark': self.benchmark,
            'families': self.families,
            'nodes': [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelTree':
        """Deserialize from C++ JSON format."""
        mt = cls(
            model_type=d.get('model_type', 'decision_tree'),
            benchmark=d.get('benchmark', ''),
            families=d.get('families', []),
        )
        mt.nodes = [ModelTreeNode.from_dict(nd) for nd in d.get('nodes', [])]
        return mt


# ============================================================================
# Training: build DT / Hybrid from benchmark data
# ============================================================================

def _build_training_data(
    bench_records: List[dict],
    graph_props: dict,
    benchmark: str,
    criterion: Criterion = Criterion.FASTEST_EXECUTION,
) -> Tuple[List[List[float]], List[str], List[str]]:
    """Build (X, y, graphs) training data for one benchmark.

    Args:
        bench_records: all benchmark result dicts (all benchmarks/graphs)
        graph_props: dict of graph_name → properties
        benchmark: which benchmark to train for
        criterion: optimization objective for labeling

    Returns:
        X: list of 12D feature vectors (one per graph)
        y: list of best algorithm labels per graph
        graphs: list of graph names (parallel to X, y)
    """
    from collections import defaultdict

    # Group records by graph for this benchmark
    by_graph = defaultdict(list)
    for r in bench_records:
        if (r.get('benchmark') == benchmark and
                r.get('success', True) and
                r.get('time_seconds', 0) > 0):
            by_graph[r.get('graph', '')].append(r)

    X, y, graph_names = [], [], []
    for graph_name in sorted(by_graph):
        if graph_name not in graph_props:
            continue
        if graph_name == 'tiny':
            continue  # skip tiny graph

        records = by_graph[graph_name]
        best_algo, _ = compute_oracle(records, criterion)

        feats = extract_dt_features(graph_props[graph_name])
        X.append(feats)
        y.append(best_algo)
        graph_names.append(graph_name)

    return X, y, graph_names


def train_decision_tree(
    bench_records: List[dict],
    graph_props: dict,
    benchmark: str,
    criterion: Criterion = Criterion.FASTEST_EXECUTION,
    max_depth: int = 4,
    min_samples_leaf: int = 1,
) -> ModelTree:
    """Train a Decision Tree classifier for one benchmark.

    Uses scikit-learn internally but exports to our ModelTree format
    for C++ compatibility.
    """
    from sklearn.tree import DecisionTreeClassifier

    X, y, graph_names = _build_training_data(
        bench_records, graph_props, benchmark, criterion)

    if len(X) < 2:
        # Not enough data — return trivial tree
        mt = ModelTree('decision_tree', benchmark,
                       families=sorted(set(y)) if y else ['ORIGINAL'])
        mt.nodes = [ModelTreeNode(leaf_class=y[0] if y else 'ORIGINAL',
                                  samples=len(X))]
        return mt

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    clf.fit(X, y)

    # Convert sklearn tree to our flat node format
    mt = _sklearn_to_model_tree(clf, benchmark, 'decision_tree')
    return mt


def train_hybrid_tree(
    bench_records: List[dict],
    graph_props: dict,
    benchmark: str,
    criterion: Criterion = Criterion.FASTEST_EXECUTION,
    max_depth: int = 3,
    min_samples_leaf: int = 1,
) -> ModelTree:
    """Train a Hybrid (Model Tree) for one benchmark.

    First trains a DT structure, then fits per-leaf perceptron weights
    on the leaf node's training samples.
    """
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    X, y, graph_names = _build_training_data(
        bench_records, graph_props, benchmark, criterion)

    if len(X) < 2:
        mt = ModelTree('hybrid', benchmark,
                       families=sorted(set(y)) if y else ['ORIGINAL'])
        mt.nodes = [ModelTreeNode(leaf_class=y[0] if y else 'ORIGINAL',
                                  samples=len(X))]
        return mt

    X_arr = np.array(X)
    families = sorted(set(y))

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    clf.fit(X_arr, y)

    # Convert to model tree
    mt = _sklearn_to_model_tree(clf, benchmark, 'hybrid')

    # For each leaf, train per-family perceptron weights
    leaf_assignments = clf.apply(X_arr)  # leaf node indices
    sklearn_to_mt_map = _build_sklearn_node_map(clf.tree_)

    for mt_idx, node in enumerate(mt.nodes):
        if not node.is_leaf():
            continue

        # Find sklearn leaf index for this mt node
        sk_idx = None
        for sk_i, mt_i in sklearn_to_mt_map.items():
            if mt_i == mt_idx:
                sk_idx = sk_i
                break
        if sk_idx is None:
            continue

        # Get samples at this leaf
        mask = leaf_assignments == sk_idx
        leaf_X = X_arr[mask]
        leaf_y = [y[i] for i in range(len(y)) if mask[i]]

        if len(leaf_X) < 1:
            continue

        # Train simple per-family weights via averaged perceptron
        weights = _train_leaf_perceptron(leaf_X, leaf_y, families)
        node.leaf_weights = weights

    mt.families = families
    return mt


def _train_leaf_perceptron(
    X: 'np.ndarray', y: List[str], families: List[str],
    epochs: int = 100,
) -> Dict[str, List[float]]:
    """Train a multi-class averaged perceptron on leaf samples.

    Returns dict of family → [w0, ..., w11, bias] (13 elements).
    """
    import numpy as np

    n_features = X.shape[1]
    n_families = len(families)
    fam_to_idx = {f: i for i, f in enumerate(families)}

    # Initialize weights [n_families × (n_features + 1)]
    W = np.zeros((n_families, n_features + 1))
    W_sum = np.zeros_like(W)
    count = 0

    for epoch in range(epochs):
        for xi, yi in zip(X, y):
            if yi not in fam_to_idx:
                continue
            true_idx = fam_to_idx[yi]

            # Score all families
            scores = W[:, :n_features] @ xi + W[:, -1]
            pred_idx = int(np.argmax(scores))

            if pred_idx != true_idx:
                # Update: promote correct, demote predicted
                W[true_idx, :n_features] += xi
                W[true_idx, -1] += 1.0
                W[pred_idx, :n_features] -= xi
                W[pred_idx, -1] -= 1.0

            count += 1
            W_sum += W

    # Average
    if count > 0:
        W_avg = W_sum / count
    else:
        W_avg = W

    result = {}
    for fam, idx in fam_to_idx.items():
        result[fam] = list(W_avg[idx])  # [w0, ..., w11, bias]
    return result


def _sklearn_to_model_tree(clf, benchmark: str, model_type: str) -> ModelTree:
    """Convert a fitted sklearn DecisionTreeClassifier to ModelTree format."""
    tree = clf.tree_
    classes = list(clf.classes_)

    # Build flat node array via BFS
    mt_nodes = []
    node_map = _build_sklearn_node_map(tree)

    # Create nodes in mt index order
    mt_size = max(node_map.values()) + 1 if node_map else 1
    mt_nodes = [ModelTreeNode() for _ in range(mt_size)]

    for sk_idx, mt_idx in node_map.items():
        node = mt_nodes[mt_idx]
        node.samples = int(tree.n_node_samples[sk_idx])

        if tree.children_left[sk_idx] == tree.children_right[sk_idx]:
            # Leaf
            node.feature_idx = -1
            class_idx = int(tree.value[sk_idx].argmax())
            node.leaf_class = classes[class_idx]
        else:
            # Split
            node.feature_idx = int(tree.feature[sk_idx])
            node.threshold = float(tree.threshold[sk_idx])
            node.left = node_map[tree.children_left[sk_idx]]
            node.right = node_map[tree.children_right[sk_idx]]

    mt = ModelTree(model_type, benchmark, families=classes, nodes=mt_nodes)
    return mt


def _build_sklearn_node_map(tree) -> Dict[int, int]:
    """Map sklearn tree node indices to flat BFS order (our format)."""
    from collections import deque
    node_map = {}
    queue = deque([0])
    mt_idx = 0
    while queue:
        sk_idx = queue.popleft()
        node_map[sk_idx] = mt_idx
        mt_idx += 1
        left = tree.children_left[sk_idx]
        right = tree.children_right[sk_idx]
        if left != right:  # not a leaf
            queue.append(left)
            queue.append(right)
    return node_map


# ============================================================================
# IO: load / save adaptive_models.json (C++ format)
# ============================================================================

def load_adaptive_models(path: Path = None) -> Dict[str, Dict[str, ModelTree]]:
    """Load adaptive_models.json.

    Returns:
        Dict with keys 'decision_tree', 'hybrid', each mapping
        benchmark → ModelTree.
    """
    if path is None:
        path = Path(RESULTS_DIR) / 'data' / 'adaptive_models.json'

    result = {'decision_tree': {}, 'hybrid': {}}
    if not path.exists():
        return result

    with open(path) as f:
        data = json.load(f)

    for subdir in ('decision_tree', 'hybrid'):
        section = data.get(subdir, {})
        for bench, tree_data in section.items():
            result[subdir][bench] = ModelTree.from_dict(tree_data)

    return result


def save_adaptive_models(
    models: Dict[str, Dict[str, ModelTree]],
    path: Path = None,
):
    """Save DT/Hybrid models to adaptive_models.json in C++ format."""
    if path is None:
        path = Path(RESULTS_DIR) / 'data' / 'adaptive_models.json'

    data = {}
    for subdir in ('decision_tree', 'hybrid'):
        data[subdir] = {}
        for bench, tree in models.get(subdir, {}).items():
            data[subdir][bench] = tree.to_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def train_all_models(
    bench_records: List[dict],
    graph_props: dict,
    benchmarks: List[str] = None,
    criterion: Criterion = Criterion.FASTEST_EXECUTION,
    save_path: Path = None,
) -> Dict[str, Dict[str, ModelTree]]:
    """Train DT and Hybrid models for all benchmarks.

    Args:
        bench_records: raw benchmark data (list of dicts)
        graph_props: graph properties lookup
        benchmarks: list of benchmarks to train for (default: all 8)
        criterion: optimization target
        save_path: where to save adaptive_models.json (default: results/data/)

    Returns:
        Dict with 'decision_tree' and 'hybrid' sub-dicts.
    """
    if benchmarks is None:
        benchmarks = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

    models = {'decision_tree': {}, 'hybrid': {}}

    for bench in benchmarks:
        dt = train_decision_tree(bench_records, graph_props, bench, criterion)
        hy = train_hybrid_tree(bench_records, graph_props, bench, criterion)
        models['decision_tree'][bench] = dt
        models['hybrid'][bench] = hy

    if save_path is not None:
        save_adaptive_models(models, save_path)

    return models


# ============================================================================
# LOGO CV for DT/Hybrid
# ============================================================================

def cross_validate_logo_model_tree(
    bench_records: List[dict],
    graph_props: dict,
    model_type: str = 'decision_tree',
    criterion: Criterion = Criterion.FASTEST_EXECUTION,
    benchmarks: List[str] = None,
) -> Dict:
    """Leave-One-Graph-Out evaluation for DT or Hybrid models.

    For each graph:
      1. Train model on all OTHER graphs
      2. Predict on held-out graph
      3. Compare to oracle per criterion

    Returns:
        Dict with accuracy, regret, per-graph breakdown
    """
    if benchmarks is None:
        benchmarks = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

    from collections import defaultdict

    # Find all graphs
    graphs = set()
    for r in bench_records:
        g = r.get('graph', '')
        if g and g != 'tiny' and r.get('success', True) and r.get('time_seconds', 0) > 0:
            graphs.add(g)
    graphs = sorted(graphs)

    if len(graphs) < 3:
        return {'accuracy': 0, 'total': 0, 'error': 'Need >= 3 graphs'}

    correct = 0
    total = 0
    regrets = []
    per_graph = {}

    for held_out in graphs:
        if held_out not in graph_props:
            continue

        # Train on all graphs except held_out
        train_records = [r for r in bench_records if r.get('graph') != held_out]
        train_fn = train_decision_tree if model_type == 'decision_tree' else train_hybrid_tree

        feats_12d = extract_dt_features(graph_props[held_out])
        g_correct = 0
        g_total = 0
        g_regrets = []

        for bench in benchmarks:
            # Records for held_out graph, this benchmark
            held_records = [r for r in bench_records
                            if r.get('graph') == held_out
                            and r.get('benchmark') == bench
                            and r.get('success', True)
                            and r.get('time_seconds', 0) > 0]
            if not held_records:
                continue

            # Oracle for this (graph, bench)
            oracle_algo, oracle_val = compute_oracle(held_records, criterion)

            # Train model on other graphs, predict
            tree = train_fn(train_records, graph_props, bench, criterion)
            predicted = tree.predict(feats_12d)

            g_total += 1
            total += 1
            if predicted == oracle_algo:
                g_correct += 1
                correct += 1

            # Regret: find predicted algo's value
            pred_val = None
            original_exec = None
            for r in held_records:
                if r.get('algorithm') == 'ORIGINAL':
                    original_exec = r.get('time_seconds', 999)
            for r in held_records:
                if r.get('algorithm') == predicted:
                    pred_val = criterion_value(r, criterion, original_exec)
                    break
            if pred_val is None:
                # Predicted algo not in held_out data, use worst
                finite_vals = [criterion_value(r, criterion, original_exec)
                               for r in held_records
                               if criterion_value(r, criterion, original_exec) < float('inf')]
                pred_val = max(finite_vals) if finite_vals else oracle_val
            if oracle_val > 0 and oracle_val < float('inf') and pred_val < float('inf'):
                reg = (pred_val - oracle_val) / oracle_val * 100
                regrets.append(reg)
                g_regrets.append(reg)

        per_graph[held_out] = {
            'correct': g_correct,
            'total': g_total,
            'accuracy': g_correct / g_total if g_total > 0 else 0,
            'avg_regret': sum(g_regrets) / len(g_regrets) if g_regrets else 0,
        }

    accuracy = correct / total if total > 0 else 0
    avg_regret = sum(regrets) / len(regrets) if regrets else 0
    sorted_regrets = sorted(regrets) if regrets else [0]
    median_regret = sorted_regrets[len(sorted_regrets) // 2]

    return {
        'model_type': model_type,
        'criterion': criterion.value,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'num_graphs': len(graphs),
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'per_graph': per_graph,
    }
