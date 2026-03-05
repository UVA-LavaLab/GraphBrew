#!/usr/bin/env python3
"""
Model Tree (Decision Tree / Hybrid) for algorithm selection.

Mirrors the C++ ModelTree / ModelTreeNode structs from reorder_types.h.
Supports:
  - Pure Decision Tree: leaf_class prediction
  - Hybrid (Model Tree): per-leaf perceptron weights for scoring

The 22-feature vector aligns with the perceptron's feat_to_weight:
  [0]  modularity
  [1]  hub_concentration
  [2]  log10(N+1)
  [3]  log10(E+1)
  [4]  density
  [5]  avg_degree/100
  [6]  clustering_coeff
  [7]  packing_factor
  [8]  forward_edge_fraction
  [9]  log2(working_set_ratio+1)
  [10] log10(community_count+1)
  [11] diameter/50
  [12] degree_variance
  [13] avg_path_length/10
  [14] vertex_significance_skewness  (DON-RL)
  [15] window_neighbor_overlap       (DON-RL)
  [16] dv×hub   (quadratic)
  [17] mod×logn (quadratic)
  [18] pf×wsr   (quadratic)
  [19] vss×hc   (quadratic)
  [20] wno×pf   (quadratic)
  [21] packing_factor_cl (IISWC'18 CL)

Training uses scikit-learn and exports to C++-compatible JSON format.
"""

import json
import math
from collections import defaultdict
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
# Feature extraction (22D — matches perceptron feat_to_weight ordering)
# ============================================================================

MODEL_TREE_N_FEATURES = 24

DT_FEATURE_NAMES = [
    'modularity', 'hub_concentration', 'log_nodes', 'log_edges',
    'density', 'avg_degree_100', 'clustering_coeff', 'packing_factor',
    'forward_edge_fraction', 'log2_wsr', 'log10_community_count',
    'diameter_50', 'degree_variance', 'avg_path_length_10',
    # DON-RL features (Zhao et al.)
    'vertex_significance_skewness', 'window_neighbor_overlap',
    # Quadratic cross-terms (paper-motivated)
    'dv_x_hub', 'mod_x_logn', 'pf_x_wsr', 'vss_x_hc', 'wno_x_pf',
    # IISWC'18 cache-line packing factor
    'packing_factor_cl',
    # Per-level WSR (P-OPT cache hierarchy)
    'log2_wsr_l1', 'log2_wsr_l2',
]


def extract_dt_features(props: dict) -> List[float]:
    """Extract the 24-element feature vector for DT/XGBoost splits.

    Aligned with perceptron feat_to_weight:
      [0-13]  14 linear features (same transforms as scoreBase)
      [14-15] 2 DON-RL features
      [16-20] 5 quadratic cross-terms
      [21]    IISWC'18 cache-line packing factor
      [22-23] Per-level WSR (P-OPT cache hierarchy)
    """
    nodes = props.get('nodes', 1000)
    edges = props.get('edges', 5000)
    avg_degree = props.get('avg_degree', 10.0)

    modularity = props.get('modularity', 0.5)
    hub_conc = props.get('hub_concentration', 0.3)
    log_nodes = math.log10(nodes + 1) if nodes > 0 else 0
    degree_var = props.get('degree_variance', 0.0)
    pf = props.get('packing_factor', 0.0)
    log_wsr = math.log2(props.get('working_set_ratio', 0.0) + 1.0)
    vss = props.get('vertex_significance_skewness', 0.0)
    wno = props.get('window_neighbor_overlap', 0.0)

    return [
        modularity,                                                      # 0
        hub_conc,                                                        # 1
        log_nodes,                                                       # 2
        math.log10(edges + 1) if edges > 0 else 0,                      # 3
        avg_degree / (nodes - 1) if nodes > 1 else 0,                   # 4: density
        avg_degree / 100.0,                                              # 5
        props.get('clustering_coefficient', 0.0),                        # 6
        pf,                                                              # 7
        props.get('forward_edge_fraction', 0.5),                         # 8
        log_wsr,                                                         # 9
        math.log10(props.get('community_count', 0.0) + 1.0),            # 10
        props.get('diameter_estimate', props.get('diameter', 0.0)) / 50, # 11
        degree_var,                                                      # 12
        props.get('avg_path_length', 0.0) / 10.0,                       # 13
        # DON-RL
        vss,                                                             # 14
        wno,                                                             # 15
        # Quadratic cross-terms
        degree_var * hub_conc,                                           # 16
        modularity * log_nodes,                                          # 17
        pf * log_wsr,                                                    # 18
        vss * hub_conc,                                                  # 19
        wno * pf,                                                        # 20
        # IISWC'18 cache-line packing factor
        props.get('packing_factor_cl', 0.0),                             # 21
        # Per-level WSR (P-OPT cache hierarchy)
        math.log2(props.get('wsr_l1', 0.0) + 1.0),                      # 22
        math.log2(props.get('wsr_l2', 0.0) + 1.0),                      # 23
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

    def predict(self, features: List[float]) -> str:
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
                            score += weights[i] * features[i]
                        if n_w > 0:
                            score += weights[-1]  # bias
                        if score > best_score:
                            best_score = score
                            best_family = fam
                    return best_family
                # Pure DT: return class directly
                return node.leaf_class

            # Split node
            feat_val = (features[node.feature_idx]
                        if 0 <= node.feature_idx < len(features) else 0.0)
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
    use_families: bool = False,
) -> Tuple[List[List[float]], List[str], List[str]]:
    """Build (X, y, graphs) training data for one benchmark.

    Args:
        bench_records: all benchmark result dicts (all benchmarks/graphs)
        graph_props: dict of graph_name → properties
        benchmark: which benchmark to train for
        criterion: optimization objective for labeling
        use_families: if True, map oracle labels to algorithm families

    Returns:
        X: list of 22D feature vectors (one per graph)
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

    if use_families:
        from scripts.lib.ml.adaptive_emulator import algo_to_family

    X, y, graph_names = [], [], []
    for graph_name in sorted(by_graph):
        if graph_name not in graph_props:
            continue
        if graph_name == 'tiny':
            continue  # skip tiny graph

        records = by_graph[graph_name]
        best_algo, _ = compute_oracle(records, criterion)

        if use_families:
            best_algo = algo_to_family(best_algo)

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
    max_depth: int = 3,
    min_samples_leaf: int = 3,
    use_families: bool = False,
) -> ModelTree:
    """Train a Decision Tree classifier for one benchmark.

    Uses scikit-learn internally but exports to our ModelTree format
    for C++ compatibility.
    """
    from sklearn.tree import DecisionTreeClassifier

    X, y, graph_names = _build_training_data(
        bench_records, graph_props, benchmark, criterion, use_families=use_families)

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
    min_samples_leaf: int = 3,
    use_families: bool = False,
) -> ModelTree:
    """Train a Hybrid (Model Tree) for one benchmark.

    First trains a DT structure, then fits per-leaf perceptron weights
    on the leaf node's training samples.
    """
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    X, y, graph_names = _build_training_data(
        bench_records, graph_props, benchmark, criterion, use_families=use_families)

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


def train_random_forest(
    bench_records: List[dict],
    graph_props: dict,
    benchmark: str,
    criterion: Criterion = Criterion.FASTEST_EXECUTION,
    n_estimators: int = 100,
    max_depth: int = 3,
    min_samples_leaf: int = 3,
    use_families: bool = False,
) -> List[ModelTree]:
    """Train a Random Forest for one benchmark.

    Returns a list of ModelTree objects (one per estimator) that can be
    ensembled via majority vote.  Each tree is exported in the same
    C++-compatible node format as the single DT.
    """
    from sklearn.ensemble import RandomForestClassifier

    X, y, graph_names = _build_training_data(
        bench_records, graph_props, benchmark, criterion, use_families=use_families)

    if len(X) < 2:
        mt = ModelTree('decision_tree', benchmark,
                       families=sorted(set(y)) if y else ['ORIGINAL'])
        mt.nodes = [ModelTreeNode(leaf_class=y[0] if y else 'ORIGINAL',
                                  samples=len(X))]
        return [mt]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight='balanced',
    )
    clf.fit(X, y)

    # Convert each estimator to a ModelTree
    # Pass parent RF's classes_ so each sub-tree uses the same class mapping
    rf_classes = list(clf.classes_)
    trees = []
    for est in clf.estimators_:
        mt = _sklearn_to_model_tree(est, benchmark, 'decision_tree',
                                    class_names=rf_classes)
        trees.append(mt)
    return trees


def predict_random_forest(trees: List[ModelTree], features: List[float]) -> str:
    """Majority-vote prediction from a list of ModelTree estimators."""
    from collections import Counter
    votes = Counter()
    for tree in trees:
        pred = tree.predict(features)
        votes[pred] += 1
    return votes.most_common(1)[0][0] if votes else 'ORIGINAL'


# ===================================================================
# XGBoost Model
# ===================================================================

def train_xgboost(
    bench_records: List[dict],
    graph_props: dict,
    benchmark: str,
    criterion: 'Criterion',
    max_depth: int = 3,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    min_child_weight: int = 3,
    use_families: bool = False,
) -> List[ModelTree]:
    """Train an XGBoost classifier and return list of ModelTree objects.

    XGBoost uses gradient-boosted trees — potentially stronger than RF
    on noisy labels but introduces a dependency on xgboost package.
    Returns list of ModelTree for API compatibility with RF.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    X, y, _graph_names = _build_training_data(
        bench_records, graph_props, benchmark, criterion, use_families=use_families)

    if len(X) < 2:
        mt = ModelTree('decision_tree', benchmark,
                       families=sorted(set(y)) if y else ['ORIGINAL'])
        mt.nodes = [ModelTreeNode(leaf_class=y[0] if y else 'ORIGINAL',
                                  samples=len(X))]
        return [mt], None

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        learning_rate=learning_rate,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0,
        use_label_encoder=False,
        n_jobs=1,  # avoid parallelism overhead on small datasets
    )
    clf.fit(X, y_encoded)

    return clf, le


def predict_xgboost(model_tuple, features: List[float]) -> str:
    """Predict class using XGBoost model.

    Args:
        model_tuple: (XGBClassifier, LabelEncoder) from train_xgboost()
        features: 22-element feature vector
    """
    import numpy as np
    clf, le = model_tuple
    if clf is None or le is None:
        return 'ORIGINAL'
    X = np.array([features])
    pred_encoded = clf.predict(X)[0]
    return str(le.inverse_transform([pred_encoded])[0])


def _summarize_per_bench(per_bench: dict) -> dict:
    """Summarise per-benchmark accumulator into accuracy / avg_regret dict."""
    return {
        b: {
            'correct': d['correct'],
            'total': d['total'],
            'accuracy': d['correct'] / d['total'] if d['total'] > 0 else 0,
            'avg_regret': (sum(d['regrets']) / len(d['regrets'])
                          if d['regrets'] else 0),
        }
        for b, d in sorted(per_bench.items())
    }


def cross_validate_logo_xgboost(
    bench_records: List[dict],
    graph_props: dict,
    criterion: 'Criterion' = None,
    benchmarks: List[str] = None,
    n_estimators: int = 100,
    max_depth: int = 3,
    min_child_weight: int = 3,
    learning_rate: float = 0.1,
    use_families: bool = False,
) -> Dict:
    """Leave-One-Graph-Out evaluation for XGBoost."""
    if criterion is None:
        from scripts.lib.ml.model_tree import Criterion
        criterion = Criterion.FASTEST_EXECUTION
    if benchmarks is None:
        benchmarks = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

    from scripts.lib.ml.adaptive_emulator import algo_to_family

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
    family_correct = 0
    per_graph = {}
    per_bench = defaultdict(lambda: {'correct': 0, 'total': 0, 'regrets': []})

    for held_out in graphs:
        if held_out not in graph_props:
            continue

        train_records = [r for r in bench_records if r.get('graph') != held_out]
        feats = extract_dt_features(graph_props[held_out])
        g_correct = 0
        g_total = 0
        g_regrets = []

        for bench in benchmarks:
            held_records = [r for r in bench_records
                            if r.get('graph') == held_out
                            and r.get('benchmark') == bench
                            and r.get('success', True)
                            and r.get('time_seconds', 0) > 0]
            if not held_records:
                continue

            oracle_algo, oracle_val = compute_oracle(held_records, criterion)
            oracle_label = algo_to_family(oracle_algo) if use_families else oracle_algo

            model_tuple = train_xgboost(
                train_records, graph_props, bench, criterion,
                n_estimators=n_estimators, max_depth=max_depth,
                min_child_weight=min_child_weight,
                learning_rate=learning_rate,
                use_families=use_families)
            predicted = predict_xgboost(model_tuple, feats)

            g_total += 1
            total += 1
            per_bench[bench]['total'] += 1
            if predicted == oracle_label:
                g_correct += 1
                correct += 1
                per_bench[bench]['correct'] += 1

            # Family-level match
            pred_family = algo_to_family(predicted) if not use_families else predicted
            oracle_family = algo_to_family(oracle_algo)
            if pred_family == oracle_family:
                family_correct += 1

            # Regret
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
                finite_vals = [criterion_value(r, criterion, original_exec)
                               for r in held_records
                               if criterion_value(r, criterion, original_exec) < float('inf')]
                pred_val = max(finite_vals) if finite_vals else oracle_val
            if oracle_val > 0 and oracle_val < float('inf') and pred_val < float('inf'):
                reg = (pred_val - oracle_val) / oracle_val * 100
                regrets.append(reg)
                g_regrets.append(reg)
                per_bench[bench]['regrets'].append(reg)

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
    within_5pct = sum(1 for r in regrets if r <= 5.0) / len(regrets) if regrets else 0

    return {
        'model_type': 'xgboost',
        'criterion': criterion.value,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'num_graphs': len(graphs),
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'within_5pct': within_5pct,
        'family_acc': family_correct / total if total > 0 else 0,
        'per_graph': per_graph,
        'per_bench': _summarize_per_bench(per_bench),
    }


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


def _sklearn_to_model_tree(clf, benchmark: str, model_type: str,
                           class_names: list = None) -> ModelTree:
    """Convert a fitted sklearn DecisionTreeClassifier to ModelTree format.

    Args:
        class_names: Explicit class name list. If None, uses clf.classes_.
            This is needed for RF estimators whose tree_.value arrays are
            indexed by the *parent* RF's classes_, not the individual tree's.
    """
    tree = clf.tree_
    classes = list(class_names) if class_names is not None else list(clf.classes_)

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
            node.leaf_class = str(classes[class_idx])
        else:
            # Split
            node.feature_idx = int(tree.feature[sk_idx])
            node.threshold = float(tree.threshold[sk_idx])
            node.left = node_map[tree.children_left[sk_idx]]
            node.right = node_map[tree.children_right[sk_idx]]

    mt = ModelTree(model_type, benchmark, families=[str(c) for c in classes],
                   nodes=mt_nodes)
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
        Dict with keys 'decision_tree', 'hybrid', 'random_forest',
        each mapping benchmark → ModelTree (or list of ModelTree for RF).
    """
    if path is None:
        path = Path(RESULTS_DIR) / 'data' / 'adaptive_models.json'

    result = {'decision_tree': {}, 'hybrid': {}, 'random_forest': {}}
    if not path.exists():
        return result

    with open(path) as f:
        data = json.load(f)

    for subdir in ('decision_tree', 'hybrid'):
        section = data.get(subdir, {})
        for bench, tree_data in section.items():
            result[subdir][bench] = ModelTree.from_dict(tree_data)

    # Random Forest: list of trees per benchmark
    rf_section = data.get('random_forest', {})
    for bench, trees_data in rf_section.items():
        if isinstance(trees_data, list):
            result['random_forest'][bench] = [ModelTree.from_dict(td) for td in trees_data]
        else:
            result['random_forest'][bench] = [ModelTree.from_dict(trees_data)]

    return result


def save_adaptive_models(
    models: Dict[str, Dict[str, ModelTree]],
    path: Path = None,
):
    """Save DT/Hybrid/RF models to adaptive_models.json in C++ format."""
    if path is None:
        path = Path(RESULTS_DIR) / 'data' / 'adaptive_models.json'

    data = {}
    for subdir in ('decision_tree', 'hybrid'):
        data[subdir] = {}
        for bench, tree in models.get(subdir, {}).items():
            data[subdir][bench] = tree.to_dict()

    # Random Forest: list of trees per benchmark
    data['random_forest'] = {}
    for bench, trees in models.get('random_forest', {}).items():
        if isinstance(trees, list):
            data['random_forest'][bench] = [t.to_dict() for t in trees]
        else:
            data['random_forest'][bench] = [trees.to_dict()]

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
    """Train DT, Hybrid, and Random Forest models for all benchmarks.

    Args:
        bench_records: raw benchmark data (list of dicts)
        graph_props: graph properties lookup
        benchmarks: list of benchmarks to train for (default: all 8)
        criterion: optimization target
        save_path: where to save adaptive_models.json (default: results/data/)

    Returns:
        Dict with 'decision_tree', 'hybrid', and 'random_forest' sub-dicts.
    """
    if benchmarks is None:
        benchmarks = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

    models = {'decision_tree': {}, 'hybrid': {}, 'random_forest': {}}

    for bench in benchmarks:
        dt = train_decision_tree(bench_records, graph_props, bench, criterion)
        hy = train_hybrid_tree(bench_records, graph_props, bench, criterion)
        rf = train_random_forest(bench_records, graph_props, bench, criterion)
        models['decision_tree'][bench] = dt
        models['hybrid'][bench] = hy
        models['random_forest'][bench] = rf

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
    use_families: bool = False,
) -> Dict:
    """Leave-One-Graph-Out evaluation for DT or Hybrid models.

    For each graph:
      1. Train model on all OTHER graphs
      2. Predict on held-out graph
      3. Compare to oracle per criterion

    Args:
        use_families: If True, train on family-level labels and compare
            predictions at the family level.

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

    from scripts.lib.ml.adaptive_emulator import algo_to_family

    correct = 0
    total = 0
    regrets = []
    family_correct = 0
    per_graph = {}
    per_bench = defaultdict(lambda: {'correct': 0, 'total': 0, 'regrets': []})

    for held_out in graphs:
        if held_out not in graph_props:
            continue

        # Train on all graphs except held_out
        train_records = [r for r in bench_records if r.get('graph') != held_out]
        train_fn = train_decision_tree if model_type == 'decision_tree' else train_hybrid_tree

        feats = extract_dt_features(graph_props[held_out])
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
            oracle_label = algo_to_family(oracle_algo) if use_families else oracle_algo

            # Train model on other graphs, predict
            tree = train_fn(train_records, graph_props, bench, criterion,
                           use_families=use_families)
            predicted = tree.predict(feats)

            g_total += 1
            total += 1
            per_bench[bench]['total'] += 1
            if predicted == oracle_label:
                g_correct += 1
                correct += 1
                per_bench[bench]['correct'] += 1

            # Family-level match
            pred_family = algo_to_family(predicted) if not use_families else predicted
            oracle_family = algo_to_family(oracle_algo)
            if pred_family == oracle_family:
                family_correct += 1

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
                per_bench[bench]['regrets'].append(reg)

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
    within_5pct = sum(1 for r in regrets if r <= 5.0) / len(regrets) if regrets else 0

    return {
        'model_type': model_type,
        'criterion': criterion.value,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'num_graphs': len(graphs),
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'within_5pct': within_5pct,
        'family_acc': family_correct / total if total > 0 else 0,
        'per_graph': per_graph,
        'per_bench': _summarize_per_bench(per_bench),
    }


# ===================================================================
# Two-Stage Classifier: (1) Reorder? → (2) Which reorder?
# ===================================================================

_NO_REORDER_ALGOS = frozenset(['ORIGINAL', 'RANDOM'])


def _build_two_stage_training_data(
    bench_records: List[dict],
    graph_props: dict,
    benchmark: str,
    criterion: Criterion = Criterion.BEST_ENDTOEND,
) -> Tuple[List[List[float]], List[str], List[str], List[str]]:
    """Build training data for two-stage classifier.

    Returns:
        X: feature vectors (one per graph)
        y_binary: 'REORDER' or 'NO_REORDER' per graph
        y_algo: actual oracle algorithm per graph (for stage 2)
        graphs: graph names
    """
    from collections import defaultdict
    from scripts.lib.ml.adaptive_emulator import algo_to_family

    by_graph = defaultdict(list)
    for r in bench_records:
        if (r.get('benchmark') == benchmark and
                r.get('success', True) and
                r.get('time_seconds', 0) > 0):
            by_graph[r.get('graph', '')].append(r)

    X, y_binary, y_algo, graph_names = [], [], [], []
    for graph_name in sorted(by_graph):
        if graph_name not in graph_props or graph_name == 'tiny':
            continue

        records = by_graph[graph_name]
        best_algo, _ = compute_oracle(records, criterion)
        best_family = algo_to_family(best_algo)

        feats = extract_dt_features(graph_props[graph_name])
        X.append(feats)
        y_binary.append('NO_REORDER' if best_algo in _NO_REORDER_ALGOS
                        else 'REORDER')
        y_algo.append(best_family)  # family label for stage 2
        graph_names.append(graph_name)

    return X, y_binary, y_algo, graph_names


def train_two_stage(
    bench_records: List[dict],
    graph_props: dict,
    benchmark: str,
    criterion: Criterion = Criterion.BEST_ENDTOEND,
    stage1_model: str = 'xgboost',
    stage2_model: str = 'xgboost',
    max_depth: int = 3,
    n_estimators: int = 100,
    noreorder_skip_threshold: float = 0.10,
):
    """Train a two-stage classifier for one benchmark.

    Stage 1: Binary classifier — should we reorder? (ORIGINAL/RANDOM vs reorder)
    Stage 2: Multi-class on reorder-only subset — which reordering family?

    If NO_REORDER fraction in training < noreorder_skip_threshold,
    Stage 1 is skipped (always predict REORDER).

    Returns:
        dict with 'stage1' (model, le) or None, 'stage2' (model, le),
              'stage1_model' str, 'skip_stage1' bool
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    X, y_binary, y_algo, graphs = _build_two_stage_training_data(
        bench_records, graph_props, benchmark, criterion)

    if len(X) < 2:
        return {'stage1': None, 'stage2': None, 'stage1_model': stage1_model,
                'skip_stage1': True}

    X_arr = np.array(X)
    noreorder_frac = sum(1 for yb in y_binary if yb == 'NO_REORDER') / len(y_binary)

    # --- Stage 1: Reorder or not? ---
    # Skip Stage 1 if NO_REORDER is very rare in training data
    skip_stage1 = noreorder_frac < noreorder_skip_threshold

    if skip_stage1:
        # Always predict REORDER — skip binary classifier
        clf1, le1 = None, None
    else:
        le1 = LabelEncoder()
        y1_enc = le1.fit_transform(y_binary)

        if stage1_model == 'xgboost':
            from xgboost import XGBClassifier
            clf1 = XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_child_weight=3, learning_rate=0.1,
                random_state=42, eval_metric='logloss',
                verbosity=0, use_label_encoder=False, n_jobs=1,
                # No scale_pos_weight: let natural distribution guide
            )
        else:
            from sklearn.tree import DecisionTreeClassifier
            clf1 = DecisionTreeClassifier(
                max_depth=max_depth, min_samples_leaf=3, random_state=42)

        clf1.fit(X_arr, y1_enc)

    # --- Stage 2: Which reorder family? (trained only on reorder samples) ---
    reorder_mask = [i for i, yb in enumerate(y_binary) if yb == 'REORDER']
    if len(reorder_mask) < 2:
        return {'stage1': (clf1, le1) if not skip_stage1 else None,
                'stage2': None, 'stage1_model': stage1_model,
                'skip_stage1': skip_stage1}

    X_reorder = X_arr[reorder_mask]
    y_reorder = [y_algo[i] for i in reorder_mask]

    le2 = LabelEncoder()
    y2_enc = le2.fit_transform(y_reorder)

    if stage2_model == 'xgboost':
        from xgboost import XGBClassifier
        n_classes = len(set(y2_enc))
        clf2 = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_child_weight=2, learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss' if n_classes > 2 else 'logloss',
            verbosity=0, use_label_encoder=False, n_jobs=1,
        )
    else:
        from sklearn.tree import DecisionTreeClassifier
        clf2 = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=2, random_state=42)

    clf2.fit(X_reorder, y2_enc)

    return {
        'stage1': (clf1, le1) if not skip_stage1 else None,
        'stage2': (clf2, le2),
        'stage1_model': stage1_model,
        'skip_stage1': skip_stage1,
    }


def predict_two_stage(model_dict, features: List[float]) -> str:
    """Predict using two-stage classifier.

    Returns predicted algorithm family or 'ORIGINAL'.
    """
    import numpy as np

    if model_dict is None:
        return 'ORIGINAL'

    X = np.array([features])

    # If Stage 1 is skipped (almost all tasks need reordering), go to Stage 2
    if model_dict.get('skip_stage1', False) or model_dict.get('stage1') is None:
        # Skip binary decision, always reorder
        if model_dict.get('stage2') is None:
            return 'ORIGINAL'
        clf2, le2 = model_dict['stage2']
        pred2_enc = clf2.predict(X)[0]
        return str(le2.inverse_transform([pred2_enc])[0])

    clf1, le1 = model_dict['stage1']

    # Stage 1: reorder or not?
    pred1_enc = clf1.predict(X)[0]
    pred1 = str(le1.inverse_transform([pred1_enc])[0])

    if pred1 == 'NO_REORDER':
        return 'ORIGINAL'

    # Stage 2: which family?
    if model_dict.get('stage2') is None:
        return 'ORIGINAL'

    clf2, le2 = model_dict['stage2']
    pred2_enc = clf2.predict(X)[0]
    return str(le2.inverse_transform([pred2_enc])[0])


def cross_validate_logo_two_stage(
    bench_records: List[dict],
    graph_props: dict,
    criterion: Criterion = Criterion.BEST_ENDTOEND,
    benchmarks: List[str] = None,
    stage1_model: str = 'xgboost',
    stage2_model: str = 'xgboost',
    max_depth: int = 3,
    n_estimators: int = 100,
) -> Dict:
    """Leave-One-Graph-Out evaluation for Two-Stage classifier.

    Stage 1: Binary (reorder or no-reorder)
    Stage 2: Multi-class (which reordering family)

    Oracle uses family-level comparison for reorder predictions,
    and checks ORIGINAL match for no-reorder predictions.
    """
    if benchmarks is None:
        benchmarks = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

    from scripts.lib.ml.adaptive_emulator import algo_to_family

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
    family_correct = 0
    stage1_correct = 0
    stage1_total = 0
    per_graph = {}
    per_bench = defaultdict(lambda: {'correct': 0, 'total': 0, 'regrets': []})

    for held_out in graphs:
        if held_out not in graph_props:
            continue

        train_records = [r for r in bench_records if r.get('graph') != held_out]
        feats = extract_dt_features(graph_props[held_out])
        g_correct = 0
        g_total = 0
        g_regrets = []

        for bench in benchmarks:
            held_records = [r for r in bench_records
                            if r.get('graph') == held_out
                            and r.get('benchmark') == bench
                            and r.get('success', True)
                            and r.get('time_seconds', 0) > 0]
            if not held_records:
                continue

            oracle_algo, oracle_val = compute_oracle(held_records, criterion)
            oracle_family = algo_to_family(oracle_algo)
            oracle_is_noreorder = oracle_algo in _NO_REORDER_ALGOS

            # Train and predict
            model = train_two_stage(
                train_records, graph_props, bench, criterion,
                stage1_model=stage1_model, stage2_model=stage2_model,
                max_depth=max_depth, n_estimators=n_estimators)
            predicted_family = predict_two_stage(model, feats)

            # Stage 1 accuracy
            stage1_total += 1
            pred_is_noreorder = (predicted_family == 'ORIGINAL')
            if pred_is_noreorder == oracle_is_noreorder:
                stage1_correct += 1

            # Overall accuracy: compare at family level
            g_total += 1
            total += 1
            per_bench[bench]['total'] += 1

            if oracle_is_noreorder:
                # Oracle says no-reorder: correct if we also predicted no-reorder
                is_correct = pred_is_noreorder
            else:
                # Oracle says reorder: correct if we predicted same family
                is_correct = (predicted_family == oracle_family)

            if is_correct:
                g_correct += 1
                correct += 1
                per_bench[bench]['correct'] += 1

            # Family match (always family-level comparison)
            pred_fam = algo_to_family(predicted_family) if predicted_family != 'ORIGINAL' else 'ORIGINAL'
            if pred_fam == oracle_family or (pred_fam == 'ORIGINAL' and oracle_is_noreorder):
                family_correct += 1

            # Regret: find the best algorithm from predicted family in held data
            pred_val = None
            original_exec = None
            for r in held_records:
                if r.get('algorithm') == 'ORIGINAL':
                    original_exec = r.get('time_seconds', 999)

            if predicted_family == 'ORIGINAL':
                # Predicted no-reorder → use ORIGINAL's value
                for r in held_records:
                    if r.get('algorithm') == 'ORIGINAL':
                        pred_val = criterion_value(r, criterion, original_exec)
                        break
            else:
                # Predicted a reorder family → find best algo of that family
                best_fam_val = float('inf')
                for r in held_records:
                    rfam = algo_to_family(r.get('algorithm', ''))
                    if rfam == predicted_family:
                        v = criterion_value(r, criterion, original_exec)
                        if v < best_fam_val:
                            best_fam_val = v
                if best_fam_val < float('inf'):
                    pred_val = best_fam_val

            if pred_val is None:
                finite_vals = [criterion_value(r, criterion, original_exec)
                               for r in held_records
                               if criterion_value(r, criterion, original_exec) < float('inf')]
                pred_val = max(finite_vals) if finite_vals else oracle_val

            if oracle_val > 0 and oracle_val < float('inf') and pred_val < float('inf'):
                reg = (pred_val - oracle_val) / oracle_val * 100
                regrets.append(reg)
                g_regrets.append(reg)
                per_bench[bench]['regrets'].append(reg)

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
    within_5pct = sum(1 for r in regrets if r <= 5.0) / len(regrets) if regrets else 0
    stage1_acc = stage1_correct / stage1_total if stage1_total > 0 else 0

    return {
        'model_type': f'two_stage_{stage1_model}',
        'criterion': criterion.value,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'num_graphs': len(graphs),
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'within_5pct': within_5pct,
        'family_acc': family_correct / total if total > 0 else 0,
        'stage1_accuracy': stage1_acc,
        'per_graph': per_graph,
        'per_bench': _summarize_per_bench(per_bench),
    }


# ===================================================================
# XGBoost Family+ORIGINAL (single-stage with ORIGINAL as a family)
# ===================================================================

def cross_validate_logo_xgboost_family(
    bench_records: List[dict],
    graph_props: dict,
    criterion: Criterion = Criterion.BEST_ENDTOEND,
    benchmarks: List[str] = None,
    max_depth: int = 3,
    n_estimators: int = 100,
) -> Dict:
    """LOGO CV for XGBoost family classifier that includes ORIGINAL as a class.

    Single-stage approach: train a multi-class XGBoost where the classes
    are algorithm families + ORIGINAL. This avoids the cascading error
    problem of the two-stage approach.

    Oracle: best algorithm at family level (ORIGINAL counts as its own family).
    Regret: best algorithm within predicted family (or ORIGINAL if predicted).
    """
    if benchmarks is None:
        benchmarks = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

    import numpy as np
    from collections import defaultdict
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    from scripts.lib.ml.adaptive_emulator import algo_to_family

    # Collect all valid graphs
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
    family_correct = 0
    per_graph = {}
    per_bench = defaultdict(lambda: {'correct': 0, 'total': 0, 'regrets': []})

    for held_out in graphs:
        if held_out not in graph_props:
            continue

        train_records = [r for r in bench_records if r.get('graph') != held_out]
        feats = extract_dt_features(graph_props[held_out])
        g_correct = 0
        g_total = 0
        g_regrets = []

        for bench in benchmarks:
            held_records = [r for r in bench_records
                            if r.get('graph') == held_out
                            and r.get('benchmark') == bench
                            and r.get('success', True)
                            and r.get('time_seconds', 0) > 0]
            if not held_records:
                continue

            oracle_algo, oracle_val = compute_oracle(held_records, criterion)
            oracle_family = algo_to_family(oracle_algo)
            if oracle_algo in _NO_REORDER_ALGOS:
                oracle_family = 'ORIGINAL'

            # Build per-benchmark training data with ORIGINAL as a family
            by_graph = defaultdict(list)
            for r in train_records:
                if (r.get('benchmark') == bench and
                        r.get('success', True) and
                        r.get('time_seconds', 0) > 0):
                    by_graph[r.get('graph', '')].append(r)

            X_train, y_train = [], []
            for gname in sorted(by_graph):
                if gname not in graph_props or gname == 'tiny':
                    continue
                recs = by_graph[gname]
                best_algo, _ = compute_oracle(recs, criterion)
                best_fam = algo_to_family(best_algo)
                if best_algo in _NO_REORDER_ALGOS:
                    best_fam = 'ORIGINAL'
                X_train.append(extract_dt_features(graph_props[gname]))
                y_train.append(best_fam)

            if len(X_train) < 3 or len(set(y_train)) < 2:
                continue

            # Train XGBoost
            le = LabelEncoder()
            y_enc = le.fit_transform(y_train)
            n_classes = len(set(y_enc))

            clf = XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                min_child_weight=2, learning_rate=0.1,
                random_state=42,
                eval_metric='mlogloss' if n_classes > 2 else 'logloss',
                verbosity=0, use_label_encoder=False, n_jobs=1,
            )
            clf.fit(np.array(X_train), y_enc)

            # Predict
            pred_enc = clf.predict(np.array([feats]))[0]
            predicted_family = str(le.inverse_transform([pred_enc])[0])

            # Accuracy (family-level)
            g_total += 1
            total += 1
            per_bench[bench]['total'] += 1
            is_correct = (predicted_family == oracle_family)
            if is_correct:
                g_correct += 1
                correct += 1
                per_bench[bench]['correct'] += 1
                family_correct += 1

            # Regret
            original_exec = None
            for r in held_records:
                if r.get('algorithm') == 'ORIGINAL':
                    original_exec = r.get('time_seconds', 999)

            pred_val = None
            if predicted_family == 'ORIGINAL':
                for r in held_records:
                    if r.get('algorithm') == 'ORIGINAL':
                        pred_val = criterion_value(r, criterion, original_exec)
                        break
            else:
                best_fam_val = float('inf')
                for r in held_records:
                    rfam = algo_to_family(r.get('algorithm', ''))
                    if rfam == predicted_family:
                        v = criterion_value(r, criterion, original_exec)
                        if v < best_fam_val:
                            best_fam_val = v
                if best_fam_val < float('inf'):
                    pred_val = best_fam_val

            if pred_val is None:
                finite_vals = [criterion_value(r, criterion, original_exec)
                               for r in held_records
                               if criterion_value(r, criterion, original_exec) < float('inf')]
                pred_val = max(finite_vals) if finite_vals else oracle_val

            if oracle_val > 0 and oracle_val < float('inf') and pred_val < float('inf'):
                reg = (pred_val - oracle_val) / oracle_val * 100
                regrets.append(reg)
                g_regrets.append(reg)
                per_bench[bench]['regrets'].append(reg)

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
    within_5pct = sum(1 for r in regrets if r <= 5.0) / len(regrets) if regrets else 0

    return {
        'model_type': 'xgboost_family_original',
        'criterion': criterion.value,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'num_graphs': len(graphs),
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'within_5pct': within_5pct,
        'family_acc': family_correct / total if total > 0 else 0,
        'per_graph': per_graph,
        'per_bench': _summarize_per_bench(per_bench),
    }


# Benchmark name → integer index for cross-benchmark feature encoding
_BENCH_INDEX = {b: i for i, b in enumerate(
    ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc'])}


def _extract_extended_features(props: dict) -> List[float]:
    """Extended features: 22D base + edge/node ratio.

    Returns 23D vector:
      [0..21] standard extract_dt_features() (14 linear + 2 DON-RL + 5 quadratic + 1 CL)
      [22]    edges / nodes ratio (complements density)
    """
    base = extract_dt_features(props)
    nodes = props.get('nodes', 1)
    edges = props.get('edges', 1)
    base.append(edges / max(nodes, 1))
    return base


def cross_validate_logo_xgboost_family_xbench(
    bench_records: List[dict],
    graph_props: dict,
    criterion: Criterion = Criterion.BEST_ENDTOEND,
    benchmarks: List[str] = None,
    max_depth: int = 3,
    n_estimators: int = 100,
) -> Dict:
    """LOGO CV for cross-benchmark XGBoost family classifier.

    Trains a SINGLE model across ALL benchmarks per LOGO fold, with
    benchmark index appended as an additional feature (22D total).
    Uses extended features: 22D base + 1 bench = 23D.
    This gives ~8× more training data per fold than per-benchmark training.

    Classes: algorithm families + ORIGINAL.
    """
    if benchmarks is None:
        benchmarks = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

    import numpy as np
    from collections import defaultdict
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    from scripts.lib.ml.adaptive_emulator import algo_to_family

    # Collect all valid graphs
    graphs = set()
    for r in bench_records:
        g = r.get('graph', '')
        if g and g != 'tiny' and r.get('success', True) and r.get('time_seconds', 0) > 0:
            graphs.add(g)
    graphs = sorted(graphs)

    if len(graphs) < 3:
        return {'accuracy': 0, 'total': 0, 'error': 'Need >= 3 graphs'}

    # Pre-index records by (graph, benchmark)
    by_graph_bench = defaultdict(list)
    for r in bench_records:
        g = r.get('graph', '')
        b = r.get('benchmark', '')
        if (g and g != 'tiny' and b in benchmarks
                and r.get('success', True) and r.get('time_seconds', 0) > 0):
            by_graph_bench[(g, b)].append(r)

    correct = 0
    total = 0
    regrets = []
    family_correct = 0
    per_graph = {}
    per_bench = defaultdict(lambda: {'correct': 0, 'total': 0, 'regrets': []})

    for held_out in graphs:
        if held_out not in graph_props:
            continue

        base_feats = extract_dt_features(graph_props[held_out])
        g_correct = 0
        g_total = 0
        g_regrets = []

        # Build cross-benchmark training data (all benchmarks, all train graphs)
        X_train, y_train = [], []
        for gname in graphs:
            if gname == held_out or gname not in graph_props:
                continue
            gfeats = extract_dt_features(graph_props[gname])
            for bench in benchmarks:
                recs = by_graph_bench.get((gname, bench), [])
                if not recs:
                    continue
                best_algo, _ = compute_oracle(recs, criterion)
                best_fam = algo_to_family(best_algo)
                if best_algo in _NO_REORDER_ALGOS:
                    best_fam = 'ORIGINAL'
                # Append benchmark index as feature
                X_train.append(gfeats + [_BENCH_INDEX.get(bench, 0)])
                y_train.append(best_fam)

        if len(X_train) < 5 or len(set(y_train)) < 2:
            continue

        # Train single cross-benchmark XGBoost
        le = LabelEncoder()
        y_enc = le.fit_transform(y_train)
        n_classes = len(set(y_enc))

        clf = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_child_weight=2, learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss' if n_classes > 2 else 'logloss',
            verbosity=0, use_label_encoder=False, n_jobs=1,
        )
        clf.fit(np.array(X_train), y_enc)

        # Predict for each benchmark on held-out graph
        for bench in benchmarks:
            held_records = by_graph_bench.get((held_out, bench), [])
            if not held_records:
                continue

            oracle_algo, oracle_val = compute_oracle(held_records, criterion)
            oracle_family = algo_to_family(oracle_algo)
            if oracle_algo in _NO_REORDER_ALGOS:
                oracle_family = 'ORIGINAL'

            # Predict with benchmark feature
            test_feats = base_feats + [_BENCH_INDEX.get(bench, 0)]
            pred_enc = clf.predict(np.array([test_feats]))[0]
            predicted_family = str(le.inverse_transform([pred_enc])[0])

            g_total += 1
            total += 1
            per_bench[bench]['total'] += 1
            is_correct = (predicted_family == oracle_family)
            if is_correct:
                g_correct += 1
                correct += 1
                per_bench[bench]['correct'] += 1
                family_correct += 1

            # Regret
            original_exec = None
            for r in held_records:
                if r.get('algorithm') == 'ORIGINAL':
                    original_exec = r.get('time_seconds', 999)

            pred_val = None
            if predicted_family == 'ORIGINAL':
                for r in held_records:
                    if r.get('algorithm') == 'ORIGINAL':
                        pred_val = criterion_value(r, criterion, original_exec)
                        break
            else:
                best_fam_val = float('inf')
                for r in held_records:
                    rfam = algo_to_family(r.get('algorithm', ''))
                    if rfam == predicted_family:
                        v = criterion_value(r, criterion, original_exec)
                        if v < best_fam_val:
                            best_fam_val = v
                if best_fam_val < float('inf'):
                    pred_val = best_fam_val

            if pred_val is None:
                finite_vals = [criterion_value(r, criterion, original_exec)
                               for r in held_records
                               if criterion_value(r, criterion, original_exec) < float('inf')]
                pred_val = max(finite_vals) if finite_vals else oracle_val

            if oracle_val > 0 and oracle_val < float('inf') and pred_val < float('inf'):
                reg = (pred_val - oracle_val) / oracle_val * 100
                regrets.append(reg)
                g_regrets.append(reg)
                per_bench[bench]['regrets'].append(reg)

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
    within_5pct = sum(1 for r in regrets if r <= 5.0) / len(regrets) if regrets else 0

    return {
        'model_type': 'xgboost_family_xbench',
        'criterion': criterion.value,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'num_graphs': len(graphs),
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'within_5pct': within_5pct,
        'family_acc': family_correct / total if total > 0 else 0,
        'per_graph': per_graph,
        'per_bench': _summarize_per_bench(per_bench),
    }


def cross_validate_logo_regression_xbench(
    bench_records: List[dict],
    graph_props: dict,
    criterion: Criterion = Criterion.BEST_ENDTOEND,
    benchmarks: List[str] = None,
    max_depth: int = 4,
    n_estimators: int = 200,
) -> Dict:
    """LOGO CV for cross-benchmark XGBoost **Learning-to-Rank** model.

    Trains an ``XGBRanker`` with ``rank:pairwise`` objective to directly
    learn which algorithm family produces the lowest cost for a given
    (graph, benchmark) pair.  Each training group contains one item per
    family with ordinal rank labels (higher = cheaper).

    The feature vector is ``[22D graph features, bench_index, family_index]``
    (24 D total).  A single model across all benchmarks and families gives
    ~8× more training data per LOGO fold than per-benchmark training.

    Why Learning-to-Rank?
    ---------------------
    * We only need the correct *ranking* of families, not exact cost values.
    * Ordinal labels avoid the calibration problems of per-family regressors.
    * Pairwise loss naturally weighs large ranking swaps more than small ones.
    * Expected to improve as more (especially larger) graphs are added.
    """
    if benchmarks is None:
        benchmarks = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

    import numpy as np
    from collections import defaultdict
    from xgboost import XGBRanker
    from scripts.lib.ml.adaptive_emulator import algo_to_family

    REORDER_FAMILIES = ['HUBSORT', 'GORDER', 'LEIDEN', 'RCM', 'SORT',
                        'RABBIT', 'GOGRAPH', 'ADAPTIVE']
    ALL_FAMILIES = REORDER_FAMILIES + ['ORIGINAL']
    FAMILY_INDEX = {f: i for i, f in enumerate(ALL_FAMILIES)}

    # ---- collect valid graphs ----
    graphs = set()
    for r in bench_records:
        g = r.get('graph', '')
        if (g and g != 'tiny' and r.get('success', True)
                and r.get('time_seconds', 0) > 0):
            graphs.add(g)
    graphs = sorted(graphs)

    if len(graphs) < 3:
        return {'accuracy': 0, 'total': 0, 'error': 'Need >= 3 graphs'}

    # Pre-index records by (graph, benchmark)
    by_graph_bench: Dict[tuple, list] = defaultdict(list)
    for r in bench_records:
        g = r.get('graph', '')
        b = r.get('benchmark', '')
        if (g and g != 'tiny' and b in benchmarks
                and r.get('success', True) and r.get('time_seconds', 0) > 0):
            by_graph_bench[(g, b)].append(r)

    correct = 0
    total = 0
    regrets = []
    per_graph = {}
    per_bench = defaultdict(lambda: {'correct': 0, 'total': 0, 'regrets': []})

    for held_out in graphs:
        if held_out not in graph_props:
            continue

        base_feats = extract_dt_features(graph_props[held_out])
        g_correct = 0
        g_total = 0
        g_regrets = []

        # ---- build LTR training data ----
        # Each group = one (graph, benchmark) pair.
        # Items within group = one per family with data.
        # Label = ordinal rank (highest = cheapest family).
        X_train: list = []
        y_train: list = []
        groups: list = []

        for gname in graphs:
            if gname == held_out or gname not in graph_props:
                continue
            gfeats = extract_dt_features(graph_props[gname])

            for bench in benchmarks:
                recs = by_graph_bench.get((gname, bench), [])
                if not recs:
                    continue

                # Original execution time for criterion_value()
                original_exec = None
                for r in recs:
                    if r.get('algorithm') == 'ORIGINAL':
                        original_exec = r.get('time_seconds', None)
                        break

                # Best criterion_value per family
                fam_best: Dict[str, float] = {}
                for r in recs:
                    algo = r.get('algorithm', '')
                    if algo in _NO_REORDER_ALGOS:
                        fam = 'ORIGINAL' if algo == 'ORIGINAL' else 'SORT'
                    else:
                        fam = algo_to_family(algo)
                    if fam not in FAMILY_INDEX:
                        continue
                    val = criterion_value(r, criterion, original_exec)
                    if val < float('inf'):
                        if fam not in fam_best or val < fam_best[fam]:
                            fam_best[fam] = val

                if len(fam_best) < 2:
                    continue

                # Sort by cost ascending → rank labels (highest = best)
                items = sorted(fam_best.items(), key=lambda x: x[1])
                n = len(items)
                base_input = gfeats + [_BENCH_INDEX.get(bench, 0)]
                for rank_i, (fam, _cost) in enumerate(items):
                    X_train.append(base_input + [FAMILY_INDEX[fam]])
                    y_train.append(n - 1 - rank_i)  # best → highest
                groups.append(n)

        if len(X_train) < 20 or len(groups) < 5:
            continue

        ranker = XGBRanker(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.1, random_state=42,
            verbosity=0, n_jobs=1,
        )
        ranker.fit(np.array(X_train), np.array(y_train), group=groups)

        # ---- predict for each benchmark on held-out graph ----
        for bench in benchmarks:
            held_records = by_graph_bench.get((held_out, bench), [])
            if not held_records:
                continue

            oracle_algo, oracle_val = compute_oracle(held_records, criterion)
            oracle_family = algo_to_family(oracle_algo)
            if oracle_algo in _NO_REORDER_ALGOS:
                oracle_family = 'ORIGINAL'

            test_base = base_feats + [_BENCH_INDEX.get(bench, 0)]

            # Score every family, pick highest-scored
            test_items = [test_base + [FAMILY_INDEX[f]]
                          for f in ALL_FAMILIES]
            scores = ranker.predict(np.array(test_items))
            predicted_family = ALL_FAMILIES[int(np.argmax(scores))]

            g_total += 1
            total += 1
            per_bench[bench]['total'] += 1
            is_correct = (predicted_family == oracle_family)
            if is_correct:
                g_correct += 1
                correct += 1
                per_bench[bench]['correct'] += 1

            # ---- compute regret using ACTUAL performance ----
            original_exec = None
            for r in held_records:
                if r.get('algorithm') == 'ORIGINAL':
                    original_exec = r.get('time_seconds', 999)

            pred_val = None
            if predicted_family == 'ORIGINAL':
                for r in held_records:
                    if r.get('algorithm') == 'ORIGINAL':
                        pred_val = criterion_value(r, criterion,
                                                   original_exec)
                        break
            else:
                best_fam_val = float('inf')
                for r in held_records:
                    rfam = algo_to_family(r.get('algorithm', ''))
                    if rfam == predicted_family:
                        v = criterion_value(r, criterion, original_exec)
                        if v < best_fam_val:
                            best_fam_val = v
                if best_fam_val < float('inf'):
                    pred_val = best_fam_val

            if pred_val is None:
                finite_vals = [
                    criterion_value(r, criterion, original_exec)
                    for r in held_records
                    if criterion_value(r, criterion, original_exec)
                    < float('inf')
                ]
                pred_val = max(finite_vals) if finite_vals else oracle_val

            if (oracle_val > 0 and oracle_val < float('inf')
                    and pred_val < float('inf')):
                reg = (pred_val - oracle_val) / oracle_val * 100
                regrets.append(reg)
                g_regrets.append(reg)
                per_bench[bench]['regrets'].append(reg)

        per_graph[held_out] = {
            'correct': g_correct,
            'total': g_total,
            'accuracy': g_correct / g_total if g_total > 0 else 0,
            'avg_regret': (sum(g_regrets) / len(g_regrets)
                          if g_regrets else 0),
        }

    accuracy = correct / total if total > 0 else 0
    avg_regret = sum(regrets) / len(regrets) if regrets else 0
    sorted_regrets = sorted(regrets) if regrets else [0]
    median_regret = sorted_regrets[len(sorted_regrets) // 2]
    within_5pct = (sum(1 for r in regrets if r <= 5.0) / len(regrets)
                   if regrets else 0)

    return {
        'model_type': 'regression_xbench',
        'criterion': criterion.value,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'num_graphs': len(graphs),
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'within_5pct': within_5pct,
        'per_graph': per_graph,
        'per_bench': _summarize_per_bench(per_bench),
    }


def cross_validate_logo_random_forest(
    bench_records: List[dict],
    graph_props: dict,
    criterion: Criterion = Criterion.FASTEST_EXECUTION,
    benchmarks: List[str] = None,
    n_estimators: int = 100,
    max_depth: int = 3,
    min_samples_leaf: int = 3,
    use_families: bool = False,
) -> Dict:
    """Leave-One-Graph-Out evaluation for Random Forest.

    For each graph:
      1. Train RF ensemble on all OTHER graphs
      2. Predict on held-out graph via majority vote
      3. Compare to oracle per criterion
    """
    if benchmarks is None:
        benchmarks = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']

    from collections import defaultdict

    from scripts.lib.ml.adaptive_emulator import algo_to_family

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
    family_correct = 0
    per_graph = {}
    per_bench = defaultdict(lambda: {'correct': 0, 'total': 0, 'regrets': []})

    for held_out in graphs:
        if held_out not in graph_props:
            continue

        train_records = [r for r in bench_records if r.get('graph') != held_out]
        feats = extract_dt_features(graph_props[held_out])
        g_correct = 0
        g_total = 0
        g_regrets = []

        for bench in benchmarks:
            held_records = [r for r in bench_records
                            if r.get('graph') == held_out
                            and r.get('benchmark') == bench
                            and r.get('success', True)
                            and r.get('time_seconds', 0) > 0]
            if not held_records:
                continue

            oracle_algo, oracle_val = compute_oracle(held_records, criterion)
            oracle_label = algo_to_family(oracle_algo) if use_families else oracle_algo

            rf_trees = train_random_forest(
                train_records, graph_props, bench, criterion,
                n_estimators=n_estimators, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                use_families=use_families)
            predicted = predict_random_forest(rf_trees, feats)

            g_total += 1
            total += 1
            per_bench[bench]['total'] += 1
            if predicted == oracle_label:
                g_correct += 1
                correct += 1
                per_bench[bench]['correct'] += 1

            # Family-level match
            pred_family = algo_to_family(predicted) if not use_families else predicted
            oracle_family = algo_to_family(oracle_algo)
            if pred_family == oracle_family:
                family_correct += 1

            # Regret
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
                finite_vals = [criterion_value(r, criterion, original_exec)
                               for r in held_records
                               if criterion_value(r, criterion, original_exec) < float('inf')]
                pred_val = max(finite_vals) if finite_vals else oracle_val
            if oracle_val > 0 and oracle_val < float('inf') and pred_val < float('inf'):
                reg = (pred_val - oracle_val) / oracle_val * 100
                regrets.append(reg)
                g_regrets.append(reg)
                per_bench[bench]['regrets'].append(reg)

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

    within_5pct = sum(1 for r in regrets if r <= 5.0) / len(regrets) if regrets else 0

    return {
        'model_type': 'random_forest',
        'criterion': criterion.value,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'num_graphs': len(graphs),
        'avg_regret': avg_regret,
        'median_regret': median_regret,
        'within_5pct': within_5pct,
        'family_acc': family_correct / total if total > 0 else 0,
        'per_graph': per_graph,
        'per_bench': _summarize_per_bench(per_bench),
    }
