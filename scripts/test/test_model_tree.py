#!/usr/bin/env python3
"""Tests for model_tree module (Decision Tree / Hybrid / Criterion)."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from scripts.lib.ml.model_tree import (
    Criterion,
    ModelTree,
    ModelTreeNode,
    DT_FEATURE_NAMES,
    MODEL_TREE_N_FEATURES,
    compute_oracle,
    criterion_value,
    extract_dt_features,
    load_adaptive_models,
    save_adaptive_models,
    train_all_models,
    train_decision_tree,
    train_hybrid_tree,
    cross_validate_logo_model_tree,
)


# ===================================================================
# Fixtures: sample data
# ===================================================================

SAMPLE_PROPS = {
    'graph_A': {
        'nodes': 100_000, 'edges': 500_000,
        'modularity': 0.65, 'hub_concentration': 0.3,
        'density': 0.001, 'avg_degree': 10.0,
        'clustering_coefficient': 0.15, 'packing_factor': 0.7,
        'forward_edge_fraction': 0.52, 'working_set_ratio': 0.3,
        'community_count': 50, 'diameter': 20,
        'vertex_significance_skewness': 1.2,
        'window_neighbor_overlap': 0.4,
    },
    'graph_B': {
        'nodes': 500_000, 'edges': 2_000_000,
        'modularity': 0.45, 'hub_concentration': 0.6,
        'density': 0.008, 'avg_degree': 8.0,
        'clustering_coefficient': 0.05, 'packing_factor': 0.3,
        'forward_edge_fraction': 0.48, 'working_set_ratio': 0.1,
        'community_count': 200, 'diameter': 35,
        'vertex_significance_skewness': -0.5,
        'window_neighbor_overlap': 0.1,
    },
    'graph_C': {
        'nodes': 50_000, 'edges': 200_000,
        'modularity': 0.80, 'hub_concentration': 0.1,
        'density': 0.002, 'avg_degree': 4.0,
        'clustering_coefficient': 0.35, 'packing_factor': 0.9,
        'forward_edge_fraction': 0.55, 'working_set_ratio': 0.5,
        'community_count': 10, 'diameter': 15,
        'vertex_significance_skewness': 0.8,
        'window_neighbor_overlap': 0.6,
    },
}


def _make_records(graphs=None, benchmarks=None, algos=None, seed=42):
    """Generate synthetic benchmark records."""
    import random
    rng = random.Random(seed)

    graphs = graphs or ['graph_A', 'graph_B', 'graph_C']
    benchmarks = benchmarks or ['pr', 'bfs']
    algos = algos or ['ORIGINAL', 'GORDER', 'RABBITORDER_csr', 'LeidenOrder']

    records = []
    for g in graphs:
        for b in benchmarks:
            for a in algos:
                exec_time = rng.uniform(0.5, 10.0) if a != 'ORIGINAL' else rng.uniform(5.0, 15.0)
                reorder_time = rng.uniform(0.1, 2.0) if a != 'ORIGINAL' else 0.0
                records.append({
                    'graph': g, 'benchmark': b, 'algorithm': a,
                    'time_seconds': exec_time,
                    'reorder_time': reorder_time,
                    'success': True,
                })
    return records


# ===================================================================
# Tests: Criterion enum
# ===================================================================

class TestCriterion:
    def test_criterion_values(self):
        assert Criterion.FASTEST_REORDER.value == "fastest-reorder"
        assert Criterion.FASTEST_EXECUTION.value == "fastest-execution"
        assert Criterion.BEST_ENDTOEND.value == "best-endtoend"
        assert Criterion.BEST_AMORTIZATION.value == "best-amortization"

    def test_all_four_criteria(self):
        assert len(Criterion) == 4


class TestCriterionValue:
    def test_fastest_execution(self):
        r = {'time_seconds': 2.5, 'reorder_time': 1.0, 'algorithm': 'GORDER'}
        val = criterion_value(r, Criterion.FASTEST_EXECUTION)
        assert val == 2.5

    def test_fastest_reorder(self):
        r = {'time_seconds': 2.5, 'reorder_time': 1.0, 'algorithm': 'GORDER'}
        val = criterion_value(r, Criterion.FASTEST_REORDER)
        assert val == 1.0

    def test_best_endtoend(self):
        r = {'time_seconds': 2.5, 'reorder_time': 1.0, 'algorithm': 'GORDER'}
        val = criterion_value(r, Criterion.BEST_ENDTOEND)
        assert val == 3.5

    def test_best_amortization_with_saving(self):
        r = {'time_seconds': 2.0, 'reorder_time': 1.0, 'algorithm': 'GORDER'}
        val = criterion_value(r, Criterion.BEST_AMORTIZATION, original_exec=5.0)
        # saving = 5.0 - 2.0 = 3.0, iterations = 1.0 / 3.0 ≈ 0.333
        assert abs(val - 1.0 / 3.0) < 1e-6

    def test_best_amortization_no_saving(self):
        r = {'time_seconds': 6.0, 'reorder_time': 1.0, 'algorithm': 'SLOW'}
        val = criterion_value(r, Criterion.BEST_AMORTIZATION, original_exec=5.0)
        assert val == float('inf')

    def test_original_reorder_is_zero(self):
        r = {'time_seconds': 5.0, 'reorder_time': 0.0, 'algorithm': 'ORIGINAL'}
        val = criterion_value(r, Criterion.FASTEST_REORDER)
        assert val == float('inf')  # ORIGINAL has 0 reorder → inf (uninteresting)


class TestComputeOracle:
    def test_finds_fastest_execution(self):
        records = [
            {'algorithm': 'ORIGINAL', 'time_seconds': 10.0, 'reorder_time': 0.0, 'success': True},
            {'algorithm': 'GORDER', 'time_seconds': 3.0, 'reorder_time': 1.0, 'success': True},
            {'algorithm': 'RABBIT', 'time_seconds': 5.0, 'reorder_time': 0.5, 'success': True},
        ]
        algo, val = compute_oracle(records, Criterion.FASTEST_EXECUTION)
        assert algo == 'GORDER'
        assert val == 3.0

    def test_finds_best_e2e(self):
        records = [
            {'algorithm': 'ORIGINAL', 'time_seconds': 10.0, 'reorder_time': 0.0, 'success': True},
            {'algorithm': 'GORDER', 'time_seconds': 3.0, 'reorder_time': 5.0, 'success': True},
            {'algorithm': 'RABBIT', 'time_seconds': 5.0, 'reorder_time': 0.5, 'success': True},
        ]
        algo, val = compute_oracle(records, Criterion.BEST_ENDTOEND)
        assert algo == 'RABBIT'
        assert val == 5.5

    def test_empty_records(self):
        algo, val = compute_oracle([], Criterion.FASTEST_EXECUTION)
        assert algo == 'ORIGINAL'


# ===================================================================
# Tests: Feature extraction
# ===================================================================

class TestFeatureExtraction:
    def test_dt_features_12D(self):
        feats = extract_dt_features(SAMPLE_PROPS['graph_A'])
        assert len(feats) == MODEL_TREE_N_FEATURES
        assert len(feats) == 12

    def test_dt_feature_names_match(self):
        assert len(DT_FEATURE_NAMES) == MODEL_TREE_N_FEATURES

    def test_modularity_first(self):
        feats = extract_dt_features(SAMPLE_PROPS['graph_A'])
        assert feats[0] == 0.65  # modularity

    def test_log_transforms(self):
        feats = extract_dt_features(SAMPLE_PROPS['graph_A'])
        assert abs(feats[2] - math.log10(100_001)) < 1e-6  # log_nodes
        assert abs(feats[3] - math.log10(500_001)) < 1e-6  # log_edges

    def test_diameter_scaling(self):
        feats = extract_dt_features(SAMPLE_PROPS['graph_A'])
        assert abs(feats[11] - 20 / 50.0) < 1e-6


# ===================================================================
# Tests: ModelTreeNode & ModelTree
# ===================================================================

class TestModelTree:
    def test_single_leaf(self):
        """Tree with just one leaf node → always predicts that class."""
        node = ModelTreeNode(
            feature_idx=-1, threshold=0.0,
            left=-1, right=-1,
            leaf_class='GORDER', samples=10,
        )
        tree = ModelTree(
            model_type='decision_tree',
            benchmark='pr',
            families=['GORDER', 'RABBIT'],
            nodes=[node],
        )
        feats = extract_dt_features(SAMPLE_PROPS['graph_A'])
        assert tree.predict(feats) == 'GORDER'

    def test_simple_split(self):
        """Tree with one split → routes left or right based on feature."""
        root = ModelTreeNode(
            feature_idx=0, threshold=0.5,  # split on modularity ≤ 0.5
            left=1, right=2,
        )
        left_leaf = ModelTreeNode(
            feature_idx=-1, threshold=0.0,
            left=-1, right=-1,
            leaf_class='RABBIT', samples=5,
        )
        right_leaf = ModelTreeNode(
            feature_idx=-1, threshold=0.0,
            left=-1, right=-1,
            leaf_class='GORDER', samples=5,
        )
        tree = ModelTree(
            model_type='decision_tree',
            benchmark='pr',
            families=['RABBIT', 'GORDER'],
            nodes=[root, left_leaf, right_leaf],
        )

        # graph_A has modularity=0.65 > 0.5 → right → GORDER
        feats_a = extract_dt_features(SAMPLE_PROPS['graph_A'])
        assert tree.predict(feats_a) == 'GORDER'

        # graph_B has modularity=0.45 ≤ 0.5 → left → RABBIT
        feats_b = extract_dt_features(SAMPLE_PROPS['graph_B'])
        assert tree.predict(feats_b) == 'RABBIT'

    def test_predict_from_props(self):
        node = ModelTreeNode(
            feature_idx=-1, threshold=0.0,
            left=-1, right=-1,
            leaf_class='SORT', samples=3,
        )
        tree = ModelTree(
            model_type='decision_tree',
            benchmark='bfs',
            families=['SORT'],
            nodes=[node],
        )
        assert tree.predict_from_props(SAMPLE_PROPS['graph_A']) == 'SORT'

    def test_to_dict_roundtrip(self):
        node = ModelTreeNode(
            feature_idx=-1, threshold=0.0,
            left=-1, right=-1,
            leaf_class='GORDER', samples=10,
        )
        tree = ModelTree(
            model_type='decision_tree',
            benchmark='pr',
            families=['GORDER'],
            nodes=[node],
        )
        d = tree.to_dict()
        assert d['model_type'] == 'decision_tree'
        assert d['benchmark'] == 'pr'
        assert len(d['nodes']) == 1
        assert d['nodes'][0]['leaf_class'] == 'GORDER'

    def test_hybrid_leaf_with_weights(self):
        """Hybrid tree: leaf uses perceptron weights for scoring."""
        node = ModelTreeNode(
            feature_idx=-1, threshold=0.0,
            left=-1, right=-1,
            leaf_class='GORDER', samples=10,
            leaf_weights={
                'GORDER': [1.0] * 12,
                'RABBIT': [-1.0] * 12,
            },
        )
        tree = ModelTree(
            model_type='hybrid',
            benchmark='pr',
            families=['GORDER', 'RABBIT'],
            nodes=[node],
        )
        feats = [0.5] * 12  # all positive → dot with [1.0]*12 > dot with [-1.0]*12
        assert tree.predict(feats) == 'GORDER'


# ===================================================================
# Tests: Training
# ===================================================================

class TestTraining:
    def test_train_decision_tree(self):
        records = _make_records()
        tree = train_decision_tree(records, SAMPLE_PROPS, 'pr',
                                    Criterion.FASTEST_EXECUTION)
        assert isinstance(tree, ModelTree)
        assert tree.model_type == 'decision_tree'
        assert tree.benchmark == 'pr'
        assert len(tree.nodes) > 0

    def test_train_hybrid_tree(self):
        records = _make_records()
        tree = train_hybrid_tree(records, SAMPLE_PROPS, 'pr',
                                  Criterion.FASTEST_EXECUTION)
        assert isinstance(tree, ModelTree)
        assert tree.model_type == 'hybrid'

    def test_trained_tree_predicts(self):
        records = _make_records()
        tree = train_decision_tree(records, SAMPLE_PROPS, 'pr',
                                    Criterion.FASTEST_EXECUTION)
        feats = extract_dt_features(SAMPLE_PROPS['graph_A'])
        pred = tree.predict(feats)
        assert isinstance(pred, str)
        assert pred in ['ORIGINAL', 'GORDER', 'RABBITORDER_csr', 'LeidenOrder']


# ===================================================================
# Tests: Save/Load
# ===================================================================

class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        records = _make_records()
        models = train_all_models(records, SAMPLE_PROPS,
                                  criterion=Criterion.FASTEST_EXECUTION,
                                  save_path=tmp_path / 'models.json')

        assert (tmp_path / 'models.json').exists()

        loaded = load_adaptive_models(tmp_path / 'models.json')
        assert isinstance(loaded, dict)
        assert 'decision_tree' in loaded
        assert 'hybrid' in loaded

        # Check structure
        for bench_trees in loaded.values():
            for bench, tree in bench_trees.items():
                assert isinstance(tree, ModelTree)


# ===================================================================
# Tests: LOGO CV
# ===================================================================

class TestLOGOCV:
    def test_logo_returns_valid_structure(self):
        records = _make_records()
        result = cross_validate_logo_model_tree(
            records, SAMPLE_PROPS,
            model_type='decision_tree',
            criterion=Criterion.FASTEST_EXECUTION,
        )
        assert 'accuracy' in result
        assert 'correct' in result
        assert 'total' in result
        assert 'per_graph' in result
        assert 0 <= result['accuracy'] <= 1.0
        assert result['total'] > 0

    def test_logo_hybrid(self):
        records = _make_records()
        result = cross_validate_logo_model_tree(
            records, SAMPLE_PROPS,
            model_type='hybrid',
            criterion=Criterion.FASTEST_EXECUTION,
        )
        assert 'accuracy' in result
        assert result['total'] > 0
