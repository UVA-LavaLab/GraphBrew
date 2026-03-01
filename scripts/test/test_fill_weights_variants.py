#!/usr/bin/env python3
"""
Test that weight filling populates entries for ALL algorithm variants.

Synthesizes small graph benchmark results for every (graph, algorithm, benchmark)
combination, then feeds them through both compute_weights_from_results and
update_type_weights_incremental, verifying:

1. All 21 trainable variant names get weight entries
2. ORIGINAL and RANDOM never leak into trained weights
3. All 7 experiment benchmarks receive training
4. No weight field stays at its default after training
5. Chained orderings are included correctly
6. Metadata (sample_count, avg_speedup, win_count) is populated
7. Per-type incremental updates produce non-zero gradients

Usage:
    pytest scripts/test/test_fill_weights_variants.py -v
"""

import json
import os
import random
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.core.utils import (
    BenchmarkResult,
    EXPERIMENT_BENCHMARKS,
    get_all_algorithm_variant_names,
    is_chained_ordering_name,
)
from scripts.lib.ml.weights import (
    assign_graph_type,
    compute_weights_from_results,
    get_best_algorithm_for_type,
    initialize_default_weights,
    load_type_weights,
    save_type_weights,
    update_type_weights_incremental,
    _create_default_weight_entry,
    PerceptronWeight,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# Synthetic small graphs with distinct structural profiles
SMALL_GRAPHS = {
    "graph_sparse": {
        "nodes": 200, "edges": 400, "avg_degree": 4.0,
        "degree_variance": 0.3, "hub_concentration": 0.05,
        "modularity": 0.7, "density": 0.02,
        "clustering_coefficient": 0.15, "clustering_coeff": 0.15,
        "avg_path_length": 4.5, "diameter": 9, "diameter_estimate": 9,
        "community_count": 8, "packing_factor": 0.3,
        "forward_edge_fraction": 0.4, "working_set_ratio": 0.6,
    },
    "graph_dense": {
        "nodes": 100, "edges": 2000, "avg_degree": 40.0,
        "degree_variance": 0.8, "hub_concentration": 0.4,
        "modularity": 0.3, "density": 0.4,
        "clustering_coefficient": 0.55, "clustering_coeff": 0.55,
        "avg_path_length": 2.0, "diameter": 4, "diameter_estimate": 4,
        "community_count": 3, "packing_factor": 0.8,
        "forward_edge_fraction": 0.7, "working_set_ratio": 1.5,
    },
    "graph_power_law": {
        "nodes": 500, "edges": 3000, "avg_degree": 12.0,
        "degree_variance": 5.0, "hub_concentration": 0.6,
        "modularity": 0.55, "density": 0.024,
        "clustering_coefficient": 0.3, "clustering_coeff": 0.3,
        "avg_path_length": 3.2, "diameter": 7, "diameter_estimate": 7,
        "community_count": 12, "packing_factor": 0.5,
        "forward_edge_fraction": 0.55, "working_set_ratio": 0.9,
    },
    "graph_road": {
        "nodes": 1000, "edges": 2500, "avg_degree": 5.0,
        "degree_variance": 0.1, "hub_concentration": 0.02,
        "modularity": 0.85, "density": 0.005,
        "clustering_coefficient": 0.05, "clustering_coeff": 0.05,
        "avg_path_length": 12.0, "diameter": 30, "diameter_estimate": 30,
        "community_count": 20, "packing_factor": 0.15,
        "forward_edge_fraction": 0.3, "working_set_ratio": 0.2,
    },
}

ALL_VARIANTS = get_all_algorithm_variant_names()
SINGLE_VARIANTS = [n for n in ALL_VARIANTS if not is_chained_ordering_name(n)]
CHAINED_VARIANTS = [n for n in ALL_VARIANTS if is_chained_ordering_name(n)]


def _make_benchmark_results() -> list[BenchmarkResult]:
    """Synthesise BenchmarkResult entries for every combination.

    Each algorithm gets a synthetic runtime drawn from a seeded random
    distribution so that different algorithms "win" for different
    (graph, benchmark) pairs.
    """
    rng = random.Random(12345)
    results: list[BenchmarkResult] = []

    # Always include ORIGINAL baseline (needed for speedup calc)
    for graph_name, props in SMALL_GRAPHS.items():
        for bench in EXPERIMENT_BENCHMARKS:
            baseline_time = rng.uniform(0.5, 2.0)
            results.append(BenchmarkResult(
                graph=graph_name,
                algorithm="ORIGINAL",
                algorithm_id=0,
                benchmark=bench,
                time_seconds=baseline_time,
                reorder_time=0.0,
                nodes=props["nodes"],
                edges=props["edges"],
                success=True,
            ))

            # All trainable variants
            for algo_name in ALL_VARIANTS:
                # Reorder time proportional to graph size
                reorder_t = rng.uniform(0.01, 0.3)
                exec_t = baseline_time * rng.uniform(0.3, 1.1)
                results.append(BenchmarkResult(
                    graph=graph_name,
                    algorithm=algo_name,
                    algorithm_id=99,  # not used by compute_weights_from_results
                    benchmark=bench,
                    time_seconds=exec_t,
                    reorder_time=reorder_t,
                    nodes=props["nodes"],
                    edges=props["edges"],
                    success=True,
                ))
    return results


@pytest.fixture
def all_variants():
    """Return the canonical set of all trainable variant names."""
    return ALL_VARIANTS


@pytest.fixture
def weights_dir(tmp_path):
    """Create an isolated weights directory."""
    wd = tmp_path / "weights"
    wd.mkdir()
    return str(wd)


@pytest.fixture
def benchmark_results():
    return _make_benchmark_results()


# ---------------------------------------------------------------------------
# Tests: compute_weights_from_results (batch training)
# ---------------------------------------------------------------------------

def _patch_graph_props(monkeypatch):
    """Monkeypatch load_graph_properties_cache to return SMALL_GRAPHS."""
    monkeypatch.setattr(
        "scripts.lib.ml.features.load_graph_properties_cache",
        lambda *_a, **_kw: dict(SMALL_GRAPHS),
    )


class TestComputeWeightsFromResults:
    """Test the batch weight computation path."""

    def test_all_variants_present(self, benchmark_results, weights_dir, monkeypatch):
        """Every trainable variant must appear in the output weights."""
        _patch_graph_props(monkeypatch)

        weights = compute_weights_from_results(
            benchmark_results=benchmark_results,
            weights_dir=weights_dir,
        )

        # All single (non-chained) canonical variant names must be present
        for name in ALL_VARIANTS:
            if is_chained_ordering_name(name):
                continue
            assert name in weights, f"Missing variant: {name}"

    def test_no_original_random(self, benchmark_results, weights_dir, monkeypatch):
        """ORIGINAL and RANDOM must NOT appear in trained weights."""
        _patch_graph_props(monkeypatch)

        weights = compute_weights_from_results(
            benchmark_results=benchmark_results,
            weights_dir=weights_dir,
        )

        for forbidden in ("ORIGINAL", "RANDOM"):
            assert forbidden not in weights, f"{forbidden} should not be in weights"

    def test_chained_not_trained(self, benchmark_results, weights_dir, monkeypatch):
        """Chained orderings get default entries but should NOT be trained (zero wins)."""
        _patch_graph_props(monkeypatch)

        weights = compute_weights_from_results(
            benchmark_results=benchmark_results,
            weights_dir=weights_dir,
        )

        for name in CHAINED_VARIANTS:
            # Chained orderings are present (from initialize_default_weights)
            # but should have zero training data (no wins, sample_count=0)
            if name in weights:
                meta = weights[name].get("_metadata", {})
                assert meta.get("sample_count", 0) == 0, (
                    f"Chained ordering {name} should not have been trained"
                )

    def test_biases_differ(self, benchmark_results, weights_dir, monkeypatch):
        """After training, not all algorithms should have the same bias."""
        _patch_graph_props(monkeypatch)

        weights = compute_weights_from_results(
            benchmark_results=benchmark_results,
            weights_dir=weights_dir,
        )

        biases = set()
        for algo_name, aw in weights.items():
            if algo_name.startswith("_") or is_chained_ordering_name(algo_name):
                continue
            biases.add(round(aw.get("bias", 0.5), 4))

        assert len(biases) > 1, "All biases are identical — training had no effect"

    def test_benchmark_weights_keys(self, benchmark_results, weights_dir, monkeypatch):
        """Each algorithm's benchmark_weights must list all 7 experiment benchmarks."""
        _patch_graph_props(monkeypatch)

        weights = compute_weights_from_results(
            benchmark_results=benchmark_results,
            weights_dir=weights_dir,
        )

        expected_benchmarks = set(EXPERIMENT_BENCHMARKS)
        for algo_name, aw in weights.items():
            if algo_name.startswith("_") or is_chained_ordering_name(algo_name):
                continue
            bw = aw.get("benchmark_weights", {})
            assert set(bw.keys()) == expected_benchmarks, (
                f"{algo_name} benchmark_weights keys: {set(bw.keys())} != {expected_benchmarks}"
            )


# ---------------------------------------------------------------------------
# Tests: update_type_weights_incremental (online / per-sample)
# ---------------------------------------------------------------------------

class TestIncrementalUpdates:
    """Test the incremental (online training) weight update path."""

    def test_all_variants_get_updated(self, weights_dir):
        """Calling update for every variant should create entries for all of them."""
        rng = random.Random(42)
        type_name = "type_0"

        for graph_name, props in SMALL_GRAPHS.items():
            for bench in EXPERIMENT_BENCHMARKS:
                for algo_name in SINGLE_VARIANTS:
                    speedup = rng.uniform(0.8, 1.5)
                    reorder_time = rng.uniform(0.01, 0.5)
                    update_type_weights_incremental(
                        type_name=type_name,
                        algorithm=algo_name,
                        benchmark=bench,
                        speedup=speedup,
                        features=props,
                        reorder_time=reorder_time,
                        weights_dir=weights_dir,
                        learning_rate=0.05,
                    )

        weights = load_type_weights(type_name, weights_dir)
        for name in SINGLE_VARIANTS:
            assert name in weights, f"Missing variant after incremental update: {name}"

    def test_original_random_skipped(self, weights_dir):
        """ORIGINAL/RANDOM calls should be silently ignored."""
        type_name = "type_0"
        features = SMALL_GRAPHS["graph_sparse"]

        for algo_name in ("ORIGINAL", "RANDOM"):
            update_type_weights_incremental(
                type_name=type_name,
                algorithm=algo_name,
                benchmark="pr",
                speedup=1.2,
                features=features,
                weights_dir=weights_dir,
            )

        weights = load_type_weights(type_name, weights_dir)
        assert "ORIGINAL" not in weights
        assert "RANDOM" not in weights

    def test_chained_skipped(self, weights_dir):
        """Chained ordering names should be silently ignored."""
        type_name = "type_0"
        features = SMALL_GRAPHS["graph_sparse"]

        for algo_name in CHAINED_VARIANTS:
            update_type_weights_incremental(
                type_name=type_name,
                algorithm=algo_name,
                benchmark="pr",
                speedup=1.2,
                features=features,
                weights_dir=weights_dir,
            )

        weights = load_type_weights(type_name, weights_dir)
        for name in CHAINED_VARIANTS:
            assert name not in weights, f"Chained ordering {name} leaked into incremental"

    def test_weights_non_zero_after_updates(self, weights_dir):
        """After multiple updates, weight fields should have moved from default."""
        rng = random.Random(99)
        type_name = "type_0"

        # Do several rounds to ensure gradients accumulate
        for _ in range(5):
            for graph_name, props in SMALL_GRAPHS.items():
                for bench in EXPERIMENT_BENCHMARKS:
                    for algo_name in SINGLE_VARIANTS:
                        speedup = rng.uniform(0.5, 2.0)
                        reorder_time = rng.uniform(0.01, 0.5)
                        update_type_weights_incremental(
                            type_name=type_name,
                            algorithm=algo_name,
                            benchmark=bench,
                            speedup=speedup,
                            features=props,
                            reorder_time=reorder_time,
                            weights_dir=weights_dir,
                            learning_rate=0.1,
                        )

        weights = load_type_weights(type_name, weights_dir)

        # Check that at least some feature weights are non-zero for every algorithm
        feature_keys = [
            "w_modularity", "w_log_nodes", "w_log_edges", "w_density",
            "w_avg_degree", "w_degree_variance", "w_hub_concentration",
            "w_clustering_coeff", "w_reorder_time", "w_packing_factor",
            "w_forward_edge_fraction", "w_working_set_ratio",
            "w_avg_path_length", "w_diameter", "w_community_count",
            "w_dv_x_hub", "w_mod_x_logn", "w_pf_x_wsr",
        ]
        for algo_name in SINGLE_VARIANTS:
            aw = weights[algo_name]
            non_zero = [k for k in feature_keys if abs(aw.get(k, 0.0)) > 1e-12]
            assert len(non_zero) > 0, (
                f"{algo_name}: all feature weights are zero after training"
            )

    def test_reorder_time_gradient(self, weights_dir):
        """w_reorder_time should become non-zero when reorder_time > 0."""
        type_name = "type_0"
        features = SMALL_GRAPHS["graph_sparse"]

        # Use varying speedups to ensure error signal is non-zero.
        # Default bias is 0.5, so speedup=1.5 → error=(1.5-1.0)-0.5=0.
        # Alternating high/low speedups keeps the error oscillating.
        rng = random.Random(77)
        for _ in range(20):
            speedup = rng.choice([0.8, 1.2, 1.8, 2.0])
            update_type_weights_incremental(
                type_name=type_name,
                algorithm="SORT",
                benchmark="pr",
                speedup=speedup,
                features=features,
                reorder_time=0.25,
                weights_dir=weights_dir,
                learning_rate=0.1,
            )

        weights = load_type_weights(type_name, weights_dir)
        assert abs(weights["SORT"]["w_reorder_time"]) > 1e-10, (
            "w_reorder_time stayed zero despite reorder_time > 0"
        )

    def test_metadata_populated(self, weights_dir):
        """Metadata (sample_count, avg_speedup, win_count) must be populated."""
        type_name = "type_0"
        features = SMALL_GRAPHS["graph_dense"]

        for i in range(5):
            update_type_weights_incremental(
                type_name=type_name,
                algorithm="GORDER",
                benchmark="bfs",
                speedup=1.1 + i * 0.1,
                features=features,
                reorder_time=0.1,
                weights_dir=weights_dir,
            )

        weights = load_type_weights(type_name, weights_dir)
        meta = weights["GORDER"].get("_metadata", {})
        assert meta.get("sample_count", 0) == 5
        assert meta.get("avg_speedup", 0) > 1.0
        assert meta.get("win_count", 0) > 0  # speedups > 1.05

    def test_cache_weights_update(self, weights_dir):
        """Cache impact weights should update when cache_stats are provided."""
        type_name = "type_0"
        features = SMALL_GRAPHS["graph_dense"]

        cache_stats = {
            "l1_hit_rate": 85.0,
            "l2_hit_rate": 10.0,
            "l3_hit_rate": 3.0,
        }

        for _ in range(10):
            update_type_weights_incremental(
                type_name=type_name,
                algorithm="HUBCLUSTER",
                benchmark="cc",
                speedup=1.3,
                features=features,
                cache_stats=cache_stats,
                reorder_time=0.05,
                weights_dir=weights_dir,
                learning_rate=0.1,
            )

        weights = load_type_weights(type_name, weights_dir)
        aw = weights["HUBCLUSTER"]
        assert abs(aw.get("cache_l1_impact", 0.0)) > 1e-10
        assert abs(aw.get("cache_l2_impact", 0.0)) > 1e-10


# ---------------------------------------------------------------------------
# Tests: default weights / initialisation
# ---------------------------------------------------------------------------

class TestDefaultWeights:
    """Test weight initialisation covers all variants."""

    def test_initialize_default_weights_complete(self, weights_dir):
        """initialize_default_weights() must create entries for all variants."""
        weights = initialize_default_weights(weights_dir)
        for name in ALL_VARIANTS:
            assert name in weights, f"Missing in default weights: {name}"

    def test_default_entry_structure(self):
        """_create_default_weight_entry must have the canonical structure."""
        entry = _create_default_weight_entry()

        # Required scalar fields
        required = [
            "bias", "w_modularity", "w_log_nodes", "w_log_edges",
            "w_density", "w_avg_degree", "w_degree_variance",
            "w_hub_concentration", "w_clustering_coeff",
            "w_reorder_time", "w_packing_factor",
            "w_forward_edge_fraction", "w_working_set_ratio",
            "w_avg_path_length", "w_diameter", "w_community_count",
            "cache_l1_impact", "cache_l2_impact", "cache_l3_impact",
            "cache_dram_penalty",
            "w_dv_x_hub", "w_mod_x_logn", "w_pf_x_wsr",
            "w_vss_x_hc", "w_wno_x_pf",
            "w_fef_convergence",
            "w_sampled_locality",
            "w_avg_reuse_distance",
            "w_vertex_significance_skewness",
            "w_window_neighbor_overlap",
            "platt_A", "platt_B",
        ]
        for key in required:
            assert key in entry, f"Missing key in default entry: {key}"

        # benchmark_weights must contain all experiment benchmarks
        bw = entry["benchmark_weights"]
        for bench in EXPERIMENT_BENCHMARKS:
            assert bench in bw, f"Missing benchmark in default entry: {bench}"

        # _metadata must be present
        meta = entry["_metadata"]
        assert meta["sample_count"] == 0
        assert meta["avg_speedup"] == 1.0

    def test_variant_count(self):
        """Sanity: variant count matches expected total."""
        # 17 base algos - ORIGINAL - RANDOM - MAP - AdaptiveOrder = 13 base IDs
        # Algo 8 (RABBITORDER) → 2 variants, algo 11 (RCM) → 2, algo 12 (GraphBrew) → 3
        # So: 13 - 3 (expanded) + 2 + 2 + 3 = 17 single variants
        # Plus chained orderings
        single = [n for n in ALL_VARIANTS if not is_chained_ordering_name(n)]
        chained = [n for n in ALL_VARIANTS if is_chained_ordering_name(n)]
        assert len(single) == 17, f"Expected 17 single variants, got {len(single)}: {single}"
        assert len(chained) == 5, f"Expected 5 chained orderings, got {len(chained)}: {chained}"
        assert len(ALL_VARIANTS) == 22


# ---------------------------------------------------------------------------
# Tests: type assignment + best algorithm selection
# ---------------------------------------------------------------------------

class TestTypeSystem:
    """Test type assignment and best-algo queries with filled weights."""

    def test_assign_distinct_types(self, weights_dir):
        """Structurally distinct graphs should get different(ish) type centroids."""
        types_seen = set()
        for graph_name, props in SMALL_GRAPHS.items():
            t = assign_graph_type(
                features=props,
                weights_dir=weights_dir,
                create_if_outlier=True,
                graph_name=graph_name,
            )
            types_seen.add(t)

        # At least 2 distinct types (some graphs may cluster together)
        assert len(types_seen) >= 2, (
            f"All {len(SMALL_GRAPHS)} graphs mapped to the same type"
        )

    def test_best_algorithm_returns_variant(self, weights_dir):
        """get_best_algorithm_for_type should return a canonical variant name."""
        type_name = "type_0"
        features = SMALL_GRAPHS["graph_sparse"]

        # Pre-populate weights with varied biases so one algo wins
        rng = random.Random(7)
        weights = {}
        for name in SINGLE_VARIANTS:
            entry = _create_default_weight_entry()
            entry["bias"] = rng.uniform(0.1, 1.0)
            weights[name] = entry
        save_type_weights(type_name, weights, weights_dir)

        best_name, best_score = get_best_algorithm_for_type(
            type_name=type_name,
            features=features,
            benchmark="bfs",
            weights_dir=weights_dir,
        )

        assert best_name in SINGLE_VARIANTS or best_name == "ORIGINAL", (
            f"Unexpected best algorithm: {best_name}"
        )

    def test_scoring_matches_perceptron(self, weights_dir):
        """PerceptronWeight.compute_score should be consistent with weight lookup."""
        type_name = "type_0"
        features = SMALL_GRAPHS["graph_dense"]

        weights = {}
        for name in SINGLE_VARIANTS:
            entry = _create_default_weight_entry()
            entry["bias"] = 0.8
            entry["w_modularity"] = 0.3
            entry["w_density"] = -0.2
            weights[name] = entry
        save_type_weights(type_name, weights, weights_dir)

        pw = PerceptronWeight(bias=0.8, w_modularity=0.3, w_density=-0.2)
        score = pw.compute_score(features, "pr")
        assert score != 0.0, "Score should be non-zero for non-trivial features"


# ---------------------------------------------------------------------------
# Tests: round-trip serialisation
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Test save → load preserves all variant weights."""

    def test_save_load_all_variants(self, weights_dir):
        """Saving weights for all variants and reloading should be identical."""
        type_name = "type_test"
        rng = random.Random(321)

        original = {}
        for name in ALL_VARIANTS:
            entry = _create_default_weight_entry()
            entry["bias"] = rng.uniform(-1.0, 1.0)
            entry["w_modularity"] = rng.uniform(-0.5, 0.5)
            entry["w_reorder_time"] = rng.uniform(-0.3, 0.0)
            original[name] = entry

        save_type_weights(type_name, original, weights_dir)
        loaded = load_type_weights(type_name, weights_dir)

        for name in ALL_VARIANTS:
            assert name in loaded, f"Lost variant on round-trip: {name}"
            assert abs(loaded[name]["bias"] - original[name]["bias"]) < 1e-10
            assert abs(loaded[name]["w_modularity"] - original[name]["w_modularity"]) < 1e-10

    def test_json_file_valid(self, weights_dir):
        """Weight files must be valid JSON."""
        type_name = "type_json"
        weights = initialize_default_weights(weights_dir)
        save_type_weights(type_name, weights, weights_dir)

        # Find and parse the file directly
        for f in os.listdir(weights_dir):
            if f.startswith("type_json") and f.endswith(".json"):
                with open(os.path.join(weights_dir, f)) as fh:
                    data = json.load(fh)
                assert isinstance(data, dict)
                assert len(data) == len(ALL_VARIANTS)


# ---------------------------------------------------------------------------
# Tests: end-to-end (assign type → train → query best)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """Full pipeline: assign type, train on all variants, pick best."""

    def test_full_pipeline(self, weights_dir):
        """Assign types, train on synthetic results, query best per bench."""
        rng = random.Random(2025)

        # Assign types for all graphs
        graph_types = {}
        for graph_name, props in SMALL_GRAPHS.items():
            t = assign_graph_type(
                features=props,
                weights_dir=weights_dir,
                create_if_outlier=True,
                graph_name=graph_name,
            )
            graph_types[graph_name] = t

        # Train: each variant on each type, several rounds
        for _round in range(3):
            for graph_name, props in SMALL_GRAPHS.items():
                type_name = graph_types[graph_name]
                for bench in EXPERIMENT_BENCHMARKS:
                    for algo_name in SINGLE_VARIANTS:
                        speedup = rng.uniform(0.7, 1.8)
                        update_type_weights_incremental(
                            type_name=type_name,
                            algorithm=algo_name,
                            benchmark=bench,
                            speedup=speedup,
                            features=props,
                            reorder_time=rng.uniform(0.01, 0.3),
                            weights_dir=weights_dir,
                            learning_rate=0.05,
                        )

        # Query best for each (type, benchmark)
        seen_best = set()
        for graph_name, props in SMALL_GRAPHS.items():
            type_name = graph_types[graph_name]
            for bench in EXPERIMENT_BENCHMARKS:
                best_name, best_score = get_best_algorithm_for_type(
                    type_name=type_name,
                    features=props,
                    benchmark=bench,
                    weights_dir=weights_dir,
                )
                seen_best.add(best_name)
                assert best_score != 0.0, (
                    f"Zero score for {type_name}/{bench} → {best_name}"
                )

        # At least 2 different algorithms should "win" somewhere
        assert len(seen_best) >= 2, (
            f"Only one algorithm ever wins: {seen_best}"
        )

    def test_all_benchmarks_exercise_benchmark_weights(self, weights_dir):
        """After training on all benchmarks, benchmark_weights must have shifted."""
        type_name = "type_0"
        algo_name = "GraphBrewOrder_leiden"
        features = SMALL_GRAPHS["graph_sparse"]

        rng = random.Random(55)
        for _ in range(10):
            for bench in EXPERIMENT_BENCHMARKS:
                speedup = rng.uniform(0.8, 1.6)
                update_type_weights_incremental(
                    type_name=type_name,
                    algorithm=algo_name,
                    benchmark=bench,
                    speedup=speedup,
                    features=features,
                    reorder_time=0.05,
                    weights_dir=weights_dir,
                    learning_rate=0.1,
                )

        weights = load_type_weights(type_name, weights_dir)
        bw = weights[algo_name]["benchmark_weights"]

        # All 7 benchmarks should have been touched (drifted from default 1.0)
        shifted = [b for b in EXPERIMENT_BENCHMARKS if abs(bw[b] - 1.0) > 1e-10]
        assert len(shifted) == len(EXPERIMENT_BENCHMARKS), (
            f"Some benchmark_weights unchanged: {[b for b in EXPERIMENT_BENCHMARKS if b not in shifted]}"
        )


# ===========================================================================
# P0 3.1c — Significance-weighted training tests
# ===========================================================================

class TestSignificanceWeighting:
    """Tests for compute_significance_weight and its effect on training."""

    def test_compute_significance_weight_high_range(self):
        """Large speedup range → high significance weight."""
        from scripts.lib.ml.training import compute_significance_weight
        from dataclasses import dataclass, field

        @dataclass
        class FakeSubcommunity:
            all_results: dict = field(default_factory=dict)

        sc = FakeSubcommunity(all_results={
            'HUBSORT': {'time': 0.5},
            'GORDER': {'time': 1.0},
            'ORIGINAL': {'time': 2.0},
        })
        w = compute_significance_weight(sc)
        # range = 2.0 / 0.5 = 4.0, but capped at max_weight=3.0
        assert w == 3.0, f"Expected 3.0 for large range, got {w}"

    def test_compute_significance_weight_low_range(self):
        """Small speedup range → low significance weight."""
        from scripts.lib.ml.training import compute_significance_weight
        from dataclasses import dataclass, field

        @dataclass
        class FakeSubcommunity:
            all_results: dict = field(default_factory=dict)

        sc = FakeSubcommunity(all_results={
            'HUBSORT': {'time': 1.0},
            'GORDER': {'time': 1.02},
        })
        w = compute_significance_weight(sc)
        # range = 1.02 / 1.0 = 1.02 → clamped to min_weight=0.1?  No, 1.02 > 0.1.
        assert abs(w - 1.02) < 1e-6, f"Expected ~1.02 for small range, got {w}"

    def test_compute_significance_weight_single_result(self):
        """Single algorithm result → default weight 1.0."""
        from scripts.lib.ml.training import compute_significance_weight
        from dataclasses import dataclass, field

        @dataclass
        class FakeSubcommunity:
            all_results: dict = field(default_factory=dict)

        sc = FakeSubcommunity(all_results={'HUBSORT': {'time': 1.0}})
        assert compute_significance_weight(sc) == 1.0

    def test_compute_significance_weight_no_results(self):
        """Empty results → default weight 1.0."""
        from scripts.lib.ml.training import compute_significance_weight
        from dataclasses import dataclass, field

        @dataclass
        class FakeSubcommunity:
            all_results: dict = field(default_factory=dict)

        sc = FakeSubcommunity(all_results={})
        assert compute_significance_weight(sc) == 1.0

    def test_compute_significance_weight_invalid_times(self):
        """All times at sentinel → default weight 1.0."""
        from scripts.lib.ml.training import compute_significance_weight
        from dataclasses import dataclass, field

        @dataclass
        class FakeSubcommunity:
            all_results: dict = field(default_factory=dict)

        sc = FakeSubcommunity(all_results={
            'HUBSORT': {'time': 999999},
            'GORDER': {'time': 999999},
        })
        assert compute_significance_weight(sc) == 1.0

    def test_significance_weight_amplifies_gradient(self, weights_dir):
        """High significance weight should produce larger weight updates."""
        type_name_1x = "type_sig_1x"
        type_name_3x = "type_sig_3x"
        features = SMALL_GRAPHS["graph_sparse"]
        algo = "HUBSORT"

        # Default bias=0.5, so error = (speedup-1.0) - 0.5.
        # Use speedup=2.0 → error = 1.0 - 0.5 = 0.5 (non-zero).
        update_type_weights_incremental(
            type_name=type_name_1x,
            algorithm=algo,
            benchmark="pr",
            speedup=2.0,
            features=features,
            weights_dir=weights_dir,
            learning_rate=0.1,
            significance_weight=1.0,
        )
        w1 = load_type_weights(type_name_1x, weights_dir)
        bias_1x = w1[algo]["bias"]

        # Same training with significance_weight=3.0 on a fresh type
        update_type_weights_incremental(
            type_name=type_name_3x,
            algorithm=algo,
            benchmark="pr",
            speedup=2.0,
            features=features,
            weights_dir=weights_dir,
            learning_rate=0.1,
            significance_weight=3.0,
        )
        w3 = load_type_weights(type_name_3x, weights_dir)
        bias_3x = w3[algo]["bias"]

        default_bias = 0.5
        delta_1x = abs(bias_1x - default_bias)
        delta_3x = abs(bias_3x - default_bias)
        assert delta_1x > 0, f"1x significance should produce non-zero update, got {delta_1x}"
        assert delta_3x > delta_1x, (
            f"3x significance ({delta_3x}) should produce larger update than 1x ({delta_1x})"
        )

    def test_significance_weight_zero_no_update(self, weights_dir):
        """significance_weight=0 should produce no gradient update."""
        type_name = "type_sig_zero"
        features = SMALL_GRAPHS["graph_dense"]
        algo = "GORDER"

        # First update to get non-default weights
        update_type_weights_incremental(
            type_name=type_name,
            algorithm=algo,
            benchmark="bfs",
            speedup=1.5,
            features=features,
            weights_dir=weights_dir,
            learning_rate=0.1,
            significance_weight=1.0,
        )
        w_before = load_type_weights(type_name, weights_dir)
        bias_before = w_before[algo]["bias"]

        # Now update with significance_weight=0
        update_type_weights_incremental(
            type_name=type_name,
            algorithm=algo,
            benchmark="bfs",
            speedup=2.0,
            features=features,
            weights_dir=weights_dir,
            learning_rate=0.1,
            significance_weight=0.0,
        )
        w_after = load_type_weights(type_name, weights_dir)
        bias_after = w_after[algo]["bias"]

        # Weight decay alone moves weights, but the gradient (lr*error) is zeroed
        # Check that bias didn't move significantly from the gradient component
        # (decay is 1e-4 per update, so trivially small)
        assert abs(bias_after - bias_before) < 0.01, (
            f"significance_weight=0 should suppress gradient, "
            f"but bias moved from {bias_before} to {bias_after}"
        )
