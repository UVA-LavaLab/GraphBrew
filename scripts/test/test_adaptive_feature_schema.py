"""Tier-0 adaptive feature schema parity tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.lib.ml.feature_schema import (
    TIER0_FEATURE_COUNT,
    TIER0_FEATURE_NAMES,
    TIER0_WEIGHT_NAMES,
    extract_tier0_features,
    informativeness_ratio,
    passes_informativeness_gate,
    residual_informativeness_ratio,
    feature_passes_acceptance_gate,
    reuse_bucket,
    tier0_feature_record,
)
from scripts.lib.ml.portfolio import (
    DEPLOYABLE_ARM_CANONICAL_NAMES,
    DEPLOYABLE_ARM_SPECS,
    CHARACTERIZATION_BASELINE_ARM_SPECS,
    CHARACTERIZATION_DENDROGRAM_ANCHOR,
    apply_portfolio_guard,
    normalize_deployable_arm,
    normalize_deployable_portfolio,
)
from scripts.lib.ml.adaptive_emulator import (
    AdaptiveOrderEmulator,
    GraphFeatures,
    SelectionCriterion,
    load_cached_features,
)
from scripts.lib.ml.working_set import (
    modeled_property_bytes,
    property_wsr_llc,
)
from scripts.lib.ml.weights import (
    LEGACY_FEATURE_TO_WEIGHT,
    PerceptronWeight,
)
from scripts.lib.ml.source_policy import (
    ADAPTIVE_PORTFOLIO_VERIFICATION_GATE_ID,
    ADAPTIVE_SOURCE_POLICY_ID,
    aggregate_source_trial_times,
    adaptive_source_record_eligible,
    require_adaptive_source_record,
    require_portfolio_gate_coverage,
)
from scripts.lib.ml.eval_weights import train_and_evaluate
from scripts.lib.pipeline.benchmark import (
    SourceContractError,
    attach_source_trial_metadata,
    format_source_list,
    parse_benchmark_output,
)
from scripts.lib.pipeline import benchmark as benchmark_module
from scripts.lib.core.utils import normalize_graph_name


def test_tier0_feature_order_is_frozen():
    assert TIER0_FEATURE_NAMES == (
        "log10_nodes",
        "log10_edges",
        "avg_degree",
        "degree_cv",
        "hub_concentration",
        "normalized_edge_span",
        "window_neighbor_overlap",
        "property_wsr_llc",
        "kernel_class",
        "reuse_bucket",
    )
    assert TIER0_FEATURE_COUNT == 10
    assert set(TIER0_WEIGHT_NAMES[1:]).isdisjoint(
        LEGACY_FEATURE_TO_WEIGHT.values()
    )


def test_deployable_portfolio_is_frozen_and_exact():
    assert DEPLOYABLE_ARM_SPECS == (
        "0",
        "5",
        "8:csr",
        "12:rabbit:compose:sg_none:comm_identity:intra_hubsort",
        "12:rabbit:compose:sg_super_rabbit:comm_identity:intra_hubsort",
    )
    assert normalize_deployable_arm("RABBITORDER_csr") == "8:csr"
    assert DEPLOYABLE_ARM_CANONICAL_NAMES[3:] == (
        "RabbitCommunities_HubSort_GraphBrewImpl",
        "RabbitCommunities_SuperRabbit_HubSort_GraphBrewImpl",
    )
    assert all(
        not name.startswith("GraphBrewOrder_")
        for name in DEPLOYABLE_ARM_CANONICAL_NAMES
    )
    assert normalize_deployable_arm(DEPLOYABLE_ARM_SPECS[3]) == (
        DEPLOYABLE_ARM_SPECS[3]
    )
    with pytest.raises(ValueError, match="non-portfolio"):
        apply_portfolio_guard("RABBIT")


def test_deployable_portfolio_rejects_conflicting_aliases():
    payload = {
        spec: {"bias": 0.0}
        for spec in DEPLOYABLE_ARM_SPECS
    }
    payload[DEPLOYABLE_ARM_CANONICAL_NAMES[0]] = {"bias": 1.0}
    with pytest.raises(ValueError, match="conflicting aliases"):
        normalize_deployable_portfolio(payload)


def test_deployable_portfolio_accepts_legacy_pipeline_alias():
    payload = {
        spec: {"bias": float(index)}
        for index, spec in enumerate(DEPLOYABLE_ARM_SPECS)
    }
    spec = DEPLOYABLE_ARM_SPECS[3]
    expected = payload.pop(spec)
    payload[
        "GraphBrewOrder_rabbit_compose_sg_none_comm_identity_intra_hubsort"
    ] = expected
    assert normalize_deployable_portfolio(payload)[spec] == expected


def test_nonportfolio_rabbit_anchor_is_ignored():
    payload = {
        spec: {"bias": float(index)}
        for index, spec in enumerate(DEPLOYABLE_ARM_SPECS)
    }
    payload[CHARACTERIZATION_DENDROGRAM_ANCHOR] = {"bias": 99.0}
    normalized = normalize_deployable_portfolio(payload)
    assert tuple(normalized) == DEPLOYABLE_ARM_SPECS


def test_characterization_baseline_superset_is_frozen():
    assert CHARACTERIZATION_BASELINE_ARM_SPECS[:8] == (
        "0",
        "5",
        "8:csr",
        "8:boost",
        "9:csr",
        "10:canonical",
        "11:mind",
        "11:bnf",
    )
    assert CHARACTERIZATION_DENDROGRAM_ANCHOR in (
        CHARACTERIZATION_BASELINE_ARM_SPECS
    )
    assert len(CHARACTERIZATION_BASELINE_ARM_SPECS) == 14


def test_tier0_feature_values_match_cpp_contract():
    values = extract_tier0_features(
        {
            "nodes": 99,
            "edges": 999,
            "avg_degree": 4.0,
            "degree_cv": 1.5,
            "hub_concentration": 0.25,
            "normalized_edge_span": 0.1,
            "window_neighbor_overlap": 0.2,
        },
        property_wsr_llc=3.0,
        kernel_class=2,
        reuse_count=20,
    )
    assert values == pytest.approx([
        2.0,
        3.0,
        4.0,
        1.5,
        0.25,
        0.1,
        0.2,
        3.0,
        2.0,
        3.0,
    ])
    assert tier0_feature_record(values)["normalized_edge_span"] == 0.1
    weight = PerceptronWeight(
        bias=1.0,
        w_t0_log10_nodes=1.0,
        w_t0_log10_edges=2.0,
        w_t0_avg_degree=3.0,
        w_t0_degree_cv=4.0,
        w_t0_hub_concentration=5.0,
        w_t0_normalized_edge_span=6.0,
        w_t0_window_neighbor_overlap=7.0,
        w_t0_property_wsr_llc=8.0,
        w_t0_kernel_class=9.0,
        w_t0_reuse_bucket=10.0,
    )
    assert weight.compute_tier0_score(
        {
            "nodes": 99,
            "edges": 999,
            "avg_degree": 4.0,
            "degree_cv": 1.5,
            "hub_concentration": 0.25,
            "normalized_edge_span": 0.1,
            "window_neighbor_overlap": 0.2,
        },
        property_wsr_llc=3.0,
        kernel_class=2,
        reuse_count=20,
    ) == pytest.approx(
        1.0 + sum(
            (index + 1) * value
            for index, value in enumerate(values)
        )
    )


@pytest.mark.parametrize(
    ("reuse_count", "expected"),
    [(1, 0), (5, 1), (10, 2), (20, 3), (50, 4), (100, 5),
     (float("inf"), 6)],
)
def test_reuse_bucket_contract(reuse_count, expected):
    assert reuse_bucket(reuse_count) == expected


def test_tier0_requires_kernel_specific_working_set():
    properties = {
        "nodes": 1,
        "edges": 1,
        "avg_degree": 1.0,
        "degree_cv": 0.0,
        "hub_concentration": 0.0,
        "normalized_edge_span": 0.0,
        "window_neighbor_overlap": 0.0,
    }
    with pytest.raises(ValueError, match="kernel-specific"):
        extract_tier0_features(
            properties,
            property_wsr_llc=-1,
            kernel_class=0,
            reuse_count=1,
        )


def test_tier0_rejects_missing_property_inputs():
    with pytest.raises(ValueError, match="normalized_edge_span"):
        extract_tier0_features(
            {
                "nodes": 1,
                "edges": 1,
                "avg_degree": 1.0,
                "degree_cv": 0.0,
                "hub_concentration": 0.0,
                "window_neighbor_overlap": 0.0,
            },
            property_wsr_llc=1.0,
            kernel_class=1,
            reuse_count=1,
        )


def test_cached_feature_loader_rejects_missing_locality(
    tmp_path,
):
    cache = tmp_path / "features.json"
    cache.write_text(
        '{"graph":{"nodes":10,"edges":20,'
        '"window_neighbor_overlap":0.2}}'
    )
    with pytest.raises(ValueError, match="normalized_edge_span"):
        load_cached_features(cache)


def _tier0_weight_entry(
    bias=0.0,
    *,
    avg_speedup=1.0,
    avg_reorder_time=0.0,
    **overrides,
):
    entry = {
        "bias": float(bias),
        **{
            f"w_t0_{name}": 0.0
            for name in TIER0_FEATURE_NAMES
        },
        "w_reorder_time": 0.0,
        "_metadata": {
            "avg_speedup": float(avg_speedup),
            "avg_reorder_time": float(avg_reorder_time),
        },
    }
    entry.update(overrides)
    return entry


def _tier0_graph_features(nodes):
    return GraphFeatures(
        name="synthetic",
        path="",
        num_nodes=nodes,
        num_edges=nodes * 4,
        avg_degree=4.0,
        degree_variance=1.0,
        hub_concentration=0.25,
        normalized_edge_span=0.1,
        window_neighbor_overlap=0.2,
        property_wsr_llc=1.0,
        kernel_class=1,
        reuse_count=1,
    )


def _emulate_with_weights(weights, features, criterion):
    emulator = AdaptiveOrderEmulator()
    emulator.algorithm_selector.load_deployable_weights = (
        lambda _matched, _benchmark: weights
    )
    selected, scores = emulator._perceptron_for_criterion(
        criterion,
        "type_0",
        features,
        "pr",
    )
    return selected, scores


def test_emulator_uses_graph_dimensions_at_the_real_call_site():
    weights = {
        spec: _tier0_weight_entry(bias=-100.0)
        for spec in DEPLOYABLE_ARM_SPECS
    }
    weights["0"] = _tier0_weight_entry(bias=4.0)
    weights["5"] = _tier0_weight_entry(w_t0_log10_nodes=1.0)
    small, _ = _emulate_with_weights(
        weights,
        _tier0_graph_features(1024),
        SelectionCriterion.FASTEST_EXECUTION,
    )
    large, _ = _emulate_with_weights(
        weights,
        _tier0_graph_features(1_000_000),
        SelectionCriterion.FASTEST_EXECUTION,
    )
    assert small == "0"
    assert large == "5"


def test_cost_criteria_match_deployable_policy():
    weights = {
        spec: _tier0_weight_entry()
        for spec in DEPLOYABLE_ARM_SPECS
    }
    features = _tier0_graph_features(10_000)

    selected, _ = _emulate_with_weights(
        weights,
        features,
        SelectionCriterion.BEST_AMORTIZATION,
    )
    assert selected == "0"

    weights["5"] = _tier0_weight_entry(
        avg_speedup=1.1,
        avg_reorder_time=0.95,
    )
    weights["8:csr"] = _tier0_weight_entry(
        avg_speedup=10.0,
        avg_reorder_time=9.5,
    )
    selected, _ = _emulate_with_weights(
        weights,
        features,
        SelectionCriterion.BEST_AMORTIZATION,
    )
    assert selected == "5"

    weights["8:csr"] = _tier0_weight_entry(
        avg_speedup=1.1,
        avg_reorder_time=0.95,
    )
    selected, _ = _emulate_with_weights(
        weights,
        features,
        SelectionCriterion.BEST_AMORTIZATION,
    )
    assert selected == "5"

    weights = {
        spec: _tier0_weight_entry(bias=-100.0)
        for spec in DEPLOYABLE_ARM_SPECS
    }
    weights["0"] = _tier0_weight_entry()
    weights["5"] = _tier0_weight_entry(bias=0.01)
    fastest, _ = _emulate_with_weights(
        weights,
        features,
        SelectionCriterion.FASTEST_EXECUTION,
    )
    end_to_end, _ = _emulate_with_weights(
        weights,
        features,
        SelectionCriterion.BEST_ENDTOEND,
    )
    assert fastest == "0"
    assert end_to_end == "5"


def test_fastest_reorder_does_not_require_a_model_artifact():
    emulator = AdaptiveOrderEmulator()
    def fail_if_loaded(_matched):
        raise AssertionError("model should not load")

    emulator.algorithm_selector.load_weights = fail_if_loaded
    selected, _ = emulator._perceptron_for_criterion(
        SelectionCriterion.FASTEST_REORDER,
        "type_0",
        _tier0_graph_features(1024),
        "pr",
    )
    assert selected == "0"


def test_emulator_matches_runtime_per_benchmark_fallback(tmp_path):
    averaged = {
        spec: _tier0_weight_entry(bias=-100.0)
        for spec in DEPLOYABLE_ARM_SPECS
    }
    averaged["0"] = _tier0_weight_entry(bias=10.0)
    per_pr = {
        spec: _tier0_weight_entry(bias=-100.0)
        for spec in DEPLOYABLE_ARM_SPECS
    }
    per_pr["5"] = _tier0_weight_entry(bias=10.0)
    model_path = tmp_path / "adaptive_models.json"
    model_path.write_text(json.dumps({
        "perceptron": {
            "schema": "adaptive-tier0/v1",
            "tier0_trained": False,
            "weights": averaged,
            "per_benchmark": {"pr": per_pr},
        },
    }))
    emulator = AdaptiveOrderEmulator(models_path=model_path)
    features = _tier0_graph_features(1024)
    selected_pr, _ = emulator._perceptron_for_criterion(
        SelectionCriterion.FASTEST_EXECUTION,
        "type_0",
        features,
        "pr",
    )
    selected_bfs, _ = emulator._perceptron_for_criterion(
        SelectionCriterion.FASTEST_EXECUTION,
        "type_0",
        features,
        "bfs",
    )
    assert selected_pr == "5"
    assert selected_bfs == "0"


def test_cpp_runtime_model_loading_is_training_free():
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "bench/include/graphbrew/reorder/reorder_database.h"
    ).read_text()
    load_body = source.split("    void load() {", 1)[1].split(
        "    void load_benchmarks()", 1
    )[0]
    assert "load_adaptive_models();" in load_body
    assert "train_all_models();" not in load_body

    adaptive_source = (
        root
        / "bench/include/graphbrew/reorder/reorder_adaptive.h"
    ).read_text()
    assert "ComputeExtendedFeatures(" not in adaptive_source
    assert "cannot trial multiple reorderers" in adaptive_source


def test_feature_informativeness_gate_compares_graph_and_seed_variation():
    informative = {
        "g1": [0.10, 0.11, 0.09],
        "g2": [0.80, 0.79, 0.81],
        "g3": [0.45, 0.46, 0.44],
    }
    noisy = {
        "g1": [0.1, 0.9, 0.5],
        "g2": [0.2, 0.8, 0.5],
        "g3": [0.3, 0.7, 0.5],
    }
    assert informativeness_ratio(informative) > 1
    assert passes_informativeness_gate(informative)
    assert informativeness_ratio(noisy) < 1
    assert not passes_informativeness_gate(noisy)


def test_residual_informativeness_rejects_size_only_features():
    sizes = {
        "g1": (1.0, 2.0),
        "g2": (2.0, 3.0),
        "g3": (3.0, 5.0),
        "g4": (4.0, 7.0),
        "g5": (5.0, 11.0),
    }
    size_only = {
        graph: [2 * nodes + 3 * edges + noise for noise in (-0.01, 0.01)]
        for graph, (nodes, edges) in sizes.items()
    }
    residual_signal = {
        graph: [
            2 * nodes + 3 * edges + signal + noise
            for noise in (-0.01, 0.01)
        ]
        for (graph, (nodes, edges)), signal in zip(
            sizes.items(),
            (0.0, 1.0, -1.0, 1.5, -1.5),
        )
    }
    assert residual_informativeness_ratio(size_only, sizes) < 1
    assert residual_informativeness_ratio(
        residual_signal, sizes,
    ) > 1
    assert not feature_passes_acceptance_gate(
        "window_neighbor_overlap", size_only, sizes,
    )
    assert feature_passes_acceptance_gate(
        "window_neighbor_overlap", residual_signal, sizes,
    )


def test_kernel_specific_property_working_sets_are_distinct():
    nodes = 100
    edges = 500
    assert modeled_property_bytes("pr", nodes, edges) == 800
    assert modeled_property_bytes("pr_spmv", nodes, edges) == 800
    assert modeled_property_bytes("bfs", nodes, edges) == 826
    assert modeled_property_bytes("cc", nodes, edges) == 413
    assert modeled_property_bytes("cc_sv", nodes, edges) == 413
    assert modeled_property_bytes("sssp", nodes, edges) == 2400
    assert modeled_property_bytes("bc", nodes, edges) == 2063
    assert property_wsr_llc("pr", nodes, edges, 400) == 2.0
    with pytest.raises(ValueError, match="unavailable"):
        modeled_property_bytes("unknown", nodes, edges)


def test_adaptive_phase_and_tier0_output_is_machine_readable():
    _avg, _reorder, extra = parse_benchmark_output("\n".join([
        "Adaptive Feature Time:0.01000",
        "Adaptive Model Time:  0.00200",
        "Adaptive Selection Time:0.01200",
        "Adaptive Arm Map Time:0.50000",
        "Adaptive Confidence:  0.75000",
        "Adaptive Predicted:   RABBIT",
        "Adaptive Applied:     0",
        "Adaptive Override Reason:non_portfolio_label",
        "Adaptive Weight Source:per-benchmark:pr",
        "Adaptive Tier0 Trained:false",
        "Property Working Set Bytes:800.00000",
        "LLC Capacity Bytes:  1024.00000",
        "Property WSR LLC:    2.00000",
        'Adaptive Tier0 Features: {"log10_nodes":2.0,"reuse_bucket":0}',
        "Source Original:     4",
        "Source Internal:     17",
        "Source Out Degree:   9",
    ]))
    assert extra["adaptive_feature_time"] == 0.01
    assert extra["adaptive_model_time"] == 0.002
    assert extra["adaptive_selection_time"] == 0.012
    assert extra["adaptive_arm_map_time"] == 0.5
    assert extra["adaptive_confidence"] == 0.75
    assert extra["adaptive_predicted"] == "RABBIT"
    assert extra["adaptive_applied"] == "0"
    assert extra["adaptive_override_reason"] == "non_portfolio_label"
    assert extra["adaptive_weight_source"] == "per-benchmark:pr"
    assert extra["adaptive_tier0_trained"] == "false"
    assert extra["property_working_set_bytes"] == 800
    assert extra["llc_capacity_bytes"] == 1024
    assert extra["property_wsr_llc"] == 2
    assert extra["adaptive_tier0_features"]["log10_nodes"] == 2
    assert extra["source_originals"] == [4]
    assert extra["source_internals"] == [17]
    assert extra["source_out_degrees"] == [9]


def test_source_list_formatter_is_strict_and_ordered():
    assert format_source_list([4, 27, 103]) == "4,27,103"
    with pytest.raises(ValueError, match="unique"):
        format_source_list([4, 4])
    with pytest.raises(ValueError, match="non-negative"):
        format_source_list([-1])


def test_pre_sprint_source_records_are_ineligible():
    assert not adaptive_source_record_eligible({
        "benchmark": "bfs",
        "source_originals": list(range(8)),
    })
    trials = [
        {
            "process_id": 3,
            "source_id": source,
            "source_internal": source + 10,
            "source_out_degree": 2,
            "repetition_index": 0,
            "measurement_mode": "cold-process",
        }
        for source in range(8)
    ]
    assert adaptive_source_record_eligible({
        "benchmark": "bfs",
        "source_policy_id": ADAPTIVE_SOURCE_POLICY_ID,
        "source_trials": trials,
    })


def test_real_output_normalizes_to_source_trial_contract():
    lines = []
    for source in range(8):
        for repetition in range(2):
            lines.extend([
                f"Trial Time: {source + repetition + 1}.0",
                f"Source Original: {source}",
                f"Source Internal: {source + 10}",
                f"Source Out Degree: {source + 2}",
            ])
    lines.append("Average Time: 4.5")
    output = "\n".join(lines)
    _average, _reorder, extra = parse_benchmark_output(output)
    attach_source_trial_metadata(
        extra,
        process_id=7,
        measurement_mode="warm-block",
        source_policy_id=ADAPTIVE_SOURCE_POLICY_ID,
        source_repeats=2,
        expected_sources=list(range(8)),
        expected_internals=[
            source + 10 for source in range(8)
        ],
        expected_out_degrees=[
            source + 2 for source in range(8)
        ],
    )
    record = {"benchmark": "bfs", **extra}
    require_adaptive_source_record(record)
    assert [
        trial["repetition_index"]
        for trial in record["source_trials"]
    ] == [0, 1] * 8


def test_run_benchmark_binds_bc_source_contract(
    tmp_path, monkeypatch,
):
    binary = tmp_path / "bc"
    binary.write_text("")
    graph = tmp_path / "graph.natural.sg"
    graph.write_text("")
    observed = {}

    def fake_run(command, **_kwargs):
        observed["command"] = command
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout="\n".join([
                "Trial Time: 1.0",
                "Source Original: 7",
                "Source Internal: 3",
                "Source Out Degree: 4",
                "Trial Time: 0.8",
                "Source Original: 7",
                "Source Internal: 3",
                "Source Out Degree: 4",
                "Average Time: 0.9",
            ]),
        )

    monkeypatch.setattr(benchmark_module, "run_command", fake_run)
    result = benchmark_module.run_benchmark(
        "bc",
        str(graph),
        trials=2,
        symmetric=False,
        bin_dir=str(tmp_path),
        source_originals=[7],
        source_repeats=2,
        source_policy_id=ADAPTIVE_SOURCE_POLICY_ID,
        process_id=0,
        measurement_mode="warm-block",
        expected_source_internals=[3],
        expected_source_out_degrees=[4],
        result_graph_name="graph",
    )
    assert result.success
    assert result.graph == "graph"
    assert observed["command"][-4:] == ["-R", "2", "-i", "1"]
    assert len(result.extra["source_trials"]) == 2

    with pytest.raises(SourceContractError, match="internals"):
        attach_source_trial_metadata(
            result.extra,
            process_id=0,
            measurement_mode="warm-block",
            source_policy_id=ADAPTIVE_SOURCE_POLICY_ID,
            source_repeats=2,
            expected_sources=[7],
            expected_internals=[99],
            expected_out_degrees=[4],
        )


def test_portfolio_gate_coverage_fails_closed():
    rows = [{
        "gate_id": ADAPTIVE_PORTFOLIO_VERIFICATION_GATE_ID,
        "graph": "g1",
        "algo_key": "0",
        "verification_state": "pass",
    }]
    require_portfolio_gate_coverage(rows, ["g1"], ["0"])
    with pytest.raises(ValueError, match="lacks verification"):
        require_portfolio_gate_coverage(rows, ["g1"], ["0", "5"])


def test_warm_source_aggregation_excludes_cold_first_trial():
    summary = aggregate_source_trial_times(
        [0.9, 0.1, 0.2, 0.8, 0.2, 0.3],
        [1, 1, 1, 2, 2, 2],
        "warm-block",
    )
    assert summary["cold_first_times"] == [0.9, 0.8]
    assert summary["warm_times"] == pytest.approx([0.15, 0.25])
    assert summary["cell_time"] == pytest.approx(0.2)


@pytest.mark.parametrize(("value", "expected"), [
    ("results/graphs/cit-Patents/cit-Patents.sg", "cit-Patents"),
    (r"C:\graphs\com-Orkut\com-Orkut.wsg", "com-Orkut"),
    ("twitter7.el.gz", "twitter7"),
    ("USA-road-d.USA", "USA-road-d.USA"),
])
def test_graph_name_normalization_contract(value, expected):
    assert normalize_graph_name(value) == expected


def test_legacy_non_nested_logo_is_retired():
    with pytest.raises(RuntimeError, match="non-nested LOGO"):
        train_and_evaluate(logo=True)
