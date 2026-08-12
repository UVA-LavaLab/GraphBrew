"""Regression tests for VLDB experiment policy and reproducibility controls."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import inspect
import os
import shutil
import statistics
import struct
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest

from scripts.experiments.vldb import figures, runner
from scripts.experiments.vldb.config import (
    ABLATION_CONTRASTS,
    ABLATION_CONFIGS,
    ALL_ALGORITHMS,
    ALGORITHM_GRAPH_EXCLUSION_EVIDENCE,
    ALGORITHM_GRAPH_EXCLUSIONS,
    CACHE_GRAPH_NAMES,
    CACHE_PR_ITERATIONS,
    CACHE_SIZES,
    CACHE_TRIALS,
    COMPOSE_VARIANTS,
    E2E_PAPER_ALGORITHM_KEYS,
    EVALUATION_BASELINES,
    GRAPHBREW_VARIANTS,
    OPTIONAL_REFINEMENT_TIMEOUT,
    PAPER_ARTIFACT_ROOT,
    PAPER_GRAPH_ROOT,
    REORDER_TIMEOUT_FULL,
    REORDER_TIMEOUT_PREVIEW,
    REORDER_TIMING_ANCHOR_ALGOS,
    REORDER_TIMING_REUSE_GRAPHS,
    SCALABILITY_GRAPH_NAMES,
    STABLE_BLOCK_ALGORITHM_KEY,
    TIMEOUT_FULL,
    TIMEOUT_PREVIEW,
    algorithm_exclusion_reason,
)
from scripts.experiments.vldb.stages import _common
from scripts.lib.pipeline import reorder_config


def test_rabbitorder_implementations_are_explicit():
    assert "8" not in EVALUATION_BASELINES
    assert EVALUATION_BASELINES["8:csr"] == "RabbitOrder (CSR)"
    assert EVALUATION_BASELINES["8:boost"] == "RabbitOrder (Boost)"


def test_gorder_paper_implementation_is_explicit():
    assert "9" not in EVALUATION_BASELINES
    assert EVALUATION_BASELINES["9:csr"] == "GORDER"


def test_hrab_candidates_are_controlled_variants():
    assert "hrab" in GRAPHBREW_VARIANTS
    assert "hrab:bfs_intra" in GRAPHBREW_VARIANTS
    rcm = runner._expected_graphbrew_config("12:hrab")
    bfs = runner._expected_graphbrew_config("12:hrab:bfs_intra")
    assert {
        key for key in rcm if rcm[key] != bfs[key]
    } == {"rcm_intra"}


def test_rabbit_supergraph_ablation_changes_one_axis():
    variants = dict(COMPOSE_VARIANTS)
    assert "SgRabH_dgd" not in variants
    base = runner._expected_graphbrew_config(variants["Rabbit-HubSort"])
    supergraph = runner._expected_graphbrew_config(
        variants["SuperRabbit-HubSort"]
    )
    differing = {
        key for key in base
        if base.get(key) != supergraph.get(key)
    }
    assert differing == {"super_graph"}


def test_published_compose_specs_pin_both_block_axes():
    variants = dict(COMPOSE_VARIANTS)
    gorder = runner._expected_graphbrew_config(
        variants["Leiden-Gorder8"]
    )
    assert gorder["intra_community_order"] == "gorder"
    assert gorder["gorder_window"] == 8
    for spec in variants.values():
        tokens = spec.split(":")
        assert any(token.startswith("sg_") for token in tokens)
        assert any(token.startswith("comm_") for token in tokens)


def test_experiment5_contrasts_change_only_registered_fields():
    for contrast in ABLATION_CONTRASTS:
        base = runner._expected_graphbrew_config(contrast["base"])
        variant = runner._expected_graphbrew_config(contrast["variant"])
        changed = {
            key for key in base
            if base.get(key) != variant.get(key)
        }
        assert changed == set(contrast["effective_fields"]), contrast["name"]


def test_twitter_two_swap_exclusion_is_explicit_and_narrow():
    refined = next(
        contrast["variant"]
        for contrast in ABLATION_CONTRASTS
        if contrast["name"] == "Refinement pass"
    )
    assert algorithm_exclusion_reason("twitter7", refined)
    assert algorithm_exclusion_reason("webbase-2001", refined) is None
    assert algorithm_exclusion_reason(
        "twitter7", ABLATION_CONTRASTS[0]["base"],
    ) is None
    assert str(OPTIONAL_REFINEMENT_TIMEOUT) in algorithm_exclusion_reason(
        "twitter7", refined,
    )
    assert ALGORITHM_GRAPH_EXCLUSIONS == {
        "twitter7": {
            refined: algorithm_exclusion_reason("twitter7", refined),
        },
    }
    assert set(ALGORITHM_GRAPH_EXCLUSION_EVIDENCE) == {
        "twitter7",
    }
    assert set(ALGORITHM_GRAPH_EXCLUSION_EVIDENCE["twitter7"]) == {
        refined,
    }


def test_stage_config_separates_kernel_and_reorder_timeouts(monkeypatch):
    for name in (
        "configure_artifact_root",
        "configure_measurement_generation",
        "configure_runtime_policy",
        "configure_cache_policy",
        "configure_algorithm_filter",
        "configure_execution_mode",
    ):
        monkeypatch.setattr(_common.V, name, lambda *args, **kwargs: None)

    parser = argparse.ArgumentParser()
    _common.add_common_args(parser)
    full = _common.resolve_config(parser.parse_args(["--exp", "2"]))
    assert full["timeout"] == TIMEOUT_FULL
    assert full["reorder_timeout"] == REORDER_TIMEOUT_FULL

    preview = _common.resolve_config(
        parser.parse_args(["--exp", "2", "--preview"])
    )
    assert preview["timeout"] == TIMEOUT_PREVIEW
    assert preview["reorder_timeout"] == REORDER_TIMEOUT_PREVIEW

    overridden = _common.resolve_config(parser.parse_args([
        "--exp", "2",
        "--timeout", "17",
        "--reorder-timeout", "19",
    ]))
    assert overridden["timeout"] == 17
    assert overridden["reorder_timeout"] == 19

    cache = _common.resolve_config(parser.parse_args(["--exp", "1"]))
    assert [graph["name"] for graph in cache["graphs"]] == CACHE_GRAPH_NAMES
    assert cache["cache_mode"] == "ultrafast"

    cache_override = _common.resolve_config(parser.parse_args([
        "--exp", "1",
        "--graphs", "soc-pokec",
    ]))
    assert [graph["name"] for graph in cache_override["graphs"]] == [
        "soc-pokec",
    ]

    scalability = _common.resolve_config(parser.parse_args(["--exp", "8"]))
    assert [graph["name"] for graph in scalability["graphs"]] == (
        SCALABILITY_GRAPH_NAMES
    )


def test_stage03_uses_reorder_timeout_for_reorder_experiments():
    source = inspect.getsource(
        __import__(
            "scripts.experiments.vldb.stages.03_cpu_perf",
            fromlist=["main"],
        ).main
    )
    assert 'cfg["exp"] in {3, 8}' in source
    assert 'cfg["reorder_timeout"]' in source
    assert "args.skip_build" in source


def test_verification_payload_binds_algorithm_exclusions(
    tmp_path,
    monkeypatch,
):
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    provenance = graph.with_suffix(".sg.meta.json")
    provenance.write_text("{}")
    monkeypatch.setattr(
        runner, "resolve_graph_path",
        lambda *args, **kwargs: str(graph),
    )
    monkeypatch.setattr(
        runner, "_overhead_algorithm_specs", lambda: [],
    )
    monkeypatch.setattr(
        runner, "_stable_file_fingerprint",
        lambda path: {"path": str(path)},
    )
    payload = runner._verification_gate_payload(
        [{"name": "tiny"}], [], str(tmp_path),
    )
    assert (
        payload["algorithm_graph_exclusions"]
        == ALGORITHM_GRAPH_EXCLUSIONS
    )
    assert payload["algorithm_graph_exclusion_evidence"] == {}


def test_exclusion_evidence_is_semantically_validated(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({
        "schema": "reorder_timeout_evidence/v1",
        "graph": "twitter7",
        "algorithm_key": "12:test",
        "applicability_timeout_seconds": 21600,
    }))
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path / "vldb_paper")
    monkeypatch.setattr(
        runner,
        "ALGORITHM_GRAPH_EXCLUSION_EVIDENCE",
        {
            "twitter7": {
                "12:test": {
                    "schema": "reorder_timeout_evidence/v1",
                    "artifact": "evidence.json",
                    "timeout_seconds": 21600,
                },
            },
        },
    )
    monkeypatch.setattr(
        runner,
        "ALGORITHM_GRAPH_EXCLUSIONS",
        {"twitter7": {"12:test": "test exclusion"}},
    )

    payload = runner._algorithm_exclusion_evidence_payload(
        [{"name": "twitter7"}],
    )
    assert payload["twitter7"]["12:test"]["artifact"] == str(evidence)

    evidence.write_text('{"schema":"wrong"}\n')
    with pytest.raises(RuntimeError, match="Invalid exclusion evidence"):
        runner._algorithm_exclusion_evidence_payload(
            [{"name": "twitter7"}],
        )


def test_structured_graphbrew_config_is_validated():
    spec = (
        "12:rabbit:compose:sg_super_rabbit:"
        "comm_identity:intra_hubsort"
    )
    config = {
        "schema": "graphbrew_config/v1",
        **runner._expected_graphbrew_config(spec),
    }
    output = (
        "GraphBrew Effective Config: "
        + json.dumps(config, separators=(",", ":"))
    )
    parsed = runner.parse_graphbrew_effective_configs(output)
    runner.validate_graphbrew_effective_configs(["-o", spec], parsed)

    parsed[0]["community_order"] = "degree-desc"
    with pytest.raises(RuntimeError, match="effective config mismatch"):
        runner.validate_graphbrew_effective_configs(["-o", spec], parsed)


def test_structured_graphbrew_realization_is_validated():
    spec = (
        "12:rabbit:compose:sg_super_rabbit:"
        "comm_identity:intra_hubsort"
    )
    effective = {
        "schema": "graphbrew_config/v1",
        **runner._expected_graphbrew_config(spec),
    }
    realized = {
        "schema": "graphbrew_realized/v1",
        "algorithm": "rabbit",
        "aggregation": "rabbit-incremental",
        "ordering": "compose",
        "super_graph": "super-rabbit",
        "community_order": "identity",
        "intra_community_order": "hubsort",
        "refinement_pass": "none",
        "resolution": None,
        "recursive_depth": None,
        "schedule_sensitive": True,
        "final_algo_id": -1,
        "sub_algo_id": 8,
        "num_passes": 1,
        "num_communities": 12,
        "fallbacks": [],
        "block_algorithms": {},
    }
    output = (
        "GraphBrew Realized Config: "
        + json.dumps(realized, separators=(",", ":"))
    )
    parsed = runner.parse_graphbrew_realized_configs(output)
    runner.validate_graphbrew_realized_configs(
        ["-o", spec], [effective], parsed,
    )

    parsed[0]["community_order"] = "degree-desc"
    with pytest.raises(RuntimeError, match="realized config mismatch"):
        runner.validate_graphbrew_realized_configs(
            ["-o", spec], [effective], parsed,
        )


def test_shared_graphbrew_config_helpers_match_vldb_runner_contract():
    spec = (
        "12:rabbit:compose:sg_super_rabbit:"
        "comm_identity:intra_hubsort"
    )
    assert runner._expected_graphbrew_config is reorder_config.expected_graphbrew_config
    assert runner.parse_graphbrew_effective_configs is reorder_config.parse_graphbrew_effective_configs
    assert runner.parse_graphbrew_realized_configs is reorder_config.parse_graphbrew_realized_configs
    assert runner.validate_graphbrew_effective_configs is reorder_config.validate_graphbrew_effective_configs
    assert runner.validate_graphbrew_realized_configs is reorder_config.validate_graphbrew_realized_configs
    assert (
        runner._expected_graphbrew_config(spec)
        == reorder_config.expected_graphbrew_config(spec)
    )

    effective = {
        "schema": "graphbrew_config/v1",
        **reorder_config.expected_graphbrew_config(spec),
    }
    realized = {
        "schema": "graphbrew_realized/v1",
        "algorithm": "rabbit",
        "aggregation": "rabbit-incremental",
        "ordering": "compose",
        "super_graph": "super-rabbit",
        "community_order": "identity",
        "intra_community_order": "hubsort",
        "refinement_pass": "none",
        "resolution": None,
        "recursive_depth": None,
        "schedule_sensitive": True,
        "final_algo_id": -1,
        "sub_algo_id": 8,
        "num_passes": 1,
        "num_communities": 12,
        "fallbacks": [],
        "block_algorithms": {},
    }
    output = "\n".join([
        reorder_config.GRAPHBREW_EFFECTIVE_CONFIG_PREFIX
        + json.dumps(effective, separators=(",", ":")),
        reorder_config.GRAPHBREW_REALIZED_CONFIG_PREFIX
        + json.dumps(realized, separators=(",", ":")),
    ])
    effective_configs = runner.parse_graphbrew_effective_configs(output)
    realized_configs = runner.parse_graphbrew_realized_configs(output)
    assert effective_configs == [effective]
    assert realized_configs == [realized]
    runner.validate_graphbrew_effective_configs(["-o", spec], effective_configs)
    runner.validate_graphbrew_realized_configs(
        ["-o", spec], effective_configs, realized_configs,
    )


def test_kernel_speedup_figure_uses_distinct_compact_styles_and_gm():
    algorithms = [
        "DBG",
        "RabbitOrder (CSR)",
        "RabbitOrder (Boost)",
        "GORDER",
        ALL_ALGORITHMS["12:leiden"],
        ALL_ALGORITHMS["12:hrab:bfs_intra"],
        ALL_ALGORITHMS["12:hrab"],
        ALL_ALGORITHMS["12:rabbit"],
        ALL_ALGORITHMS["12:hubcluster"],
        "GoGraphOrder",
        "RCM",
    ]
    labels = [
        figures.kernel_speedup_label(algorithm)
        for algorithm in algorithms
    ]
    styles = figures.kernel_speedup_styles(algorithms)
    assert len(labels) == len(set(labels))
    assert max(map(len, labels)) <= 12
    assert len({style[0] for style in styles.values()}) == len(algorithms)
    assert all(style[1] == "" for style in styles.values())
    assert figures.append_graph_geomean([1.0, 4.0]) == pytest.approx(
        [1.0, 4.0, 2.0]
    )
    assert figures._amortization_value({
        "break_even": {"status": "finite", "point": 7},
    }) == (7.0, "finite")
    assert math.isinf(figures._amortization_value({
        "break_even": {"status": "never", "point": None},
    })[0])
    assert figures.FIGURES[9][1] is figures.fig9_amortization_trials


def test_compose_supergraph_requires_explicit_community_order():
    with pytest.raises(RuntimeError, match="community-order"):
        runner._expected_graphbrew_config(
            "12:rabbit:compose:sg_super_rabbit:intra_hubsort"
        )


def test_algorithm_filter_restricts_paper_matrix():
    runner.configure_algorithm_filter(["0", "8:csr", "12:hrab"])
    try:
        keys = {
            key for key, _name, _flags
            in runner._paper_algorithm_specs(include_compose=True)
        }
        assert keys == {"0", "8:csr", "12:hrab"}
    finally:
        runner.configure_algorithm_filter(None)


def test_cache_filter_accepts_explicit_compose_candidate():
    compose = dict(COMPOSE_VARIANTS)["SuperRabbit-HubSort"]
    runner.configure_algorithm_filter(["8:csr", compose])
    runner.configure_cache_policy(
        preview=False,
        mode="sampled",
        sample_rate=64,
        all_algorithms=False,
    )
    try:
        keys = {key for key, _name, _flags in runner._cache_algorithm_specs()}
        assert keys == {"8:csr", compose}
    finally:
        runner.configure_algorithm_filter(None)


def test_cache_filter_resolves_ablation_and_chain_keys():
    ablation = ABLATION_CONTRASTS[0]["base"]
    runner.configure_algorithm_filter([
        ablation,
        "chain:GB-Leiden+DBG",
    ])
    try:
        specs = runner._cache_algorithm_specs()
        assert {key for key, _name, _flags in specs} == {
            ablation,
            "chain:GB-Leiden+DBG",
        }
    finally:
        runner.configure_algorithm_filter(None)


def test_cache_trials_are_cold_process_runs():
    assert CACHE_TRIALS == 1


def test_final_cache_matrix_is_faithful_and_representative():
    assert CACHE_PR_ITERATIONS == 5
    assert CACHE_SIZES == [
        2 * 1024**2,
        8 * 1024**2,
        22 * 1024**2,
        32 * 1024**2,
        64 * 1024**2,
    ]
    assert CACHE_GRAPH_NAMES == [
        "cit-Patents",
        "com-Orkut",
        "hollywood-2009",
        "USA-road-d.USA",
    ]


def test_end_to_end_paper_subset_is_predeclared():
    assert len(E2E_PAPER_ALGORITHM_KEYS) == len(
        set(E2E_PAPER_ALGORITHM_KEYS)
    )
    assert {
        f"12:{variant}" for variant in GRAPHBREW_VARIANTS
    }.issubset(E2E_PAPER_ALGORITHM_KEYS)
    assert STABLE_BLOCK_ALGORITHM_KEY not in E2E_PAPER_ALGORITHM_KEYS


def test_cache_hierarchy_lookup_metric_and_styles():
    assert figures.cache_hierarchy_lookups({
        "total_accesses": 100,
        "l1_misses": 40,
        "l2_misses": 20,
        "l3_misses": 5,
    }) == 165
    with pytest.raises(ValueError, match="l2_misses"):
        figures.cache_hierarchy_lookups({
            "total_accesses": 100,
            "l1_misses": 40,
            "l2_misses": None,
            "l3_misses": 5,
        })
    styles = [
        (color, marker, linestyle)
        for _label, color, marker, linestyle
        in figures.CACHE_ALGO_STYLES.values()
    ]
    assert len(styles) == len(set(styles))


def test_cache_environment_is_single_threaded_and_monotonic():
    runner.configure_cache_policy(
        preview=True,
        mode="ultrafast",
        sample_rate=64,
        all_algorithms=False,
        sizes_kib=[256, 8192],
    )
    assert runner._CACHE_SIZE_OVERRIDE == [256 * 1024, 8192 * 1024]
    env = runner._cache_env(64 * 1024)
    assert env["OMP_NUM_THREADS"] == "1"
    assert env["CACHE_MULTICORE"] == "0"
    assert env["CACHE_ULTRAFAST"] == "1"
    assert env["CACHE_POLICY"] == "CLOCK"
    assert env["ECG_PREFETCH_MODE"] == "0"
    assert env["ECG_PFX_BITS"] == "0"
    assert int(env["CACHE_L1_SIZE"]) <= int(env["CACHE_L2_SIZE"])
    assert int(env["CACHE_L2_SIZE"]) <= int(env["CACHE_L3_SIZE"])
    llc_env = runner._cache_env(22 * 1024**2)
    assert llc_env["CACHE_L3_WAYS"] == "11"
    with pytest.raises(ValueError, match="power of two"):
        runner._cache_env(20 * 1024**2)


def test_runtime_policy_reserves_and_slices_cpus():
    runner.configure_runtime_policy(4, "24-27")
    assert runner._cpu_list_for_threads(1) == "24"
    assert runner._cpu_list_for_threads(2) == "24,25"
    assert runner._cpu_list_for_threads(4) == "24,25,26,27"
    assert runner._RUNTIME_ENV["GRAPHBREW_DB_DIR"] == ""
    assert runner._RUNTIME_ENV["GRAPHBREW_TOPOLOGY_ANALYSIS"] == "0"


def test_run_cmd_propagates_semantic_failure(monkeypatch):
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=3, stdout="", stderr="semantic failure",
        ),
    )
    with pytest.raises(RuntimeError, match="Semantic benchmark"):
        runner.run_cmd(["false"])


def test_measurement_policy_changes_with_trials_and_affinity():
    runner.configure_runtime_policy(4, "24-27")
    one_trial = runner.measurement_policy_id("kernel", trials=1)
    five_trials = runner.measurement_policy_id("kernel", trials=5)
    other_cpu = runner.measurement_policy_id(
        "kernel", trials=1, cpu_list="20-23",
    )
    assert one_trial != five_trials
    assert one_trial != other_cpu


def test_mapping_reuse_does_not_depend_on_generation_threads():
    runner.configure_runtime_policy(4, "24-27")
    first = runner.measurement_policy_id("mapping-reuse", trials=1)
    runner.configure_runtime_policy(16, "0-15")
    second = runner.measurement_policy_id(
        "mapping-reuse",
        trials=1,
        cpu_list="24-27",
        env={
            "OMP_NUM_THREADS": "4",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "OMP_DYNAMIC": "FALSE",
            "GRAPHBREW_DB_DIR": "",
            "GRAPHBREW_TOPOLOGY_ANALYSIS": "0",
        },
    )
    assert first == second


def test_rabbit_mapping_draw_classification():
    assert runner._mapping_draw_count(["-o", "8:csr"]) > 1
    assert runner._mapping_draw_count(["-o", "12:rabbit"]) > 1
    assert runner._mapping_draw_count(["-o", "12:hrab"]) > 1
    assert runner._mapping_draw_count([
        "-o",
        "12:rabbit:compose:sg_none:comm_identity:intra_hubsort",
    ]) > 1
    assert runner._mapping_draw_count(["-o", "5"]) == 1
    assert runner._mapping_draw_count([
        "-o", "12:leiden:compose:intra_hubsort",
    ]) == 1


def test_mapping_dry_run_reports_applicability_matrix_with_stale_provenance(
    tmp_path,
    monkeypatch,
    caplog,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "converter").write_text("")
    graph_path = tmp_path / "graph.sg"
    graph_path.write_text("")
    monkeypatch.setattr(runner, "BIN_DIR", bin_dir)
    monkeypatch.setattr(
        runner, "resolve_graph_path",
        lambda *args, **kwargs: str(graph_path),
    )
    monkeypatch.setattr(
        runner, "_graph_provenance_valid",
        lambda path, **kwargs: False,
    )
    runner.configure_algorithm_filter(None)

    with caplog.at_level("INFO"):
        runner._pregenerate_mappings(
            [{"name": "cit-Patents"}, {"name": "twitter7"}],
            str(tmp_path),
            dry_run=True,
        )

    output = caplog.text
    assert "provenance must be refreshed before execution" in output
    assert "cit-Patents: planned 40 mapping(s), 80 named draw(s)" in output
    assert "twitter7: planned 39 mapping(s), 77 named draw(s)" in output
    assert "EXCLUDED: twitter7" in output


def test_measurement_policy_tracks_algorithm_environment(monkeypatch):
    monkeypatch.setenv("GORDER_WINDOW", "5")
    first = runner.measurement_policy_id("kernel", trials=1)
    monkeypatch.setenv("GORDER_WINDOW", "11")
    second = runner.measurement_policy_id("kernel", trials=1)
    assert first != second


def test_measurement_policy_tracks_timing_machine(monkeypatch):
    monkeypatch.setattr(
        runner,
        "timing_machine_metadata",
        lambda: {
            "cpu_governors": ["powersave"],
            "intel_pstate_no_turbo": "0",
        },
    )
    first = runner.measurement_policy_id("kernel", trials=1)
    monkeypatch.setattr(
        runner,
        "timing_machine_metadata",
        lambda: {
            "cpu_governors": ["performance"],
            "intel_pstate_no_turbo": "1",
        },
    )
    second = runner.measurement_policy_id("kernel", trials=1)
    assert first != second


def test_verification_machine_identity_ignores_timing_controls():
    powersave = {
        "cpu_model": "cpu",
        "logical_cpus": 32,
        "cpu_list": "0-15",
        "threads": 16,
        "omp_env": {"OMP_NUM_THREADS": "16"},
        "rabbit_enable_env": "1",
        "cpu_governors": ["powersave"],
        "intel_pstate_no_turbo": "0",
    }
    performance = {
        **powersave,
        "cpu_governors": ["performance"],
        "intel_pstate_no_turbo": "1",
    }
    assert (
        runner.verification_machine_identity(powersave)
        == runner.verification_machine_identity(performance)
    )


def test_timing_policy_fails_fast_and_preview_can_bypass(monkeypatch):
    runner.configure_runtime_policy(16, "0-15")
    monkeypatch.setattr(runner.platform, "node", lambda: "jaguar")
    monkeypatch.setattr(
        runner,
        "timing_machine_metadata",
        lambda: {
            "cpu_governors": ["powersave"],
            "intel_pstate_no_turbo": "0",
        },
    )
    with pytest.raises(RuntimeError, match="Final timing requires"):
        runner.require_timing_machine_policy()
    runner.require_timing_machine_policy(preview=True)
    monkeypatch.setattr(
        runner,
        "timing_machine_metadata",
        lambda: {
            "cpu_governors": ["performance"],
            "intel_pstate_no_turbo": "1",
        },
    )
    runner.require_timing_machine_policy()


def test_measurement_policy_tracks_executable_file_identity(
    tmp_path,
):
    executable = tmp_path / "kernel"
    executable.write_bytes(b"first")
    first = runner.measurement_policy_id(
        "kernel",
        trials=1,
        executable=executable,
    )
    stat = executable.stat()
    executable.write_bytes(b"other")
    os.utime(
        executable,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )
    second = runner.measurement_policy_id(
        "kernel",
        trials=1,
        executable=executable,
    )
    assert first != second


def test_cache_cohort_ignores_timing_machine(monkeypatch):
    monkeypatch.setattr(
        runner,
        "timing_machine_metadata",
        lambda: {
            "cpu_governors": ["powersave"],
            "intel_pstate_no_turbo": "0",
        },
    )
    first = runner.measurement_cohort_id("cache", trials=1)
    monkeypatch.setattr(
        runner,
        "timing_machine_metadata",
        lambda: {
            "cpu_governors": ["performance"],
            "intel_pstate_no_turbo": "1",
        },
    )
    second = runner.measurement_cohort_id("cache", trials=1)
    assert first == second


def test_measurement_cohort_ignores_batch_selection_scope():
    runner._CAMPAIGN_ID = None
    runner.configure_algorithm_filter(["8:csr"])
    runner.configure_cache_policy(
        preview=False,
        mode="sampled",
        sample_rate=64,
        all_algorithms=False,
        sizes_kib=[256],
    )
    first = runner.measurement_cohort_id("cache", trials=1)
    runner.configure_algorithm_filter(["12:hrab"])
    runner.configure_cache_policy(
        preview=False,
        mode="sampled",
        sample_rate=64,
        all_algorithms=False,
        sizes_kib=[8192],
    )
    second = runner.measurement_cohort_id("cache", trials=1)
    runner.configure_algorithm_filter(None)
    assert first == second


def test_kernel_cohort_is_shared_across_benchmarks(tmp_path):
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    runner.configure_measurement_generation("campaign-generation")
    try:
        pr_cohort, _ = runner._kernel_policy_ids(
            graph_path=str(graph),
            kind="kernel",
            trials=1,
            executable=runner.BIN_DIR / "pr",
        )
        bfs_cohort, _ = runner._kernel_policy_ids(
            graph_path=str(graph),
            kind="kernel",
            trials=1,
            executable=runner.BIN_DIR / "bfs",
        )
    finally:
        runner.configure_measurement_generation(None)
    assert pr_cohort == bfs_cohort


def test_kernel_policy_does_not_depend_on_batch_cohort(tmp_path):
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    runner.configure_measurement_generation("generation-a")
    _cohort_a, policy_a = runner._kernel_policy_ids(
        graph_path=str(graph),
        kind="kernel",
        trials=1,
        executable=runner.BIN_DIR / "pr",
    )
    runner.configure_measurement_generation("generation-b")
    _cohort_b, policy_b = runner._kernel_policy_ids(
        graph_path=str(graph),
        kind="kernel",
        trials=1,
        executable=runner.BIN_DIR / "pr",
    )
    runner.configure_measurement_generation(None)
    assert policy_a == policy_b


def test_kernel_policy_uses_canonical_graph_identity(tmp_path):
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10) + b"first")
    _cohort, first = runner._kernel_policy_ids(
        graph_path=str(graph), kind="kernel", trials=1,
        executable=runner.BIN_DIR / "pr",
    )
    graph.write_bytes(struct.pack("<?qq", False, 20, 10) + b"second")
    _cohort, second = runner._kernel_policy_ids(
        graph_path=str(graph), kind="kernel", trials=1,
        executable=runner.BIN_DIR / "pr",
    )
    assert first == second


def test_parse_timing_adds_robust_summary():
    output = "\n".join([
        "Trial Time: 3.00000",
        "Trial Time: 1.00000",
        "Trial Time: 2.00000",
        "Average Time: 2.00000",
        "Iterations: 16.00000",
    ])
    parsed = runner.parse_timing(output)
    assert parsed["median_time"] == 2.0
    assert parsed["mean_time"] == 2.0
    assert parsed["stddev_time"] == 1.0
    assert parsed["iterations"] == 16


def test_reorder_time_uses_validation_and_apply_total():
    output = "\n".join([
        "Reorder Time: 1.0",
        "Reorder Validation Time: 0.2",
        "Reorder Apply Time: 0.3",
        "Reorder End-to-End Time: 1.5",
    ])
    parsed = runner.parse_timing(output)
    assert parsed["mapping_generation_time"] == 1.0
    assert parsed["reorder_validation_time"] == 0.2
    assert parsed["reorder_apply_time"] == 0.3
    assert parsed["reorder_time"] == 1.5


def test_parse_timing_captures_weighted_sssp_contract():
    output = "\n".join([
        "Weight Scheme:          hash",
        "Weight Checksum:        0123456789abcdef0123456789abcdef",
        "Delta:                     32",
        "Trial Time: 1.0",
        "Source Original:          17",
        "Source Internal:            4",
        "Distance Fingerprint:   fedcba9876543210fedcba9876543210",
        "Average Time: 1.0",
    ])
    parsed = runner.parse_timing(output)
    assert parsed["weight_scheme"] == "hash"
    assert parsed["weight_checksum"] == "0123456789abcdef0123456789abcdef"
    assert parsed["delta"] == 32
    assert parsed["source_originals"] == [17]
    assert parsed["source_internals"] == [4]
    assert parsed["distance_fingerprints"] == [
        "fedcba9876543210fedcba9876543210",
    ]


def test_parse_timing_captures_fixed_work_pagerank():
    output = "\n".join([
        "PR Mode:             fixed-work",
        "Iterations:                  20",
        "Final Error:             0.005",
        "Trial Time:               2.0",
        "Iterations:                  20",
        "Final Error:             0.004",
        "Trial Time:               1.0",
        "Average Time:             1.5",
    ])
    parsed = runner.parse_timing(output)
    assert parsed["pr_mode"] == "fixed-work"
    assert parsed["iteration_counts"] == [20, 20]
    assert parsed["final_errors"] == [0.005, 0.004]
    assert parsed["time_per_iteration"] == [0.1, 0.05]
    assert parsed["median_time_per_iteration"] == pytest.approx(0.075)


def test_parse_timing_verification_is_tri_state():
    assert runner.parse_timing("") == {}
    passed = runner.parse_timing("Verification: PASS")
    failed = runner.parse_timing("Verification: FAIL")
    not_run = runner.parse_timing("Average Time: 1.0")
    assert passed["verification_state"] == "pass"
    assert failed["verification_state"] == "fail"
    assert not_run["verification_state"] == "not-run"


def test_parse_timing_captures_executed_work():
    output = "\n".join([
        "Source Original: 7",
        "Source Internal: 3",
        "BFS TD Edges: 10",
        "BFS BU Edges: 20",
        "BFS Edges Examined: 30",
        "BFS Steps: 4",
        "SSSP Edges Examined: 40",
        "SSSP Relax Successes: 5",
        "SSSP Frontier Entries: 6",
        "SSSP Bucket Iterations: 7",
        "CC Sampled Edges: 8",
        "CC Final Edges: 9",
        "CC Compress Steps: 10",
        "CC Skipped Vertices: 11",
        "CC-SV Iterations: 12",
        "CC-SV Edges Examined: 13",
        "CC-SV Compress Steps: 14",
    ])
    parsed = runner.parse_timing(output)
    assert parsed["bfs_edges_examined"] == [30]
    assert parsed["sssp_relax_successes"] == [5]
    assert parsed["cc_final_edges"] == [9]
    assert parsed["cc_sv_iterations"] == [12]


def test_paper_pagerank_command_is_fixed_work():
    cmd = runner.build_benchmark_cmd(
        "pr", "/tmp/tiny.sg", ["-o", "0"], trials=3,
    )
    assert "-F" in cmd
    assert cmd[cmd.index("-i") + 1] == str(
        runner.PR_FIXED_ITERATIONS
    )
    convergence = runner.build_benchmark_cmd(
        "pr_convergence", "/tmp/tiny.sg", ["-o", "0"], trials=3,
    )
    assert "-F" not in convergence
    assert convergence[0].endswith("/pr")
    assert convergence[convergence.index("-i") + 1] == str(
        runner.PR_CONVERGENCE_MAX_ITERATIONS
    )


def test_pagerank_policy_validation_distinguishes_modes():
    fixed = {
        "pr_mode": "fixed-work",
        "iteration_counts": [runner.PR_FIXED_ITERATIONS],
        "final_errors": [0.01],
    }
    runner._validate_kernel_policy(
        benchmark="pr",
        graph_name="tiny",
        timing=fixed,
        trials=1,
    )
    convergence = {
        "pr_mode": "convergence",
        "iteration_counts": [7],
        "final_errors": [runner.PR_TOLERANCE / 2],
    }
    runner._validate_kernel_policy(
        benchmark="pr_convergence",
        graph_name="tiny",
        timing=convergence,
        trials=1,
    )
    with pytest.raises(RuntimeError, match="PageRank"):
        runner._validate_kernel_policy(
            benchmark="pr_convergence",
            graph_name="tiny",
            timing=fixed,
            trials=1,
        )


def test_sssp_command_requires_and_records_frozen_policy(monkeypatch):
    graph = "tiny-policy"
    with pytest.raises(RuntimeError, match="No frozen weighted-SSSP policy"):
        runner.build_benchmark_cmd("sssp", f"/tmp/{graph}.sg", ["-o", "0"])

    monkeypatch.setitem(runner.SSSP_POLICY, graph, {
        "weight_scheme": "hash",
        "weight_checksum": "0123456789abcdef0123456789abcdef",
        "delta": 32,
        "conversion_policy_id": "0123456789abcdef",
    })
    cmd = runner.build_benchmark_cmd(
        "sssp", f"/tmp/{graph}.sg", ["-o", "0"], trials=3,
    )
    assert cmd[cmd.index("-W") + 1] == "hash"
    assert cmd[cmd.index("-d") + 1] == "32"


def test_sssp_policy_rejects_boolean_delta(monkeypatch):
    monkeypatch.setitem(runner.SSSP_POLICY, "bad", {
        "weight_scheme": "hash",
        "weight_checksum": "0" * 32,
        "delta": True,
        "conversion_policy_id": "policy",
    })
    with pytest.raises(RuntimeError, match="Invalid SSSP delta"):
        runner.sssp_policy_for_graph("bad")
    monkeypatch.setitem(runner.SSSP_POLICY, "bad-checksum", {
        "weight_scheme": "hash",
        "weight_checksum": 12345678901234567890123456789012,
        "delta": 1,
        "conversion_policy_id": "policy",
    })
    with pytest.raises(RuntimeError, match="weight checksum"):
        runner.sssp_policy_for_graph("bad-checksum")


def test_cross_ordering_sssp_answers_fail_closed(tmp_path):
    runner.configure_artifact_root(tmp_path / "artifacts")
    timing = {
        "weight_scheme": "hash",
        "weight_checksum": "0123456789abcdef0123456789abcdef",
        "delta": 32,
        "source_originals": [17],
        "distance_fingerprints": [
            "fedcba9876543210fedcba9876543210",
        ],
    }
    runner._validate_cross_ordering_answers(
        graph_name="tiny",
        algo_key="0",
        benchmark="sssp",
        cohort_id="cohort",
        timing=timing,
    )
    runner._validate_cross_ordering_answers(
        graph_name="tiny",
        algo_key="8:csr",
        benchmark="sssp",
        cohort_id="cohort",
        timing=dict(timing),
    )
    changed = dict(timing)
    changed["distance_fingerprints"] = [
        "00000000000000000000000000000000",
    ]
    with pytest.raises(RuntimeError, match="answer mismatch"):
        runner._validate_cross_ordering_answers(
            graph_name="tiny",
            algo_key="12:hrab",
            benchmark="sssp",
            cohort_id="cohort",
            timing=changed,
        )


def test_parse_sampled_cache_accepts_extrapolated_counts():
    parsed = runner.parse_cache_sim("\n".join([
        "║ L1 Cache (32KB, 8-way)",
        "║   Hits: 123 (extrapolated)",
        "║   Misses: 7 (extrapolated)",
        "║ SUMMARY",
        "║ Overall Hit Rate: 94.6154%",
    ]))
    assert parsed["cache_schema"] == "cache_metrics/v2"
    assert parsed["cache_rate_unit"] == "percent"
    assert parsed["l1_hits"] == 123.0
    assert parsed["l1_misses"] == 7.0
    assert parsed["overall_hit_rate"] == 94.6154


def test_precomputed_mapping_time_is_not_reorder_ssot():
    timing = {
        "reorder_time": 0.01,
        "mapping_generation_time": 0.002,
    }
    merged = runner._merge_pregen(
        timing,
        3.5,
        precomputed=True,
    )
    assert "reorder_time" not in merged
    assert merged["map_load_time"] == 0.002
    assert merged["mapping_application_time"] == 0.01
    assert merged["mapping_generation_time"] == 3.5


def test_promoted_mapping_with_zero_time_is_still_precomputed():
    timing = {
        "reorder_time": 0.01,
        "mapping_generation_time": 0.002,
    }
    merged = runner._merge_pregen(
        timing,
        0.0,
        precomputed=True,
    )
    assert merged["reorder_source"] == "precomputed-map"
    assert merged["mapping_generation_time"] == 0.0


def test_mapping_sidecar_timing_reuses_named_draws():
    meta = {
        "mapping_draws": [
            {
                "representation_build_time": 3.0,
                "mapping_generation_time": 10.0,
                "reorder_validation_time": 1.0,
                "reorder_apply_time": 2.0,
                "total_preprocessing_time": 16.0,
            },
            {
                "representation_build_time": 3.5,
                "mapping_generation_time": 14.0,
                "reorder_validation_time": 1.5,
                "reorder_apply_time": 2.5,
                "total_preprocessing_time": 21.5,
            },
            {
                "representation_build_time": 3.2,
                "mapping_generation_time": 12.0,
                "reorder_validation_time": 1.2,
                "reorder_apply_time": 2.2,
                "total_preprocessing_time": 18.6,
            },
        ],
    }
    timing = runner._mapping_sidecar_timing(meta)
    assert timing is not None
    assert timing["timing_source"] == "stage02-sidecar"
    assert timing["mapping_generation_time"] == 10.0
    assert timing["mapping_generation_times"] == [10.0, 14.0, 12.0]
    assert timing["reorder_core_time"] == 10.0
    assert timing["representation_build_time"] == 3.0
    assert timing["total_preprocessing_time"] == 16.0
    assert timing["reorder_time"] == pytest.approx(13.0)


def test_mapping_sidecar_timing_rejects_unmeasured_promotion():
    assert runner._mapping_sidecar_timing({
        "mapping_draws": [{
            "mapping_generation_time": 0.0,
            "reorder_validation_time": 0.0,
            "reorder_apply_time": 0.0,
        }],
    }) is None


def test_measure_reorder_records_only_true_timeouts(monkeypatch):
    def timeout_run(*args, **kwargs):
        kwargs["failure_details"].update({
            "failure_mode": "timeout",
            "elapsed_seconds": 9.8,
        })
        return None

    monkeypatch.setattr(runner, "run_cmd", timeout_run)
    timing = runner._measure_reorder(
        ["converter"],
        repeats=1,
        dry_run=False,
        timeout=10,
        allow_timeout=True,
    )
    assert timing["overhead_timeout"] is True
    assert timing["failure_mode"] == "timeout"

    def failed_run(*args, **kwargs):
        kwargs["failure_details"].update({
            "failure_mode": "nonzero-exit",
            "elapsed_seconds": 1.0,
            "returncode": 9,
        })
        return None

    monkeypatch.setattr(runner, "run_cmd", failed_run)
    with pytest.raises(RuntimeError, match="nonzero-exit"):
        runner._measure_reorder(
            ["converter"],
            repeats=1,
            dry_run=False,
            timeout=10,
            allow_timeout=True,
        )


def test_overhead_checkpoint_retries_only_with_larger_timeout():
    censored = [{
        "overhead_timeout": True,
        "timeout_seconds": 100,
    }]
    assert runner._overhead_checkpoint_is_complete(censored, 100)
    assert runner._overhead_checkpoint_is_complete(censored, 50)
    assert not runner._overhead_checkpoint_is_complete(censored, 101)
    assert runner._overhead_checkpoint_is_complete(
        [{"mapping_generation_time": 1.0}],
        100,
    )
    assert not runner._overhead_checkpoint_is_complete(
        [{"mapping_generation_time": 1.0}],
        100,
        require_equivalence=True,
    )
    assert not runner._overhead_checkpoint_is_complete(
        [{
            "weighted_apply_timeout": True,
            "weighted_timeout_seconds": 100,
        }],
        100,
        require_equivalence=True,
    )
    assert runner._overhead_checkpoint_is_complete(
        [{
            "weighted_apply_timeout": True,
            "weighted_timeout_seconds": 100,
            "mapping_equivalence_checked": True,
        }],
        100,
        require_equivalence=True,
    )
    assert runner._overhead_checkpoint_is_complete(
        [{
            "mapping_generation_time": 1.0,
            "mapping_equivalence_checked": True,
        }],
        100,
        require_equivalence=True,
    )


def test_figure_metric_prefers_trial_median():
    row = {"average_time": 100.0, "trial_times": [3.0, 1.0, 2.0]}
    assert figures.run_metric(row) == 2.0


def test_figures_select_one_measurement_cohort():
    rows = [
        {
            "cohort_id": "old", "cell_key": "a",
            "measured_at": "2026-01-01", "value": 1,
        },
        {
            "cohort_id": "new", "cell_key": "a",
            "measured_at": "2026-02-01", "value": 2,
        },
        {
            "cohort_id": "new", "cell_key": "b",
            "measured_at": "2026-02-01", "value": 3,
        },
    ]
    selected = figures.select_measurement_cohort(rows, "test")
    assert [row["value"] for row in selected] == [2, 3]


def test_figures_reject_split_graph_cohorts():
    rows = [
        {"cohort_id": "graph-a", "cell_key": "a", "value": 1},
        {"cohort_id": "graph-b", "cell_key": "b", "value": 2},
    ]
    with pytest.raises(RuntimeError, match="no complete measurement cohort"):
        figures.select_measurement_cohort(rows, "test")


def test_pr_working_set_uses_exact_sg_dimensions(tmp_path):
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    working_set = runner._pr_working_set(str(graph))
    assert working_set["nodes"] == 10
    assert working_set["edges"] == 20
    assert working_set["property_working_set_bytes"] == 80


def test_mapping_metadata_is_bound_to_algorithm_not_binary(tmp_path):
    artifact_root = tmp_path / "artifacts"
    runner.configure_artifact_root(artifact_root)
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10) + b"first")
    lo = runner._lo_path("tiny", "8:csr")
    meta = runner._meta_path("tiny", "8:csr")
    lo.parent.mkdir(parents=True)
    lo.write_text("0\n")
    flags = ["-o", "8:csr"]
    draw_records = []
    for draw in range(runner._mapping_draw_count(flags)):
        draw_path = lo.with_name(f"{lo.stem}.draw{draw}.lo")
        draw_path.write_text("0\n")
        draw_records.append({
            "draw": draw,
            "path": draw_path.name,
            "graphbrew_effective_configs": [],
            "graphbrew_realized_configs": [],
        })
    meta.write_text(json.dumps({
        "schema": "reorder_meta/v4",
        "graph": "tiny",
        "graph_info": runner._serialized_graph_info(graph),
        "converter_flags": flags,
        "lo_path": lo.name,
        "lo_bytes": lo.stat().st_size,
        "mapping_draw_count": runner._mapping_draw_count(flags),
        "mapping_draws": draw_records,
        "cmd": [
            str(runner.BIN_DIR / "converter"),
            "-f", str(graph), "-o", "8:csr",
            "-q", str(lo.with_name(f"{lo.stem}.draw0.lo").resolve()),
        ],
        "cmd_template": [
            str(runner.BIN_DIR / "converter"),
            "-f", str(graph), "-o", "8:csr",
            "-q", f"{lo.stem}.draw0.lo",
        ],
        "reorder_time": 1.0,
        "graphbrew_effective_configs": [],
        "graphbrew_realized_configs": [],
    }))
    assert runner._mapping_is_valid("tiny", "8:csr", graph, flags)
    relocated_root = tmp_path / "relocated"
    shutil.copytree(artifact_root, relocated_root)
    runner.configure_artifact_root(relocated_root)
    assert runner._mapping_is_valid("tiny", "8:csr", graph, flags)
    _mapped_flags, _time, first_identity = runner.algo_flags_or_map(
        "8:csr", flags, "tiny", str(graph),
    )
    mapped_lo = runner._lo_path("tiny", "8:csr")
    mapped_stat = mapped_lo.stat()
    mapped_lo.write_text("1\n")
    os.utime(
        mapped_lo,
        ns=(mapped_stat.st_atime_ns, mapped_stat.st_mtime_ns + 1_000_000),
    )
    _mapped_flags, _time, second_identity = runner.algo_flags_or_map(
        "8:csr", flags, "tiny", str(graph),
    )
    assert first_identity != second_identity
    assert not runner._mapping_is_valid(
        "tiny", "8:csr", graph, ["-o", "8:boost"],
    )
    graph.write_bytes(struct.pack("<?qq", False, 20, 10) + b"second")
    assert runner._mapping_is_valid("tiny", "8:csr", graph, flags)
    other_graph = tmp_path / "other.sg"
    other_graph.write_bytes(struct.pack("<?qq", False, 22, 11))
    assert not runner._mapping_is_valid(
        "tiny", "8:csr", other_graph, flags,
    )


def test_mapping_metadata_v5_requires_complete_timing_contract(tmp_path):
    artifact_root = tmp_path / "artifacts"
    runner.configure_artifact_root(artifact_root)
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    flags = ["-o", "5"]
    lo = runner._lo_path("tiny", "5")
    lo.parent.mkdir(parents=True)
    lo.write_text("0\n")
    draw_path = lo.with_name(f"{lo.stem}.draw0.lo")
    draw_path.write_text("0\n")
    meta_path = runner._meta_path("tiny", "5")
    payload = {
        "schema": "reorder_meta/v5",
        "graph": "tiny",
        "graph_info": runner._serialized_graph_info(graph),
        "converter_flags": flags,
        "lo_path": lo.name,
        "lo_bytes": lo.stat().st_size,
        "mapping_draw_count": 1,
        "mapping_draws": [{
            "draw": 0,
            "path": draw_path.name,
            "graphbrew_effective_configs": [],
            "graphbrew_realized_configs": [],
        }],
        "cmd": [
            str(runner.BIN_DIR / "converter"),
            "-f",
            str(graph),
            "-o",
            "5",
            "-q",
            str(draw_path.resolve()),
        ],
        "cmd_template": [
            str(runner.BIN_DIR / "converter"),
            "-f",
            str(graph),
            "-o",
            "5",
            "-q",
            draw_path.name,
        ],
        "representation_build_time": 1.0,
        "reorder_core_time": 2.0,
        "reorder_validation_time": 0.1,
        "reorder_apply_time": 0.4,
        "total_preprocessing_time": 3.6,
        "graphbrew_effective_configs": [],
        "graphbrew_realized_configs": [],
    }
    meta_path.write_text(json.dumps(payload))
    assert runner._mapping_is_valid("tiny", "5", graph, flags)

    payload["total_preprocessing_time"] = 3.0
    meta_path.write_text(json.dumps(payload))
    assert not runner._mapping_is_valid("tiny", "5", graph, flags)


def test_graph_provenance_uses_semantic_conversion_policy(tmp_path):
    runner.configure_runtime_policy(4, "24-27")
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10) + b"payload")
    source = tmp_path / "tiny.el"
    source.write_text("x")
    provenance_path = graph.with_suffix(".sg.meta.json")
    provenance = {
        "schema": "graph_source/v1",
        "graph": "tiny",
        "source_path": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "output_path": str(graph.resolve()),
        "output_bytes": graph.stat().st_size,
        "directed": False,
        "symmetrized": True,
        "nodes": 10,
        "directed_edges": 20,
        "random_order_algorithm": "1",
        "random_seed": 0,
        "converter_args": [
            str(runner.BIN_DIR / "converter"),
            "-f", str(source.resolve()), "-s", "-o", "1",
            "-b", str(graph.resolve()),
        ],
        "omp_num_threads": "4",
    }
    provenance["conversion_policy_id"] = (
        runner._graph_conversion_policy_id(provenance)
    )
    provenance_path.write_text(json.dumps(provenance))
    assert runner._graph_provenance_valid(graph)
    provenance["random_seed"] = 1
    provenance_path.write_text(json.dumps(provenance))
    assert not runner._graph_provenance_valid(graph)


def test_verification_gate_manifest_requires_complete_rows(tmp_path):
    artifact_root = tmp_path / "artifacts"
    runner.configure_artifact_root(artifact_root)
    graph_root = tmp_path / "graphs"
    graph_root.mkdir()
    graph = graph_root / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    graph.with_suffix(".sg.meta.json").write_text("{}")
    graphs = [{"name": "tiny"}]
    policy = runner._verification_gate_payload(
        graphs, ["pr"], str(graph_root),
    )
    gate_id = runner.verification_gate_id(
        graphs, ["pr"], str(graph_root),
    )
    row = {
        "gate_id": gate_id,
        "graph": "tiny",
        "algo_key": "0",
        "benchmark": "pr",
        "mapping": policy["mapping_identities"]["tiny"]["0"],
    }
    gate_dir = artifact_root / "vldb_paper" / "verification_gate"
    gate_dir.mkdir(parents=True)
    results = [row]
    (gate_dir / "verification_results.json").write_text(
        json.dumps(results)
    )
    (gate_dir / f"manifest-{gate_id}.json").write_text(json.dumps({
        "schema": "verification_gate/v1",
        "gate_id": gate_id,
        "expected_cells": 1,
        "completed_cells": 1,
        "policy": policy,
        "simulator_smoke": {"verification_state": "pass"},
    }))
    runner.require_verification_gate(
        graphs, ["pr"], str(graph_root),
    )
    (gate_dir / "verification_results.json").write_text(
        json.dumps([])
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.require_verification_gate(
            graphs, ["pr"], str(graph_root),
        )


def test_verification_policy_tracks_binary_file_identity(
    tmp_path,
    monkeypatch,
):
    graph_root = tmp_path / "graphs"
    graph_root.mkdir()
    graph = graph_root / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    graph.with_suffix(".sg.meta.json").write_text("{}")
    bin_dir = tmp_path / "bin"
    work_dir = tmp_path / "work"
    sim_dir = tmp_path / "sim"
    for directory in (bin_dir, work_dir, sim_dir):
        directory.mkdir()
    (bin_dir / "bfs").write_bytes(b"timed")
    (work_dir / "bfs").write_bytes(b"work!")
    monkeypatch.setattr(runner, "BIN_DIR", bin_dir)
    monkeypatch.setattr(runner, "BIN_WORK_DIR", work_dir)
    monkeypatch.setattr(runner, "BIN_SIM_DIR", sim_dir)
    monkeypatch.setattr(
        runner, "_overhead_algorithm_specs", lambda: [],
    )

    first = runner._verification_gate_payload(
        [{"name": "tiny"}], ["bfs"], str(graph_root),
    )
    binary = bin_dir / "bfs"
    stat = binary.stat()
    binary.write_bytes(b"other")
    os.utime(
        binary,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )
    second = runner._verification_gate_payload(
        [{"name": "tiny"}], ["bfs"], str(graph_root),
    )
    assert first["binaries"] != second["binaries"]


def test_verification_gate_accepts_verified_graph_superset(tmp_path):
    artifact_root = tmp_path / "artifacts"
    runner.configure_artifact_root(artifact_root)
    runner.configure_algorithm_filter(["0"])
    graph_root = tmp_path / "graphs"
    graph_root.mkdir()
    graphs = [{"name": "a"}, {"name": "b"}]
    for graph in graphs:
        path = graph_root / f"{graph['name']}.sg"
        path.write_bytes(struct.pack("<?qq", False, 20, 10))
        path.with_suffix(".sg.meta.json").write_text("{}")
    full_id = runner.verification_gate_id(
        graphs, ["bfs"], str(graph_root),
    )
    policy = runner._verification_gate_payload(
        graphs, ["bfs"], str(graph_root),
    )
    rows = [
        {
            "gate_id": full_id,
            "graph": graph["name"],
            "algo_key": "0",
            "benchmark": "bfs",
            "mapping":
                policy["mapping_identities"][graph["name"]]["0"],
        }
        for graph in graphs
    ]
    gate_dir = artifact_root / "vldb_paper" / "verification_gate"
    gate_dir.mkdir(parents=True)
    (gate_dir / "verification_results.json").write_text(
        json.dumps(rows)
    )
    (gate_dir / f"manifest-{full_id}.json").write_text(json.dumps({
        "schema": "verification_gate/v1",
        "gate_id": full_id,
        "expected_cells": 2,
        "completed_cells": 2,
        "policy": policy,
    }))
    try:
        runner.require_verification_gate(
            [{"name": "a"}], ["bfs"], str(graph_root),
        )
    finally:
        runner.configure_algorithm_filter(None)


def test_exp4_materializes_end_to_end_reuse_results(tmp_path):
    runner.configure_artifact_root(tmp_path / "artifacts")
    speedup_dir = runner.RESULTS_DIR / "exp2_speedup"
    overhead_dir = runner.RESULTS_DIR / "exp3_overhead"
    speedup_dir.mkdir(parents=True)
    overhead_dir.mkdir(parents=True)
    (speedup_dir / "speedup_results.json").write_text(json.dumps([
        {
            "graph": "tiny", "benchmark": "pr", "algo_id": 0,
            "algorithm": "Original", "trial_times": [1.0],
            "cohort_id": "kernel", "policy_id": "p0",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "original",
            "mapping_identity": {"source": "direct"},
        },
        {
            "graph": "tiny", "benchmark": "pr", "algo_id": 5,
            "algorithm": "DBG", "trial_times": [0.5],
            "cohort_id": "kernel", "policy_id": "p5",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "dbg",
            "mapping_identity": {"mapping_build_id": "dbg-build"},
        },
        {
            "graph": "tiny", "benchmark": "sssp", "algo_id": 0,
            "algorithm": "Original", "trial_times": [1.0],
            "cohort_id": "kernel", "policy_id": "s0",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "original",
            "mapping_identity": {"source": "direct"},
        },
        {
            "graph": "tiny", "benchmark": "sssp", "algo_id": 5,
            "algorithm": "DBG", "trial_times": [0.5],
            "cohort_id": "kernel", "policy_id": "s5",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "dbg",
            "mapping_identity": {"mapping_build_id": "dbg-build"},
        },
        {
            "graph": "live", "benchmark": "pr", "algo_id": 0,
            "algorithm": "SHUFFLED", "trial_times": [1.0],
            "cohort_id": "kernel", "policy_id": "live-p0",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "live-original",
            "mapping_identity": {"source": "direct"},
        },
        {
            "graph": "live", "benchmark": "pr", "algo_id": 5,
            "algorithm": "DBG", "trial_times": [0.5],
            "cohort_id": "kernel", "policy_id": "live-p5",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "live-dbg",
            "mapping_identity": {"mapping_build_id": "live-dbg"},
        },
        {
            "graph": "live", "benchmark": "sssp", "algo_id": 0,
            "algorithm": "SHUFFLED", "trial_times": [1.0],
            "cohort_id": "kernel", "policy_id": "live-s0",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "live-original",
            "mapping_identity": {"source": "direct"},
        },
        {
            "graph": "live", "benchmark": "sssp", "algo_id": 5,
            "algorithm": "DBG", "trial_times": [0.5],
            "cohort_id": "kernel", "policy_id": "live-s5",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "live-dbg",
            "mapping_identity": {"mapping_build_id": "live-dbg"},
        },
        {
            "graph": "censored", "benchmark": "pr", "algo_id": 0,
            "algorithm": "SHUFFLED", "trial_times": [1.0],
            "cohort_id": "kernel", "policy_id": "censored-p0",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "censored-original",
            "mapping_identity": {"source": "direct"},
        },
        {
            "graph": "censored", "benchmark": "pr", "algo_id": 5,
            "algorithm": "DBG", "trial_times": [0.5],
            "cohort_id": "kernel", "policy_id": "censored-p5",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "censored-dbg",
            "mapping_identity": {
                "mapping_build_id": "censored-dbg",
            },
        },
    ]))
    (overhead_dir / "overhead_results.json").write_text(json.dumps([
        {
            "graph": "tiny", "algo_id": 5,
            "mapping_generation_time": 0.1,
            "reorder_validation_time": 0.02,
            "reorder_apply_time": 0.08,
            "weighted_reorder_apply_time": 0.16,
            "reorder_time": 0.2,
            "timing_source": "stage02-sidecar",
            "measured_at": "2026-01-01T00:00:00",
            "cohort_id": "reorder",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "dbg",
            "mapping_identity": {"mapping_build_id": "dbg-build"},
        },
        {
            "graph": "tiny", "algo_id": 5,
            "overhead_timeout": True,
            "timeout_seconds": 43200,
            "failure_mode": "timeout",
            "measured_at": "2026-02-01T00:00:00",
            "cohort_id": "reorder",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "dbg",
            "mapping_identity": {"mapping_build_id": "dbg-build"},
        },
        {
            "graph": "calibration", "algo_id": 5,
            "mapping_generation_time": 0.3,
            "reorder_validation_time": 0.04,
            "reorder_apply_time": 0.16,
            "weighted_reorder_apply_time": 0.2,
            "reorder_time": 0.5,
            "timing_source": "live-final",
            "sidecar_reference": {
                "mapping_generation_time": 1.0,
            },
            "mapping_generation_calibration_ratio": 3.0,
            "complete_reorder_calibration_ratio": 2.0,
            "cohort_id": "reorder",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "calibration",
            "mapping_identity": {"mapping_build_id": "calibration"},
        },
        {
            "graph": "live", "algo_id": 5,
            "mapping_generation_time": 0.3,
            "reorder_validation_time": 0.04,
            "reorder_apply_time": 0.16,
            "weighted_reorder_apply_time": 0.2,
            "reorder_time": 0.5,
            "timing_source": "live-final",
            "sidecar_reference": {
                "mapping_generation_time": 1.0,
            },
            "mapping_generation_calibration_ratio": 3.0,
            "complete_reorder_calibration_ratio": 2.0,
            "cohort_id": "reorder",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "live-dbg",
            "mapping_identity": {"mapping_build_id": "live-dbg"},
        },
        {
            "graph": "censored", "algo_id": 5,
            "overhead_timeout": True,
            "timeout_seconds": 43200,
            "elapsed_seconds": 43201,
            "failure_mode": "timeout",
            "cohort_id": "reorder",
            "e2e_join_context_id": "join",
            "mapping_identity_id": "censored-dbg",
            "mapping_identity": {
                "mapping_build_id": "censored-dbg",
            },
        },
    ]))
    runner.exp4_end_to_end(
        graphs=[], benchmarks=[], trials=1, timeout=1,
        dry_run=False, graph_dir=".",
    )
    payload = json.loads(
        (
            runner.RESULTS_DIR / "exp4_e2e" / "e2e_results.json"
        ).read_text()
    )
    dbg = next(
        row for row in payload["rows"]
        if str(row["algo_id"]) == "5" and row["benchmark"] == "pr"
    )
    assert dbg["one_run_end_to_end_speedup"] == pytest.approx(1 / 0.9)
    assert dbg["break_even_runs"] == 1
    assert dbg["reuse_counts"]["10"]["speedup"] == pytest.approx(10 / 5.4)
    assert dbg["reorder_time_uncalibrated"] == pytest.approx(0.2)
    assert dbg["reorder_calibration_scope"] == "same-algorithm"
    assert dbg["reorder_calibration_sample_size"] == 2
    assert dbg["reorder_apply_profile"] == "unweighted-csr"
    sssp = next(
        row for row in payload["rows"]
        if str(row["algo_id"]) == "5" and row["benchmark"] == "sssp"
    )
    assert sssp["reorder_time"] == pytest.approx(0.5)
    assert sssp["reorder_time_uncalibrated"] == pytest.approx(0.28)
    assert sssp["mapping_generation_calibration_factor"] == 3.0
    assert sssp["reorder_apply_profile"] == "weighted-csr"
    live_sssp = next(
        row for row in payload["rows"]
        if row["graph"] == "live"
        and str(row["algo_id"]) == "5"
        and row["benchmark"] == "sssp"
    )
    assert live_sssp["reorder_time"] == pytest.approx(0.54)
    assert live_sssp["reorder_time_uncalibrated"] == pytest.approx(0.54)
    assert live_sssp["reorder_calibration_scope"] == "live"
    assert not any(
        row["graph"] == "censored"
        and str(row["algo_id"]) == "5"
        for row in payload["rows"]
    )
    assert any(
        row["graph"] == "censored"
        and str(row["algo_id"]) == "5"
        for row in payload["censored_cells"]
    )


def test_exp4_rejects_mismatched_mapping_campaigns(tmp_path):
    runner.configure_artifact_root(tmp_path / "artifacts")
    speedup_dir = runner.RESULTS_DIR / "exp2_speedup"
    overhead_dir = runner.RESULTS_DIR / "exp3_overhead"
    speedup_dir.mkdir(parents=True)
    overhead_dir.mkdir(parents=True)
    (speedup_dir / "speedup_results.json").write_text(json.dumps([
        {
            "graph": "tiny", "benchmark": "pr", "algo_id": 0,
            "trial_times": [1.0], "cohort_id": "kernel",
            "e2e_join_context_id": "new",
            "mapping_identity_id": "original",
            "mapping_identity": {"source": "direct"},
        },
        {
            "graph": "tiny", "benchmark": "pr", "algo_id": 5,
            "trial_times": [0.5], "cohort_id": "kernel",
            "e2e_join_context_id": "new",
            "mapping_identity_id": "new-dbg",
            "mapping_identity": {"mapping_build_id": "new"},
        },
    ]))
    (overhead_dir / "overhead_results.json").write_text(json.dumps([
        {
            "graph": "tiny", "algo_id": 5, "cohort_id": "reorder",
            "e2e_join_context_id": "old",
            "mapping_identity_id": "old-dbg",
            "mapping_identity": {"mapping_build_id": "old"},
            "mapping_generation_time": 0.1,
            "reorder_validation_time": 0.01,
            "reorder_apply_time": 0.02,
            "weighted_reorder_apply_time": 0.03,
        },
    ]))
    with pytest.raises(RuntimeError, match="no complete compatible"):
        runner.exp4_end_to_end(
            graphs=[], benchmarks=[], trials=1, timeout=1,
            dry_run=False, graph_dir=".",
        )


def test_figure4_accepts_sample_flag_and_writes_table(
    tmp_path,
    monkeypatch,
):
    results_dir = tmp_path / "results"
    tables_dir = tmp_path / "tables"
    exp4_dir = results_dir / "exp4_e2e"
    exp4_dir.mkdir(parents=True)
    (exp4_dir / "e2e_results.json").write_text(json.dumps({
        "schema": "end_to_end_results/v3",
        "primary_cohort": {
            "benchmarks": ["pr"],
        },
        "rows": [{
            "algorithm": "DBG",
            "algo_id": 5,
            "graph": "g1",
            "benchmark": "pr",
            "one_run_end_to_end_speedup": 1.1,
            "break_even_runs": 3,
            "amortization_status": "finite",
            "break_even": {
                "status": "finite",
                "point": 3,
            },
            "reuse_counts": {
                "10": {"point": 1.2},
                "100": {"point": 1.3},
            },
        }],
    }))
    monkeypatch.setattr(figures, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(figures, "TABLES_DIR", tables_dir)
    figures.table_end_to_end(sample=True)
    table = (tables_dir / "table_end_to_end.tex").read_text()
    assert "100 runs" in table
    assert "DBG" in table
    assert "3 & 0.0\\% & 0.0\\% & 1" in table


def test_end_to_end_table_rejects_empty_primary_cohort(
    tmp_path,
    monkeypatch,
):
    results_dir = tmp_path / "results"
    exp4_dir = results_dir / "exp4_e2e"
    exp4_dir.mkdir(parents=True)
    (exp4_dir / "e2e_results.json").write_text(json.dumps({
        "schema": "end_to_end_results/v3",
        "primary_cohort": {"benchmarks": ["pr"]},
        "rows": [],
    }))
    monkeypatch.setattr(figures, "RESULTS_DIR", results_dir)
    with pytest.raises(RuntimeError, match="no algorithms"):
        figures.table_end_to_end()


def test_figure5_reports_kernel_and_reorder_graph_counts(
    tmp_path,
    monkeypatch,
):
    tables_dir = tmp_path / "tables"
    contrast = ABLATION_CONTRASTS[0]
    names = {
        config["algo"]: config["name"]
        for config in ABLATION_CONFIGS
    }
    base = names[contrast["base"]]
    variant = names[contrast["variant"]]
    records = [
        {
            "graph": "g1",
            "config": base,
            "algo": contrast["base"],
            "avg_time": 2.0,
        },
        {
            "graph": "g1",
            "config": variant,
            "algo": contrast["variant"],
            "avg_time": 1.0,
        },
        {
            "graph": "g2",
            "config": base,
            "algo": contrast["base"],
            "avg_time": 4.0,
        },
        {
            "graph": "g2",
            "config": variant,
            "algo": contrast["variant"],
            "avg_time": 2.0,
        },
    ]
    overhead = [
        {
            "graph": "g1",
            "algo_id": contrast["base"],
            "reorder_time": 1.0,
        },
        {
            "graph": "g1",
            "algo_id": contrast["variant"],
            "reorder_time": 2.0,
        },
    ]
    monkeypatch.setattr(figures, "TABLES_DIR", tables_dir)
    monkeypatch.setattr(
        figures,
        "load_json",
        lambda path: (
            overhead
            if "exp3_overhead" in str(path)
            else records
        ),
    )
    monkeypatch.setattr(
        figures, "select_measurement_cohort",
        lambda data, experiment: data,
    )
    monkeypatch.setattr(
        figures, "run_metric",
        lambda record: record["avg_time"],
    )

    figures.fig5_ablation()

    table = (tables_dir / "table_ablation.tex").read_text()
    assert "\\textbf{Wins}" in table
    assert "\\textbf{Kernel graphs}" in table
    assert "\\textbf{Reorder graphs}" in table
    vertex_layout_row = next(
        line for line in table.splitlines()
        if "Vertex layout" in line
    )
    assert vertex_layout_row.endswith("& 2 & 1 \\\\")


def test_complete_reorder_calibration_uses_algo_then_global():
    rows = [
        {
            "algo_id": "a",
            "complete_reorder_calibration_ratio": 2.0,
        },
        {
            "algo_id": "a",
            "complete_reorder_calibration_ratio": 4.0,
        },
        {
            "algo_id": "b",
            "complete_reorder_calibration_ratio": 8.0,
        },
        {
            "graph": next(iter(REORDER_TIMING_REUSE_GRAPHS)),
            "algo_id": next(
                key for key in ("not-an-anchor", "also-not-an-anchor")
                if key not in REORDER_TIMING_ANCHOR_ALGOS
            ),
            "complete_reorder_calibration_ratio": 100.0,
        },
    ]
    factors, global_factor = (
        figures._complete_reorder_calibration_factors(rows)
    )
    assert factors == {"a": 3.0, "b": 8.0}
    assert global_factor == 4.0
    assert figures._calibrated_complete_reorder_time(
        {
            "algo_id": "a",
            "timing_source": "stage02-sidecar",
            "reorder_time": 10.0,
        },
        factors,
        global_factor,
    ) == 30.0
    assert figures._calibrated_complete_reorder_time(
        {
            "algo_id": "missing",
            "timing_source": "stage02-sidecar",
            "reorder_time": 10.0,
        },
        factors,
        global_factor,
    ) == 40.0
    assert figures._calibrated_complete_reorder_time(
        {
            "algo_id": "a",
            "timing_source": "live-final",
            "reorder_time": 10.0,
        },
        factors,
        global_factor,
    ) == 10.0


def test_bootstrap_geo_ci_is_order_pinned_and_repeatable():
    values = [1.1, 0.9, 1.4, 1.2]
    first = figures._bootstrap_geo_ci(
        values,
        resamples=1000,
        seed=7,
    )
    second = figures._bootstrap_geo_ci(
        values,
        resamples=1000,
        seed=7,
    )
    adjusted = figures._bootstrap_geo_ci(
        values,
        resamples=1000,
        seed=7,
        alpha=0.05 / 8,
    )
    assert first == second
    assert first[0] < figures._geo_mean(values) < first[1]
    assert adjusted[0] <= first[0]
    assert adjusted[1] >= first[1]


def test_published_table_uses_exact_paper_path(tmp_path, monkeypatch):
    table_dir = tmp_path / "results" / "tables"
    paper_dir = tmp_path / "research" / "dataCharts"
    monkeypatch.setattr(figures, "PUBLISH_TO_PAPER", True)
    monkeypatch.setattr(figures, "PAPER_CHARTS_DIR", paper_dir)
    figures.save_latex_table(
        "\\begin{table}\\end{table}\n",
        table_dir / "table_ablation.tex",
    )
    assert (
        paper_dir / "tables" / "table_ablation.tex"
    ).read_text() == "\\begin{table}\\end{table}\n"


def test_paper_package_is_complete_and_deterministic(tmp_path):
    aggregate = importlib.import_module(
        "scripts.experiments.vldb.stages.05_aggregate"
    )
    paper_dir = tmp_path / "research"
    for index, relative in enumerate(aggregate.PAPER_PACKAGE_FILES):
        path = paper_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"file-{index}\n".encode())

    package = aggregate.build_paper_package(paper_dir)
    first = package.read_bytes()
    aggregate.build_paper_package(paper_dir)
    assert package.read_bytes() == first
    with zipfile.ZipFile(package) as archive:
        assert archive.namelist() == sorted(
            aggregate.PAPER_PACKAGE_FILES
        )
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0)
            for info in archive.infolist()
        )
        assert all(
            (info.external_attr >> 16) & 0o777 == 0o644
            for info in archive.infolist()
        )


def test_graph_conversion_transaction_restores_backups(
    tmp_path,
    monkeypatch,
):
    graph = tmp_path / "tiny.sg"
    provenance = runner._graph_provenance_path(graph)
    backup_graph = graph.with_suffix(".sg.previous")
    backup_provenance = provenance.with_suffix(".json.previous")
    transaction = runner._graph_conversion_transaction_path(graph)
    candidate = tmp_path / ".tiny.candidate-dead.sg"
    candidate_provenance = runner._graph_provenance_path(candidate)
    backup_graph.write_text("old graph")
    backup_provenance.write_text("old provenance")
    candidate.write_text("candidate")
    candidate_provenance.write_text("candidate provenance")
    runner._write_graph_conversion_transaction(
        transaction_path=transaction,
        graph_name="tiny",
        phase="installing",
        candidate_path=candidate,
        candidate_provenance=candidate_provenance,
        backup_graph=backup_graph,
        backup_provenance=backup_provenance,
    )
    monkeypatch.setattr(
        runner, "_graph_provenance_valid", lambda *args, **kwargs: False,
    )
    runner._recover_graph_conversion_transaction(
        graph_name="tiny",
        graph_path=graph,
        provenance_path=provenance,
        backup_graph=backup_graph,
        backup_provenance=backup_provenance,
        transaction_path=transaction,
    )
    assert graph.read_text() == "old graph"
    assert provenance.read_text() == "old provenance"
    assert not candidate.exists()
    assert not candidate_provenance.exists()
    assert not transaction.exists()


def test_graph_conversion_transaction_keeps_valid_install(
    tmp_path,
    monkeypatch,
):
    graph = tmp_path / "tiny.sg"
    provenance = runner._graph_provenance_path(graph)
    backup_graph = graph.with_suffix(".sg.previous")
    backup_provenance = provenance.with_suffix(".json.previous")
    transaction = runner._graph_conversion_transaction_path(graph)
    candidate = tmp_path / ".tiny.candidate-dead.sg"
    candidate_provenance = runner._graph_provenance_path(candidate)
    graph.write_text("new graph")
    provenance.write_text("new provenance")
    backup_graph.write_text("old graph")
    backup_provenance.write_text("old provenance")
    candidate.write_text("orphan")
    candidate_provenance.write_text("orphan provenance")
    runner._write_graph_conversion_transaction(
        transaction_path=transaction,
        graph_name="tiny",
        phase="installing",
        candidate_path=candidate,
        candidate_provenance=candidate_provenance,
        backup_graph=backup_graph,
        backup_provenance=backup_provenance,
    )
    monkeypatch.setattr(
        runner, "_graph_provenance_valid", lambda *args, **kwargs: True,
    )
    runner._recover_graph_conversion_transaction(
        graph_name="tiny",
        graph_path=graph,
        provenance_path=provenance,
        backup_graph=backup_graph,
        backup_provenance=backup_provenance,
        transaction_path=transaction,
    )
    assert graph.read_text() == "new graph"
    assert provenance.read_text() == "new provenance"
    assert not backup_graph.exists()
    assert not backup_provenance.exists()
    assert not candidate.exists()
    assert not candidate_provenance.exists()


def test_slurm_stages_share_external_graph_root():
    slurm_dir = (
        runner.PROJECT_ROOT
        / "scripts/experiments/vldb/stages/slurm"
    )
    for name in (
        "01_prep.sbatch",
        "02_reorder.sbatch",
        "03_cpu_perf.sbatch",
        "04_cache_sim.sbatch",
        "05_aggregate.sbatch",
    ):
        content = (slurm_dir / name).read_text()
        assert (
            "GRAPH_DIR=\"${GRAPH_DIR:-"
            "/media/Data/00_GraphDatasets/GraphBrew}\""
        ) in content
        assert "--graph-dir \"$GRAPH_DIR\"" in content

    cpu_stage = (slurm_dir / "03_cpu_perf.sbatch").read_text()
    assert "#SBATCH --nodelist=jaguar" in cpu_stage
    assert 'EXPECTED_HOST="${EXPECTED_HOST:-jaguar}"' in cpu_stage


def test_direct_stage_defaults_use_external_roots():
    parser = argparse.ArgumentParser()
    _common.add_common_args(parser)
    args = parser.parse_args(["--exp", "2"])
    assert args.graph_dir == str(PAPER_GRAPH_ROOT)
    assert args.artifact_root == str(PAPER_ARTIFACT_ROOT)
    runner_main = inspect.getsource(runner.main)
    assert "default=str(PAPER_GRAPH_ROOT)" in runner_main
    assert "default=str(PAPER_ARTIFACT_ROOT)" in runner_main

    stage05 = (
        runner.PROJECT_ROOT
        / "scripts/experiments/vldb/stages/05_aggregate.py"
    )
    result = subprocess.run(
        [sys.executable, str(stage05), "--dry-run"],
        cwd=runner.PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key != "GRAPHBREW_VLDB_ROOT"
        },
    )
    assert str(PAPER_GRAPH_ROOT) in result.stdout
    assert str(PAPER_ARTIFACT_ROOT) in result.stdout


def test_sssp_tuner_reuses_completed_graph_checkpoint(
    tmp_path,
    monkeypatch,
):
    runner.configure_artifact_root(tmp_path / "artifacts")
    runner.configure_runtime_policy(1, None)
    runner.configure_cache_policy(
        preview=True,
        mode="ultrafast",
        sample_rate=64,
        all_algorithms=False,
    )
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    policy = {
        "weight_scheme": runner.SSSP_WEIGHT_SCHEME,
        "weight_checksum": "0" * 32,
        "delta": 1,
        "conversion_policy_id": None,
    }
    artifact = {
        "schema": "sssp_delta_tuning/v2",
        "preview": True,
        "eligible_for_freeze": False,
        "weight_scheme": runner.SSSP_WEIGHT_SCHEME,
        "selection_rule_id": runner.SSSP_SELECTION_RULE_ID,
        "delta_candidates": runner.SSSP_DELTA_CANDIDATES,
        "trials_per_candidate": 1,
        "trials_per_invocation": 1,
        "source_count": 1,
        "repeats_per_source": 1,
        "invocation_replicates": 1,
        "candidate_order_policy": runner.SSSP_TUNING_ORDER_POLICY,
        "practical_tie_ratio":
            runner.SSSP_TUNING_PRACTICAL_TIE_RATIO,
        "paired_t_critical": runner.SSSP_TUNING_T_CRITICAL_95_DF8,
        "runtime_env": runner._effective_env(),
        "cpu_list": None,
        "measurement_protocol_id":
            runner._sssp_measurement_protocol_id(
                trials=1,
                sources=1,
                repeats=1,
                replicates=1,
            ),
        "graphs": {
            "tiny": {
                "graph_info": runner._serialized_graph_info(graph),
                "conversion_policy_id": None,
                "candidate_execution_order": [
                    runner.SSSP_DELTA_CANDIDATES,
                ],
                "candidates": [
                    {
                        "delta": delta,
                        "invocations": [{
                            "replicate": 0,
                            "execution_index": index,
                            "trial_times": [float(delta)],
                            "source_originals": [0],
                            "distance_fingerprints": ["a" * 32],
                            "weight_checksum": "0" * 32,
                        }],
                    }
                    for index, delta in enumerate(
                        runner.SSSP_DELTA_CANDIDATES
                    )
                ],
                "selected_policy": policy,
            },
        },
        "preview_candidates": {"tiny": policy},
    }
    path = (
        runner.RESULTS_DIR / "sssp_delta_tuning_preview.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(
        runner,
        "run_cmd",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed graph should not rerun")
        ),
    )
    try:
        result = runner.tune_sssp_deltas(
            graphs=[{"name": "tiny"}],
            graph_dir=str(tmp_path),
            timeout=1,
            dry_run=False,
            trials=1,
        )
    finally:
        runner.configure_cache_policy(
            preview=False,
            mode="sampled",
            sample_rate=64,
            all_algorithms=False,
        )
    assert result == {}
    updated = json.loads(path.read_text())
    assert updated["selection_rule_id"] == runner.SSSP_SELECTION_RULE_ID
    assert (
        updated["graphs"]["tiny"]["candidates"][0]["median_time"]
        == 1.0
    )
    updated["graphs"]["tiny"]["candidates"][0]["invocations"][0][
        "trial_times"
    ] = [True]
    path.write_text(json.dumps(updated))
    runner.configure_cache_policy(
        preview=True,
        mode="ultrafast",
        sample_rate=64,
        all_algorithms=False,
    )
    with pytest.raises(
        RuntimeError, match="Invalid cached SSSP invocation data",
    ):
        runner.tune_sssp_deltas(
            graphs=[{"name": "tiny"}],
            graph_dir=str(tmp_path),
            timeout=1,
            dry_run=False,
            trials=1,
        )
    runner.configure_cache_policy(
        preview=False,
        mode="sampled",
        sample_rate=64,
        all_algorithms=False,
    )


def _sssp_v2_test_artifact(
    graph,
    provenance,
    candidate_rows,
    deltas,
):
    orders = runner._sssp_candidate_execution_orders(deltas, 3)
    return {
        "schema": "sssp_delta_tuning/v2",
        "preview": False,
        "eligible_for_freeze": True,
        "weight_scheme": runner.SSSP_WEIGHT_SCHEME,
        "selection_rule_id": runner.SSSP_SELECTION_RULE_ID,
        "delta_candidates": deltas,
        "trials_per_candidate": 9,
        "trials_per_invocation": 3,
        "source_count": 3,
        "repeats_per_source": 1,
        "invocation_replicates": 3,
        "candidate_order_policy": runner.SSSP_TUNING_ORDER_POLICY,
        "practical_tie_ratio":
            runner.SSSP_TUNING_PRACTICAL_TIE_RATIO,
        "paired_t_critical": runner.SSSP_TUNING_T_CRITICAL_95_DF8,
        "runtime_env": runner._effective_env(),
        "cpu_list": None,
        "measurement_protocol_id":
            runner._sssp_measurement_protocol_id(
                trials=9,
                sources=3,
                repeats=1,
                replicates=3,
                candidates=deltas,
            ),
        "graphs": {
            "tiny": {
                "graph_info": runner._serialized_graph_info(graph),
                "conversion_policy_id": None,
                "candidate_execution_order": orders,
                "candidates": candidate_rows,
            },
        },
        "recommendations": {},
    }


def test_sssp_final_policy_rejects_unbracketed_upper_bound(
    tmp_path,
    monkeypatch,
):
    runner.configure_artifact_root(tmp_path / "artifacts")
    runner.configure_runtime_policy(1, None)
    runner.configure_cache_policy(
        preview=False,
        mode="sampled",
        sample_rate=64,
        all_algorithms=False,
    )
    monkeypatch.setattr(runner, "SSSP_DELTA_CANDIDATES", [1, 2])
    candidate_rows = []
    sources = [0, 1, 2]
    fingerprints = ["a" * 32, "b" * 32, "c" * 32]
    orders = runner._sssp_candidate_execution_orders([1, 2], 3)
    for delta, times in (
        (1, [2.0] * 3),
        (2, [1.0] * 3),
    ):
        candidate_rows.append({
            "delta": delta,
            "invocations": [
                {
                    "replicate": replicate,
                    "execution_index": orders[replicate].index(delta),
                    "trial_times": times,
                    "source_originals": sources,
                    "distance_fingerprints": fingerprints,
                    "weight_checksum": "0" * 32,
                }
                for replicate in range(3)
            ],
        })
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    provenance = graph.with_suffix(".sg.meta.json")
    provenance.write_text("{}")
    artifact = _sssp_v2_test_artifact(
        graph, provenance, candidate_rows, [1, 2],
    )
    path = runner.RESULTS_DIR / "sssp_delta_tuning.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(
        runner,
        "_graph_provenance_valid",
        lambda *args, **kwargs: True,
    )
    with pytest.raises(RuntimeError, match="not bracketed"):
        runner.tune_sssp_deltas(
            graphs=[{"name": "tiny"}],
            graph_dir=str(tmp_path),
            timeout=1,
            dry_run=False,
            trials=9,
        )


def test_sssp_measurement_protocol_excludes_selection_rule(monkeypatch):
    before = runner._sssp_measurement_protocol_id()
    monkeypatch.setattr(
        runner, "SSSP_SELECTION_RULE_ID", "analysis-only-change",
    )
    assert runner._sssp_measurement_protocol_id() == before


def test_sssp_candidate_order_is_cyclically_shifted():
    orders = runner._sssp_candidate_execution_orders(
        list(runner.SSSP_DELTA_CANDIDATES), 3,
    )
    assert [order[0] for order in orders] == [1, 16, 256]
    assert all(
        sorted(order) == sorted(runner.SSSP_DELTA_CANDIDATES)
        for order in orders
    )


def test_sssp_tie_rule_selects_smallest_equivalent_delta():
    fastest_times = [1.0] * 9
    lower_times = [
        0.99, 0.99, 0.99, 0.99,
        1.019, 1.019, 1.019, 1.019, 1.019,
    ]
    rows = [
        {
            "delta": 1,
            "trial_times": lower_times,
            "median_time": statistics.median(lower_times),
        },
        {
            "delta": 2,
            "trial_times": fastest_times,
            "median_time": statistics.median(fastest_times),
        },
        {
            "delta": 4,
            "trial_times": [1.1] * 9,
            "median_time": 1.1,
        },
    ]
    fastest, winner, tie_set = runner._select_sssp_delta(rows)
    assert fastest["delta"] == 2
    assert tie_set == [1, 2]
    assert winner["delta"] == 1
    assert rows[0]["paired_one_sided_95_lower_bound"] <= 0
    assert not rows[2]["within_practical_band"]


def test_sssp_tuner_resumes_only_missing_invocation(
    tmp_path,
    monkeypatch,
):
    runner.configure_artifact_root(tmp_path / "artifacts")
    runner.configure_runtime_policy(1, None)
    runner.configure_cache_policy(
        preview=False,
        mode="sampled",
        sample_rate=64,
        all_algorithms=False,
    )
    monkeypatch.setattr(runner, "SSSP_DELTA_CANDIDATES", [1, 2])
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    provenance = graph.with_suffix(".sg.meta.json")
    provenance.write_text("{}")
    orders = runner._sssp_candidate_execution_orders([1, 2], 3)
    sources = [0, 1, 2]
    fingerprints = ["a" * 32, "b" * 32, "c" * 32]

    def invocation(delta, replicate, value):
        return {
            "replicate": replicate,
            "execution_index": orders[replicate].index(delta),
            "trial_times": [value] * 3,
            "source_originals": sources,
            "distance_fingerprints": fingerprints,
            "weight_checksum": "0" * 32,
        }

    candidates = [
        {
            "delta": 1,
            "invocations": [
                invocation(1, replicate, 1.0)
                for replicate in range(3)
            ],
        },
        {
            "delta": 2,
            "invocations": [
                invocation(2, replicate, 1.01)
                for replicate in range(2)
            ],
        },
    ]
    artifact = _sssp_v2_test_artifact(
        graph, provenance, candidates, [1, 2],
    )
    path = runner.RESULTS_DIR / "sssp_delta_tuning.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(
        runner, "_graph_provenance_valid", lambda *args, **kwargs: True,
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return "\n".join([
            "Weight Checksum: " + "0" * 32,
            "Trial Time: 1.01",
            "Source Original: 0",
            "Distance Fingerprint: " + "a" * 32,
            "Trial Time: 1.01",
            "Source Original: 1",
            "Distance Fingerprint: " + "b" * 32,
            "Trial Time: 1.01",
            "Source Original: 2",
            "Distance Fingerprint: " + "c" * 32,
        ])

    monkeypatch.setattr(runner, "run_cmd", fake_run)
    result = runner.tune_sssp_deltas(
        graphs=[{"name": "tiny"}],
        graph_dir=str(tmp_path),
        timeout=1,
        dry_run=False,
        trials=9,
    )
    assert result["tiny"]["delta"] == 1
    assert len(calls) == 1
    assert calls[0][calls[0].index("-n") + 1] == "3"
    assert calls[0][calls[0].index("-R") + 1] == "1"
    assert calls[0][calls[0].index("-d") + 1] == "2"
    updated = json.loads(path.read_text())
    delta2 = next(
        row for row in updated["graphs"]["tiny"]["candidates"]
        if row["delta"] == 2
    )
    assert len(delta2["invocations"]) == 3


def test_sssp_freeze_is_validation_only(
    tmp_path,
    monkeypatch,
):
    runner.configure_artifact_root(tmp_path / "artifacts")
    runner.configure_runtime_policy(1, None)
    runner.configure_cache_policy(
        preview=False,
        mode="sampled",
        sample_rate=64,
        all_algorithms=False,
    )
    monkeypatch.setattr(runner, "SSSP_DELTA_CANDIDATES", [1, 2])
    graph = tmp_path / "tiny.sg"
    graph.write_bytes(struct.pack("<?qq", False, 20, 10))
    provenance = graph.with_suffix(".sg.meta.json")
    provenance.write_text("{}")
    orders = runner._sssp_candidate_execution_orders([1, 2], 3)
    sources = [0, 1, 2]
    fingerprints = ["a" * 32, "b" * 32, "c" * 32]
    candidates = []
    for delta, value in ((1, 1.0), (2, 1.1)):
        candidates.append({
            "delta": delta,
            "invocations": [
                {
                    "replicate": replicate,
                    "execution_index": orders[replicate].index(delta),
                    "trial_times": [value] * 3,
                    "source_originals": sources,
                    "distance_fingerprints": fingerprints,
                    "weight_checksum": "0" * 32,
                }
                for replicate in range(3)
            ],
        })
    artifact = _sssp_v2_test_artifact(
        graph, provenance, candidates, [1, 2],
    )
    path = runner.RESULTS_DIR / "sssp_delta_tuning.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(
        runner, "_graph_provenance_valid", lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        runner,
        "run_cmd",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("complete tuning artifact must not benchmark")
        ),
    )
    runner.tune_sssp_deltas(
        graphs=[{"name": "tiny"}],
        graph_dir=str(tmp_path),
        timeout=1,
        dry_run=False,
        trials=9,
    )
    reviewed = path.read_bytes()
    snapshot = tmp_path / "sssp_delta_tuning.json"
    policy = tmp_path / "sssp_policy.json"
    monkeypatch.setattr(runner, "SSSP_TUNING_SNAPSHOT_PATH", snapshot)
    monkeypatch.setattr(runner, "SSSP_POLICY_PATH", policy)
    monkeypatch.setattr(runner, "EVAL_GRAPHS", [{"name": "tiny"}])
    runner.tune_sssp_deltas(
        graphs=[{"name": "tiny"}],
        graph_dir=str(tmp_path),
        timeout=1,
        dry_run=False,
        trials=9,
        freeze_policy=True,
    )
    assert path.read_bytes() == reviewed
    assert snapshot.read_bytes() == reviewed
    frozen = json.loads(policy.read_text())
    assert frozen["selection_rule_id"] == runner.SSSP_SELECTION_RULE_ID
    assert frozen["policies"]["tiny"]["delta"] == 1


def test_failed_conversion_preserves_existing_graph(tmp_path, monkeypatch):
    runner.configure_runtime_policy(4, "24-27")
    graph_root = tmp_path / "graphs"
    graph_dir = graph_root / "tiny"
    graph_dir.mkdir(parents=True)
    source = graph_dir / "tiny.el"
    source.write_text("0 1\n1 0\n")
    graph = graph_dir / "tiny.sg"
    original_graph = b"existing-canonical-graph"
    graph.write_bytes(original_graph)
    provenance = graph.with_suffix(".sg.meta.json")
    original_provenance = '{"existing": true}\n'
    provenance.write_text(original_provenance)
    monkeypatch.setattr(runner, "run_cmd", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="conversion"):
        runner._setup_convert_graphs(
            [{"name": "tiny"}],
            graph_root,
            timeout=1,
        )

    assert graph.read_bytes() == original_graph
    assert provenance.read_text() == original_provenance


def test_dry_run_results_store_does_not_write(tmp_path):
    path = tmp_path / "results.json"
    runner.configure_execution_mode(dry_run=True)
    try:
        store = runner.ResultsStore(path, key_fields=["id"])
        store.add({"id": "cell"})
        assert not path.exists()
    finally:
        runner.configure_execution_mode(dry_run=False)


def test_results_store_merges_concurrent_snapshots(tmp_path):
    path = tmp_path / "results.json"
    first = runner.ResultsStore(path, key_fields=["id"])
    second = runner.ResultsStore(path, key_fields=["id"])
    first.add({"id": "a"})
    second.add({"id": "b"})
    rows = json.loads(path.read_text())
    assert {row["id"] for row in rows} == {"a", "b"}


def test_graph_source_selection_prefers_normalized_exact_name(tmp_path):
    graph_dir = tmp_path / "soc-pokec"
    aux_dir = graph_dir / "soc-Pokec"
    aux_dir.mkdir(parents=True)
    graph = graph_dir / "soc-Pokec.mtx"
    auxiliary = aux_dir / "soc-Pokec_completion_percentage.mtx"
    graph.write_bytes(b"graph")
    auxiliary.write_bytes(b"auxiliary" * 100)
    assert runner._select_graph_input(graph_dir, "soc-pokec") == graph


def test_graph_source_selection_prefers_exact_edge_list_over_mtx_fallback(tmp_path):
    graph_dir = tmp_path / "target-graph"
    graph_dir.mkdir()
    exact = graph_dir / "target_graph.el"
    fallback = graph_dir / "unrelated.mtx"
    exact.write_text("0 1\n")
    fallback.write_text("%%MatrixMarket\n")
    assert runner._select_graph_input(graph_dir, "target-graph") == exact


def test_graph_source_selection_reuses_recorded_external_source(tmp_path):
    source = tmp_path / "external" / "graph_0.sg"
    source.parent.mkdir()
    source.write_bytes(b"canonical source")
    source_stat = source.stat()
    provenance = tmp_path / "target.sg.meta.json"
    provenance.write_text(json.dumps({
        "schema": "graph_source/v1",
        "graph": "target",
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
    }))

    assert runner._recorded_graph_input(provenance, "target") == source

    source.write_bytes(b"modified source!")
    os.utime(
        source,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
    )
    assert runner._recorded_graph_input(provenance, "target") == source

    source.write_bytes(b"modified source!!")
    with pytest.raises(RuntimeError, match="recorded graph source changed"):
        runner._recorded_graph_input(provenance, "target")

    wrong_graph = json.loads(provenance.read_text())
    wrong_graph["graph"] = "other"
    provenance.write_text(json.dumps(wrong_graph))
    assert runner._recorded_graph_input(provenance, "target") is None


def test_chain_filter_is_accepted():
    runner.configure_algorithm_filter(["chain:GB-Leiden+DBG"])
    runner.configure_algorithm_filter(None)
