import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT / "research/ecg-hpca/preregistration/"
    "proposal_k2m_sota_pr_screen_v1.json")
V2_CONFIG_PATH = (
    ROOT / "research/ecg-hpca/preregistration/"
    "proposal_k2m_sota_pr_screen_v2.json")
GATE_PATH = (
    ROOT / "scripts/experiments/ecg/analysis/proposal_sota_gate.py")
PAPER_RUN_PATH = (
    ROOT / "scripts/experiments/ecg/flows/paper_run.py")
MANIFEST_PATH = (
    ROOT / "scripts/experiments/ecg/final_paper_manifest.json")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def config(path=CONFIG_PATH):
    return json.loads(path.read_text())


def gate():
    return load_module("proposal_sota_gate_test", GATE_PATH)


def expected_popt(cfg, graph, iterations):
    model = cfg["popt_model"]
    line_size = int(graph["line_size"])
    lines = (
        int(graph["vertices"]) * int(model["property_bytes"]) +
        line_size - 1) // line_size
    matrix_bytes = int(model["reserved_column_slots"]) * lines
    bytes_per_way = (
        gate().parse_size(graph["l3_size"]) //
        int(graph["l3_ways"]))
    reserved_ways = (matrix_bytes + bytes_per_way - 1) // bytes_per_way
    stream_bytes_per_iteration = int(model["epochs"]) * lines
    stream_requests_per_iteration = (
        stream_bytes_per_iteration + line_size - 1) // line_size
    target_stream_bytes = (
        stream_requests_per_iteration * iterations * line_size)
    return (
        matrix_bytes, reserved_ways,
        stream_bytes_per_iteration, target_stream_bytes)


def synthetic_rows(primary_ratio=0.94, cfg=None):
    cfg = cfg or config()
    mod = gate()
    roles = mod.policy_roles(cfg)
    rows = []
    for graph in cfg["graphs"]:
        for iterations in cfg["iterations"]:
            (matrix_bytes, reserved_ways, stream_bytes_per_iteration,
             target_stream_bytes) = expected_popt(
                cfg, graph, iterations)
            ordinary = target_stream_bytes * 2 + 1000
            metrics = {
                "LRU": (110.0, ordinary),
                "GRASP": (100.0, ordinary),
                "POPT": (90.0, ordinary + target_stream_bytes),
                "POPT_UNCHARGED": (85.0, ordinary),
                roles["transport"]: (
                    105.0, ordinary * 1.01),
                roles["primary"]: (
                    90.0 * primary_ratio, ordinary * 1.01),
                roles["characterization"][0]: (
                    90.0 * (primary_ratio + 0.01), ordinary * 1.015),
            }
            for policy, (ticks, traffic) in metrics.items():
                row = {
                    "status": "ok",
                    "final_output_status": "ok",
                    "final_graph": graph["name"],
                    "final_job_id": (
                        f"job-{graph['name']}-i{iterations}"),
                    "benchmark": "pr",
                    "options": cfg["options_template"].format(
                        graph_path=str((ROOT / graph["path"]).resolve()),
                        iterations=iterations),
                    "policy_label": policy,
                    "timing_valid_for_speedup": "1",
                    "timing_model": "simulated_target_time",
                    "timing_caveat": "",
                    "simulator": "gem5",
                    "gem5_cpu_type": "O3",
                    "prefetcher": "none",
                    "pr_result_matched": "1",
                    "l3_exercised": "True",
                    "l1d_size": graph["l1d_size"],
                    "l1d_ways": str(graph["l1d_ways"]),
                    "l2_size": graph["l2_size"],
                    "l2_ways": str(graph["l2_ways"]),
                    "l3_size": graph["l3_size"],
                    "l3_ways": str(graph["l3_ways"]),
                    "line_size": str(graph["line_size"]),
                    "l3_effective_size": graph["l3_size"],
                    "l3_effective_ways": str(graph["l3_ways"]),
                    "gem5_l3_size_actual": graph["l3_size"],
                    "gem5_l3_ways_actual": str(graph["l3_ways"]),
                    "popt_effective_l3_size": graph["l3_size"],
                    "popt_effective_l3_ways": str(graph["l3_ways"]),
                    "popt_reserved_ways": "0",
                    "proposal_path_active": "0",
                    "ecg_schedule_k": "0",
                    "graph_edge_bytes": "4",
                    "edge_stream_bytes_per_edge": "4",
                    "ecg_record_replaces_edge": "0",
                    "pr_iterations": str(iterations),
                    "pr_semantic_edges": str(
                        graph["semantic_receipts"][str(iterations)]["edges"]),
                    "pr_score_checksum":
                        graph["semantic_receipts"][str(iterations)]["checksum"],
                    "sim_ticks": str(ticks),
                    "dram_offchip_bytes": str(traffic),
                    "l3_misses": "100",
                    "roi_insts": "1000",
                    "ipc": "1.0",
                    "dram_bus_util_pct": "1.0",
                }
                if policy == "GRASP":
                    row.update({
                        "grasp_context_loaded": "1",
                        "grasp_regions_loaded": "2",
                        "grasp_hot_property_accesses": "100",
                    })
                elif policy == "POPT":
                    row.update({
                        "popt_overhead_charged": "1",
                        "popt_reserve_model": "size_correct",
                        "popt_matrix_fits": "1",
                        "popt_property_bytes":
                            str(cfg["popt_model"]["property_bytes"]),
                        "popt_matrix_active_columns":
                            str(cfg["popt_model"]["reserved_column_slots"]),
                        "popt_num_epochs":
                            str(cfg["popt_model"]["epochs"]),
                        "popt_min_data_ways":
                            str(cfg["popt_model"]["minimum_data_ways"]),
                        "popt_reload_each_iteration": "1",
                        "popt_initial_columns_charged": "1",
                        "popt_target_time_charged": "0",
                        "popt_timing_optimistic": "1",
                        "timing_model":
                            "optimistic_popt_analytic_stream",
                        "timing_caveat":
                            "Matrix-stream latency is omitted; timing "
                            "therefore favors P-OPT.",
                        "popt_matrix_stream_mode": "analytic_cumulative",
                        "popt_offchip_includes_matrix_stream": "1",
                        "popt_policy_active": "1",
                        "popt_context_loaded": "1",
                        "popt_rereference_loaded": "1",
                        "popt_runtime_epochs":
                            str(cfg["popt_model"]["epochs"]),
                        "popt_runtime_cache_lines": str(
                            stream_bytes_per_iteration //
                            int(cfg["popt_model"]["epochs"])),
                        "popt_roi_rereference_queries": "100",
                        "popt_matrix_bytes": str(matrix_bytes),
                        "popt_reserved_ways": str(reserved_ways),
                        "popt_effective_l3_ways": str(
                            int(graph["l3_ways"]) - reserved_ways),
                        "popt_effective_l3_size": str(
                            mod.parse_size(graph["l3_size"]) *
                            (int(graph["l3_ways"]) - reserved_ways) //
                            int(graph["l3_ways"])),
                        "l3_effective_ways": str(
                            int(graph["l3_ways"]) - reserved_ways),
                        "l3_effective_size": str(
                            mod.parse_size(graph["l3_size"]) *
                            (int(graph["l3_ways"]) - reserved_ways) //
                            int(graph["l3_ways"])),
                        "gem5_l3_ways_actual": str(
                            int(graph["l3_ways"]) - reserved_ways),
                        "gem5_l3_size_actual": str(
                            mod.parse_size(graph["l3_size"]) *
                            (int(graph["l3_ways"]) - reserved_ways) //
                            int(graph["l3_ways"])),
                        "popt_matrix_stream_bytes":
                            str(stream_bytes_per_iteration),
                        "popt_cumulative_stream_bytes":
                            str(target_stream_bytes),
                        "popt_matrix_stream_iterations": str(iterations),
                        "popt_matrix_stream_requests": str(
                            target_stream_bytes // int(graph["line_size"])),
                        "popt_dram_offchip_bytes_without_matrix_stream":
                            str(ordinary),
                        "popt_matrix_stream_dram_bytes":
                            str(target_stream_bytes),
                        "popt_stream_requestor_dram_bytes":
                            str(target_stream_bytes),
                    })
                elif policy == roles["oracle"]:
                    row.update({
                        "popt_overhead_charged": "0",
                        "popt_reserved_ways": "0",
                        "popt_target_time_charged": "0",
                        "popt_matrix_stream_mode": "none",
                        "popt_matrix_stream_bytes": "0",
                        "popt_matrix_stream_requests": "0",
                        "popt_cumulative_stream_bytes": "0",
                        "popt_matrix_stream_dram_bytes": "0",
                        "popt_stream_requestor_dram_bytes": "0",
                        "popt_dram_offchip_bytes_without_matrix_stream":
                            str(ordinary),
                        "popt_nonstream_requestor_dram_bytes": str(ordinary),
                        "popt_effective_l3_ways": str(graph["l3_ways"]),
                        "popt_effective_l3_size": graph["l3_size"],
                        "popt_policy_active": "1",
                        "popt_context_loaded": "1",
                        "popt_rereference_loaded": "1",
                        "popt_runtime_epochs":
                            str(cfg["popt_model"]["epochs"]),
                        "popt_runtime_cache_lines": str(
                            stream_bytes_per_iteration //
                            int(cfg["popt_model"]["epochs"])),
                        "popt_roi_rereference_queries": "100",
                    })
                elif policy.startswith("ECG_K2_"):
                    receipt = cfg["variant_receipts"][policy]
                    row.update({
                        "proposal_path_active": "1",
                        "proposal_performance_mode_active": "1",
                        "gem5_compact_k2m_streamshield_active": "1",
                        "gem5_compact_k2m_performance_requested": "1",
                        "gem5_ecg_delivery":
                            "ecg.stream.load2.compact+ecg.k2.mload.f32",
                        "gem5_k2_binding_model": "request",
                        "ecg_schedule_k": "2",
                        "ecg_record_bytes": "4",
                        "ecg_record_replaces_edge": "1",
                        "edge_stream_bytes_per_edge": "4",
                        "k2_metadata_bits_per_line": "49",
                        "l3_effective_ways": str(graph["l3_ways"]),
                        "l3_effective_size": graph["l3_size"],
                        "ecg_isa_variant": cfg["isa_variant"],
                        "ecg_epochs": str(cfg["k2_epochs"]),
                        "gem5_k2_delivery_trace_limit": "0",
                        "gem5_stream_bypass_trace_limit": "0",
                        "proposal_compact_id_bits":
                            str(graph["compact_id_bits"]),
                        "proposal_compact_epoch_bits":
                            str(graph["compact_epoch_bits"]),
                        "proposal_compact_tier_bits":
                            str(cfg["compact_tier_bits"]),
                        "gem5_variant_requested_receipt":
                            receipt["requested"],
                        "gem5_variant_effective_receipt":
                            str(receipt["effective"]),
                        "gem5_variant_dueling_receipt":
                            str(receipt["dueling"]),
                    })
                    if int(receipt["dueling"]) == 1:
                        row.update({
                            "gem5_k2_dueling_request_bound_victims": "100",
                            "gem5_k2_dueling_leader_samples": "2048",
                            "gem5_k2_dueling_follower_selections": "90",
                            "gem5_k2_dueling_completed_windows": "2",
                            "gem5_k2_dueling_winner_changes": "0",
                            "gem5_k2_dueling_follower_variant_overrides": "0",
                        })
                rows.append(row)
    return rows


def test_ssot_is_compact_and_has_no_hash_qualification():
    cfg = config()
    text = CONFIG_PATH.read_text()
    assert len(text.splitlines()) < 300
    assert "sha256" not in text.lower()
    assert len(cfg["graphs"]) == 3
    assert cfg["iterations"] == [1, 2, 4, 8]
    assert len(cfg["policies"]["all"]) == 7
    assert cfg["policies"]["primary_candidate"] == (
        "ECG:K2_STREAMSHIELD")
    assert cfg["compact_tier_bits"] == 2


def test_v2_preserves_failed_v1_and_preregisters_static_rrip():
    v1 = config()
    v2 = config(V2_CONFIG_PATH)
    assert len(V2_CONFIG_PATH.read_text().splitlines()) < 300
    assert "sha256" not in V2_CONFIG_PATH.read_text().lower()
    assert v2["lineage"]["prior_screen"] == v1["id"]
    assert v2["blockers"] == []
    assert v2["execution"]["ready"] is True
    assert v2["execution"]["maximum_policy_runtime_seconds"] == 86400
    assert v1["policies"]["primary_candidate"] == (
        "ECG:K2_STREAMSHIELD")
    assert v2["policies"]["primary_candidate"] == (
        "ECG:K2_RRIP_STREAMSHIELD")
    assert v2["policies"]["all"] == [
        "LRU", "GRASP", "POPT", "POPT:UNCHARGED",
        "ECG:K2_LRU_STREAMSHIELD",
        "ECG:K2_RRIP_STREAMSHIELD",
        "ECG:K2_ONLINE_STREAMSHIELD",
    ]
    changed = {
        key for key in set(v1) | set(v2)
        if v1.get(key) != v2.get(key)
    }
    assert changed == {
        "version", "id", "scope", "lineage",
        "policies", "variant_receipts", "blockers", "execution",
    }
    assert {
        key: value for key, value in v2["policies"].items()
        if key not in ("all", "primary_candidate")
    } == {
        key: value for key, value in v1["policies"].items()
        if key not in ("all", "primary_candidate")
    }
    assert {
        key: value for key, value in v2["variant_receipts"].items()
        if key != "ECG_K2_RRIP_STREAMSHIELD"
    } == {
        key: value for key, value in v1["variant_receipts"].items()
        if key != "ECG_K2_STREAMSHIELD"
    }
    result = gate().evaluate(synthetic_rows(cfg=v2), v2)
    assert result["primary_candidate"] == "ECG_K2_RRIP_STREAMSHIELD"
    assert result["screen_passes"] is True


def test_profile_expands_to_twelve_whole_cells(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/flows/paper_run.py",
            "--profile", "ecg_proposal_k2m_sota_pr_screen",
            "--run-dir", str(tmp_path / "run"),
            "--list", "--dry-run", "--no-build",
            "--allow-missing-graphs",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "jobs=12" in text
    for iterations in (1, 2, 4, 8):
        assert text.count(f"-i {iterations} -t 0'") == 3
    assert text.count(
        "--policies LRU GRASP POPT POPT:UNCHARGED "
        "ECG:K2_LRU_STREAMSHIELD ECG:K2_STREAMSHIELD "
        "ECG:K2_ONLINE_STREAMSHIELD") == 12
    assert text.count("--popt-active-columns 3") == 12
    assert text.count("--popt-matrix-stream analytic") == 12
    assert text.count("--timeout-gem5 43200") == 12
    assert text.count("--gem5-compact-k2m-performance") == 12

    blocked = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/flows/paper_run.py",
            "--profile", "ecg_proposal_k2m_sota_pr_screen",
            "--run-dir", str(tmp_path / "blocked"),
            "--limit", "1", "--no-build", "--allow-missing-graphs",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert blocked.returncode != 0
    assert "blocked by screen config" in (
        blocked.stdout + blocked.stderr)

    paper = load_module("proposal_sota_paper_run_test", PAPER_RUN_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text())
    stage = next(
        value for value in manifest["stages"]
        if value["name"] == "70_gem5_proposal_sota_pr_i1")
    settings, _ = paper.apply_screen_config(
        paper.merged_defaults(manifest, stage))
    assert settings["env"]["ECG_RECORD_VARIABLE_WIDTH"] == "1"
    assert settings["env"]["ECG_RECORD_TIER_BITS"] == str(
        config()["compact_tier_bits"])


def test_v2_profile_expands_with_static_rrip_primary(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/flows/paper_run.py",
            "--profile", "ecg_proposal_k2m_sota_pr_screen_v2",
            "--run-dir", str(tmp_path / "run-v2"),
            "--list", "--dry-run", "--no-build",
            "--allow-missing-graphs",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "jobs=12" in text
    assert text.count(
        "--policies LRU GRASP POPT POPT:UNCHARGED "
        "ECG:K2_LRU_STREAMSHIELD ECG:K2_RRIP_STREAMSHIELD "
        "ECG:K2_ONLINE_STREAMSHIELD") == 12
    assert "proposal_k2m_sota_pr_screen_v2.json" in (
        MANIFEST_PATH.read_text())


def test_popt_model_matches_roi_matrix_producer():
    cfg = config()
    mod = gate()
    roi = load_module(
        "proposal_sota_roi_matrix_test",
        ROOT / "scripts/experiments/ecg/roi_matrix.py")
    spec = roi.parse_policy_spec("POPT")
    for graph in cfg["graphs"]:
        args = SimpleNamespace(
            options=f"-f {graph['path']} -i 1",
            line_size=str(graph["line_size"]),
            l3_ways=str(graph["l3_ways"]),
            popt_property_bytes=str(cfg["popt_model"]["property_bytes"]),
            popt_active_columns=str(
                cfg["popt_model"]["reserved_column_slots"]),
            popt_num_epochs=str(cfg["popt_model"]["epochs"]),
            popt_min_data_ways=str(
                cfg["popt_model"]["minimum_data_ways"]),
            popt_reserve_model="size_correct",
        )
        charge = roi.popt_charge_metadata(args, spec, graph["l3_size"])
        expected = mod.expected_popt(cfg, graph, 1)
        assert expected["reserved_ways"] == cfg["popt_model"][
            "expected_reserved_ways_in_screen"]
        assert expected["effective_ways"] == cfg["popt_model"][
            "expected_effective_data_ways_in_screen"]
        assert charge["popt_matrix_bytes"] == expected["matrix_bytes"]
        assert charge["popt_reserved_ways"] == expected["reserved_ways"]
        assert charge["popt_matrix_stream_bytes"] == (
            expected["stream_bytes_per_iteration"])


def test_policy_sharding_is_rejected(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/slurm/make_slurm_shards.py",
            "--profile", "ecg_proposal_k2m_sota_pr_screen",
            "--run-tag", "screen",
            "--out", str(tmp_path / "shards.tsv"),
            "--allow-blocked",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "whole-cell jobs" in (result.stdout + result.stderr)

    local = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/flows/paper_run.py",
            "--profile", "ecg_proposal_k2m_sota_pr_screen",
            "--policy", "LRU",
            "--run-dir", str(tmp_path / "policy-filter"),
            "--dry-run", "--no-build", "--allow-missing-graphs",
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert local.returncode != 0
    assert "complete policy roster" in (local.stdout + local.stderr)


def test_valid_screen_passes_and_reports_attribution():
    result = gate().evaluate(synthetic_rows(), config())
    primary = result["candidates"]["ECG_K2_STREAMSHIELD"]
    online = result["candidates"]["ECG_K2_ONLINE_STREAMSHIELD"]
    assert result["cell_count"] == 12
    assert result["row_count"] == 84
    assert result["screen_valid"] is True
    assert result["screen_result"] == "go"
    assert result["screen_passes"] is True
    assert result["stop_broad_campaign"] is False
    assert primary["passes"] is True
    assert primary["replacement_policy_contribution"] is True
    assert online["decision_role"] == "characterization_only"
    assert "POPT_UNCHARGED" in primary["comparisons"]
    assert len(result["popt_stream_accounting"]) == 12
    assert result["decision"] == config()["decision"]
    assert result["popt_model"] == config()["popt_model"]


def test_online_characterization_cannot_pass_screen_alone():
    rows = synthetic_rows()
    for row in rows:
        if row["policy_label"] == "ECG_K2_STREAMSHIELD":
            row["sim_ticks"] = "100"
    result = gate().evaluate(rows, config())
    assert result["candidates"]["ECG_K2_STREAMSHIELD"]["passes"] is False
    assert result["candidates"]["ECG_K2_ONLINE_STREAMSHIELD"]["passes"] is True
    assert result["screen_passes"] is False


def test_baseline_activity_and_popt_accounting_fail_closed():
    rows = synthetic_rows()
    grasp = next(row for row in rows if row["policy_label"] == "GRASP")
    grasp["grasp_hot_property_accesses"] = "0"
    with pytest.raises(ValueError, match="must be positive"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    grasp = next(row for row in rows if row["policy_label"] == "GRASP")
    grasp["grasp_regions_loaded"] = "0"
    with pytest.raises(ValueError, match="must be positive"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    popt = next(row for row in rows if row["policy_label"] == "POPT")
    popt["popt_matrix_stream_dram_bytes"] = "0"
    with pytest.raises(
            ValueError, match="components|decomposition"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    popt = next(row for row in rows if row["policy_label"] == "POPT")
    popt["popt_matrix_stream_iterations"] = "99"
    with pytest.raises(ValueError, match="popt_matrix_stream_iterations"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    popt = next(row for row in rows if row["policy_label"] == "POPT")
    popt["popt_roi_rereference_queries"] = "0"
    with pytest.raises(ValueError, match="must be positive"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    oracle = next(
        row for row in rows
        if row["policy_label"] == "POPT_UNCHARGED")
    oracle["popt_policy_active"] = "0"
    with pytest.raises(ValueError, match="popt_policy_active"):
        gate().evaluate(rows, config())


def test_k2_performance_mode_and_online_dueling_fail_closed():
    rows = synthetic_rows()
    primary = next(
        row for row in rows
        if row["policy_label"] == "ECG_K2_STREAMSHIELD")
    primary["proposal_performance_mode_active"] = "0"
    with pytest.raises(ValueError, match="proposal_performance_mode_active"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    online = next(
        row for row in rows
        if row["policy_label"] == "ECG_K2_ONLINE_STREAMSHIELD")
    online["gem5_variant_dueling_receipt"] = "0"
    with pytest.raises(ValueError, match="gem5_variant_dueling_receipt"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    online = next(
        row for row in rows
        if row["policy_label"] == "ECG_K2_ONLINE_STREAMSHIELD")
    online["gem5_k2_dueling_completed_windows"] = "0"
    with pytest.raises(ValueError, match="must be positive"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    online = next(
        row for row in rows
        if row["policy_label"] == "ECG_K2_ONLINE_STREAMSHIELD")
    online["gem5_k2_dueling_leader_samples"] = "1023"
    with pytest.raises(ValueError, match="full leader-sample window"):
        gate().evaluate(rows, config())


def test_incomplete_duplicate_and_extra_cells_are_rejected():
    rows = synthetic_rows()
    with pytest.raises(ValueError, match="incomplete policy roster"):
        gate().evaluate(rows[:-1], config())

    rows = synthetic_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate policy"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    rows[0]["final_graph"] = "other-graph"
    with pytest.raises(ValueError, match="outside screen graph set"):
        gate().evaluate(rows, config())


def test_per_cell_guard_prevents_masking():
    rows = synthetic_rows(primary_ratio=0.80)
    for row in rows:
        if (
                row["policy_label"] == "ECG_K2_STREAMSHIELD" and
                row["final_graph"] == "web-Google-n16" and
                "-i 1" in row["options"]):
            row["sim_ticks"] = str(90.0 * 1.021)
    result = gate().evaluate(rows, config())
    assert result["candidates"]["ECG_K2_STREAMSHIELD"]["passes"] is False
    assert result["screen_passes"] is False


def test_i8_guard_prevents_short_run_masking():
    rows = synthetic_rows(primary_ratio=0.80)
    for row in rows:
        if (
                row["policy_label"] == "ECG_K2_STREAMSHIELD" and
                "-i 8" in row["options"]):
            row["sim_ticks"] = str(90.0 * 0.98)
    result = gate().evaluate(rows, config())
    assert result["candidates"]["ECG_K2_STREAMSHIELD"]["passes"] is False


def test_leave_one_graph_out_guard_prevents_one_graph_masking():
    rows = synthetic_rows()
    for row in rows:
        if row["policy_label"] == "ECG_K2_STREAMSHIELD":
            ratio = (
                0.70 if row["final_graph"] == "web-Google-n16"
                else 0.99)
            row["sim_ticks"] = str(90.0 * ratio)
    result = gate().evaluate(rows, config())
    assert result["candidates"]["ECG_K2_STREAMSHIELD"]["passes"] is False


def test_oracle_sanity_is_checked_per_cell():
    rows = synthetic_rows()
    oracle = next(
        row for row in rows
        if (
            row["policy_label"] == "POPT_UNCHARGED" and
            row["final_graph"] == "web-Google-n16" and
            "-i 1" in row["options"]
        ))
    oracle["sim_ticks"] = "1000"
    result = gate().evaluate(rows, config())
    assert result["oracle_sanity_passes"] is False
    assert result["screen_passes"] is True
    assert result["stop_broad_campaign"] is False


def test_invalid_baseline_is_inconclusive_not_stop():
    rows = synthetic_rows(primary_ratio=0.80)
    for row in rows:
        if row["policy_label"] == "GRASP":
            row["sim_ticks"] = "176"
    result = gate().evaluate(rows, config())
    assert result["baseline_sanity_passes"] is False
    assert result["screen_valid"] is False
    assert result["screen_result"] == "inconclusive_invalid_baselines"
    assert result["screen_passes"] is False
    assert result["stop_broad_campaign"] is False
    assert result["candidates"]["ECG_K2_STREAMSHIELD"][
        "performance_guards_pass"] is True


def test_faithfully_expensive_charged_popt_is_not_invalid():
    rows = synthetic_rows(primary_ratio=0.80)
    for row in rows:
        if row["policy_label"] == "POPT":
            row["sim_ticks"] = "176"
    result = gate().evaluate(rows, config())
    assert result["baseline_sanity"]["POPT"]["passes"] is True
    assert result["screen_valid"] is True


def test_transport_claim_has_leave_one_graph_out_guard():
    rows = synthetic_rows()
    candidate_ticks = 90.0 * 0.94
    ratios = {
        "web-Google-n16": 1.10,
        "soc-pokec-n16": 0.90,
        "cit-Patents-n18-sym": 0.90,
    }
    for row in rows:
        if row["policy_label"] == "ECG_K2_LRU_STREAMSHIELD":
            row["sim_ticks"] = str(
                candidate_ticks / ratios[row["final_graph"]])
    result = gate().evaluate(rows, config())
    primary = result["candidates"]["ECG_K2_STREAMSHIELD"]
    assert primary["passes"] is True
    assert primary["comparisons"]["ECG_K2_LRU_STREAMSHIELD"][
        "aggregate_time_ratio"] <= 0.98
    assert primary["replacement_policy_contribution"] is False
    assert result["replacement_policy_claim_allowed"] is False


def test_transport_only_win_does_not_authorize_policy_claim():
    rows = synthetic_rows()
    for row in rows:
        if row["policy_label"] == "ECG_K2_LRU_STREAMSHIELD":
            row["sim_ticks"] = str(90.0 * 0.94)
    result = gate().evaluate(rows, config())
    primary = result["candidates"]["ECG_K2_STREAMSHIELD"]
    assert result["screen_passes"] is True
    assert primary["claim_classification"] == (
        "complete_design_transport_or_layout_only")
    assert result["replacement_policy_claim_allowed"] is False


def test_wrong_semantics_or_geometry_is_rejected():
    rows = synthetic_rows()
    rows[0]["pr_score_checksum"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    rows[0]["l2_ways"] = "4"
    with pytest.raises(ValueError, match="failed l2_ways"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    lru = next(row for row in rows if row["policy_label"] == "LRU")
    lru["l3_effective_ways"] = "8"
    with pytest.raises(ValueError, match="l3_effective_ways"):
        gate().evaluate(rows, config())

    rows = synthetic_rows()
    popt = next(row for row in rows if row["policy_label"] == "POPT")
    popt["popt_effective_l3_size"] = "64kB"
    with pytest.raises(ValueError, match="popt_effective_l3_size"):
        gate().evaluate(rows, config())

    bad_config = config()
    bad_config["graphs"][0]["compact_epoch_bits"] = 4
    with pytest.raises(ValueError, match="too few compact epoch bits"):
        gate().evaluate(synthetic_rows(), bad_config)

    bad_config = config()
    bad_config["popt_model"]["expected_reserved_ways_in_screen"] = 1
    with pytest.raises(ValueError, match="reservation differs"):
        gate().evaluate(synthetic_rows(), bad_config)
