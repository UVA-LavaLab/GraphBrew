import argparse
import csv
import hashlib
import importlib.util
import json
import pytest
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def output_descriptor(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    rows = None
    if path.suffix == ".csv":
        rows = max(len(path.read_text().splitlines()) - 1, 0)
    return {
        "sha256": digest,
        "size": path.stat().st_size,
        "rows": rows,
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_k2_policy_aliases_are_first_class(monkeypatch):
    module = load_module(
        "roi_matrix_paper_ssot",
        ROOT / "scripts/experiments/ecg/roi_matrix.py",
    )
    monkeypatch.delenv("ECG_EDGE_MASK_SCHED", raising=False)
    monkeypatch.delenv("ECG_STREAM_BYPASS", raising=False)

    k2 = module.parse_policy_spec("ECG:K2")
    assert k2.label == "ECG_K2"
    assert k2.ecg_mode == "ECG_GRASP_POPT"
    assert module.ecg_transport_for(
        k2, "pr") == module.EcgTransport(2, False, False)
    with pytest.raises(RuntimeError):
        module.ecg_transport_for(k2, "cc")
    with pytest.raises(RuntimeError):
        module.ecg_transport_for(k2, "sssp")
    with pytest.raises(RuntimeError):
        module.ecg_transport_for(k2, "bc")

    streamshield = module.parse_policy_spec("ECG:K2_STREAMSHIELD")
    assert streamshield.label == "ECG_K2_STREAMSHIELD"
    assert module.ecg_transport_for(
        streamshield, "pr") == module.EcgTransport(2, True, False)
    with pytest.raises(RuntimeError):
        module.ecg_transport_for(streamshield, "bfs")
    k1 = module.parse_policy_spec("ECG:K1")
    k1_ss = module.parse_policy_spec("ECG:K1_STREAMSHIELD")
    assert module.ecg_transport_for(
        k1, "pr") == module.EcgTransport(0, False, False)
    assert module.ecg_transport_for(
        k1_ss, "pr") == module.EcgTransport(0, True, False)
    monkeypatch.setenv("ECG_VARIANT", "grasp_only")
    assert module.effective_ecg_variant(
        argparse.Namespace(benchmark="pr"), 2, k2) == "epoch_first"
    monkeypatch.setenv("ECG_K2_DELIVERY_TRACE", "32")
    paper_env = {}
    module.apply_ecg_transport_env(
        paper_env, module.ecg_transport_for(k2, "pr"))
    assert "ECG_K2_DELIVERY_TRACE" not in paper_env
    assert paper_env["ECG_EDGE_MASKS"] == "1"

    baseline_env = {
        "ECG_EDGE_MASK_SCHED": "2",
        "ECG_EDGE_MASKS": "1",
        "ECG_STREAM_BYPASS": "1",
    }
    module.apply_ecg_transport_env(
        baseline_env, module.EcgTransport())
    assert "ECG_EDGE_MASK_SCHED" not in baseline_env
    assert "ECG_EDGE_MASKS" not in baseline_env
    assert "ECG_STREAM_BYPASS" not in baseline_env
    contaminated = {
        "ECG_PREFETCH_MODE": "6",
        "ECG_MODE": "ECG_GRASP_POPT",
        "GEM5_FORCE_ECG_LOAD2": "1",
        "SNIPER_ECG_FUSED_K2": "1",
        "GRASP_HOT_FRACTION": "0.40",
        "POPT_DUAL_REREF": "1",
        "ECG_DEBUG": "1",
    }
    monkeypatch.setenv("ECG_DEBUG", "1")
    module.scrub_cell_mechanism_env(contaminated)
    assert contaminated == {"ECG_DEBUG": "1"}
    monkeypatch.setenv(
        "GRAPHBREW_EXPLICIT_CELL_ENV",
        json.dumps({
            "ECG_EDGE_MASK_AMPLIFY": "3",
            "ECG_EDGE_MASK_SCHED": "4",
            "GRASP_HOT_FRACTION": "0.25",
        }),
    )
    explicit_env = {}
    module.apply_explicit_cell_mechanism_env(explicit_env, k2)
    assert explicit_env == {
        "ECG_EDGE_MASK_AMPLIFY": "3",
        "ECG_EDGE_MASK_SCHED": "4",
        "GRASP_HOT_FRACTION": "0.25",
    }
    module.apply_ecg_transport_env(
        explicit_env, module.ecg_transport_for(k2, "pr"))
    assert explicit_env["ECG_EDGE_MASK_AMPLIFY"] == "3"
    assert explicit_env["ECG_EDGE_MASK_SCHED"] == "2"
    assert explicit_env["GRASP_HOT_FRACTION"] == "0.25"
    baseline = module.parse_policy_spec("LRU")
    baseline_env = {}
    module.apply_explicit_cell_mechanism_env(baseline_env, baseline)
    assert baseline_env == {}
    grasp_env = {}
    module.apply_explicit_cell_mechanism_env(
        grasp_env, module.parse_policy_spec("GRASP"))
    assert grasp_env == {"GRASP_HOT_FRACTION": "0.25"}


def test_streamshield_manifest_is_complete():
    manifest = json.loads(
        (ROOT / "scripts/experiments/ecg/final_paper_manifest.json").read_text())
    assert "streamshield_sniper_realgraph" in manifest["profiles"]
    assert "ecg_cache_sim_factorial" in manifest["profiles"]
    assert "gem5_streamshield_mechanism" in manifest["profiles"]
    assert "sniper_streamshield_mechanism" in manifest["profiles"]
    stage = next(
        stage for stage in manifest["stages"]
        if stage["name"] == "40_sniper_streamshield_realgraph")
    assert stage["policies"] == [
        "LRU", "SRRIP", "GRASP", "POPT",
        "ECG:K2", "ECG:K2_STREAMSHIELD",
    ]
    assert stage["prefetcher"] == "STRIDE"
    assert stage["popt_reserve_model"] == "size_correct"
    graph = manifest["graph_sets"]["web_google_streamshield"][0]
    assert graph["structure_prefetch_degree"] == 8
    assert stage["sniper_roi_icount"] == 100000000
    assert stage["sniper_workload"] == "sg_kernel"
    assert stage["sniper_frontend"] == "sift"
    assert stage["require_sniper_aslr_disable"] is True
    assert "env" not in stage
    factorial = next(
        stage for stage in manifest["stages"]
        if stage["name"] == "20_cache_sim_streamshield_factorial")
    assert factorial["policies"] == [
        "LRU", "SRRIP", "GRASP", "POPT",
        "ECG:K1", "ECG:K1_STREAMSHIELD",
        "ECG:K2", "ECG:K2_STREAMSHIELD",
    ]
    gem5_mechanism = next(
        stage for stage in manifest["stages"]
        if stage["name"] == "30_gem5_streamshield_mechanism")
    assert gem5_mechanism["env"]["GEM5_KERNEL_SUFFIX"] == "_riscv_m5ops"
    sniper_mechanism = next(
        stage for stage in manifest["stages"]
        if stage["name"] == "31_sniper_streamshield_mechanism")
    assert sniper_mechanism["sniper_workload"] == "sg_kernel"


def test_sniper_sg_kernel_supports_synthetic_profiles():
    source = (
        ROOT / "bench/src_sniper/sg_kernel.cc"
    ).read_text()
    assert "options.scale = std::atoi" in source
    assert '"-g", std::to_string(opt.scale)' in source
    assert "requires -f graph.sg or -g scale" in source

    module = load_module(
        "roi_matrix_synthetic_sg",
        ROOT / "scripts/experiments/ecg/roi_matrix.py",
    )
    binary, options = module.sniper_binary_and_options(argparse.Namespace(
        sniper_workload="sg_kernel",
        benchmark="pr",
        options="-g 16 -k 16 -o 5 -n 1 -i 1",
    ))
    assert binary.name == "sg_kernel"
    assert options[:2] == ["--benchmark", "pr"]
    assert "-g" in options


def test_k2_cache_sim_paths_do_not_build_popt_matrix():
    helper = (
        ROOT / "bench/include/cache_sim/graph_sim.h"
    ).read_text()
    assert "GraphSimMatrixFreeK2" in helper
    for relative in (
        "bench/src_sim/pr.cc",
        "bench/src_sim/bfs.cc",
    ):
        source = (ROOT / relative).read_text()
        assert "GraphSimMatrixFreeK2" in source, relative
    bfs = (ROOT / "bench/src_sim/bfs.cc").read_text()
    assert bfs.count("SIM_CACHE_READ_EDGE_K2") == 2


def test_streamshield_profile_and_slurm_shards(tmp_path):
    run_dir = tmp_path / "dryrun"
    listed = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/flows/paper_run.py",
            "--profile", "streamshield_sniper_realgraph",
            "--run-dir", str(run_dir),
            "--list", "--dry-run", "--no-build",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0, listed.stdout + listed.stderr
    for policy in (
        "LRU", "SRRIP", "GRASP", "POPT",
        "ECG:K2", "ECG:K2_STREAMSHIELD",
    ):
        assert policy in listed.stdout
    assert "--sniper-roi-icount 100000000" in listed.stdout
    assert "--structure-prefetch-degree 8" in listed.stdout
    assert "--popt-reserve-model size_correct" in listed.stdout
    assert "--require-sniper-aslr-disable" in listed.stdout

    shards = tmp_path / "shards.tsv"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/slurm/make_slurm_shards.py",
            "--profile", "streamshield_sniper_realgraph",
            "--run-tag", "ecg_successor_test",
            "--out", str(shards),
            "--allow-missing-graphs",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    rows = [line.split("\t") for line in shards.read_text().splitlines()]
    assert len(rows) == 6
    assert [row[4] for row in rows] == [
        "LRU", "SRRIP", "GRASP", "POPT",
        "ECG:K2", "ECG:K2_STREAMSHIELD",
    ]
    sbatch = (
        ROOT / "scripts/experiments/ecg/slurm/slurm_final_shard.sbatch"
    ).read_text()
    assert "${profile}_${safe_stage}_${graph}_${benchmark}_${safe_policy}" in sbatch


def test_paper_pipeline_uses_canonical_runner():
    pipeline = (
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py"
    ).read_text()
    assert 'ECG_DIR / "flows" / "paper_run.py"' in pipeline
    assert (
        ROOT / "scripts/experiments/ecg/flows/paper_run.py"
    ).is_file()
    runner = (
        ROOT / "scripts/experiments/ecg/flows/paper_run.py"
    ).read_text()
    assert "GRAPHBREW_EXPLICIT_CELL_ENV" in runner
    for removed_wrapper in (
        "final_paper_run.py",
        "paper_pipeline.py",
        "make_slurm_shards.py",
    ):
        assert not (
            ROOT / "scripts/experiments/ecg" / removed_wrapper
        ).exists()


def test_partial_policy_matrix_is_not_resumable(tmp_path):
    module = load_module(
        "paper_run_partial_matrix",
        ROOT / "scripts/experiments/ecg/flows/paper_run.py",
    )
    out_dir = tmp_path / "matrix"
    out_dir.mkdir()
    csv_path = out_dir / "roi_matrix.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["status", "policy_label"])
        writer.writeheader()
        writer.writerow({"status": "ok", "policy_label": "LRU"})
    job = module.Job(
        job_id="matrix",
        stage="stage",
        kind="roi_matrix",
        command=[],
        out_dir=out_dir,
        log_path=tmp_path / "matrix.log",
        metadata={"policies": ["LRU", "SRRIP"]},
    )
    status, detail = module.job_csv_status(job)
    assert status == "partial"
    assert "SRRIP" in detail


def test_complete_matrix_requires_matching_marker(tmp_path):
    module = load_module(
        "paper_run_complete_matrix",
        ROOT / "scripts/experiments/ecg/flows/paper_run.py",
    )
    out_dir = tmp_path / "matrix"
    out_dir.mkdir()
    with (out_dir / "roi_matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["status", "policy_label"])
        writer.writeheader()
        writer.writerow({"status": "ok", "policy_label": "LRU"})
    job = module.Job(
        job_id="matrix",
        stage="stage",
        kind="roi_matrix",
        command=[],
        out_dir=out_dir,
        log_path=tmp_path / "matrix.log",
        metadata={
            "policies": ["LRU"],
            "l3_sizes": ["2MB"],
            "threads": ["1"],
            "structure_prefetch_degree": 8,
            "config_hash": "abc",
        },
    )
    assert module.job_csv_status(job)[0] == "partial"
    (out_dir / "roi_matrix.complete.json").write_text(json.dumps({
        "complete": True,
        "all_rows_ok": True,
        "policy_labels": ["LRU"],
        "l3_sizes": ["2MB"],
        "threads": ["1"],
        "structure_prefetch_degree": 8,
        "config_hash": "stale",
        "outputs": {
            "roi_matrix.csv": output_descriptor(
                out_dir / "roi_matrix.csv"),
        },
    }))
    assert module.job_csv_status(job)[0] == "partial"
    marker = json.loads(
        (out_dir / "roi_matrix.complete.json").read_text())
    marker["config_hash"] = "abc"
    (out_dir / "roi_matrix.complete.json").write_text(json.dumps(marker))
    assert module.job_csv_status(job)[0] == "ok"
    with (out_dir / "roi_matrix.csv").open("a") as handle:
        handle.write("\n")
    assert module.job_csv_status(job)[0] == "partial"


def test_policy_labels_share_one_parser():
    module = load_module(
        "policy_specs_ssot",
        ROOT / "scripts/experiments/ecg/policy_specs.py",
    )
    assert module.policy_output_label("POPT") == "POPT"
    assert module.policy_output_label("POPT_CHARGED") == "POPT"
    assert module.policy_output_label("POPT:UNCHARGED") == "POPT_UNCHARGED"
    assert module.policy_output_label("ECG:K2") == "ECG_K2"


def test_sharded_policies_share_comparison_scope():
    module = load_module(
        "paper_pipeline_shard_scope",
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py",
    )
    common = {
        "status": "ok",
        "final_shard_group": "run_tag",
        "final_matrix_id": "web_pr",
        "final_matrix_config_hash": "same-config",
        "simulator": "sniper",
        "benchmark": "pr",
        "prefetcher": "STRIDE",
        "l3_size": "2MB",
        "threads": "1",
        "section": "1",
        "timing_valid_for_speedup": "1",
        "final_expected_policy_labels": json.dumps(["LRU", "ECG_K2"]),
    }
    rows = [
        {**common, "pipeline_run_name": "lru", "final_job_id": "lru",
         "policy_label": "LRU", "sim_ticks": "100", "l3_misses": "50"},
        {**common, "pipeline_run_name": "k2", "final_job_id": "k2",
         "policy_label": "ECG_K2", "sim_ticks": "80", "l3_misses": "40"},
        {**common, "pipeline_run_name": "partial", "final_job_id": "partial",
         "final_output_status": "partial", "policy_label": "SRRIP",
         "sim_ticks": "90", "l3_misses": "45"},
    ]
    relative = module.roi_relative_metrics(rows)
    assert len(relative) == 2
    k2 = next(row for row in relative if row["policy_label"] == "ECG_K2")
    assert k2["speedup_vs_lru"] == 1.25


def test_missing_policy_shards_produce_no_relative_rows():
    module = load_module(
        "paper_pipeline_missing_shards",
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py",
    )
    expected = json.dumps([
        "LRU", "SRRIP", "GRASP", "POPT",
        "ECG_K2", "ECG_K2_STREAMSHIELD",
    ])
    common = {
        "status": "ok",
        "final_output_status": "ok",
        "final_shard_group": "group",
        "final_matrix_id": "matrix",
        "final_matrix_config_hash": "same-config",
        "final_expected_policy_labels": expected,
        "simulator": "sniper",
        "benchmark": "pr",
        "prefetcher": "STRIDE",
        "l3_size": "2MB",
        "threads": "1",
        "section": "1",
    }
    rows = [
        {**common, "policy_label": "LRU", "sim_ticks": "100"},
        {**common, "policy_label": "ECG_K2", "sim_ticks": "80"},
    ]
    assert module.roi_relative_metrics(rows) == []


def test_mismatched_matrix_hashes_do_not_compare():
    module = load_module(
        "paper_pipeline_mismatched_hashes",
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py",
    )
    expected = json.dumps(["LRU", "ECG_K2"])
    common = {
        "status": "ok",
        "final_output_status": "ok",
        "final_shard_group": "group",
        "final_matrix_id": "matrix",
        "final_expected_policy_labels": expected,
        "simulator": "sniper",
        "benchmark": "pr",
        "prefetcher": "STRIDE",
        "l3_size": "2MB",
        "threads": "1",
        "section": "1",
    }
    rows = [
        {**common, "final_matrix_config_hash": "a",
         "policy_label": "LRU", "sim_ticks": "100"},
        {**common, "final_matrix_config_hash": "b",
         "policy_label": "ECG_K2", "sim_ticks": "80"},
    ]
    assert module.roi_relative_metrics(rows) == []


def test_raw_partial_matrix_is_not_collected(tmp_path):
    module = load_module(
        "paper_pipeline_partial_collection",
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py",
    )
    matrix = tmp_path / "roi_matrix.csv"
    with matrix.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["status", "policy_label"])
        writer.writeheader()
        writer.writerow({"status": "ok", "policy_label": "LRU"})
    roi, proof = module.collect_csvs([], [matrix])
    assert roi == []
    assert proof == []


def test_failed_proof_matrix_is_not_collected(tmp_path):
    module = load_module(
        "paper_pipeline_failed_proof",
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py",
    )
    matrix = tmp_path / "proof_matrix.csv"
    with matrix.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["status", "policy_label"])
        writer.writeheader()
        writer.writerow({"status": "ok", "policy_label": "LRU"})
    (tmp_path / "proof_matrix.complete.json").write_text(json.dumps({
        "complete": True,
        "all_rows_ok": False,
    }))
    roi, proof = module.collect_csvs([], [matrix])
    assert roi == []
    assert proof == []


def test_stale_combined_csv_requires_run_marker(tmp_path):
    module = load_module(
        "paper_pipeline_stale_combined",
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py",
    )
    with (tmp_path / "combined_roi_matrix.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["status", "policy_label"])
        writer.writeheader()
        writer.writerow({"status": "ok", "policy_label": "LRU"})
    roi, proof = module.collect_csvs([tmp_path], [])
    assert roi == []
    assert proof == []


def test_fallback_matrix_must_match_resolved_job_hash(tmp_path):
    module = load_module(
        "paper_pipeline_stale_fallback",
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py",
    )
    matrix_dir = tmp_path / "matrices" / "stage" / "graph" / "pr"
    matrix_dir.mkdir(parents=True)
    with (matrix_dir / "roi_matrix.csv").open(
            "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["status", "policy_label"])
        writer.writeheader()
        writer.writerow({"status": "ok", "policy_label": "LRU"})
    (matrix_dir / "roi_matrix.complete.json").write_text(json.dumps({
        "complete": True,
        "all_rows_ok": True,
        "config_hash": "old",
        "matrix_config_hash": "matrix",
        "matrix_id": "matrix",
        "shard_group": "group",
        "expected_policy_labels": ["LRU"],
        "outputs": {
            "roi_matrix.csv": output_descriptor(
                matrix_dir / "roi_matrix.csv"),
        },
    }))
    (tmp_path / "resolved_manifest.json").write_text(json.dumps({
        "run_config_hash": "new-run",
        "jobs": [{
            "out_dir": str(matrix_dir),
            "metadata": {"config_hash": "new"},
        }],
    }))
    roi, proof = module.collect_csvs([tmp_path], [])
    assert roi == []
    assert proof == []


def test_charged_metrics_use_uniform_overhead_field():
    module = load_module(
        "paper_pipeline_charged_metrics",
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py",
    )
    assert module.effective_l3_misses({
        "l3_misses": "100",
        "l3_misses_with_overhead": "150",
    }) == 150
    common = {
        "status": "ok",
        "final_shard_group": "group",
        "final_matrix_id": "matrix",
        "final_matrix_config_hash": "same-config",
        "simulator": "sniper",
        "benchmark": "pr",
        "prefetcher": "STRIDE",
        "l3_size": "2MB",
        "threads": "1",
        "section": "1",
        "final_expected_policy_labels": json.dumps(
            ["POPT", "POPT_UNCHARGED"]),
    }
    rows = [
        {**common, "policy_label": "POPT",
         "l3_misses": "100", "l3_misses_with_overhead": "150"},
        {**common, "policy_label": "POPT_UNCHARGED",
         "l3_misses": "100", "l3_misses_with_overhead": "100"},
    ]
    overhead = module.charged_overhead(rows)
    assert len(overhead) == 1
    assert overhead[0]["l3_miss_delta"] == 50


def test_thread_scaling_uses_series_and_per_core_llc():
    module = load_module(
        "paper_pipeline_scaling_scope",
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py",
    )
    common = {
        "status": "ok",
        "final_shard_group": "group",
        "final_scaling_series_id": "series",
        "simulator": "sniper",
        "benchmark": "pr",
        "prefetcher": "none",
        "per_core_l3_size": "2MB",
        "policy_label": "LRU",
        "section": "1",
        "final_expected_policy_labels": json.dumps(["LRU"]),
    }
    rows = [
        {**common, "threads": "1", "sim_ticks": "100"},
        {**common, "threads": "2", "sim_ticks": "60"},
    ]
    scaling = module.thread_scaling_metrics(rows)
    assert len(scaling) == 2
    assert scaling[1]["thread_speedup_vs_1t"] == 100 / 60


def test_policy_filter_uses_shared_labels():
    module = load_module(
        "paper_run_policy_filter",
        ROOT / "scripts/experiments/ecg/flows/paper_run.py",
    )
    assert module.filter_policy_specs(
        ["LRU", "POPT"], ["POPT_CHARGED"]) == ["POPT"]


def test_sniper_policy_marker_is_content_validated(tmp_path):
    module = load_module(
        "roi_matrix_sniper_marker",
        ROOT / "scripts/experiments/ecg/roi_matrix.py",
    )
    relative = Path("common/core/memory_subsystem/cache/cache_set_ecg.cc")
    overlay = tmp_path / "bench/include/sniper_sim/overlays" / relative
    installed_root = tmp_path / "snipersim"
    installed = installed_root / relative
    overlay.parent.mkdir(parents=True)
    installed.parent.mkdir(parents=True)
    overlay.write_text("same")
    installed.write_text("same")
    binary = installed_root / "lib/sniper"
    binary.parent.mkdir(parents=True)
    binary.write_text("binary")
    subprocess.run(["git", "init", "-q"], cwd=installed_root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=installed_root, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=installed_root, check=True)
    subprocess.run(["git", "add", "."], cwd=installed_root, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"],
        cwd=installed_root, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=installed_root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    module.PINNED_SNIPER_HEAD = head
    marker = tmp_path / ".sniper_overlays.json"
    marker.write_text(json.dumps({
        "sniper_head": head,
        "policies": ["grasp", "popt", "ecg"],
        "copied_files": [str(relative)],
        "patched_files": [],
        "file_hashes": {
            str(relative): hashlib.sha256(
                installed.read_bytes()).hexdigest(),
        },
        "binary": {
            "path": "lib/sniper",
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "size": binary.stat().st_size,
        },
    }))
    module.PROJECT_ROOT = tmp_path
    module.SNIPER_OVERLAY_STATUS = marker
    args = argparse.Namespace(
        sniper_enable_graph_policies=False,
        sniper_root=str(installed_root),
    )
    assert module.sniper_graph_policies_enabled(args)
    extra = overlay.parent / "extra.cc"
    extra.write_text("new overlay")
    assert not module.sniper_graph_policies_enabled(args)
    extra.unlink()
    installed.write_text("stale")
    assert not module.sniper_graph_policies_enabled(args)


def test_failed_rerun_invalidates_completion_marker(tmp_path):
    module = load_module(
        "paper_run_failed_rerun",
        ROOT / "scripts/experiments/ecg/flows/paper_run.py",
    )
    out_dir = tmp_path / "matrix"
    out_dir.mkdir()
    marker = out_dir / "roi_matrix.complete.json"
    marker.write_text("{}")
    job = module.Job(
        job_id="failed",
        stage="stage",
        kind="roi_matrix",
        command=[sys.executable, "-c", "raise SystemExit(1)"],
        out_dir=out_dir,
        log_path=tmp_path / "failed.log",
        metadata={"policies": ["LRU"]},
    )
    args = argparse.Namespace(
        force=True,
        resume=True,
        skip_failed=False,
        dry_run=False,
    )
    assert module.run_job(job, tmp_path, args) == 1
    assert not marker.exists()
    module.write_run_completion(tmp_path, [job], successful=False)
    payload = json.loads((tmp_path / "run.complete.json").read_text())
    assert payload["complete"] is False


def test_sniper_fingerprint_covers_sift_stack():
    module = load_module(
        "paper_run_sniper_fingerprint",
        ROOT / "scripts/experiments/ecg/flows/paper_run.py",
    )
    args = argparse.Namespace(
        manifest=str(
            ROOT / "scripts/experiments/ecg/final_paper_manifest.json"))
    settings = {
        "suite": "sniper",
        "sniper_root": "bench/include/sniper_sim/snipersim",
        "sniper_workload": "sg_kernel",
    }
    fingerprints = module.roi_input_fingerprints(
        args,
        settings,
        ROOT / "results/graphs/web-Google/web-Google.sg",
        "pr",
    )
    for key in (
        "sniper_runner",
        "sniper_record_trace",
        "sniper_binary",
        "sniper_config",
        "sniper_runtime_scripts",
        "sniper_tools",
        "sniper_sde",
        "sniper_sift_recorder",
        "setarch",
        "benchmark_binary",
    ):
        assert key in fingerprints


def test_proof_hash_tracks_material_environment(tmp_path, monkeypatch):
    module = load_module(
        "paper_run_proof_environment",
        ROOT / "scripts/experiments/ecg/flows/paper_run.py",
    )
    args = argparse.Namespace(
        manifest=str(
            ROOT / "scripts/experiments/ecg/final_paper_manifest.json"),
        no_build=True,
        dry_run=False,
    )
    settings = {
        "name": "proof",
        "benchmarks": ["pr"],
        "l1d_size": "1kB",
        "l2_size": "2kB",
        "l3_sizes": ["4kB"],
        "l3_ways": "16",
        "line_size": "64",
        "timeout_cache": 60,
        "no_build": True,
    }
    monkeypatch.setenv("CACHE_FAST", "0")
    first = module.make_proof_job(args, tmp_path, settings)
    monkeypatch.setenv("CACHE_FAST", "1")
    second = module.make_proof_job(args, tmp_path, settings)
    assert first.metadata["config_hash"] != second.metadata["config_hash"]


def test_pipeline_dry_run_succeeds_without_rows(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/flows/paper_pipeline.py",
            "--profiles", "ecg_smoke",
            "--dry-run",
            "--no-build",
            "--run-root", str(tmp_path / "pipeline"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_proof_matrix_uses_supported_pfx_cli():
    source = (
        ROOT / "scripts/experiments/ecg/flows/proof_matrix.py"
    ).read_text()
    assert '"--prefetcher", "ECG_PFX"' in source
    assert '"--ecg-pfx-mode", pfx_mode' in source
    assert '"ECG_PREFETCH_MODE"' not in source


def test_direct_complete_matrix_has_standalone_hash(tmp_path):
    module = load_module(
        "paper_pipeline_standalone_hash",
        ROOT / "scripts/experiments/ecg/flows/paper_pipeline.py",
    )
    matrix = tmp_path / "roi_matrix.csv"
    with matrix.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["status", "policy_label"])
        writer.writeheader()
        writer.writerow({"status": "ok", "policy_label": "LRU"})
    (tmp_path / "roi_matrix.complete.json").write_text(json.dumps({
        "complete": True,
        "all_rows_ok": True,
        "matrix_id": "direct",
        "shard_group": "direct",
        "matrix_config_hash": "standalone-hash",
        "expected_policy_labels": ["LRU"],
        "outputs": {
            "roi_matrix.csv": output_descriptor(matrix),
        },
    }))
    roi, proof = module.collect_csvs([], [matrix])
    assert len(roi) == 1
    assert roi[0]["final_matrix_config_hash"] == "standalone-hash"
    assert proof == []
