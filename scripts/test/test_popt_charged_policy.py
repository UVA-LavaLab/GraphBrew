#!/usr/bin/env python3
"""Regression tests for P-OPT charged-overhead policy expansion."""

from argparse import Namespace
import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROI_MATRIX_PATH = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "roi_matrix.py"
spec = importlib.util.spec_from_file_location("roi_matrix", ROI_MATRIX_PATH)
roi_matrix = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["roi_matrix"] = roi_matrix
spec.loader.exec_module(roi_matrix)


def test_popt_charged_reserves_matrix_columns_as_llc_ways():
    args = Namespace(
        options="-g 12 -k 16 -o 5 -n 1 -i 2",
        line_size="64",
        l3_ways="16",
        popt_property_bytes="4",
        popt_active_columns="2",
        popt_min_data_ways="1",
        popt_num_epochs="256",
    )

    spec = roi_matrix.parse_policy_spec("POPT_CHARGED")
    charge = roi_matrix.popt_charge_metadata(args, spec, "4kB")

    assert spec.policy == "POPT"
    assert spec.label == "POPT_CHARGED"
    assert spec.charge_popt_overhead
    assert charge["popt_estimated_vertices"] == 4096
    assert charge["popt_matrix_column_bytes"] == 256
    assert charge["popt_reserved_ways"] == 2
    assert charge["popt_reserved_bytes"] == 512
    assert charge["popt_effective_l3_ways"] == "14"
    assert charge["popt_effective_l3_size"] == "3584B"
    assert charge["popt_matrix_stream_cache_lines"] == 1024


def test_charged_ecg_policy_maps_to_underlying_ecg_mode():
    spec = roi_matrix.parse_policy_spec("ECG:DBG_PRIMARY_CHARGED")

    assert spec.label == "ECG_DBG_PRIMARY_CHARGED"
    assert spec.policy == "ECG"
    assert spec.ecg_mode == "DBG_PRIMARY"
    assert spec.charge_popt_overhead


def test_sniper_defaults_to_virtual_address_domain():
    args = roi_matrix.parse_args(["--suite", "sniper"])

    assert args.sniper_address_domain == "virtual"
    assert args.sniper_root == "bench/include/sniper_sim/snipersim"
    assert roi_matrix.sniper_root_path(args) == PROJECT_ROOT / "bench" / "include" / "sniper_sim" / "snipersim"
    assert args.sniper_frontend == "live"
    assert args.sniper_omp_wait_policy == "passive"
    assert args.sniper_base_config == "graphbrew/graph_sniper"
    assert args.sniper_memory_limit_gb == 16.0
    assert args.sniper_mimicos_memory_mb == "4096"
    assert args.sniper_mimicos_kernel_mb == "128"
    assert not args.allow_sniper_sg_kernel_workload


def test_sniper_root_accepts_absolute_path():
    args = roi_matrix.parse_args(["--suite", "sniper", "--sniper-root", "/tmp/snipersim-test"])

    assert roi_matrix.sniper_root_path(args) == Path("/tmp/snipersim-test")
    assert roi_matrix.sniper_runner_path(args) == Path("/tmp/snipersim-test/run-sniper")


def test_sniper_frontend_accepts_sift():
    args = roi_matrix.parse_args(["--suite", "sniper", "--sniper-frontend", "sift"])

    assert args.sniper_frontend == "sift"


def test_ecg_pfx_prefetcher_sets_cache_sim_env(tmp_path):
    args = roi_matrix.parse_args([
        "--suite", "cache-sim",
        "--prefetcher", "ECG_PFX",
        "--ecg-pfx-mode", "popt",
        "--ecg-pfx-window", "12",
        "--ecg-pfx-lookahead", "6",
    ])
    spec = roi_matrix.parse_policy_spec("ECG:POPT_PRIMARY")

    env = roi_matrix.cache_sim_env(args, spec, "4kB", "16", tmp_path / "out.json")
    row = roi_matrix.base_row("cache_sim", args, spec, "4kB")

    assert env["ECG_PREFETCH_MODE"] == "2"
    assert env["ECG_PREFETCH_WINDOW"] == "12"
    assert env["ECG_PREFETCH_LOOKAHEAD"] == "6"
    assert row["prefetcher"] == "ECG_PFX"
    assert row["ecg_prefetch_mode"] == "2"
    assert row["ecg_prefetch_window"] == "12"
    assert row["ecg_prefetch_lookahead"] == "6"


def test_ecg_pfx_gem5_returns_unsupported_without_launch(tmp_path):
    args = roi_matrix.parse_args(["--suite", "gem5", "--prefetcher", "ECG_PFX"])
    spec = roi_matrix.parse_policy_spec("LRU")

    rows = roi_matrix.run_gem5(args, tmp_path, spec, "4kB")

    assert rows[0]["status"] == "unsupported"
    assert "experimental" in rows[0]["error"]


def test_ecg_pfx_gem5_can_be_explicitly_enabled(monkeypatch, tmp_path):
    args = roi_matrix.parse_args([
        "--suite", "gem5",
        "--prefetcher", "ECG_PFX",
        "--allow-gem5-ecg-pfx",
        "--dry-run",
    ])
    spec = roi_matrix.parse_policy_spec("LRU")

    monkeypatch.setattr(roi_matrix, "run_command", lambda *call_args, **kwargs: None)

    assert roi_matrix.run_gem5(args, tmp_path, spec, "4kB") == []


def test_ecg_pfx_sniper_requires_overlays_without_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(roi_matrix, "SNIPER_OVERLAY_STATUS", tmp_path / "missing_overlays.json")
    args = roi_matrix.parse_args(["--suite", "sniper", "--prefetcher", "ECG_PFX"])
    spec = roi_matrix.parse_policy_spec("LRU")

    rows = roi_matrix.run_sniper(args, tmp_path, spec, "4kB")

    assert rows[0]["status"] == "unsupported"
    assert "requires overlays" in rows[0]["error"]


def test_unsafe_sniper_memory_limit_wraps_command(monkeypatch):
    monkeypatch.setattr(roi_matrix.shutil, "which", lambda name: "/usr/bin/prlimit" if name == "prlimit" else None)

    limited = roi_matrix.memory_limited_command(["run-sniper", "--", "bench"], 1.5)

    assert limited == ["/usr/bin/prlimit", "--as=1610612736", "--", "run-sniper", "--", "bench"]


def test_sniper_memory_limit_can_be_disabled():
    cmd = ["run-sniper", "--", "bench"]

    assert roi_matrix.memory_limited_command(cmd, 0.0) == cmd


def test_run_command_timeout_returns_error_code(tmp_path):
    log_path = tmp_path / "timeout.log"

    result = roi_matrix.run_command(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        PROJECT_ROOT,
        None,
        1,
        log_path,
        False,
    )

    assert result is not None
    assert result.returncode == 124
    text = log_path.read_text()
    assert "[timeout_s] 1" in text
    assert "[timeout_action] SIGTERM process group" in text
    assert "[exit_code] 124" in text


def test_gem5_sideband_paths_are_per_output_directory(tmp_path):
    paths = roi_matrix.gem5_sideband_paths(tmp_path / "gem5" / "row")

    assert paths["context"] == tmp_path / "gem5" / "row" / "graphbrew_sidebands" / "gem5_graphbrew_ctx.json"
    assert paths["popt_matrix"].parent == paths["context"].parent
    assert paths["out_edges"].name == "gem5_graphbrew_out_edges.bin"
    assert paths["in_edges"].name == "gem5_graphbrew_in_edges.bin"