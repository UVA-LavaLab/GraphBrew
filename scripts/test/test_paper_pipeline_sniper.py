#!/usr/bin/env python3
"""Regression tests for Sniper-aware ECG paper aggregation."""

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "paper_pipeline.py"
spec = importlib.util.spec_from_file_location("paper_pipeline", PIPELINE_PATH)
paper_pipeline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["paper_pipeline"] = paper_pipeline
spec.loader.exec_module(paper_pipeline)


def roi_row(policy: str, threads: str, ticks: str, simulator: str = "sniper") -> dict[str, str]:
    return {
        "status": "ok",
        "pipeline_run_name": "run",
        "final_job_id": "job",
        "simulator": simulator,
        "benchmark": "pr",
        "prefetcher": "none",
        "l3_size": "4kB",
        "threads": threads,
        "section": "1",
        "policy_label": policy,
        "sim_ticks": ticks,
        "l3_misses": "100",
    }


def test_roi_relative_metrics_keep_sniper_threads_separate():
    rows = [
        roi_row("LRU", "1", "100"),
        roi_row("GRASP", "1", "80"),
        roi_row("LRU", "2", "50"),
        roi_row("GRASP", "2", "25"),
    ]

    relative = paper_pipeline.roi_relative_metrics(rows)
    speedups = {
        (row["policy_label"], row["threads"]): row["speedup_vs_lru"]
        for row in relative
    }

    assert speedups[("GRASP", "1")] == 1.25
    assert speedups[("GRASP", "2")] == 2.0


def test_backend_direction_agreement_compares_normalized_directions():
    relative = [
        {
            "benchmark": "pr",
            "prefetcher": "none",
            "l3_size": "4kB",
            "threads": "1",
            "policy_label": "GRASP",
            "section": "1",
            "simulator": "gem5",
            "speedup_vs_lru": 1.10,
            "l3_miss_reduction_vs_lru_pct": 2.0,
        },
        {
            "benchmark": "pr",
            "prefetcher": "none",
            "l3_size": "4kB",
            "threads": "1",
            "policy_label": "GRASP",
            "section": "1",
            "simulator": "sniper",
            "speedup_vs_lru": 1.05,
            "l3_miss_reduction_vs_lru_pct": 3.0,
        },
        {
            "benchmark": "pr",
            "prefetcher": "none",
            "l3_size": "4kB",
            "threads": "1",
            "policy_label": "GRASP",
            "section": "1",
            "simulator": "cache_sim",
            "speedup_vs_lru": 0.95,
            "l3_miss_reduction_vs_lru_pct": -1.0,
        },
    ]

    agreement = paper_pipeline.backend_direction_agreement(relative)
    by_pair = {
        (row["metric"], row["simulator_a"], row["simulator_b"]): row["direction_agrees"]
        for row in agreement
    }

    assert by_pair[("speedup_vs_lru", "gem5", "sniper")] == "yes"
    assert by_pair[("l3_miss_reduction_vs_lru_pct", "gem5", "sniper")] == "yes"
    assert by_pair[("speedup_vs_lru", "cache_sim", "sniper")] == "no"
    assert by_pair[("l3_miss_reduction_vs_lru_pct", "cache_sim", "sniper")] == "no"


def test_prefetch_quality_preserves_sniper_source():
    rows = [{
        "status": "ok",
        "simulator": "sniper",
        "benchmark": "pr",
        "prefetcher": "DROPLET",
        "l3_size": "4kB",
        "threads": "1",
        "policy_label": "LRU",
        "pf_issued": "2",
        "pf_useful": "1",
    }]

    summary = paper_pipeline.prefetch_quality_summary(rows, [])

    assert summary[0]["source"] == "sniper"
    assert summary[0]["prefetch_accuracy_pct"] == 50.0


def test_sniper_cpi_stack_summary_computes_component_percentages():
    rows = [{
        "status": "ok",
        "simulator": "sniper",
        "benchmark": "pr",
        "prefetcher": "none",
        "l3_size": "4kB",
        "threads": "1",
        "policy_label": "LRU",
        "sniper_cpi_base": "10",
        "sniper_cpi_branch": "5",
        "sniper_cpi_data_cache": "20",
        "sniper_cpi_data_llc": "4",
        "sniper_cpi_data_dram": "8",
        "sniper_cpi_sync": "0",
        "sniper_cpi_unknown": "5",
    }]

    summary = paper_pipeline.sniper_cpi_stack_summary(rows)

    assert summary[0]["avg_sniper_cpi_stack_total"] == 40.0
    assert summary[0]["avg_sniper_cpi_data_cache_pct"] == 20.0 / 40.0 * 100.0


def test_prefetch_quality_preserves_ecg_pfx_label():
    rows = [{
        "status": "ok",
        "simulator": "gem5",
        "benchmark": "bfs",
        "prefetcher": "ECG_PFX",
        "l3_size": "4kB",
        "threads": "1",
        "policy_label": "LRU",
        "pf_issued": "2",
        "pf_useful": "1",
    }]

    summary = paper_pipeline.prefetch_quality_summary(rows, [])

    assert summary[0]["source"] == "gem5"
    assert summary[0]["prefetcher"] == "ECG_PFX"
    assert summary[0]["prefetch_accuracy_pct"] == 50.0


def test_ecg_pfx_hint_timing_is_not_used_for_speedup():
    rows = [
        {
            **roi_row("LRU", "1", "100"),
            "prefetcher": "ECG_PFX",
            "timing_model": "prototype_explicit_hint_delivery",
            "timing_valid_for_speedup": "0",
            "timing_caveat": "explicit hint delivery",
            "l3_misses": "100",
        },
        {
            **roi_row("ECG_POPT_PRIMARY", "1", "80"),
            "prefetcher": "ECG_PFX",
            "timing_model": "prototype_explicit_hint_delivery",
            "timing_valid_for_speedup": "0",
            "timing_caveat": "explicit hint delivery",
            "l3_misses": "50",
        },
    ]

    relative = paper_pipeline.roi_relative_metrics(rows)
    pfx_row = next(row for row in relative if row["policy_label"] == "ECG_POPT_PRIMARY")

    assert "speedup_vs_lru" not in pfx_row
    assert "normalized_ticks_vs_lru" not in pfx_row
    assert pfx_row["l3_miss_reduction_vs_lru_pct"] == 50.0
    assert pfx_row["timing_valid_for_speedup"] == "0"


def test_legacy_ecg_pfx_rows_are_inferred_timing_invalid():
    rows = [
        {
            **roi_row("LRU", "1", "100"),
            "prefetcher": "ECG_PFX",
            "l3_misses": "100",
        },
        {
            **roi_row("ECG_POPT_PRIMARY", "1", "80"),
            "prefetcher": "ECG_PFX",
            "l3_misses": "50",
        },
    ]

    relative = paper_pipeline.roi_relative_metrics(rows)
    pfx_row = next(row for row in relative if row["policy_label"] == "ECG_POPT_PRIMARY")

    assert "speedup_vs_lru" not in pfx_row
    assert pfx_row["timing_valid_for_speedup"] == "0"
    assert pfx_row["timing_model"] == "prototype_explicit_hint_delivery"
