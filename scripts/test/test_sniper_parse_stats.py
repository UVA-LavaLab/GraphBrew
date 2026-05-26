#!/usr/bin/env python3
"""Regression tests for GraphBrew Sniper stat parsing."""

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARSER_PATH = PROJECT_ROOT / "bench" / "include" / "sniper_sim" / "scripts" / "parse_stats.py"
spec = importlib.util.spec_from_file_location("sniper_parse_stats", PARSER_PATH)
sniper_parse_stats = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["sniper_parse_stats"] = sniper_parse_stats
spec.loader.exec_module(sniper_parse_stats)


def test_extract_graphbrew_metrics_includes_sniper_cpi_stack():
    metrics = sniper_parse_stats.extract_graphbrew_metrics({
        "success": True,
        "core.instructions": 100,
        "performance_model.elapsed_time": 200,
        "rob_timer.cpiBase": 10,
        "rob_timer.cpiBranchPredictor": 5,
        "rob_timer.cpiDataCacheL1": 1,
        "rob_timer.cpiDataCacheL2": 2,
        "rob_timer.cpiDataCachenuca-cache": 3,
        "rob_timer.cpiDataCachedram-local": 4,
        "performance_model.cpiSyncFutex": 6,
        "performance_model.cpiSyncPthreadBarrier": 7,
        "performance_model.cpiUnknown": 8,
    })

    assert metrics["sniper_cpi_base"] == 10
    assert metrics["sniper_cpi_branch"] == 5
    assert metrics["sniper_cpi_data_cache"] == 10.0
    assert metrics["sniper_cpi_data_l1"] == 1.0
    assert metrics["sniper_cpi_data_l2"] == 2.0
    assert metrics["sniper_cpi_data_llc"] == 3.0
    assert metrics["sniper_cpi_data_dram"] == 4.0
    assert metrics["sniper_cpi_sync"] == 13.0
    assert metrics["sniper_cpi_unknown"] == 8.0
