import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_rows(module):
    rows = []
    for simulator in module.SIMULATORS:
        for benchmark in module.BENCHMARKS:
            for policy in module.POLICIES:
                row = {
                    "simulator": simulator,
                    "benchmark": benchmark,
                    "policy_label": policy,
                    "final_graph": "kron_s12_k4",
                    "l3_size": "32kB",
                    "l3_ways": "8",
                    "status": "ok",
                    "final_output_status": "ok",
                    "l3_exercised": "1",
                    "l3_misses": "1",
                    "l3_miss_rate": "0.5",
                    "timing_valid_for_speedup": "1",
                    "timing_model": "model",
                }
                if simulator == "cache_sim":
                    row.update({
                        "total_memory_traffic_with_overhead": "1",
                        "total_accesses": "2",
                        "l3_hits": "1",
                        "l3_prop_misses": "1",
                    })
                elif simulator == "gem5":
                    row.update({
                        "l3_accesses": "2",
                        "dram_read_bytes": "64",
                        "dram_write_bytes": "0",
                        "sim_ticks": "10",
                        "ipc": "1",
                    })
                else:
                    row.update({
                        "l3_accesses": "2",
                        "instructions": "10",
                        "sim_ticks": "10",
                        "ipc": "1",
                        "sniper_cpi_base": "0",
                        "sniper_cpi_data_cache": "0",
                        "sniper_cpi_data_llc": "0",
                        "sniper_cpi_data_dram": "0",
                    })
                if policy in module.K2_POLICIES:
                    row.update({
                        "ecg_schedule_k": "2",
                        "ecg_epochs_effective": "32768",
                    })
                    if simulator == "gem5":
                        row["gem5_ecg_delivery"] = (
                            "ecg.stream.load2"
                            if policy in module.SS_POLICIES
                            else "ecg.load2")
                        if policy in module.SS_POLICIES:
                            row["gem5_stream_bypass_trace_events"] = "1"
                    if simulator == "sniper":
                        row.update({
                            "sniper_ecg_delivery": "fused-k2-model",
                            "sniper_fused_k2_receipts": "1",
                            "sniper_fused_k2_bad_receipts": "0",
                        })
                        if policy in module.SS_POLICIES:
                            row.update({
                                "sniper_stream_bypass_reads": "1",
                                "sniper_stream_bypass_writes": "1",
                            })
                rows.append(row)
    return rows


def test_smoke_coverage_accepts_complete_matrix():
    module = load_module(
        "smoke_coverage_complete",
        ROOT / "scripts/experiments/ecg/verify/smoke_coverage.py")
    assert module.validate(make_rows(module)) == []


def test_smoke_coverage_rejects_missing_backend_metric():
    module = load_module(
        "smoke_coverage_missing",
        ROOT / "scripts/experiments/ecg/verify/smoke_coverage.py")
    rows = make_rows(module)
    sniper = next(row for row in rows if row["simulator"] == "sniper")
    del sniper["instructions"]
    errors = module.validate(rows)
    assert any("missing instructions" in error for error in errors)


def test_smoke_coverage_rejects_incomplete_matrix():
    module = load_module(
        "smoke_coverage_incomplete",
        ROOT / "scripts/experiments/ecg/verify/smoke_coverage.py")
    rows = make_rows(module)[:-1]
    errors = module.validate(rows)
    assert any("expected 120 rows, found 119" in error for error in errors)


def test_smoke_coverage_rejects_wrong_gem5_delivery():
    module = load_module(
        "smoke_coverage_delivery",
        ROOT / "scripts/experiments/ecg/verify/smoke_coverage.py")
    rows = make_rows(module)
    row = next(
        row for row in rows
        if row["simulator"] == "gem5" and
        row["policy_label"] == "ECG_K2_STREAMSHIELD")
    row["gem5_ecg_delivery"] = "ecg.load2"
    errors = module.validate(rows)
    assert any("expected='ecg.stream.load2'" in error for error in errors)


def test_smoke_coverage_rejects_missing_bad_receipt_count():
    module = load_module(
        "smoke_coverage_receipts",
        ROOT / "scripts/experiments/ecg/verify/smoke_coverage.py")
    rows = make_rows(module)
    row = next(
        row for row in rows
        if row["simulator"] == "sniper" and
        row["policy_label"] == "ECG_K2")
    del row["sniper_fused_k2_bad_receipts"]
    errors = module.validate(rows)
    assert any(
        "missing sniper_fused_k2_bad_receipts" in error for error in errors)
