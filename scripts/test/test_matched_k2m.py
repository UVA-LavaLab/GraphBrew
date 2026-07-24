import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.ecg.verify.matched_k2m import validate  # noqa: E402


def write_rows(path: Path, ratio: float = 1.001) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "policy_label", "status", "instructions",
        "sniper_transport_matched", "sniper_k2_exact_bind",
        "sniper_k2_epoch_context_bound",
        "ecg_isa_variant", "sniper_workload",
        "sniper_roi_icount", "timing_valid_for_speedup",
        "sniper_workload_sha256", "benchmark", "options", "prefetcher",
        "l1d_size", "l2_size", "l3_size", "l3_ways", "threads",
        "sniper_cores", "sniper_cache_warming",
        "sniper_transport_record_bytes", "sniper_semantic_result", "log_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "policy_label": "LRU",
            "status": "ok",
            "instructions": "100000",
            "sniper_transport_matched": "1",
            "sniper_k2_exact_bind": "1",
            "sniper_k2_epoch_context_bound": "1",
            "ecg_isa_variant": "baseline",
            "sniper_workload": "sg_kernel",
            "sniper_roi_icount": "0",
            "timing_valid_for_speedup": "0",
            "sniper_workload_sha256": "abc",
            "benchmark": path.parent.name,
            "options": "-f graph.sg",
            "prefetcher": "none",
            "l1d_size": "2kB",
            "l2_size": "4kB",
            "l3_size": "16kB",
            "l3_ways": "8",
            "threads": "1",
            "sniper_cores": "1",
            "sniper_cache_warming": "1",
            "sniper_transport_record_bytes": "8",
            "sniper_semantic_result": "same",
            "log_path": str(path.parent / "lru.log"),
        })
        writer.writerow({
            "policy_label": "ECG_K2",
            "status": "ok",
            "instructions": str(round(100000 * ratio)),
            "sniper_transport_matched": "1",
            "sniper_k2_exact_bind": "1",
            "sniper_k2_epoch_context_bound": "1",
            "ecg_isa_variant": "mask",
            "sniper_workload": "sg_kernel",
            "sniper_roi_icount": "0",
            "timing_valid_for_speedup": "0",
            "sniper_workload_sha256": "abc",
            "benchmark": path.parent.name,
            "options": "-f graph.sg",
            "prefetcher": "none",
            "l1d_size": "2kB",
            "l2_size": "4kB",
            "l3_size": "16kB",
            "l3_ways": "8",
            "threads": "1",
            "sniper_cores": "1",
            "sniper_cache_warming": "1",
            "sniper_transport_record_bytes": "8",
            "sniper_semantic_result": "same",
            "log_path": str(path.parent / "k2.log"),
        })
    (path.parent / "lru.log").write_text(
        "[K2_TRANSPORT_MATCHED]\n[K2_EXACT_BIND]\n")
    (path.parent / "k2.log").write_text(
        "[K2_TRANSPORT_MATCHED]\n[K2_EXACT_BIND]\n")
    json_path = path.parent / "roi_matrix.json"
    json_rows = list(csv.DictReader(path.open()))
    json_path.write_text(json.dumps(json_rows))
    marker = {
        "complete": True,
        "all_rows_ok": True,
        "outputs": {
            "roi_matrix.csv": {
                "rows": 2,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "roi_matrix.json": {
                "rows": 2,
                "sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
            },
        },
    }
    (path.parent / "roi_matrix.complete.json").write_text(json.dumps(marker))


def test_matched_rows_pass(tmp_path: Path):
    for kernel in ("pr", "bfs", "sssp", "bc", "cc"):
        write_rows(tmp_path / kernel / "roi_matrix.csv")
    assert validate(tmp_path) == []


def test_instruction_drift_fails(tmp_path: Path):
    write_rows(tmp_path / "pr" / "roi_matrix.csv", ratio=1.01)
    errors = validate(tmp_path, ("pr",))
    assert len(errors) == 1
    assert "instruction ratio" in errors[0]
