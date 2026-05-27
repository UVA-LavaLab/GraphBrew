import csv
import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "ecg_validation_gates.py"
spec = importlib.util.spec_from_file_location("ecg_validation_gates", SCRIPT)
ecg_validation_gates = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["ecg_validation_gates"] = ecg_validation_gates
spec.loader.exec_module(ecg_validation_gates)


def row(benchmark: str, ablation: str, memory: int, useful: int = 0, fills: int = 0) -> dict[str, str]:
    return {
        "benchmark": benchmark,
        "ablation": ablation,
        "status": "ok",
        "memory_accesses": str(memory),
        "total_memory_traffic": str(memory),
        "l3_misses": str(memory),
        "prefetch_useful": str(useful),
        "prefetch_fills": str(fills),
    }


def test_gate_report_classifies_pass_and_activation_only():
    rows = [
        row("pr", "LRU_cache_only", 100),
        row("pr", "GRASP_DBG_only", 80),
        row("pr", "POPT_only", 70),
        row("pr", "ECG_DBG_only", 82),
        row("pr", "ECG_POPT_primary", 72),
        row("pr", "ECG_DBG_POPT", 69),
        row("pr", "ECG_EMBEDDED", 75),
        row("pr", "ECG_COMBINED", 95),
        row("pr", "PFX_POPT_only", 110, useful=10, fills=20),
        row("pr", "DBG_PFX", 81, useful=4, fills=8),
        row("pr", "POPT_PFX", 71, useful=3, fills=7),
        row("pr", "DBG_POPT_PFX", 68, useful=5, fills=9),
    ]

    gates = ecg_validation_gates.evaluate(
        rows,
        metric="memory_accesses",
        parity_tolerance=0.05,
        benefit_tolerance=0.0,
        embedded_tolerance=0.10,
    )
    by_gate = {(gate["benchmark"], gate["gate"]): gate for gate in gates}

    assert by_gate[("pr", "grasp_parity")]["status"] == "pass"
    assert by_gate[("pr", "popt_parity")]["status"] == "pass"
    assert by_gate[("pr", "embedded_quality")]["status"] == "pass"
    assert by_gate[("pr", "ecg_hybrid_value")]["status"] == "pass"
    assert by_gate[("pr", "PFX_POPT_only_pfx")]["status"] == "activation_only"
    assert by_gate[("pr", "DBG_POPT_PFX_pfx")]["status"] == "pass"


def test_cli_writes_csv_and_markdown(tmp_path):
    proof_csv = tmp_path / "proof_matrix.csv"
    fieldnames = ["benchmark", "ablation", "status", "memory_accesses", "total_memory_traffic", "l3_misses", "prefetch_useful", "prefetch_fills"]
    rows = [
        row("bfs", "LRU_cache_only", 100),
        row("bfs", "GRASP_DBG_only", 80),
        row("bfs", "POPT_only", 70),
        row("bfs", "ECG_DBG_only", 80),
        row("bfs", "ECG_POPT_primary", 70),
        row("bfs", "ECG_DBG_POPT", 69),
        row("bfs", "ECG_EMBEDDED", 72),
        row("bfs", "ECG_COMBINED", 90),
        row("bfs", "PFX_POPT_only", 95, useful=1, fills=1),
        row("bfs", "DBG_PFX", 80, useful=1, fills=1),
        row("bfs", "POPT_PFX", 70, useful=1, fills=1),
        row("bfs", "DBG_POPT_PFX", 68, useful=1, fills=1),
    ]
    with proof_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_csv = tmp_path / "gates.csv"
    out_md = tmp_path / "gates.md"
    rc = ecg_validation_gates.main([str(proof_csv), "--out-csv", str(out_csv), "--out-md", str(out_md)])

    assert rc == 0
    assert out_csv.exists()
    assert "ECG Validation Gate Report" in out_md.read_text()