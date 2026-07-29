import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/experiments/ecg/flows/popt_artifact_repro.py"
SPEC = importlib.util.spec_from_file_location("popt_artifact_repro", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["popt_artifact_repro"] = MOD
SPEC.loader.exec_module(MOD)


def test_parser_sums_artifact_llc_totals():
    text = """
[LLC-STAT] Total Misses = 100
[LLC-STAT] Total Misses = 7
"""
    assert MOD.parse_total_llc_misses(text) == 107


def test_public_artifact_gate_does_not_claim_grasp_or_speedup():
    src = PATH.read_text()
    assert '"popt_vs_grasp_figure12_exact": False' in src
    assert '"execution_time": False' in src
    assert "llc_demand_misses" in src


def test_commands_pin_one_pagerank_sweep(tmp_path):
    root = tmp_path / "artifact"
    command = MOD.build_command(
        root, root / "pin", root / "tools", "uk-2002", "popt-8b")
    assert command[-6:] == [
        "-f", str(root / "input-graphs/uk-2002.sg"),
        "-n", "1", "-i", "1",
    ]
    assert str(root / "applications/popt/pr") in command
    assert str(root / "tools/popt-8b/cache_pinsim.so") in command


def test_resume_rows_round_trip_integer_metrics(tmp_path):
    rows = [{
        "graph": "g", "policy": "lru", "exit_code": 0,
        "llc_demand_misses": 123, "status": "ok",
    }]
    path = tmp_path / "results.csv"
    MOD.write_csv(path, rows)
    text = path.read_text()
    assert "123" in text and "lru" in text


def test_dbg_grasp_mode_uses_official_grasp_as_reference():
    src = PATH.read_text()
    assert 'reference = "drrip" if public_gate else "grasp"' in src
    assert '"popt_vs_grasp_direction": not public_gate' in src
    assert src.index('reference = "drrip"') < src.index(
        'if "popt-8b" in policies and reference in policies')
