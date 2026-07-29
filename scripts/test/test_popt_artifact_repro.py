import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/experiments/ecg/flows/popt_artifact_repro.py"
SPEC = importlib.util.spec_from_file_location("popt_artifact_repro", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["popt_artifact_repro"] = MOD
SPEC.loader.exec_module(MOD)


def test_parser_requires_normal_completion_and_sums_llc_totals():
    text = """
~~~ PINTOOL STATS BEGIN ~~~
[LLC-STAT] Total Misses = 100
[LLC-STAT] Total Misses = 7
~~~ PINTOOL STATS END ~~~
[APP] Error = 0.25
[PIN-FINI] App Exit Code = 0
"""
    assert MOD.validate_completed_output(text) == (107, 0.25)
    with pytest.raises(ValueError):
        MOD.validate_completed_output(text.replace(
            "~~~ PINTOOL STATS END ~~~", ""))


def test_commands_pin_one_pagerank_sweep(tmp_path):
    root = tmp_path / "artifact"
    app_root = tmp_path / "apps"
    command = MOD.build_command(
        root, root / "pin", root / "tools", app_root,
        "uk-2002", "popt-8b", "/usr/bin/setarch")
    assert command[:3] == ["/usr/bin/setarch", "x86_64", "-R"]
    assert command[-6:] == [
        "-f", str(root / "input-graphs/uk-2002.sg"),
        "-n", "1", "-i", "1",
    ]
    assert str(app_root / "popt/pr") in command
    assert str(root / "tools/popt-8b/cache_pinsim.so") in command


def test_graph_provenance_must_match_actual_graph(tmp_path):
    root = tmp_path / "artifact"
    graphs = root / "input-graphs"
    graphs.mkdir(parents=True)
    graph = graphs / "g.sg"
    graph.write_bytes(b"graph")
    receipt = tmp_path / "graphs.json"
    receipt.write_text(json.dumps({
        "artifact_commit": MOD.PINNED_ARTIFACT_COMMIT,
        "graphs": {"g.sg": {"sha256": MOD.sha256(graph)}},
    }))

    MOD.verify_graph_provenance(
        receipt, root, ["g"], MOD.PINNED_ARTIFACT_COMMIT)
    graph.write_bytes(b"changed")
    with pytest.raises(SystemExit):
        MOD.verify_graph_provenance(
            receipt, root, ["g"], MOD.PINNED_ARTIFACT_COMMIT)


def test_port_manifest_binds_script_pin_binary_app_and_sources(tmp_path):
    port = tmp_path / "port"
    pin_root = tmp_path / "pin"
    tool_root = port / "bin"
    app_root = port / "apps"
    script = tmp_path / "setup.py"
    script.write_text("setup\n")
    (pin_root / "intel64").mkdir(parents=True)
    (pin_root / "source/tools/Config").mkdir(parents=True)
    (pin_root / "pin").write_text("pin\n")
    (pin_root / "intel64/runtime").write_text("runtime\n")
    (pin_root / "source/tools/Config/rules").write_text("rules\n")
    (tool_root / "lru").mkdir(parents=True)
    tool = tool_root / "lru/cache_pinsim.so"
    tool.write_text("tool\n")
    (port / "src/lru").mkdir(parents=True)
    (port / "src/lru/source.cpp").write_text("source\n")
    (app_root / "baseline").mkdir(parents=True)
    app = app_root / "baseline/pr"
    app.write_text("app\n")
    (port / "app-src/baseline").mkdir(parents=True)
    (port / "app-src/baseline/pr.cc").write_text("app source\n")
    manifest = {
        "schema_version": 2,
        "setup_script_sha256": MOD.sha256(script),
        "popt_repository": {"commit": MOD.PINNED_ARTIFACT_COMMIT},
        "pin": {
            "pin_sha256": MOD.sha256(pin_root / "pin"),
            "intel64_tree_sha256": MOD.hash_tree(pin_root / "intel64"),
            "config_tree_sha256": MOD.hash_tree(
                pin_root / "source/tools/Config"),
        },
        "smoke": {
            "passed": True,
            "normal_application_completion": True,
            "semantic_error_match": True,
        },
        "binaries": {"lru": MOD.sha256(tool)},
        "generated_source_trees": {
            "lru": MOD.hash_tree(port / "src/lru"),
        },
        "applications": {
            "baseline": {"pr": MOD.sha256(app)},
        },
        "application_source_trees": {
            "baseline": MOD.hash_tree(port / "app-src/baseline"),
        },
    }
    receipt = port / "port_build_manifest.json"
    receipt.write_text(json.dumps(manifest))

    MOD.verify_port_build_manifest(
        receipt, script, MOD.PINNED_ARTIFACT_COMMIT,
        pin_root, tool_root, app_root, ["lru"])
    tool.write_text("different\n")
    with pytest.raises(SystemExit):
        MOD.verify_port_build_manifest(
            receipt, script, MOD.PINNED_ARTIFACT_COMMIT,
            pin_root, tool_root, app_root, ["lru"])


def test_resume_rows_round_trip_integer_metrics(tmp_path):
    rows = [{
        "graph": "g", "policy": "lru", "exit_code": 0,
        "llc_demand_misses": 123, "app_error": 0.5, "status": "ok",
    }]
    path = tmp_path / "results.csv"
    MOD.write_csv(path, rows)
    text = path.read_text()
    assert "123" in text and "lru" in text


def test_grasp_mode_is_explicitly_nonclaimable_rules_proxy():
    assert MOD.PROXY_GATE == "dbg-grasp-rules-proxy"
    assert MOD.GRASP_PROXY == "grasp-rules-proxy"
    source = PATH.read_text()
    assert '"popt_vs_grasp_direction": False' in source
    assert '"popt_vs_grasp_figure12_exact": False' in source
    assert '"grasp_rules_proxy_direction": False' in source


def test_hash_tree_rejects_missing_port_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        MOD.hash_tree(tmp_path / "missing")
