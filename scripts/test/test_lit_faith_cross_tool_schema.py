"""Pytest for gate 262 — ECG cross-tool aggregator schema registry."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN = REPO_ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_cross_tool_schema.py"
ARTIFACT = REPO_ROOT / "wiki" / "data" / "lit_faith_cross_tool_schema.json"


def _load():
    spec = importlib.util.spec_from_file_location("gate262", GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gate262"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load()


@pytest.fixture(scope="module")
def data(gate):
    return gate.audit()


# ----- Module shape ---------------------------------------------------------


def test_module_loads(gate):
    assert hasattr(gate, "audit")
    assert hasattr(gate, "main")
    assert hasattr(gate, "CROSS_TOOL_AGGREGATORS")
    assert hasattr(gate, "CANONICAL_TOOLS")
    assert hasattr(gate, "Aggregator")


def test_canonical_tools_nonempty(gate):
    assert len(gate.CANONICAL_TOOLS) >= 3
    assert "gem5" in gate.CANONICAL_TOOLS
    assert "sniper" in gate.CANONICAL_TOOLS


def test_canonical_tools_include_cache_sim_underscore(gate):
    # gate 257 backend registry declares both punctuation variants
    assert "cache_sim" in gate.CANONICAL_TOOLS


def test_aggregator_count_floor(gate):
    # At least 6 today; the gate forbids silent shrinkage by listing
    # every aggregator here. Adding a 7th must be a deliberate edit.
    assert len(gate.CROSS_TOOL_AGGREGATORS) == 6


def test_aggregator_names_unique(gate):
    names = [a.name for a in gate.CROSS_TOOL_AGGREGATORS]
    assert len(names) == len(set(names))


def test_every_aggregator_has_purpose(gate):
    for agg in gate.CROSS_TOOL_AGGREGATORS:
        assert agg.purpose.strip(), f"{agg.name} has empty purpose"
        assert len(agg.purpose) >= 40, f"{agg.name} purpose too short"


def test_every_aggregator_has_top_keys(gate):
    for agg in gate.CROSS_TOOL_AGGREGATORS:
        assert agg.top_keys, f"{agg.name} has empty top_keys"


# ----- audit() shape --------------------------------------------------------


def test_active(data):
    assert data["status"] == "active"


def test_no_violations(data):
    assert data["violations"] == [], data["violations"]


def test_data_shape(data):
    for k in (
        "status",
        "n_aggregators",
        "n_evidence_rows_total",
        "canonical_tools",
        "aggregators",
        "aggregator_status",
        "rules",
        "violations",
    ):
        assert k in data, f"missing key {k}"


def test_rules_present(data):
    for rid in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"):
        assert rid in data["rules"], f"missing rule {rid}"


def test_n_aggregators_matches_constant(data, gate):
    assert data["n_aggregators"] == len(gate.CROSS_TOOL_AGGREGATORS)


def test_total_evidence_rows_positive(data):
    # All 6 aggregators today carry at least one row of evidence.
    assert data["n_evidence_rows_total"] >= 6


# ----- Per-aggregator S1-S7 -------------------------------------------------


def test_all_aggregator_artifacts_exist(data):
    for s in data["aggregator_status"]:
        assert s["exists"], f"{s['name']} artifact missing"


def test_all_top_keys_ok(data):
    for s in data["aggregator_status"]:
        assert s["top_keys_ok"], f"{s['name']} top_keys not OK"


def test_all_evidence_nonempty(data):
    for s in data["aggregator_status"]:
        assert s["evidence_nonempty"], f"{s['name']} evidence empty"


def test_all_row_keys_ok(data):
    for s in data["aggregator_status"]:
        assert s["row_keys_ok"], f"{s['name']} row keys not OK"


def test_all_tools_ok(data):
    for s in data["aggregator_status"]:
        assert s["tools_ok"], f"{s['name']} tools not OK"


def test_all_verdicts_ok(data):
    for s in data["aggregator_status"]:
        assert s["verdict_ok"], f"{s['name']} verdict not OK"


# ----- Required aggregators are present -------------------------------------


REQUIRED_AGGREGATORS = (
    "cross_tool_lru_regime",
    "cross_tool_saturation",
    "cross_tool_slope_ordering",
    "cross_tool_slope_universality",
    "cross_tool_winners",
    "anchor_cross_tool_agreement",
)


@pytest.mark.parametrize("name", REQUIRED_AGGREGATORS)
def test_required_aggregator_declared(gate, name):
    declared = {a.name for a in gate.CROSS_TOOL_AGGREGATORS}
    assert name in declared, f"{name} not in CROSS_TOOL_AGGREGATORS"


@pytest.mark.parametrize("name", REQUIRED_AGGREGATORS)
def test_required_aggregator_status_ok(data, name):
    by_name = {s["name"]: s for s in data["aggregator_status"]}
    assert name in by_name, f"{name} not audited"
    s = by_name[name]
    assert s["exists"] and s["top_keys_ok"] and s["evidence_nonempty"], s


# ----- Artifact round-trip --------------------------------------------------


def test_artifact_exists():
    assert ARTIFACT.exists(), f"missing {ARTIFACT}"


def test_artifact_no_violations():
    doc = json.loads(ARTIFACT.read_text())
    assert doc["violations"] == [], doc["violations"]


def test_artifact_active():
    doc = json.loads(ARTIFACT.read_text())
    assert doc["status"] == "active"


def test_artifact_aggregator_count_matches():
    doc = json.loads(ARTIFACT.read_text())
    assert doc["n_aggregators"] == 6


# ----- main() exit code -----------------------------------------------------


def test_main_zero_exit(tmp_path):
    j = tmp_path / "j.json"
    m = tmp_path / "m.md"
    c = tmp_path / "c.csv"
    proc = subprocess.run(
        [sys.executable, str(GEN), "--json-out", str(j), "--md-out", str(m), "--csv-out", str(c)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert j.exists() and m.exists() and c.exists()
    doc = json.loads(j.read_text())
    assert doc["violations"] == []


# ----- Schema integrity -----------------------------------------------------


def test_no_duplicate_top_keys_per_aggregator(gate):
    for agg in gate.CROSS_TOOL_AGGREGATORS:
        assert len(agg.top_keys) == len(set(agg.top_keys)), \
            f"{agg.name} has duplicate top_keys: {agg.top_keys}"


def test_no_duplicate_row_required_keys(gate):
    for agg in gate.CROSS_TOOL_AGGREGATORS:
        if agg.row_required_keys:
            assert len(agg.row_required_keys) == len(set(agg.row_required_keys)), \
                f"{agg.name} has duplicate row_required_keys"


def test_verdict_type_is_python_type(gate):
    for agg in gate.CROSS_TOOL_AGGREGATORS:
        if agg.verdict_path:
            assert agg.verdict_type is not None, \
                f"{agg.name} has verdict_path but no verdict_type"
            assert isinstance(agg.verdict_type, type), \
                f"{agg.name}.verdict_type is not a Python type"


def test_evidence_path_or_summary_only(gate):
    # Every aggregator must either have an evidence_path OR an explicit
    # empty tuple (summary-only); never a partial declaration.
    for agg in gate.CROSS_TOOL_AGGREGATORS:
        assert isinstance(agg.evidence_path, tuple)
