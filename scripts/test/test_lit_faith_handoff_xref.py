"""Tests for gate 253 — HANDOFF gate-reference registry."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_handoff_xref.py"
JSON_OUT = ROOT / "wiki" / "data" / "lit_faith_handoff_xref.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load("lit_faith_handoff_xref_dyn", GEN_PATH)


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


# --- shape -----------------------------------------------------------

def test_generator_imports(gen):
    assert hasattr(gen, "audit")
    assert hasattr(gen, "GATE_REF_RE")
    assert hasattr(gen, "DASHBOARD_LABEL_RE")
    assert hasattr(gen, "HEADLINE_RE")
    assert hasattr(gen, "REFRESH_AT_RE")
    assert hasattr(gen, "REFRESH_DUE_RE")
    assert hasattr(gen, "REFRESH_CADENCE")


def test_audit_returns_active(audit):
    assert audit["status"] == "active"


def test_audit_zero_violations(audit):
    assert audit["violations"] == [], (
        f"violations: {audit['violations'][:5]}")


def test_audit_advertises_7_rules(audit):
    assert set(audit["rules"].keys()) == {
        "H1", "H2", "H3", "H4", "H5", "H6", "H7"}


def test_refresh_cadence_is_5(gen):
    assert gen.REFRESH_CADENCE == 5


# --- per-rule live checks --------------------------------------------

def test_h1_no_malformed_gate_refs(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "H1"]
    assert not bad, f"H1: {bad}"


def test_h2_no_orphan_dashboard_labels(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "H2"]
    assert not bad, f"H2: {bad}"


def test_h3_headline_matches_suite_count(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "H3"]
    assert not bad, f"H3: {bad}"


def test_h4_refresh_at_matches_suite_count(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "H4"]
    assert not bad, f"H4: {bad}"


def test_h5_refresh_due_equals_refresh_at_plus_cadence(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "H5"]
    assert not bad, f"H5: {bad}"


def test_h6_no_duplicate_gate_labels(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "H6"]
    assert not bad, f"H6: {bad}"


def test_h7_max_labeled_gate_equals_suite_count(audit):
    bad = [v for v in audit["violations"] if v["rule"] == "H7"]
    assert not bad, f"H7: {bad}"


# --- concrete floors -------------------------------------------------

def test_at_least_100_handoff_gate_refs(audit):
    # Today: 138.  >=100 leaves comfortable headroom for HANDOFF
    # cleanup without flipping the gate red.
    assert audit["totals"]["handoff_gate_refs"] >= 100


def test_at_least_5_labeled_dashboard_gates(audit):
    # Today: 11 (gates 242..252).  >=5 catches accidental label
    # stripping while leaving room for label-format changes.
    assert audit["totals"]["labeled_dashboard_gates"] >= 5


def test_max_labeled_gate_at_least_240(audit):
    # The "(gate N)" label format was introduced around gate 242.
    # Lock that the labeling system stays in active use.
    assert audit["max_labeled_dashboard_gate"] is not None
    assert audit["max_labeled_dashboard_gate"] >= 240


def test_labeled_gates_are_contiguous_at_top(audit):
    # The most recent labeled gates must be a contiguous range
    # ending at len(PYTEST_SUITES) — no holes in [max-4 .. max].
    n = audit["totals"]["pytest_suites"]
    labeled = set(audit["labeled_dashboard_gates"])
    for k in range(max(0, n - 4), n + 1):
        # Allow the bottom of the recent range to skip if labelling
        # only began part-way; the rule that matters is the upper
        # cap.  But test that the upper 5 numbers are all labeled.
        pass
    # Just assert the upper 5 are labeled:
    for k in range(n - 4, n + 1):
        assert k in labeled, (
            f"gate {k} missing from labeled set "
            f"(top window must be contiguous)")


def test_handoff_refs_include_max_label(audit):
    handoff_refs = set(audit["handoff_gate_refs"])
    assert audit["max_labeled_dashboard_gate"] in handoff_refs


# --- regex spot-checks -----------------------------------------------

def test_gate_ref_regex_matches_singletons(gen):
    assert gen.GATE_REF_RE.search("we just landed gate 252 today")
    assert gen.GATE_REF_RE.search("see gate 107 for the canonical map")


def test_gate_ref_regex_matches_ranges(gen):
    m = gen.GATE_REF_RE.search("gates 226-228 introduced LIT-Tol")
    assert m and m.group(1) == "226-228"


def test_gate_ref_regex_is_case_sensitive(gen):
    # Should only match lowercase "gate" (it's how HANDOFF writes).
    assert not gen.GATE_REF_RE.search("Gate 1 introduced foo")
    assert not gen.GATE_REF_RE.search("GATE 1 introduced foo")


def test_dashboard_label_regex_extracts_gate_number(gen):
    s = "Slurm SBATCH schema registry (gate 252) — locks every sbatch"
    m = gen.DASHBOARD_LABEL_RE.search(s)
    assert m and m.group(1) == "252"


def test_headline_regex_matches_canonical_format(gen):
    s = "**252 gates, all GREEN, exit 0**"
    m = gen.HEADLINE_RE.search(s)
    assert m and m.group(1) == "252"


def test_refresh_at_regex_matches(gen):
    s = "Refresh status: Refresh complete at gate 252. Next refresh due"
    m = gen.REFRESH_AT_RE.search(s)
    assert m and m.group(1) == "252"


def test_refresh_due_regex_matches(gen):
    s = "Next refresh due at gate 257.\n"
    m = gen.REFRESH_DUE_RE.search(s)
    assert m and m.group(1) == "257"


# --- harvesting -------------------------------------------------------

def test_harvest_expands_ranges(gen):
    refs = gen._harvest_handoff_gate_refs(
        "see gates 226-228 and gate 250.")
    assert refs == {226, 227, 228, 250}


def test_harvest_dedups_repeated_refs(gen):
    refs = gen._harvest_handoff_gate_refs(
        "gate 1 and gate 1 and gate 2 and gate 2.")
    assert refs == {1, 2}


# --- artifact-on-disk parity -----------------------------------------

def test_audit_serialisable(audit):
    json.dumps(audit)


def test_on_disk_json_matches_live_audit(audit):
    if not JSON_OUT.exists():
        pytest.skip(
            f"{JSON_OUT} not yet generated; run `make lit-handoff-xref`.")
    on_disk = json.loads(JSON_OUT.read_text())
    assert on_disk["status"] == audit["status"]
    assert on_disk["totals"] == audit["totals"]
    assert on_disk["violations"] == audit["violations"]


def test_on_disk_md_mentions_gate():
    md = ROOT / "wiki" / "data" / "lit_faith_handoff_xref.md"
    if md.exists():
        txt = md.read_text()
        assert "HANDOFF gate-reference registry" in txt
        assert "gate 253" in txt


def test_on_disk_csv_has_expected_columns():
    csvp = ROOT / "wiki" / "data" / "lit_faith_handoff_xref.csv"
    if csvp.exists():
        head = csvp.read_text().splitlines()[0]
        assert "metric" in head and "value" in head
