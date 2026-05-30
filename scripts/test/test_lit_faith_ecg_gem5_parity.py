"""Tests for gate 239 — ECG-Gem5-Parity.

Sibling to gate 238 (cache_sim ECG-Parity). Locks the POPT-arm
faithfulness of the ECG substrate under cycle-accurate gem5 timing:
ECG_POPT_PRIMARY must agree with stock POPT on L3 miss-rate to within
2e-3 (≈ 0.2 pp) on every matched (benchmark, section, L3) cell. DBG arm
and PFX activation are explicitly out-of-scope today (see generator
docstring); this gate documents that scope.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_JSON = ROOT / "wiki" / "data" / "lit_faith_ecg_gem5_parity.json"
ARTIFACT_MD   = ROOT / "wiki" / "data" / "lit_faith_ecg_gem5_parity.md"
ARTIFACT_CSV  = ROOT / "wiki" / "data" / "lit_faith_ecg_gem5_parity.csv"
POSTFIX_JSON  = ROOT / "wiki" / "data" / "ecg_gem5_parity_postfix.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    assert ARTIFACT_JSON.exists(), (
        f"Missing {ARTIFACT_JSON}. Run `make lit-ecg-gem5-parity`."
    )
    return json.loads(ARTIFACT_JSON.read_text())


@pytest.fixture(scope="module")
def postfix() -> dict:
    assert POSTFIX_JSON.exists(), (
        f"Missing {POSTFIX_JSON}. Curate gem5 ROI matrix into the postfix "
        "JSON before re-running gate 239."
    )
    return json.loads(POSTFIX_JSON.read_text())


# --- schema --------------------------------------------------------------

def test_schema_has_top_level_sections(audit):
    for key in ("rules", "constants", "totals", "parity_popt", "violations"):
        assert key in audit, f"missing top-level section: {key}"


def test_seven_rules_present(audit):
    expected = {"G1", "G2", "G3", "G4", "G5", "G6", "G7"}
    assert set(audit["rules"].keys()) == expected


def test_constants_pinned(audit):
    c = audit["constants"]
    assert c["eps_popt_parity"] == 0.002, (
        "POPT parity tolerance must be 2e-3 (2× observed gem5 drift max 1.09e-3). "
        "Loosening this is a paper-level decision."
    )
    assert c["section_floor"] == 2
    assert c["sim_tick_floor"] == 1
    assert c["ipc_floor"] == 0.0
    assert set(c["required_policies"]) == {"LRU", "POPT", "ECG_POPT_PRIMARY"}
    assert set(c["baseline_policies"]) == {"LRU"}
    assert set(c["bench_floor"]) == {"pr"}


# --- core invariant ------------------------------------------------------

def test_no_violations(audit):
    assert audit["violations"] == [], (
        f"ECG-Gem5-Parity violations: {audit['violations']}"
    )


# --- per-rule no-violation guards ---------------------------------------

def _rule_violations(audit, rule_prefix):
    return [v for v in audit["violations"] if v["rule"].startswith(rule_prefix)]


def test_g1_roster_complete(audit):
    assert _rule_violations(audit, "G1") == []


def test_g2_popt_parity_within_tolerance(audit):
    assert _rule_violations(audit, "G2") == []


def test_g3_backend_is_gem5_everywhere(audit):
    assert _rule_violations(audit, "G3") == []


def test_g4_sim_ticks_and_ipc_nonzero(audit):
    assert _rule_violations(audit, "G4") == []


def test_g5_lru_baseline_nonzero(audit):
    assert _rule_violations(audit, "G5") == []


def test_g6_l3_hierarchy_sane(audit):
    assert _rule_violations(audit, "G6") == []


def test_g7_section_coverage_floor_met(audit):
    assert _rule_violations(audit, "G7") == []


# --- phenomenology (load-bearing on real gem5 data) ----------------------

def test_totals_floor(audit):
    t = audit["totals"]
    assert t["rows"] >= 12, "gem5 bracket sweep should emit >= 12 ok rows"
    assert t["cells"] >= 4, "must cover >= 4 (benchmark, section, L3) cells"
    assert set(t["benchmarks"]) == {"pr"}
    assert t["backends"] == ["gem5"]
    assert set(t["sections"]) >= {1, 2}
    assert set(t["policies_present"]) >= {"LRU", "POPT", "ECG_POPT_PRIMARY"}


def test_popt_parity_drift_bounded(audit):
    """Even tighter than the gate tolerance: every observed drift must
    sit within the real-data max (1.5e-3 = 50 % headroom over observed
    1.09e-3) so a regression in gem5 ECG_POPT_PRIMARY is caught early."""
    for row in audit["parity_popt"]:
        d = row["abs_delta"]
        assert d is not None, f"missing parity delta: {row}"
        assert d <= 1.5e-3, (
            f"gem5 POPT parity drift {d:.6e} exceeds observed-headroom "
            f"floor 1.5e-3 at cell={row}"
        )


def test_popt_and_ecg_popt_primary_both_present_per_cell(audit):
    """Without both, parity is undefined. Catch dropped rows."""
    for row in audit["parity_popt"]:
        assert row["popt"] is not None, f"missing POPT row in cell {row}"
        assert row["ecg_popt_primary"] is not None, (
            f"missing ECG_POPT_PRIMARY row in cell {row}"
        )


# --- postfix-source invariants ------------------------------------------

def test_postfix_schema(postfix):
    assert postfix["schema_version"] == 1
    assert postfix["gate_id"] == "ECG-Gem5-Parity"
    assert "source" in postfix
    src = postfix["source"]
    assert src["backend"] == "gem5"
    assert src["tool"] == "roi_matrix"
    assert "roi_matrix_path" in src
    assert "roi_matrix_sha256" in src
    assert len(src["roi_matrix_sha256"]) == 64


def test_postfix_has_required_policies(postfix):
    rows = postfix["per_observation"]
    pols = {r["policy_label"] for r in rows}
    assert {"LRU", "POPT", "ECG_POPT_PRIMARY"}.issubset(pols)


def test_postfix_documents_out_of_scope(postfix):
    """Every deferred dimension must be explicitly named in the postfix.
    Without this, a future reviewer can't tell what's intentionally
    missing vs accidentally dropped."""
    oos = postfix.get("out_of_scope", [])
    joined = " ".join(oos).lower()
    assert "dbg" in joined, "DBG arm deferral must be documented"
    assert "pfx" in joined or "prefetcher" in joined, (
        "PFX activation deferral must be documented"
    )
    assert "droplet" in joined, "DROPLET comparison deferral must be documented"


def test_postfix_required_policies_field_matches_audit(postfix, audit):
    """The postfix's declared required_policies must match what the
    audit rules actually enforce. Drift between these two lists is a
    config bug."""
    assert set(postfix["required_policies"]) == set(
        audit["constants"]["required_policies"]
    )


def test_postfix_epsilon_matches_audit(postfix, audit):
    assert postfix["epsilon_popt_parity"] == audit["constants"]["eps_popt_parity"]


# --- artifact byte-level checks -----------------------------------------

def test_md_artifact_exists_and_documents_scope():
    assert ARTIFACT_MD.exists()
    text = ARTIFACT_MD.read_text()
    assert "Gate 239" in text
    assert "ECG-Gem5-Parity" in text
    assert "POPT-arm" in text
    assert "cache_sim" in text  # cross-reference to gate 238
    assert "G1" in text and "G7" in text


def test_csv_artifact_exists_and_has_expected_columns():
    assert ARTIFACT_CSV.exists()
    with ARTIFACT_CSV.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = list(rdr)
    assert header == ["benchmark", "section", "l3_size",
                      "popt", "ecg_popt_primary", "abs_delta"]
    assert len(rows) >= 4, "csv must have >= 4 cells of parity data"


def test_json_summary_consistent_with_postfix(audit, postfix):
    """The audit row count must match the postfix observation count.
    Catches accidental row filtering in the audit."""
    assert audit["totals"]["rows"] == len(postfix["per_observation"])
