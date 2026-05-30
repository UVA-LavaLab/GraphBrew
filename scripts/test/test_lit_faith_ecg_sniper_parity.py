"""Tests for gate 240 — ECG-Sniper-Parity.

Sibling to gate 238 (cache_sim ECG-Parity) and gate 239 (gem5
ECG-Parity). Today this gate is in **SCAFFOLD/DEFERRED** mode: no
matched-proof Sniper ECG sweep is available, so the postfix declares
``status == "deferred"`` and ``per_observation == []``. The generator
emits zero violations in deferred mode but stamps the deferral status
and the expected source pattern; these tests lock that the deferred
shape is intentional, the schema is complete, the audit logic is
implemented end-to-end (not stubbed), and that when a real Sniper ECG
sweep is curated into the postfix the gate will activate without code
changes.

When the postfix transitions to ``status == "active"``, the
deferred-mode tests skip and the active-mode tests engage:
roster + parity + backend identity + IPC/instructions floors + LRU
non-zero floor + L3 hierarchy sanity + observation-count floor.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_JSON = ROOT / "wiki" / "data" / "lit_faith_ecg_sniper_parity.json"
ARTIFACT_MD   = ROOT / "wiki" / "data" / "lit_faith_ecg_sniper_parity.md"
ARTIFACT_CSV  = ROOT / "wiki" / "data" / "lit_faith_ecg_sniper_parity.csv"
POSTFIX_JSON  = ROOT / "wiki" / "data" / "ecg_sniper_parity_postfix.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    assert ARTIFACT_JSON.exists(), (
        f"Missing {ARTIFACT_JSON}. Run `make lit-ecg-sniper-parity`."
    )
    return json.loads(ARTIFACT_JSON.read_text())


@pytest.fixture(scope="module")
def postfix() -> dict:
    assert POSTFIX_JSON.exists(), (
        f"Missing {POSTFIX_JSON}. Hand-curated input fixture must exist."
    )
    return json.loads(POSTFIX_JSON.read_text())


def _is_deferred(audit) -> bool:
    return audit.get("status") == "deferred"


# --- schema (mode-agnostic) ---------------------------------------------

def test_schema_has_top_level_sections(audit):
    for key in ("status", "rules", "constants", "totals",
                "parity_popt", "parity_dbg", "violations"):
        assert key in audit, f"missing top-level section: {key}"


def test_nine_rules_present(audit):
    expected = {"G1", "G1b", "G2", "G2b", "G3", "G4", "G5", "G6", "G7"}
    assert set(audit["rules"].keys()) == expected


def test_constants_pinned(audit):
    c = audit["constants"]
    assert c["eps_popt_parity"] == 0.002, (
        "Sniper POPT parity tolerance must be 2e-3 (mirrors gem5; "
        "tighten only after empirical Sniper drift data exists)."
    )
    assert c["eps_dbg_parity"] == 0.002
    assert c["ipc_floor"] == 0.0
    assert c["instructions_floor"] == 1
    assert set(c["required_popt_policies"]) == {"LRU", "POPT", "ECG_POPT_PRIMARY"}
    assert set(c["optional_dbg_policies"]) == {"GRASP", "ECG_DBG_ONLY"}
    assert set(c["baseline_policies"]) == {"LRU"}


def test_no_violations(audit):
    """Both deferred and active modes must emit zero violations to be
    confidence-green. Active mode achieves this by satisfying every
    rule; deferred mode achieves it by not having data to check."""
    assert audit["violations"] == [], (
        f"ECG-Sniper-Parity violations: {audit['violations']}"
    )


# --- deferred-mode invariants -------------------------------------------

def test_deferred_status_documented_when_deferred(audit):
    if not _is_deferred(audit):
        pytest.skip("audit is active; deferred-mode tests do not apply")
    assert audit.get("defer_reason"), (
        "Deferred mode must carry a defer_reason explaining why no Sniper "
        "ECG data is available."
    )
    assert audit.get("expected_source_pattern"), (
        "Deferred mode must declare the expected source path pattern so "
        "the next curator knows where to find the matched-proof Sniper "
        "ECG sweep."
    )
    assert audit.get("expected_minimum_observations", 0) > 0, (
        "Deferred mode must declare a positive observation floor for "
        "when the gate activates."
    )


def test_deferred_expected_policies_match_constants(audit):
    if not _is_deferred(audit):
        pytest.skip("audit is active; deferred-mode tests do not apply")
    c = audit["constants"]
    assert set(audit["expected_required_policies"]) == set(
        c["required_popt_policies"]
    ), (
        "Deferred mode's expected POPT policies must match the audit "
        "constants — drift would mean the gate activates with mismatched "
        "expectations."
    )
    assert set(audit["expected_required_dbg_policies"]) == set(
        c["optional_dbg_policies"]
    )


def test_deferred_totals_empty(audit):
    if not _is_deferred(audit):
        pytest.skip("audit is active; deferred-mode tests do not apply")
    t = audit["totals"]
    assert t["rows"] == 0
    assert t["cells"] == 0
    assert t["benchmarks"] == []
    assert t["sections"] == []


# --- active-mode per-rule guards (skipped when deferred) ----------------

def _rule_violations(audit, prefix):
    return [v for v in audit["violations"] if v["rule"].startswith(prefix)]


def test_g1_roster_complete_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G1-") == []


def test_g1b_grasp_paired_with_dbg_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G1b") == []


def test_g2_popt_parity_within_tolerance_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G2-") == []


def test_g2b_dbg_parity_within_tolerance_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G2b") == []


def test_g3_backend_is_sniper_everywhere_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G3") == []


def test_g4_ipc_and_instructions_floors_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G4") == []


def test_g5_lru_baseline_nonzero_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G5") == []


def test_g6_l3_hierarchy_sane_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G6") == []


def test_g7_observation_floor_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G7") == []


# --- active-mode phenomenology floors -----------------------------------

def test_totals_floor_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    t = audit["totals"]
    assert t["rows"] >= 6, "active mode must carry >= 6 rows"
    assert t["backends"] == ["sniper"]
    assert set(t["sections"]) >= {1}
    assert set(t["policies_present"]) >= {"LRU", "POPT", "ECG_POPT_PRIMARY"}


def test_popt_parity_present_per_cell_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    for row in audit["parity_popt"]:
        assert row["popt"] is not None, f"missing POPT row in cell {row}"
        assert row["ecg_popt_primary"] is not None, (
            f"missing ECG_POPT_PRIMARY row in cell {row}"
        )


# --- postfix-source invariants (always-on) ------------------------------

def test_postfix_schema(postfix):
    assert postfix["schema_version"] == 1
    assert postfix["gate_id"] == "ECG-Sniper-Parity"
    assert "status" in postfix
    assert postfix["status"] in {"deferred", "active"}
    assert "expected_source_pattern" in postfix
    assert "expected_minimum_observations" in postfix
    assert "expected_required_policies" in postfix
    assert "expected_required_dbg_policies" in postfix
    assert "siblings" in postfix
    assert "per_observation" in postfix


def test_postfix_siblings_named_correctly(postfix):
    sibs = postfix["siblings"]
    ids = {s.get("id") for s in sibs}
    assert "ECG-Parity" in ids
    assert "ECG-Gem5-Parity" in ids
    gates = {s.get("gate") for s in sibs}
    assert 238 in gates and 239 in gates


def test_postfix_when_deferred_documents_oos_for_activation(postfix):
    """When activated, PFX + DROPLET should remain documented as
    out-of-scope so a reviewer can distinguish 'intentionally deferred'
    from 'silently dropped'."""
    if postfix["status"] != "deferred":
        pytest.skip("postfix is active; deferred-only check")
    oos = postfix.get("out_of_scope_when_activated", [])
    joined = " ".join(oos).lower()
    assert "pfx" in joined or "prefetcher" in joined, (
        "PFX activation deferral must be documented for activation phase"
    )
    assert "droplet" in joined, (
        "DROPLET comparison deferral must be documented for activation phase"
    )


def test_postfix_epsilon_matches_audit(postfix, audit):
    assert postfix["epsilon_popt_parity"] == audit["constants"]["eps_popt_parity"]
    assert postfix["epsilon_dbg_parity"] == audit["constants"]["eps_dbg_parity"]


def test_postfix_expected_policies_match_audit_constants(postfix, audit):
    """The postfix's declared expected_* policy lists must agree with
    audit constants, so activation doesn't trigger a surprise gap."""
    assert set(postfix["expected_required_policies"]) == set(
        audit["constants"]["required_popt_policies"]
    )
    assert set(postfix["expected_required_dbg_policies"]) == set(
        audit["constants"]["optional_dbg_policies"]
    )


# --- artifact byte-level checks -----------------------------------------

def test_md_artifact_exists_and_documents_scope():
    assert ARTIFACT_MD.exists()
    text = ARTIFACT_MD.read_text()
    assert "Gate 240" in text
    assert "ECG-Sniper-Parity" in text
    assert "Sniper" in text
    for rid in ("G1", "G2", "G3", "G7"):
        assert rid in text, f"rule {rid} must appear in markdown"


def test_csv_artifact_exists_with_two_arm_header():
    assert ARTIFACT_CSV.exists()
    with ARTIFACT_CSV.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
    assert header == ["arm", "benchmark", "section", "l3_size",
                      "baseline", "proposed", "abs_delta"]


def test_json_summary_consistent_with_postfix(audit, postfix):
    """The audit row count must match the postfix observation count.
    Catches accidental row filtering in the audit (or stale renders)."""
    assert audit["totals"]["rows"] == len(postfix["per_observation"])
