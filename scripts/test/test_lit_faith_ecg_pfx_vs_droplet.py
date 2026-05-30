"""Tests for gate 241 — ECG-Pfx-vs-DROPLET.

Sibling family to gates 238/239/240 (substrate parity). This gate
audits the *prefetcher* comparison story: ECG's PFX vs DROPLET on
the same baseline. Today: SCAFFOLD/DEFERRED — no matched-proof
ECG-PFX-vs-DROPLET sweep is available (runtime prefetcher counters
are zero across all /tmp corpora). The postfix declares
status="deferred" + per_observation=[]; the audit emits zero
violations while preserving the schema and the activation contract.

When the postfix flips to status="active" with at least
``expected_minimum_observations`` rows, the gate activates against:
arm completeness (G1), baseline neutrality (G2), useful-fraction
floor (G3), backend identity (G5), and observation floor (G6).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_JSON = ROOT / "wiki" / "data" / "lit_faith_ecg_pfx_vs_droplet.json"
ARTIFACT_MD   = ROOT / "wiki" / "data" / "lit_faith_ecg_pfx_vs_droplet.md"
ARTIFACT_CSV  = ROOT / "wiki" / "data" / "lit_faith_ecg_pfx_vs_droplet.csv"
POSTFIX_JSON  = ROOT / "wiki" / "data" / "ecg_pfx_vs_droplet_postfix.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    assert ARTIFACT_JSON.exists(), (
        f"Missing {ARTIFACT_JSON}. Run `make lit-ecg-pfx-vs-droplet`."
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
                "head_to_head", "violations"):
        assert key in audit, f"missing top-level section: {key}"


def test_six_rules_present(audit):
    expected = {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert set(audit["rules"].keys()) == expected


def test_constants_pinned(audit):
    c = audit["constants"]
    assert c["eps_l3_miss_rate_neutral_floor"] == 0.005, (
        "Neutral floor 0.5 pp is the paper-anchored slop budget "
        "for prefetcher-quiet baseline drift."
    )
    assert c["eps_useful_prefetch_floor"] == 0.05, (
        "Useful-fraction floor 5 % is the noise floor — a prefetcher "
        "below this is essentially random."
    )
    assert set(c["required_arms"]) == {"LRU", "DROPLET", "ECG_PFX"}


def test_no_violations(audit):
    """Both deferred and active modes must emit zero violations."""
    assert audit["violations"] == [], (
        f"ECG-Pfx-vs-DROPLET violations: {audit['violations']}"
    )


# --- deferred-mode invariants -------------------------------------------

def test_deferred_status_documented_when_deferred(audit):
    if not _is_deferred(audit):
        pytest.skip("audit is active; deferred-mode tests do not apply")
    assert audit.get("defer_reason"), (
        "Deferred mode must carry a defer_reason."
    )
    assert audit.get("expected_source_pattern"), (
        "Deferred mode must declare the expected source pattern."
    )
    assert audit.get("expected_minimum_observations", 0) > 0, (
        "Deferred mode must declare a positive observation floor."
    )


def test_deferred_expected_arms_match_constants(audit):
    if not _is_deferred(audit):
        pytest.skip("audit is active; deferred-mode tests do not apply")
    c = audit["constants"]
    assert set(audit["expected_required_arms"]) == set(c["required_arms"]), (
        "Deferred mode's expected arms must match audit constants."
    )


def test_deferred_totals_empty(audit):
    if not _is_deferred(audit):
        pytest.skip("audit is active; deferred-mode tests do not apply")
    t = audit["totals"]
    assert t["rows"] == 0
    assert t["cells"] == 0
    assert t["benchmarks"] == []


# --- active-mode per-rule guards ----------------------------------------

def _rule_violations(audit, prefix):
    return [v for v in audit["violations"] if v["rule"].startswith(prefix)]


def test_g1_arm_completeness_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G1") == []


def test_g2_baseline_neutrality_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G2") == []


def test_g3_useful_fraction_floor_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G3") == []


def test_g5_backend_identity_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G5") == []


def test_g6_observation_floor_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert _rule_violations(audit, "G6") == []


def test_head_to_head_populated_when_active(audit):
    if _is_deferred(audit):
        pytest.skip("deferred")
    assert audit["head_to_head"], (
        "Active mode must populate head_to_head — that's the gate's "
        "primary output."
    )
    for row in audit["head_to_head"]:
        assert row["lru_miss_rate"] is not None, (
            f"LRU baseline missing in head-to-head row {row}"
        )


# --- postfix-source invariants (always-on) ------------------------------

def test_postfix_schema(postfix):
    assert postfix["schema_version"] == 1
    assert postfix["gate_id"] == "ECG-Pfx-vs-DROPLET"
    assert postfix["status"] in {"deferred", "active"}
    for k in ("expected_source_pattern", "expected_minimum_observations",
              "expected_required_arms", "siblings", "per_observation"):
        assert k in postfix, f"missing postfix key: {k}"


def test_postfix_siblings_reference_substrate_parity_trinity(postfix):
    sibs = postfix["siblings"]
    ids = {s.get("id") for s in sibs}
    assert {"ECG-Parity", "ECG-Gem5-Parity", "ECG-Sniper-Parity"}.issubset(ids), (
        "Gate 241's siblings field must reference all three substrate-"
        "parity gates (238/239/240) — this is the cross-gate audit trail."
    )
    gates = {s.get("gate") for s in sibs}
    assert {238, 239, 240}.issubset(gates)


def test_postfix_when_deferred_documents_oos(postfix):
    if postfix["status"] != "deferred":
        pytest.skip("postfix is active")
    oos = postfix.get("out_of_scope_when_activated", [])
    joined = " ".join(oos).lower()
    assert "energy" in joined or "area" in joined, (
        "Energy/area cost comparison deferral must be documented"
    )


def test_postfix_arms_match_audit_constants(postfix, audit):
    assert set(postfix["expected_required_arms"]) == set(
        audit["constants"]["required_arms"]
    )


def test_postfix_epsilons_match_audit(postfix, audit):
    assert (postfix["epsilon_l3_miss_rate_neutral_floor"]
            == audit["constants"]["eps_l3_miss_rate_neutral_floor"])
    assert (postfix["epsilon_useful_prefetch_floor"]
            == audit["constants"]["eps_useful_prefetch_floor"])


# --- artifact byte-level checks -----------------------------------------

def test_md_artifact_exists_and_documents_scope():
    assert ARTIFACT_MD.exists()
    text = ARTIFACT_MD.read_text()
    assert "Gate 241" in text
    assert "ECG-Pfx-vs-DROPLET" in text
    for rid in ("G1", "G2", "G3", "G5", "G6"):
        assert rid in text, f"rule {rid} must appear in markdown"
    assert "DROPLET" in text
    assert "ECG_PFX" in text or "PFX" in text


def test_csv_artifact_exists_with_expected_header():
    assert ARTIFACT_CSV.exists()
    with ARTIFACT_CSV.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
    assert header == [
        "benchmark", "section", "l3_size",
        "lru", "droplet", "ecg_pfx",
        "droplet_vs_lru", "ecg_vs_lru", "ecg_vs_droplet",
        "droplet_useful", "ecg_useful",
    ]


def test_json_summary_consistent_with_postfix(audit, postfix):
    """Catch stale renders: audit row count == postfix observation count."""
    assert audit["totals"]["rows"] == len(postfix["per_observation"])
