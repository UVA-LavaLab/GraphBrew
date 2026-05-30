"""Tests for gate 242 — paper-label-map integrity.

Unlike the substrate-parity gates (238/239/240) and the prefetcher
head-to-head gate (241), this gate has NO scaffold-deferred mode.
The source-of-truth is the code itself (POLICY_LABELS in
paper_pipeline.py), not a hand-curated data fixture. So this test
suite is always active: every rule (G1-G5) has both a "no violations"
assertion and a "violation count == 0" assertion against the
rendered artifact.

The gate catches a real failure mode: someone adds a new policy
to the simulator without (a) adding a paper-figure label for it
(violates G3), (b) adding a description and color (G1), or (c)
regenerating the committed paper_pipeline CSV (G4). Any of these
would mean the paper ships with a missing or wrong legend entry.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT          = Path(__file__).resolve().parents[2]
ARTIFACT_JSON = ROOT / "wiki" / "data" / "lit_faith_paper_label_map.json"
ARTIFACT_MD   = ROOT / "wiki" / "data" / "lit_faith_paper_label_map.md"
ARTIFACT_CSV  = ROOT / "wiki" / "data" / "lit_faith_paper_label_map.csv"
PAPER_PIPELINE = ROOT / "scripts" / "experiments" / "ecg" / "paper_pipeline.py"


@pytest.fixture(scope="module")
def audit() -> dict:
    assert ARTIFACT_JSON.exists(), (
        f"Missing {ARTIFACT_JSON}. Run `make lit-paper-label-map`."
    )
    return json.loads(ARTIFACT_JSON.read_text())


# --- schema --------------------------------------------------------------

def test_status_is_active(audit):
    assert audit["status"] == "active", (
        "Gate 242 has no deferred mode — source-of-truth is the code, "
        "not a curated fixture."
    )


def test_schema_has_top_level_sections(audit):
    for key in ("status", "rules", "constants", "totals",
                "policy_label_map", "source_scan", "violations"):
        assert key in audit, f"missing top-level section: {key}"


def test_five_rules_present(audit):
    expected = {"G1", "G2", "G3", "G4", "G5"}
    assert set(audit["rules"].keys()) == expected


def test_no_violations(audit):
    assert audit["violations"] == [], (
        f"Paper-label-map violations: {audit['violations']}"
    )


# --- per-rule guards -----------------------------------------------------

def _rule_violations(audit, rid):
    return [v for v in audit["violations"] if v.get("rule") == rid]


def test_g1_no_partial_additions(audit):
    """Every POLICY_LABELS key has matching description and color."""
    assert _rule_violations(audit, "G1") == []


def test_g2_figure_labels_unique(audit):
    """No two internal policy_labels map to the same figure label."""
    assert _rule_violations(audit, "G2") == []
    # Belt-and-suspenders: re-verify directly against the rendered map.
    fig_labels = [row["figure_label"] for row in audit["policy_label_map"]]
    assert len(fig_labels) == len(set(fig_labels)), (
        f"figure_label collision: {fig_labels}"
    )


def test_g3_no_unknown_policy_labels(audit):
    """No tracked-source policy_label appears outside POLICY_LABELS
    or the allowlist."""
    assert _rule_violations(audit, "G3") == []


def test_g4_committed_csv_matches_code(audit):
    """Latest paper_pipeline_*/policy_label_map.csv equals what we'd
    rebuild from code."""
    assert _rule_violations(audit, "G4") == []


def test_g5_no_orphan_paper_labels(audit):
    """Every POLICY_LABELS key appears in at least one tracked source."""
    assert _rule_violations(audit, "G5") == []


# --- counts / completeness ----------------------------------------------

def test_at_least_seven_paper_labels(audit):
    """A sanity floor: the paper has the canonical 5+ baselines + the
    4 ECG modes today, so total >= 7 always."""
    assert audit["constants"]["policy_labels_count"] >= 7


def test_descriptions_and_colors_count_matches_labels(audit):
    c = audit["constants"]
    assert c["policy_descriptions_count"] == c["policy_labels_count"]
    assert c["policy_colors_count"] == c["policy_labels_count"]


def test_canonical_baselines_present(audit):
    """LRU, SRRIP, GRASP, POPT MUST appear in POLICY_LABELS — they
    are the four canonical replacement-policy baselines the paper
    compares everything against."""
    keys = {row["policy_label"] for row in audit["policy_label_map"]}
    for baseline in ("LRU", "SRRIP", "GRASP", "POPT"):
        assert baseline in keys, f"missing canonical baseline {baseline}"


def test_ecg_variants_present(audit):
    """All four ECG modes must appear in POLICY_LABELS so paper
    legends are complete."""
    keys = {row["policy_label"] for row in audit["policy_label_map"]}
    for v in ("ECG_DBG_ONLY", "ECG_DBG_PRIMARY",
              "ECG_DBG_PRIMARY_CHARGED", "ECG_POPT_PRIMARY"):
        assert v in keys, f"missing ECG variant {v}"


def test_all_sources_readable(audit):
    """A tracked source going missing or unreadable is a regression."""
    t = audit["totals"]
    assert t["sources_missing"] == 0, (
        f"{t['sources_missing']} tracked sources are missing"
    )
    assert t["sources_ok"] == t["sources_scanned"], (
        "Some tracked sources are not OK"
    )


def test_at_least_eight_sources_scanned(audit):
    """Sanity: minimum tracked source count. As of gate 242, we scan
    8 JSON sources + 5 paper_pipeline CSVs = 13. Floor at 8 keeps
    flexibility while catching accidental TRACKED_SOURCES truncation."""
    assert audit["totals"]["sources_scanned"] >= 8


# --- artifact byte-level checks -----------------------------------------

def test_md_artifact_exists_and_documents_rules():
    assert ARTIFACT_MD.exists()
    text = ARTIFACT_MD.read_text()
    assert "gate 242" in text
    for rid in ("G1", "G2", "G3", "G4", "G5"):
        assert rid in text, f"rule {rid} must appear in markdown"


def test_csv_artifact_header_pinned():
    assert ARTIFACT_CSV.exists()
    with ARTIFACT_CSV.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
    assert header == ["policy_label", "figure_label",
                      "description", "color"]


def test_audit_csv_row_count_matches_policy_labels(audit):
    with ARTIFACT_CSV.open() as f:
        rdr = csv.reader(f)
        next(rdr)
        rows = list(rdr)
    assert len(rows) == audit["constants"]["policy_labels_count"]


# --- regression: catch the "paper_pipeline_dir is missing" failure mode --

def test_audit_did_find_a_paper_pipeline_dir(audit):
    """At least one source row should come from a paper_pipeline_*/
    directory — i.e., the latest snapshot exists and has been
    scanned. Catches the failure where no snapshot is present at all."""
    pp_sources = [r for r in audit["source_scan"]
                  if r["source"].startswith("paper_pipeline_")]
    assert pp_sources, (
        "No paper_pipeline_*/ source rows in audit — snapshot is "
        "missing or _find_latest_paper_pipeline_dir is broken."
    )
