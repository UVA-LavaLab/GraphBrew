"""Tests for gate 238 — ECG-Parity (ECG substrate-parity audit).

Locks the cache_sim component-proof matrix's load-bearing invariants:
ECG re-implementations match the policies they shadow, the prefetch
path actually fires, and the encoding bookkeeping is consistent. This
is the confidence-floor that must hold before any cluster-scale ECG
sweep is launched.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_JSON = ROOT / "wiki" / "data" / "lit_faith_ecg_parity.json"
ARTIFACT_MD   = ROOT / "wiki" / "data" / "lit_faith_ecg_parity.md"
ARTIFACT_CSV  = ROOT / "wiki" / "data" / "lit_faith_ecg_parity.csv"
POSTFIX_JSON  = ROOT / "wiki" / "data" / "ecg_substrate_parity_postfix.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    assert ARTIFACT_JSON.exists(), (
        f"Missing {ARTIFACT_JSON}. Run `make lit-ecg-parity`."
    )
    return json.loads(ARTIFACT_JSON.read_text())


@pytest.fixture(scope="module")
def postfix() -> dict:
    assert POSTFIX_JSON.exists(), (
        f"Missing {POSTFIX_JSON}. Curate ECG proof-matrix output into the "
        "postfix JSON before re-running gate 238."
    )
    return json.loads(POSTFIX_JSON.read_text())


# --- schema --------------------------------------------------------------

def test_schema_has_top_level_sections(audit):
    for key in (
        "rules", "constants", "totals",
        "parity_dbg", "parity_popt", "pfx_activation", "dedup_audit",
        "violations",
    ):
        assert key in audit, f"missing top-level section: {key}"


def test_eight_rules_present(audit):
    expected = {"E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"}
    assert set(audit["rules"].keys()) == expected


def test_constants_pinned(audit):
    c = audit["constants"]
    assert c["eps_dbg_parity"]   == 0.0005
    assert c["eps_popt_parity"]  == 0.0005
    assert c["pfx_issued_floor"] == 1
    assert c["backend_floor"]    == 1
    assert set(c["required_ablations"]) >= {
        "LRU_cache_only", "SRRIP_cache_only",
        "GRASP_DBG_only", "POPT_only",
        "ECG_DBG_only", "ECG_POPT_primary",
        "PFX_degree_only", "PFX_POPT_only", "DBG_PFX", "POPT_PFX",
    }
    assert set(c["pfx_ablations"]) == {
        "PFX_degree_only", "PFX_POPT_only", "DBG_PFX", "POPT_PFX",
    }
    assert set(c["baseline_ablations"]) == {
        "LRU_cache_only", "SRRIP_cache_only",
        "GRASP_DBG_only", "POPT_only",
    }
    assert set(c["bench_floor"]) == {"pr", "bfs", "sssp"}


# --- core invariant ------------------------------------------------------

def test_zero_violations(audit):
    assert audit["violations"] == [], (
        f"unexpected ECG substrate-parity violations: {audit['violations']}"
    )


def test_no_dbg_parity_drift(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "E2-dbg-parity-drift"]
    assert bad == []


def test_no_popt_parity_drift(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "E3-popt-parity-drift"]
    assert bad == []


def test_no_pfx_issued_floor_violation(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "E4-pfx-issued-floor"]
    assert bad == []


def test_no_pfx_useful_zero_on_pr(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "E5-pfx-useful-zero-on-pr"]
    assert bad == []


def test_no_useful_exceeds_requests(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "E5-pfx-useful-exceeds-requests"]
    assert bad == []


def test_no_encoding_hygiene_violation(audit):
    bad = [v for v in audit["violations"]
           if v.get("rule") in ("E6-encoded-exceeds-candidates", "E6-negative-counter")]
    assert bad == []


def test_no_baseline_zero(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "E7-baseline-zero"]
    assert bad == []


def test_backend_coverage_floor_met(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "E8-backend-coverage-floor"]
    assert bad == []


# --- phenomenology asserts -----------------------------------------------

def test_three_benchmarks_present(audit):
    assert set(audit["totals"]["benchmarks"]) >= {"pr", "bfs", "sssp"}


def test_dbg_parity_exact_or_near_exact(audit):
    """ECG_DBG_only is a re-implementation of GRASP's DBG-only mode and
    should produce bit-identical miss-rate. Any |Δ| > 1e-6 is a red flag."""
    for row in audit["parity_dbg"]:
        delta = row["abs_delta"]
        assert delta is not None
        assert delta <= 1e-6, (
            f"ECG_DBG_only drifted from GRASP_DBG_only on {row['benchmark']}: "
            f"|Δ|={delta}"
        )


def test_popt_parity_tight(audit):
    """ECG_POPT_primary is a substrate over the same POPT advice and
    should match within 5e-4 (POPT's tie-break order may legitimately
    differ at borderline cells)."""
    for row in audit["parity_popt"]:
        delta = row["abs_delta"]
        assert delta is not None
        assert delta <= 5e-4, (
            f"ECG_POPT_primary drifted from POPT_only on {row['benchmark']}: "
            f"|Δ|={delta}"
        )


def test_pr_pfx_useful_floor(audit):
    """On the dense-hub anchor (PR + email-Eu-core), every PFX ablation
    must register >= 500 useful prefetches. PR has dense L3 reuse, so a
    useful count below the floor implies the prefetch path is broken."""
    pr_rows = [r for r in audit["pfx_activation"] if r["benchmark"] == "pr"]
    assert pr_rows, "no PFX activation rows recorded for benchmark=pr"
    for r in pr_rows:
        useful = r["prefetch_useful"] or 0
        assert useful >= 500, (
            f"PR/{r['ablation']} useful={useful} below the 500 floor"
        )


def test_pr_pfx_issued_dominates_bfs(audit):
    """Phenomenology: PR (dense traversal) must issue >> than BFS
    (sparse frontier) for the same ablation. Catches a regression where
    PR mistakenly behaves like a sparse benchmark."""
    by_key = {(r["benchmark"], r["ablation"]): r["ecg_runtime_issued"]
              for r in audit["pfx_activation"]}
    for abl in ("PFX_POPT_only", "DBG_PFX", "POPT_PFX"):
        pr_issued = by_key.get(("pr", abl), 0) or 0
        bfs_issued = by_key.get(("bfs", abl), 0) or 0
        assert pr_issued >= 10 * max(bfs_issued, 1), (
            f"{abl}: PR issued ({pr_issued}) should be at least 10x BFS "
            f"({bfs_issued}); the dense vs sparse phenomenology has inverted"
        )


def test_required_ablations_present_on_every_benchmark(audit):
    """E1 phenomenology check — make sure the required-ablation roster
    is actually populated and we haven't lost rows during curation."""
    required = set(audit["constants"]["required_ablations"])
    for bench in ("pr", "bfs", "sssp"):
        present_pfx = {r["ablation"] for r in audit["pfx_activation"]
                       if r["benchmark"] == bench}
        # only the PFX subset is inspectable from pfx_activation; for the
        # rest we rely on the violation list being empty (covered above)
        assert {"PFX_degree_only", "PFX_POPT_only", "DBG_PFX", "POPT_PFX"} <= present_pfx


# --- postfix-source invariants -------------------------------------------

def test_postfix_has_source_metadata(postfix):
    src = postfix.get("source", {})
    assert src.get("graph")   == "email-Eu-core"
    assert src.get("backend") == "cache_sim"
    assert "tool" in src and "proof_matrix" in src["tool"]
    assert "note" in src and src["note"]


def test_postfix_every_row_is_ok(postfix):
    rows = postfix.get("per_observation", [])
    assert rows, "postfix carries no observations"
    non_ok = [r for r in rows if r.get("status") != "ok"]
    assert non_ok == [], (
        f"postfix carries non-ok status rows (curation drift): {non_ok}"
    )


def test_postfix_row_count_matches_audit(postfix, audit):
    assert len(postfix["per_observation"]) == audit["totals"]["rows"]


# --- artifact byte-level checks ------------------------------------------

def test_csv_artifact_parseable():
    rows = list(csv.DictReader(ARTIFACT_CSV.open()))
    assert rows
    expected_cols = {"benchmark", "ablation", "issued", "useful", "requests"}
    assert expected_cols <= set(rows[0].keys())


def test_md_artifact_has_signature_sections():
    text = ARTIFACT_MD.read_text()
    for section in (
        "# ECG substrate-parity audit",
        "## Rules",
        "## DBG parity",
        "## POPT parity",
        "## PFX activation",
        "## Violations",
    ):
        assert section in text, f"md artifact missing section: {section}"
