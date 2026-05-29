"""Pytest gate: every tracked wiki/data aggregator artifact is bit-
reproducible from inputs already on disk.

This catches the failure mode where a generator silently goes stale
(e.g. a downstream registry rebuilds and quietly contradicts the
committed dashboard).  See `scripts/experiments/ecg/reproduce_smoke.py`
for the underlying engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_JSON = REPO_ROOT / "wiki" / "data" / "reproduce_smoke.json"

ARTIFACT_FLOOR = 124  # bump explicitly when a new aggregator is added


@pytest.fixture(scope="module")
def doc() -> dict:
    if not DOC_JSON.exists():
        pytest.skip(f"{DOC_JSON} missing; run `make reproduce-smoke`")
    return json.loads(DOC_JSON.read_text())


def test_schema(doc):
    expected = {"n_artifacts", "ok", "drift", "missing", "passed", "rows"}
    assert expected.issubset(doc.keys()), f"missing top-level keys: {expected - doc.keys()}"


def test_artifact_count_floor(doc):
    assert doc["n_artifacts"] >= ARTIFACT_FLOOR, (
        f"reproduce_smoke tracks only {doc['n_artifacts']} artifacts; "
        f"floor is {ARTIFACT_FLOOR}. Did a new aggregator land without "
        "being added to TRACKED_ARTIFACTS?"
    )


def test_no_drift(doc):
    assert doc["drift"] == [], (
        f"{len(doc['drift'])} tracked artifacts changed after `make lit-claims "
        f"lit-catalog` re-ran from disk: {doc['drift']}. This means a "
        "committed wiki/data file silently went stale relative to its "
        "generator; regenerate and re-commit."
    )


def test_no_missing(doc):
    assert doc["missing"] == [], (
        f"{len(doc['missing'])} tracked artifacts missing before or after "
        f"regen: {doc['missing']}. Either the generator failed or the "
        "tracked-artifact list references a stale path."
    )


def test_passed_flag_agrees_with_row_status(doc):
    rows_ok = all(r["status"] == "ok" for r in doc["rows"])
    assert doc["passed"] == rows_ok, (
        "top-level passed flag disagrees with per-row status; "
        "the report is internally inconsistent"
    )


def test_passed(doc):
    assert doc["passed"], (
        f"reproduce_smoke verdict is not GREEN; "
        f"ok={doc['ok']}/{doc['n_artifacts']} drift={doc['drift']} "
        f"missing={doc['missing']}"
    )


def test_dashboard_is_a_tracked_artifact(doc):
    """If we're not tracking the dashboard itself, the gate is mostly
    decorative."""
    names = {r["artifact"] for r in doc["rows"]}
    assert "confidence_dashboard.json" in names
    assert "confidence_dashboard.md" in names


def test_load_bearing_artifacts_tracked(doc):
    """Specific aggregators whose drift would silently break paper claims."""
    names = {r["artifact"] for r in doc["rows"]}
    must_have = {
        "paper_claims.json",
        "literature_faithfulness_postfix.json",
        "oracle_gap.json",
        "oracle_gap_by_app.json",
        "bootstrap_ci.json",
        "policy_winner_table.json",
        "popt_vs_grasp_delta.json",
        "winning_regime_taxonomy.json",
    }
    missing = must_have - names
    assert not missing, (
        f"tracked-artifact list is missing load-bearing files: {missing}. "
        "Without these, reproducibility drift in critical claims would "
        "go silently undetected."
    )
