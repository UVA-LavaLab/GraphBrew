"""Sanity gate for the literature-deviations inventory.

Pins the *structural* invariants of
``wiki/data/literature_deviations.json`` so the inventory keeps
producing a sound paper-grade picture as the corpus grows.

Central claims pinned today:
* Every ``status=known_deviation`` row in the reproduction summary
  is represented exactly once.
* Every deviation carries a mechanism label from the closed
  vocabulary; no "unclassified" leakage is allowed.
* The ``popt_overhead_dominates`` bucket dominates today (all 30
  cells); the test enforces this remains the majority mechanism.
* The mechanism × family cross-tab counts are consistent with the
  per-cell records.
* Today's KNOWN_DEVIATIONS table is exclusively about POPT claims
  (``POPT_GE_GRASP`` / ``POPT_NEAR_GRASP_IF_BIG_GAP``); the gate
  guards against new policy families silently slipping in
  un-explained.

Run via ``pytest scripts/test/test_literature_deviations.py``.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "wiki" / "data"
JSON_PATH = DATA_DIR / "literature_deviations.json"
CSV_PATH = DATA_DIR / "literature_deviations.csv"
MD_PATH = DATA_DIR / "literature_deviations.md"
REPRO_CSV = DATA_DIR / "literature_reproduction_summary.csv"

VALID_MECHANISMS = {
    "popt_overhead_dominates",
    "within_extended_tolerance",
    "policy_data_missing",
    "unclassified",
}


def _ensure_report() -> None:
    if JSON_PATH.exists() and CSV_PATH.exists() and MD_PATH.exists():
        return
    cmd = [
        sys.executable,
        "-m",
        "scripts.experiments.ecg.literature_deviations_report",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@pytest.fixture(scope="module")
def report() -> dict:
    _ensure_report()
    return json.loads(JSON_PATH.read_text())


@pytest.fixture(scope="module")
def known_deviation_rows() -> list[dict]:
    with REPRO_CSV.open() as f:
        return [r for r in csv.DictReader(f) if r.get("status") == "known_deviation"]


def test_report_artifacts_exist() -> None:
    _ensure_report()
    assert JSON_PATH.exists(), JSON_PATH
    assert CSV_PATH.exists(), CSV_PATH
    assert MD_PATH.exists(), MD_PATH


def test_summary_has_required_keys(report: dict) -> None:
    summary = report["summary"]
    for key in (
        "n_deviations",
        "by_mechanism",
        "by_graph",
        "by_family",
        "by_app",
        "by_policy",
        "mechanism_family_cross_tab",
    ):
        assert key in summary, key


def test_inventory_matches_repro_known_deviation_count(
    report: dict, known_deviation_rows: list[dict]
) -> None:
    assert report["summary"]["n_deviations"] == len(known_deviation_rows)
    assert len(report["deviations"]) == len(known_deviation_rows)


def test_every_mechanism_label_is_valid(report: dict) -> None:
    for r in report["deviations"]:
        assert r["mechanism"] in VALID_MECHANISMS, r


def test_no_unclassified_deviations(report: dict) -> None:
    """An ``unclassified`` deviation means the paper has no canned
    explanation. We refuse to ship the paper with any.

    If new known_deviation rows are added with novel mechanisms,
    extend ``_classify`` in
    ``scripts/experiments/ecg/literature_deviations_report.py``
    before this gate will accept them.
    """
    bad = [r for r in report["deviations"] if r["mechanism"] == "unclassified"]
    assert not bad, f"{len(bad)} unclassified deviation(s): {bad[:3]}"


def test_popt_overhead_is_majority_mechanism(report: dict) -> None:
    counts = report["summary"]["by_mechanism"]
    total = report["summary"]["n_deviations"]
    if total == 0:
        pytest.skip("no deviations in corpus yet")
    popt_overhead = counts.get("popt_overhead_dominates", 0)
    assert popt_overhead / total >= 0.5, (
        "popt_overhead_dominates was expected to be the majority "
        f"mechanism, got {popt_overhead}/{total}"
    )


def test_mechanism_family_cross_tab_consistent(report: dict) -> None:
    """The mechanism×family cross-tab must equal the per-record counts."""
    expected: dict[tuple[str, str], int] = {}
    for r in report["deviations"]:
        key = (r["mechanism"], r["graph_family"])
        expected[key] = expected.get(key, 0) + 1
    actual = {
        tuple(k.split("|", 1)): v
        for k, v in report["summary"]["mechanism_family_cross_tab"].items()
    }
    # Cross-tab only stores keys with nonzero counts.
    assert {k: v for k, v in actual.items() if v > 0} == expected


def test_deviation_policy_set_is_known(report: dict) -> None:
    """Today only two computed policy names show up. The gate
    catches the case where new computed-claim policies appear without
    a matching classifier rule.
    """
    known_today = {"POPT_GE_GRASP", "POPT_NEAR_GRASP_IF_BIG_GAP"}
    seen = {r["policy"] for r in report["deviations"]}
    unknown = seen - known_today
    if unknown:
        pytest.fail(
            "deviation table references new policy names "
            f"{sorted(unknown)} - extend the classifier in "
            "scripts/experiments/ecg/literature_deviations_report.py "
            "before merging."
        )
