"""Sanity gate for the POPT-vs-GRASP delta report.

These tests pin the *structural* invariants of
``wiki/data/popt_vs_grasp_delta.json`` so the report keeps producing a
sound paper-grade picture as the lit-faith corpus grows. The numbers
themselves shift with every new (graph, app, L3) cell — the gate
therefore asserts shape, completeness, and a few crisp scientific
claims rather than exact numerical values.

Central claims pinned today:
* Sign convention is consistent: ``delta_pp`` matches the recomputed
  ``(POPT − GRASP) × 100``.
* Every cell with a delta lands in exactly one of the four
  classifications used by the script (``popt_better`` /
  ``grasp_better`` / ``tie`` / ``missing``).
* The road family — where POPT's offline lookahead has the strongest
  theoretical reason to dominate — must show at least one cell where
  POPT improves on GRASP by ≥ 5 pp.
* The social family must show at least one cell where GRASP improves
  on POPT by ≥ 5 pp (this is the canonical "hub graphs let GRASP
  match POPT without the overhead" claim).
* The classification counts sum to ``n_cells``.

Run via ``pytest scripts/test/test_popt_vs_grasp_delta.py``.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "wiki" / "data"
JSON_PATH = DATA_DIR / "popt_vs_grasp_delta.json"
CSV_PATH = DATA_DIR / "popt_vs_grasp_delta.csv"
MD_PATH = DATA_DIR / "popt_vs_grasp_delta.md"

VALID_CLASSES = {"popt_better", "grasp_better", "tie", "missing"}


def _ensure_report() -> None:
    """Regenerate the artifacts if any of the three are missing."""
    if JSON_PATH.exists() and CSV_PATH.exists() and MD_PATH.exists():
        return
    cmd = [sys.executable, "-m", "scripts.experiments.ecg.popt_vs_grasp_report"]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@pytest.fixture(scope="module")
def report() -> dict:
    _ensure_report()
    return json.loads(JSON_PATH.read_text())


def test_report_artifacts_exist() -> None:
    _ensure_report()
    assert JSON_PATH.exists(), JSON_PATH
    assert CSV_PATH.exists(), CSV_PATH
    assert MD_PATH.exists(), MD_PATH


def test_summary_has_required_keys(report: dict) -> None:
    summary = report["summary"]
    for key in (
        "n_cells",
        "overall",
        "classification_counts",
        "by_family",
        "by_regime",
        "by_app",
        "by_family_regime",
        "popt_top5_helps",
        "grasp_top5_helps",
    ):
        assert key in summary, key


def test_class_counts_sum_to_n_cells(report: dict) -> None:
    summary = report["summary"]
    cc = summary["classification_counts"]
    assert sum(cc.values()) == summary["n_cells"]
    # Every label must come from the closed vocabulary.
    assert set(cc.keys()).issubset(VALID_CLASSES), cc


def test_per_cell_delta_consistent_with_inputs(report: dict) -> None:
    """Recompute (POPT − GRASP) × 100 and confirm the stored delta_pp
    matches to floating-point tolerance. Also confirms classification
    obeys the 0.5 pp floor exactly.
    """
    floor = 0.5
    for cell in report["cells"]:
        gm = float(cell["grasp_miss_rate"])
        pm = float(cell["popt_miss_rate"])
        delta = (pm - gm) * 100.0
        stored = float(cell["delta_pp"])
        assert math.isclose(stored, delta, abs_tol=1e-3), cell
        cls = cell["classification"]
        if delta < -floor:
            assert cls == "popt_better", cell
        elif delta > floor:
            assert cls == "grasp_better", cell
        else:
            assert cls == "tie", cell


def test_road_family_has_strong_popt_win(report: dict) -> None:
    """On road-like graphs POPT's offline trace is *exactly* the
    knowledge GRASP lacks, so the paper's central claim requires at
    least one road cell where POPT improves on GRASP by >= 5 pp.

    This is the headline result for the "POPT shines on diffuse
    locality" story — if this ever flips, the paper's storyline
    needs to change.
    """
    road_deltas = [
        float(c["delta_pp"]) for c in report["cells"]
        if c["graph_family"] == "road"
    ]
    if not road_deltas:
        pytest.skip("no road-family cells in lit-faith yet")
    big_wins = [d for d in road_deltas if d <= -5.0]
    assert big_wins, (
        f"expected ≥1 road cell with POPT improving GRASP by ≥5 pp; "
        f"deltas seen: {road_deltas}"
    )


def test_social_family_has_at_least_one_grasp_win(report: dict) -> None:
    """The companion claim: on hub-heavy social graphs at large L3 the
    permutation cost of POPT is not always recovered, so at least one
    social cell must have GRASP improving on POPT by ≥ 5 pp.
    """
    social_deltas = [
        float(c["delta_pp"]) for c in report["cells"]
        if c["graph_family"] == "social"
    ]
    assert social_deltas, "no social-family cells in lit-faith"
    grasp_wins = [d for d in social_deltas if d >= 5.0]
    assert grasp_wins, (
        f"expected ≥1 social cell with GRASP improving POPT by ≥5 pp; "
        f"deltas seen: {sorted(social_deltas)[-5:]}"
    )


def test_overall_mean_within_sane_band(report: dict) -> None:
    """Sanity guard: the overall mean delta must stay inside a sane
    [-20, +20] pp band. Anything outside that signals a data-corruption
    bug (e.g. units mismatch, swapped GRASP/POPT columns).
    """
    mean_pp = report["summary"]["overall"]["mean_pp"]
    assert -20.0 <= mean_pp <= 20.0, mean_pp


def test_family_stats_match_per_cell_counts(report: dict) -> None:
    """The per-family ``n`` in the summary must equal the number of
    cells with that family in the per-cell table.
    """
    per_family_counts: dict[str, int] = {}
    for c in report["cells"]:
        per_family_counts[c["graph_family"]] = per_family_counts.get(
            c["graph_family"], 0
        ) + 1
    for fam, st in report["summary"]["by_family"].items():
        assert st["n"] == per_family_counts.get(fam, 0), fam
