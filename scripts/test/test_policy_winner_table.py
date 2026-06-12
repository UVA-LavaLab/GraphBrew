#!/usr/bin/env python3
"""Sanity gate for the policy-winner table.

Why this exists
---------------
The policy-winner table summarises every (graph, app, L3 size) cell in
the lit-faithfulness CSV onto a winning replacement policy. We treat
that as a paper-grade artifact, so we lock down the following load-
bearing properties:

1. The CSV / JSON / MD files exist on disk and are mutually consistent.
2. Every cell has a winner that came from the lit-faith CSV (no stray
   policies, no NaN miss rates).
3. GRASP-paper-aligned hub graphs (web-Google, soc-LiveJournal1,
   com-orkut, cit-Patents) yield at least one GRASP or POPT win at the
   large L3 regime — the paper's central claim. If this ever falls to
   zero, something has gone wrong with our GRASP wiring.
4. The road family does *not* show a GRASP win larger than 0.5pp over
   LRU at any cell. This is the road-like invariant projected onto the
   winner table.

The test re-runs `policy_winner_table.py` against the on-disk lit-faith
CSV to make sure the test is always evaluating up-to-date data.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI = REPO_ROOT / "wiki" / "data"
LIT_FAITH_CSV = WIKI / "literature_faithfulness_postfix.csv"
WINNER_CSV = WIKI / "policy_winner_table.csv"
WINNER_JSON = WIKI / "policy_winner_table.json"
WINNER_MD = WIKI / "policy_winner_table.md"

KNOWN_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT", "POPT_CHARGED"}
HUB_GRAPHS = {"web-Google", "soc-LiveJournal1", "com-orkut", "cit-Patents"}


@pytest.fixture(scope="module")
def regenerated_winner_table() -> dict:
    """Regenerate the winner table from the current lit-faith CSV so
    the assertions below see up-to-date data even if the on-disk
    snapshot drifts. Returns the JSON payload.
    """
    if not LIT_FAITH_CSV.exists():
        pytest.skip(f"missing lit-faith CSV at {LIT_FAITH_CSV}")
    subprocess.run(  # noqa: S603 — fixed argv
        [sys.executable, "-m", "scripts.experiments.ecg.policy_winner_table"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(WINNER_JSON.read_text())


def test_winner_artifacts_exist(regenerated_winner_table: dict) -> None:
    for p in (WINNER_CSV, WINNER_JSON, WINNER_MD):
        assert p.exists(), f"missing winner artifact: {p}"
        assert p.stat().st_size > 0, f"empty winner artifact: {p}"


def test_winner_csv_and_json_have_same_cells(
    regenerated_winner_table: dict,
) -> None:
    rows = list(csv.DictReader(WINNER_CSV.open()))
    json_cells = regenerated_winner_table.get("cells", [])
    assert len(rows) == len(json_cells), (
        f"CSV/JSON cell count mismatch: csv={len(rows)} json={len(json_cells)}"
    )
    assert rows, "winner table has zero cells — lit-faith CSV may be empty"


def test_every_winner_is_a_known_policy(regenerated_winner_table: dict) -> None:
    rows = list(csv.DictReader(WINNER_CSV.open()))
    bad = {r["winner_policy"] for r in rows} - KNOWN_POLICIES
    assert not bad, (
        f"winner table reports unknown winners {bad}; "
        f"add to KNOWN_POLICIES or fix lit-faith CSV"
    )


def test_every_winner_has_a_numeric_miss_rate(
    regenerated_winner_table: dict,
) -> None:
    rows = list(csv.DictReader(WINNER_CSV.open()))
    for r in rows:
        mr = r["winner_miss_rate"]
        try:
            v = float(mr)
        except ValueError as exc:
            pytest.fail(
                f"non-numeric winner_miss_rate={mr!r} for cell "
                f"{r['graph']}/{r['app']}@{r['l3_size']}: {exc}"
            )
        assert 0.0 <= v <= 1.0, (
            f"winner_miss_rate out of range: {v} for "
            f"{r['graph']}/{r['app']}@{r['l3_size']}"
        )


def test_at_least_one_hub_large_l3_grasp_or_popt_win(
    regenerated_winner_table: dict,
) -> None:
    """The whole point of GRASP / POPT is to win at a useful L3 size on
    a hub-skewed graph. If this drops to zero, the wiring is broken."""
    rows = list(csv.DictReader(WINNER_CSV.open()))
    hub_large_wins = [
        r for r in rows
        if r["graph"] in HUB_GRAPHS
        and r["l3_regime"] == "large"
        and r["winner_policy"] in {"GRASP", "POPT", "POPT_CHARGED"}
    ]
    assert hub_large_wins, (
        "no hub graph has a GRASP / POPT / POPT_CHARGED win at the large-"
        "L3 regime — this contradicts the paper's central locality claim"
    )


def test_road_winner_grasp_wins_are_documented_regimes(
    regenerated_winner_table: dict,
) -> None:
    """Where GRASP wins on the road family it must be in a DOCUMENTED
    regime: sub-WSS caches (< 1MB, where GRASP's biased retention is an
    anti-thrashing win) or the cc kernel (edge-driven component-representative
    reuse). See docs/findings/grasp_road_anti_thrashing.md. A GRASP road win
    of >= 0.5pp at >= 1MB on a NON-cc kernel would be the surprising result
    the gate must still catch."""
    rows = list(csv.DictReader(WINNER_CSV.open()))
    sub_wss = {"4kB", "16kB", "64kB", "256kB"}
    violations = []
    for r in rows:
        if r["graph_family"] != "road":
            continue
        if r["winner_policy"] != "GRASP":
            continue
        try:
            margin = float(r["margin_pp"])
        except ValueError:
            margin = float("nan")
        if not (margin == margin and margin >= 0.5):
            continue
        documented = r["l3_size"] in sub_wss or r["app"] == "cc"
        if not documented:
            violations.append((r, margin))
    assert not violations, (
        "GRASP wins road by >=0.5pp OUTSIDE the documented anti-thrashing "
        "(<1MB) / cc-reuse regimes at:\n  "
        + "\n  ".join(
            f"{r['graph']}/{r['app']}@{r['l3_size']} margin={m:.3f}pp"
            for r, m in violations
        )
    )


def test_summary_counts_sum_to_total(regenerated_winner_table: dict) -> None:
    summary = regenerated_winner_table["summary"]
    n_cells = summary["n_cells"]
    sum_by_policy = sum(summary["wins_by_policy"].values())
    assert sum_by_policy == n_cells, (
        f"wins_by_policy sums to {sum_by_policy} but n_cells={n_cells}"
    )
    for label, group in (
        ("wins_by_family", summary["wins_by_family"]),
        ("wins_by_regime", summary["wins_by_regime"]),
        ("wins_by_app", summary["wins_by_app"]),
    ):
        sub_sum = sum(sum(c.values()) for c in group.values())
        assert sub_sum == n_cells, (
            f"{label} sums to {sub_sum} but n_cells={n_cells}"
        )
