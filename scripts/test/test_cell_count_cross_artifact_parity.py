"""Gate 91 — cell-census cross-artifact cell-count parity gate.

Cross-artifact integrity gate locking that the four winner-class
artifacts agree on:

  * total cell count (n_cells = 114)
  * per-app cell breakdown (bc/bfs/cc/pr/sssp counts)
  * winner-typology accounting (unique + tied = total; no_winner == 0)
  * the specific 15 known tied cells

Artifacts:
  * wiki/data/cell_winner_census.json
  * wiki/data/policy_winner_table.json
  * wiki/data/winning_regime_taxonomy.json
  * wiki/data/popt_vs_grasp_delta.json

Gate 85 already locks the *winner-counts* aggregate across these
sibling artifacts. This gate locks the *cell-set accounting* — that
they all measure the same 114 cells, that the cell census's per-app
counts agree with the winner table's per-app sums, and that the
no-winner/tied breakdown follows the load-bearing 0/15/99 partition.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI = REPO_ROOT / "wiki" / "data"

CWC_JSON = WIKI / "cell_winner_census.json"
PWT_JSON = WIKI / "policy_winner_table.json"
TAX_JSON = WIKI / "winning_regime_taxonomy.json"
PVG_JSON = WIKI / "popt_vs_grasp_delta.json"

EXPECTED_TOTAL_CELLS = 114
EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_NO_WINNER_COUNT = 0
EXPECTED_TIED_WINNER_COUNT = 15
EXPECTED_UNIQUE_WINNER_COUNT = 99

# The 15 known tied cells in the regenerated deterministic corpus.
# Locking this exact set catches a comparator regression that could
# silently flip one cell to "no winner" or to a different (graph, app, L3).
EXPECTED_TIED_CELLS = {
    ("bc", "email-Eu-core", "1MB"),
    ("bc", "email-Eu-core", "4MB"),
    ("bc", "email-Eu-core", "8MB"),
    ("bfs", "email-Eu-core", "1MB"),
    ("bfs", "email-Eu-core", "4MB"),
    ("bfs", "email-Eu-core", "8MB"),
    ("cc", "soc-pokec", "8MB"),
    ("cc", "web-Google", "4MB"),
    ("cc", "web-Google", "8MB"),
    ("pr", "email-Eu-core", "1MB"),
    ("pr", "email-Eu-core", "4MB"),
    ("pr", "email-Eu-core", "8MB"),
    ("sssp", "soc-pokec", "8MB"),
    ("sssp", "web-Google", "4MB"),
    ("sssp", "web-Google", "8MB"),
}


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _cwc() -> dict:
    return _load(CWC_JSON)


def _pwt() -> dict:
    return _load(PWT_JSON)


def _tax() -> dict:
    return _load(TAX_JSON)


def _pvg() -> dict:
    return _load(PVG_JSON)


# ---------------------------------------------------------------------------
# Total cell-count parity across four artifacts
# ---------------------------------------------------------------------------

def test_cell_census_total_cells_equals_expected_total():
    m = _cwc()["meta"]
    assert m["n_cells_total"] == EXPECTED_TOTAL_CELLS, (
        f"cell_winner_census n_cells_total drifted: "
        f"got {m['n_cells_total']}, expected {EXPECTED_TOTAL_CELLS}"
    )


def test_policy_winner_table_n_cells_equals_expected_total():
    n = _pwt()["summary"]["n_cells"]
    assert n == EXPECTED_TOTAL_CELLS, (
        f"policy_winner_table summary.n_cells drifted: got {n}, "
        f"expected {EXPECTED_TOTAL_CELLS}"
    )


def test_winning_regime_taxonomy_n_cells_equals_expected_total():
    n = _tax()["summary"]["n_cells"]
    assert n == EXPECTED_TOTAL_CELLS, (
        f"winning_regime_taxonomy summary.n_cells drifted: got {n}, "
        f"expected {EXPECTED_TOTAL_CELLS}"
    )


def test_popt_vs_grasp_delta_n_cells_equals_expected_total():
    n = _pvg()["summary"]["n_cells"]
    assert n == EXPECTED_TOTAL_CELLS, (
        f"popt_vs_grasp_delta summary.n_cells drifted: got {n}, "
        f"expected {EXPECTED_TOTAL_CELLS}"
    )


def test_top_level_cell_list_lengths_match_summary_counts():
    """Each of the three artifacts also stores the per-cell rows at
    top level. Locking the list length == summary.n_cells catches
    silent drift between the body and the summary roll-up."""
    for path_key, getter in [
        ("policy_winner_table", _pwt),
        ("winning_regime_taxonomy", _tax),
        ("popt_vs_grasp_delta", _pvg),
    ]:
        d = getter()
        body_len = len(d["cells"])
        summary_n = d["summary"]["n_cells"]
        assert body_len == summary_n, (
            f"{path_key}: cells list ({body_len}) != summary.n_cells "
            f"({summary_n}) — silent body/summary drift"
        )


# ---------------------------------------------------------------------------
# Winner typology accounting (unique + tied = total, no_winner == 0)
# ---------------------------------------------------------------------------

def test_cell_census_no_winner_count_is_zero():
    m = _cwc()["meta"]
    assert m["n_no_winner"] == EXPECTED_NO_WINNER_COUNT, (
        f"cell_winner_census n_no_winner regressed to {m['n_no_winner']} "
        f"(was {EXPECTED_NO_WINNER_COUNT}) — at least one cell can no "
        "longer pick a winning policy"
    )
    assert m["pct_no_winner"] == 0.0


def test_cell_census_tied_winner_count_matches_expected():
    m = _cwc()["meta"]
    assert m["n_tied_winners"] == EXPECTED_TIED_WINNER_COUNT, (
        f"cell_winner_census n_tied_winners drifted: "
        f"got {m['n_tied_winners']}, expected {EXPECTED_TIED_WINNER_COUNT}"
    )


def test_cell_census_unique_winner_count_matches_expected():
    m = _cwc()["meta"]
    assert m["n_unique_winner"] == EXPECTED_UNIQUE_WINNER_COUNT, (
        f"cell_winner_census n_unique_winner drifted: "
        f"got {m['n_unique_winner']}, expected {EXPECTED_UNIQUE_WINNER_COUNT}"
    )


def test_cell_census_winner_partition_is_consistent():
    """unique + tied + no_winner must equal n_cells_total — internal
    consistency check that catches a counter-increment bug."""
    m = _cwc()["meta"]
    parts = m["n_unique_winner"] + m["n_tied_winners"] + m["n_no_winner"]
    assert parts == m["n_cells_total"], (
        f"cell_winner_census partition does not sum to total: "
        f"unique({m['n_unique_winner']}) + tied({m['n_tied_winners']}) "
        f"+ no_winner({m['n_no_winner']}) = {parts} != "
        f"total({m['n_cells_total']})"
    )


# ---------------------------------------------------------------------------
# Per-app cell-count parity (cell_winner_census ↔ policy_winner_table)
# ---------------------------------------------------------------------------

def test_per_app_cell_counts_match_winner_table_sums():
    cwc = _cwc()["per_app"]
    pwt_by_app = _pwt()["summary"]["wins_by_app"]
    bad = []
    for app in EXPECTED_APPS:
        cwc_n = cwc[app]["n_cells"]
        pwt_n = sum(pwt_by_app.get(app, {}).values())
        if cwc_n != pwt_n:
            bad.append((app, cwc_n, pwt_n))
    assert not bad, (
        f"cell_winner_census per_app n_cells != policy_winner_table "
        f"wins_by_app sum for: {bad}"
    )


def test_per_app_cell_counts_sum_to_total():
    cwc = _cwc()["per_app"]
    s = sum(cwc[a]["n_cells"] for a in EXPECTED_APPS)
    assert s == EXPECTED_TOTAL_CELLS, (
        f"per-app cell counts sum to {s}, expected "
        f"{EXPECTED_TOTAL_CELLS}"
    )


def test_per_app_unique_plus_tied_plus_no_winner_equals_n_cells():
    """Same internal-consistency check, but applied per-app to catch
    bugs that only show up on one app."""
    cwc = _cwc()["per_app"]
    bad = []
    for app in EXPECTED_APPS:
        d = cwc[app]
        parts = d["unique_winner"] + d["tied_winners"] + d["no_winner"]
        if parts != d["n_cells"]:
            bad.append(
                (app, d["unique_winner"], d["tied_winners"],
                 d["no_winner"], parts, d["n_cells"])
            )
    assert not bad, (
        f"per-app winner partition does not sum to per-app n_cells: {bad}"
    )


# ---------------------------------------------------------------------------
# Lock the specific tied-cell set
# ---------------------------------------------------------------------------

def test_tied_cells_match_known_set():
    """The tied cell set is pinned exactly for corpus-level decisiveness."""
    cwc = _cwc()
    tied = {
        (c["app"], c["graph"], c["l3"])
        for c in cwc["all_tied_cells"]
    }
    assert tied == EXPECTED_TIED_CELLS, (
        f"tied-cell set drifted: extra={tied - EXPECTED_TIED_CELLS}, "
        f"missing={EXPECTED_TIED_CELLS - tied}"
    )


def test_tied_cells_per_app_breakdown_consistent_with_top_level():
    """Per-app tied counts should match the top-level tied-cell list."""
    cwc = _cwc()
    expected_per_app = {a: 0 for a in EXPECTED_APPS}
    for c in cwc["all_tied_cells"]:
        expected_per_app[c["app"]] += 1
    bad = [
        (app, cwc["per_app"][app]["tied_winners"], expected)
        for app, expected in expected_per_app.items()
        if cwc["per_app"][app]["tied_winners"] != expected
    ]
    assert not bad, (
        f"per-app tied_winners diverges from all_tied_cells distribution: "
        f"{bad}"
    )
