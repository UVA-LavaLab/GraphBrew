"""Pytest gate: per-policy oracle-gap report invariants.

Oracle = empirical per-cell minimum miss rate across
{LRU, SRRIP, GRASP, POPT}. This gate pins the structural
invariants of the gap projection so a regression in the join /
aggregation logic is caught and the load-bearing paper narrative
(POPT smallest mean gap, GRASP biggest road-family gap) stays
anchored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPORT.exists():
        pytest.skip(f"{REPORT} not generated; run `make lit-oracle-gap`")
    return json.loads(REPORT.read_text())


def test_top_level_schema(report):
    assert {"summary", "rows"}.issubset(report.keys())
    for k in (
        "n_cells", "n_rows",
        "overall_by_policy", "by_policy_family", "by_policy_regime",
    ):
        assert k in report["summary"], f"missing summary key {k!r}"


def test_overall_covers_all_known_policies(report):
    keys = set(report["summary"]["overall_by_policy"].keys())
    missing = set(POLICIES) - keys
    assert not missing, f"overall_by_policy missing policies: {sorted(missing)}"


def test_no_negative_gap(report):
    """A miss rate cannot be lower than the per-cell minimum across
    the same policies; any negative gap_pp means the oracle was
    computed wrong (e.g. selecting min over the wrong policy set)."""
    bad = [
        r for r in report["rows"]
        if float(r["gap_pp"]) < -1e-9
    ]
    assert not bad, (
        f"{len(bad)} rows have negative gap_pp; first 3: "
        f"{[(r['graph'], r['app'], r['l3_size'], r['policy'], r['gap_pp']) for r in bad[:3]]}"
    )


def test_winners_have_zero_gap(report):
    """Every is_winner=1 row must have gap_pp ≤ 1e-3 pp (floating
    point slack). Catches a regression where the winner flag is
    computed against the wrong reference."""
    for r in report["rows"]:
        if r["is_winner"] == "1":
            assert float(r["gap_pp"]) <= 1e-3, r


def test_at_least_one_winner_per_cell(report):
    """Every cell must yield at least one policy with is_winner=1
    (the policy that achieved the oracle). If zero policies are
    flagged, the empirical-oracle floor became disconnected from
    the row miss rates."""
    seen = set()
    winners_per_cell: dict[tuple, int] = {}
    for r in report["rows"]:
        key = (r["graph"], r["app"], r["l3_size"])
        seen.add(key)
        winners_per_cell[key] = winners_per_cell.get(key, 0) + (
            1 if r["is_winner"] == "1" else 0
        )
    no_winner = [k for k, v in winners_per_cell.items() if v == 0]
    assert not no_winner, f"{len(no_winner)} cells have no winner row"
    assert len(seen) == report["summary"]["n_cells"]


def test_popt_has_smallest_overall_mean_gap(report):
    """Load-bearing paper claim: across the whole corpus POPT is
    closest to the empirical oracle on average. If GRASP or SRRIP
    ever overtakes POPT here, the headline 'POPT defines the
    achievable floor' statement is wrong and must be revisited."""
    means = {
        p: report["summary"]["overall_by_policy"][p]["mean"]
        for p in POLICIES
    }
    best_policy = min(means, key=lambda p: means[p])
    assert best_policy == "POPT", (
        f"expected POPT to have the smallest mean oracle gap; "
        f"got means={means}, best={best_policy}"
    )


def test_grasp_road_family_gap_is_large(report):
    """Counterpart to the POPT-vs-GRASP delta gate: GRASP on the
    road family must show a large gap to the empirical oracle.
    Threshold 5 pp is well below the observed ~12 pp mean — if
    this fails the road-graph counter-narrative has eroded."""
    s = report["summary"]["by_policy_family"].get("GRASP/road")
    assert s is not None, "GRASP/road bin missing"
    assert s["mean"] >= 5.0, (
        f"expected GRASP/road mean gap ≥ 5 pp; got {s['mean']} (n={s['n']})"
    )


def test_road_family_present(report):
    """Sanity: the road family must contribute rows to the report.
    Without it the GRASP-vs-road narrative becomes invisible."""
    fams = {
        k.split("/", 1)[1]
        for k in report["summary"]["by_policy_family"].keys()
    }
    assert "road" in fams, "road family missing from oracle-gap breakdown"
    assert "mesh" in fams, "mesh family missing from oracle-gap breakdown"


def test_overall_n_matches_rows_per_policy(report):
    """For each policy, the n in overall_by_policy must equal the
    number of rows with that policy. Catches a slice/group-by drift
    between the per-policy aggregation and the raw row stream."""
    counts: dict[str, int] = {}
    for r in report["rows"]:
        counts[r["policy"]] = counts.get(r["policy"], 0) + 1
    for p in POLICIES:
        expected = counts.get(p, 0)
        got = report["summary"]["overall_by_policy"][p]["n"]
        assert got == expected, (
            f"policy {p}: overall n={got} but {expected} rows in stream"
        )
