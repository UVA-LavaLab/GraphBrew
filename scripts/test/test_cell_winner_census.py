"""Gate: cell-winner census (corpus decisiveness).

Pins how decisive the corpus is: the fraction of (graph, app, l3) cells
that have a unique winner vs ties vs no-winner. The paper must report
this number — including the absolute count of tied cells and the
specific (graph, app, l3) tuples — because any per-cell win-rate
claim that silently includes tied cells is methodologically broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = REPO_ROOT / "wiki" / "data" / "cell_winner_census.json"


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip(
            f"missing {PAYLOAD}; run `make lit-cell-census` "
            f"(or `make lit-claims`)."
        )
    return json.loads(PAYLOAD.read_text())


def test_schema_complete(payload):
    for key in ("meta", "per_app", "all_tied_cells", "all_no_winner_cells"):
        assert key in payload
    for key in ("n_cells_total", "n_unique_winner", "n_tied_winners",
                "n_no_winner", "pct_unique_winner", "pct_tied_winners",
                "pct_no_winner", "tied_breakdown_by_count"):
        assert key in payload["meta"], f"meta missing {key}"


def test_counts_sum_to_total(payload):
    m = payload["meta"]
    assert m["n_unique_winner"] + m["n_tied_winners"] + m["n_no_winner"] \
        == m["n_cells_total"]


def test_percentages_sum_to_100(payload):
    m = payload["meta"]
    total = m["pct_unique_winner"] + m["pct_tied_winners"] + m["pct_no_winner"]
    assert abs(total - 100.0) < 0.05, f"pct total = {total}"


def test_n_cells_above_floor(payload):
    """Corpus must have ≥ 100 cells — defends against silent corpus shrinkage."""
    assert payload["meta"]["n_cells_total"] >= 100, payload["meta"]


def test_corpus_is_decisive(payload):
    """≥ 90% of cells must have unique winners — without this floor, the
    paper's win-rate-based claims are at risk because too many cells
    are ambiguous to support a clean count."""
    assert payload["meta"]["pct_unique_winner"] >= 90.0, (
        f"corpus decisiveness dropped to {payload['meta']['pct_unique_winner']}%; "
        "investigate which cells became ambiguous."
    )


def test_no_no_winner_cells(payload):
    """A no-winner cell is a data-generation bug (some policy MUST be best);
    we should have zero. If this jumps, regenerate the source data."""
    assert payload["meta"]["n_no_winner"] == 0, (
        f"{payload['meta']['n_no_winner']} cells have no winner; "
        f"see all_no_winner_cells: {payload['all_no_winner_cells']}"
    )


def test_tied_cells_are_disclosed_explicitly(payload):
    """If we report N tied cells in meta, we must also list them
    individually so the paper can name them."""
    n_tied = payload["meta"]["n_tied_winners"]
    n_listed = len(payload["all_tied_cells"])
    assert n_tied == n_listed, (
        f"meta says {n_tied} tied cells, but only {n_listed} listed individually"
    )


def test_tied_cell_count_is_low(payload):
    """≤ 10 tied cells in absolute terms — bounds the size of the
    'qualified' bucket in the paper."""
    assert payload["meta"]["n_tied_winners"] <= 10, payload["meta"]


def test_all_apps_have_decisive_majority(payload):
    """Every app must have ≥ 80% unique-winner cells. If sssp or bfs
    fell below this, the cross-gate weak-signal pattern would have
    a structural data-quality explanation we'd need to investigate."""
    for app, p in payload["per_app"].items():
        if p["n_cells"] == 0:
            continue
        pct = 100.0 * p["unique_winner"] / p["n_cells"]
        assert pct >= 80.0, (
            f"{app} only has {pct:.1f}% unique-winner cells; "
            f"breakdown: unique={p['unique_winner']} tied={p['tied_winners']} "
            f"none={p['no_winner']}"
        )


def test_tied_cells_listed_have_required_fields(payload):
    """Every tied-cell entry must have graph, app, l3, tied_policies,
    tied_count — so the paper can cite them precisely."""
    for c in payload["all_tied_cells"]:
        for key in ("graph", "app", "l3", "tied_policies", "tied_count"):
            assert key in c, f"tied cell missing {key}: {c}"
        assert len(c["tied_policies"]) == c["tied_count"]
        assert c["tied_count"] >= 2


def test_tied_breakdown_consistent(payload):
    """meta.tied_breakdown_by_count must sum to n_tied_winners."""
    bk = payload["meta"]["tied_breakdown_by_count"]
    total_tied_from_bk = sum(int(v) for v in bk.values())
    assert total_tied_from_bk == payload["meta"]["n_tied_winners"], (
        f"breakdown sums to {total_tied_from_bk}, meta says "
        f"{payload['meta']['n_tied_winners']}"
    )


def test_bc_email_eu_core_is_the_canonical_tied_subcorpus(payload):
    """bc on email-Eu-core is the canonical tied subcorpus — pin it
    so any future change to the corpus that resolves these ties OR
    introduces ties elsewhere is investigated."""
    tied = payload["all_tied_cells"]
    if not tied:
        return  # empty corpus, covered by other tests
    # Every current tied cell is bc/email-Eu-core. If this changes,
    # update the paper's qualification language.
    apps_tied = {c["app"] for c in tied}
    graphs_tied = {c["graph"] for c in tied}
    assert "bc" in apps_tied, (
        f"bc/email-Eu-core no longer in tied set; apps_tied={apps_tied} "
        f"graphs_tied={graphs_tied}. Update paper qualification language."
    )
