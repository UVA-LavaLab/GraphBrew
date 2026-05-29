"""Gate 145+ — cell_winner_census derivation parity.

cell_winner_census.json is derived from oracle_gap.json by:

  1. Group rows by (graph, app, l3_size) → each group = one cell
  2. Per cell: collect is_winner=='1' rows → winner_policies (set)
  3. Classify: no_winner (0 winners) / unique_winner (1) / tied_winners (>=2)
  4. Per-app aggregation: count by class; collect tied_cells +
     no_winner_cells lists (cell payloads with policies and tied count)
  5. Corpus-level meta: n_total, n_unique, n_tied, n_none, pct (×100,
     rounded to 2 dp), tied_breakdown_by_count (Counter of tied widths)

Why this gate matters
---------------------
This artifact pins corpus decisiveness — the headline number 'X% of
cells have a unique winner; remaining are tied between policies and
reported separately'. If we ever silently include tied cells in
win-rate counts, that is a credibility-killing methodology bug.

Load-bearing rules:
- is_winner comparison uses STRING '1' (NOT bool / int — the generator
  treats values literally as strings). Mismatch on the string-vs-int
  question would silently miscount everything downstream.
- Cell key = (graph, app, l3_size); l3_size is NOT restricted to paper
  L3 sizes — this gate counts ALL cells (e.g. probe L3 sizes too).
  Note: meta has n_cells_total=114, not 60 (paper grid).
- pct values use round(100*x/n, 2) — different from cohens_h's 4dp.
- tied_breakdown_by_count uses INTEGER counts as keys (Counter), but
  serialized to JSON those become string keys.

Invariants (19 tests, 5 groups):

Group A — Structural & scope
  1. Top-level keys: meta, per_app, all_tied_cells, all_no_winner_cells
  2. n_cells_total == sum of per_app n_cells
  3. n_unique + n_tied + n_none == n_cells_total
  4. per_app keys are subset of 5 known apps
  5. all_tied_cells len == n_tied_winners; all_no_winner_cells len == n_no_winner

Group B — Cell classification (oracle_gap → per-cell winner status)
  6. For every cell in oracle_gap: count of is_winner=='1' policies
     matches generator classification (no/unique/tied)
  7. per_app[app] n_cells == count of distinct (graph, l3) cells for app
  8. per_app[app] sum(unique+tied+none) == n_cells per app
  9. Tied cell payloads have tied_count == len(tied_policies)
     and tied_policies sorted ascending and tied_count >= 2

Group C — Cross-list consistency
  10. all_tied_cells == flattened union of per_app[*].tied_cells (same set)
  11. all_no_winner_cells == flattened union of per_app[*].no_winner_cells
  12. Every tied cell appears in exactly ONE per_app bucket (matches its app field)
  13. No cell classified as both tied AND no-winner (mutually exclusive)

Group D — Percentage + breakdown reproduction
  14. pct_unique_winner == round(100 * n_unique / n_total, 2)
  15. pct_tied_winners == round(100 * n_tied / n_total, 2)
  16. pct_no_winner == round(100 * n_none / n_total, 2)
  17. tied_breakdown_by_count: sum(values) == n_tied; keys cover the
      observed tied_count distribution exactly

Group E — Headline consistency with paper claim
  18. pct_unique_winner is the decisiveness number — must be in (50, 100]
      (paper claim that corpus is mostly decisive)
  19. all_tied_cells entries have valid (app, graph, l3, tied_count, tied_policies)
      shape; tied_policies subset of {LRU, SRRIP, GRASP, POPT}
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

VALID_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}
KNOWN_APPS = {"pr", "bc", "cc", "bfs", "sssp"}


@pytest.fixture(scope="module")
def og() -> dict:
    return json.loads((WIKI_DATA / "oracle_gap.json").read_text())


@pytest.fixture(scope="module")
def cwc() -> dict:
    return json.loads((WIKI_DATA / "cell_winner_census.json").read_text())


@pytest.fixture(scope="module")
def cells_by_winners(og) -> dict:
    """{(graph, app, l3): sorted-unique winner policies via is_winner=='1'}."""
    cells = defaultdict(list)
    for r in og["rows"]:
        cells[(r["graph"], r["app"], r["l3_size"])].append(r)
    out = {}
    for key, rs in cells.items():
        winners = sorted({r["policy"] for r in rs if r.get("is_winner") == "1"})
        out[key] = winners
    return out


# ─── Group A — Structural ────────────────────────────────────────────


def test_top_level_keys(cwc):
    assert set(cwc.keys()) >= {"meta", "per_app", "all_tied_cells", "all_no_winner_cells"}


def test_n_cells_total_equals_sum_per_app(cwc):
    s = sum(p["n_cells"] for p in cwc["per_app"].values())
    assert cwc["meta"]["n_cells_total"] == s


def test_meta_class_counts_partition_total(cwc):
    m = cwc["meta"]
    assert m["n_unique_winner"] + m["n_tied_winners"] + m["n_no_winner"] == m["n_cells_total"]


def test_per_app_keys_subset_of_known(cwc):
    assert set(cwc["per_app"].keys()) <= KNOWN_APPS


def test_global_lists_length_match_counts(cwc):
    assert len(cwc["all_tied_cells"]) == cwc["meta"]["n_tied_winners"]
    assert len(cwc["all_no_winner_cells"]) == cwc["meta"]["n_no_winner"]


# ─── Group B — Cell classification ───────────────────────────────────


def test_per_cell_classification_matches_oracle(cwc, cells_by_winners):
    """Recompute classification independently; tied counts must match."""
    exp_unique = exp_tied = exp_none = 0
    for winners in cells_by_winners.values():
        if not winners:
            exp_none += 1
        elif len(winners) == 1:
            exp_unique += 1
        else:
            exp_tied += 1
    m = cwc["meta"]
    assert m["n_unique_winner"] == exp_unique
    assert m["n_tied_winners"] == exp_tied
    assert m["n_no_winner"] == exp_none


def test_per_app_n_cells_count_distinct_graph_l3(cwc, cells_by_winners):
    by_app = defaultdict(int)
    for (graph, app, l3) in cells_by_winners:
        by_app[app] += 1
    mism = []
    for app, payload in cwc["per_app"].items():
        if payload["n_cells"] != by_app.get(app, 0):
            mism.append((app, payload["n_cells"], by_app.get(app, 0)))
    assert not mism, mism


def test_per_app_sum_of_classes_equals_n_cells(cwc):
    mism = []
    for app, p in cwc["per_app"].items():
        s = p["unique_winner"] + p["tied_winners"] + p["no_winner"]
        if s != p["n_cells"]:
            mism.append((app, s, p["n_cells"]))
    assert not mism, mism


def test_tied_cell_payload_invariants(cwc):
    bad = []
    for tc in cwc["all_tied_cells"]:
        if tc["tied_count"] != len(tc["tied_policies"]):
            bad.append(("count_mismatch", tc))
        if tc["tied_count"] < 2:
            bad.append(("low_count", tc))
        if tc["tied_policies"] != sorted(tc["tied_policies"]):
            bad.append(("not_sorted", tc))
    assert not bad, bad


# ─── Group C — Cross-list consistency ───────────────────────────────


def _tied_key(tc):
    return (tc["app"], tc["graph"], tc["l3"], tuple(tc["tied_policies"]))


def test_all_tied_equals_union_of_per_app(cwc):
    global_keys = sorted(_tied_key(tc) for tc in cwc["all_tied_cells"])
    per_app_keys = sorted(
        _tied_key(tc)
        for p in cwc["per_app"].values()
        for tc in p["tied_cells"]
    )
    assert global_keys == per_app_keys


def test_all_no_winner_equals_union_of_per_app(cwc):
    global_keys = sorted(
        (c["app"], c["graph"], c["l3"]) for c in cwc["all_no_winner_cells"]
    )
    per_app_keys = sorted(
        (c["app"], c["graph"], c["l3"])
        for p in cwc["per_app"].values()
        for c in p["no_winner_cells"]
    )
    assert global_keys == per_app_keys


def test_tied_cells_each_in_exactly_one_app_bucket(cwc):
    bad = []
    for tc in cwc["all_tied_cells"]:
        app = tc["app"]
        bucket = cwc["per_app"].get(app, {}).get("tied_cells", [])
        matches = [b for b in bucket if (b["graph"], b["l3"]) == (tc["graph"], tc["l3"])]
        if len(matches) != 1:
            bad.append((tc, len(matches)))
    assert not bad, bad


def test_tied_and_no_winner_sets_disjoint(cwc):
    tied_keys = {(tc["app"], tc["graph"], tc["l3"]) for tc in cwc["all_tied_cells"]}
    none_keys = {(c["app"], c["graph"], c["l3"]) for c in cwc["all_no_winner_cells"]}
    overlap = tied_keys & none_keys
    assert not overlap, overlap


# ─── Group D — Percentage + breakdown reproduction ──────────────────


def test_pct_unique_winner_reproduces(cwc):
    m = cwc["meta"]
    exp = round(100.0 * m["n_unique_winner"] / m["n_cells_total"], 2) if m["n_cells_total"] else 0
    assert m["pct_unique_winner"] == exp


def test_pct_tied_winners_reproduces(cwc):
    m = cwc["meta"]
    exp = round(100.0 * m["n_tied_winners"] / m["n_cells_total"], 2) if m["n_cells_total"] else 0
    assert m["pct_tied_winners"] == exp


def test_pct_no_winner_reproduces(cwc):
    m = cwc["meta"]
    exp = round(100.0 * m["n_no_winner"] / m["n_cells_total"], 2) if m["n_cells_total"] else 0
    assert m["pct_no_winner"] == exp


def test_tied_breakdown_by_count_matches_observed(cwc):
    """Sum of breakdown values == n_tied; keys == distinct tied_count values seen."""
    breakdown = cwc["meta"]["tied_breakdown_by_count"]
    assert sum(int(v) for v in breakdown.values()) == cwc["meta"]["n_tied_winners"]
    observed = Counter(str(tc["tied_count"]) for tc in cwc["all_tied_cells"])
    # Breakdown keys serialize as strings; reproduce same dict
    assert {str(k): int(v) for k, v in breakdown.items()} == dict(observed)


# ─── Group E — Headline + shape sanity ──────────────────────────────


def test_pct_unique_winner_in_decisive_range(cwc):
    """Paper claim: corpus is mostly decisive (pct_unique_winner > 50%)."""
    assert 50.0 < cwc["meta"]["pct_unique_winner"] <= 100.0


def test_tied_cells_have_valid_policy_set(cwc):
    bad = []
    for tc in cwc["all_tied_cells"]:
        if not set(tc["tied_policies"]).issubset(VALID_POLICIES):
            bad.append(tc)
        for required in ("app", "graph", "l3", "tied_count", "tied_policies"):
            if required not in tc:
                bad.append((required, tc))
    assert not bad, bad
