"""Gate 138+ — per_graph_rollup arithmetic + meta count parity.

per_graph_app_stability.json includes a per_graph_rollup section that
counts how many of each graph's (graph,app) cells fall into each
stability bucket. It also exposes meta.n_stable_unique /
n_stable_partial / n_regime_change / n_insufficient_l3 which roll
up the per_graph_app classifications across the whole corpus.

Gate 135 (winner identification parity) locked the per-cell
classification decisions but NOT the per-graph and per-corpus
rollups. This gate closes that gap: if the generator silently
shifts a classification into the wrong rollup bucket, or the meta
counts drift from per_graph_app, this gate flips.

Critical generator subtlety (lines 145-150 of per_graph_app_stability.py):
the rollup MERGES 'stable_unique' AND 'stable_unique_with_ties' into
the single 'stable_unique' bucket. The meta count meta.n_stable_unique
ALSO sums both. The test mirrors this merge.

Invariants (18 tests, 5 groups):

Group A — per_graph_rollup structural
  1. per_graph_rollup keys == set(graph for each per_graph_app entry)
  2. Every rollup entry has exactly {n_apps, stable_unique,
     regime_change, partial}

Group B — Per-graph count reproduction (per_graph_app → per_graph_rollup)
  3. rollup[g].n_apps == count of per_graph_app entries with graph==g
  4. rollup[g].stable_unique == count of per_graph_app entries with
     classification ∈ {stable_unique, stable_unique_with_ties}
     for graph==g
  5. rollup[g].regime_change == count of per_graph_app entries with
     classification == regime_change for graph==g
  6. rollup[g].partial == count of per_graph_app entries with
     classification == stable_partial for graph==g
  7. (stable_unique + regime_change + partial) <= n_apps (insufficient
     cells are NOT tallied into any bucket but count toward n_apps)
  8. The leftover (n_apps - sum-of-3-buckets) equals count of
     classification == insufficient_l3 for that graph

Group C — Corpus-level meta count reproduction
  9. meta.n_graph_app_pairs == len(per_graph_app)
  10. meta.n_stable_unique == sum of rollup[g].stable_unique
      ALSO == count of per_graph_app entries with classification
      ∈ {stable_unique, stable_unique_with_ties}
  11. meta.n_stable_partial == sum of rollup[g].partial
      ALSO == count of per_graph_app entries with
      classification == stable_partial
  12. meta.n_regime_change == sum of rollup[g].regime_change
      ALSO == count of per_graph_app entries with
      classification == regime_change
  13. meta.n_insufficient_l3 == count of per_graph_app entries with
      classification == insufficient_l3
  14. (4 meta counts) sum to meta.n_graph_app_pairs

Group D — meta.stability_fraction_excl_insufficient
  15. stability_fraction_excl_insufficient == round(n_stable_unique
      / max(n_graph_app_pairs - n_insufficient_l3, 1), 3)

Group E — Headline list count parity
  16. len(stable_unique_cells) ==
      n_stable_unique + n_stable_partial - n_stable_partial
      i.e. exactly n_stable_unique (since stable_partial has its own
      list). Tests: len(stable_unique_cells) == meta.n_stable_unique.
  17. len(stable_partial_cells) == meta.n_stable_partial
  18. len(regime_change_cells) == meta.n_regime_change AND
      len(insufficient_cells) == meta.n_insufficient_l3
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

STABLE_UNIQUE_SET = {"stable_unique", "stable_unique_with_ties"}


@pytest.fixture(scope="module")
def pgas() -> dict:
    return json.loads((WIKI_DATA / "per_graph_app_stability.json").read_text())


@pytest.fixture(scope="module")
def by_graph_cls(pgas) -> dict:
    """{graph: Counter(classification)}"""
    d: dict[str, Counter] = {}
    for r in pgas["per_graph_app"]:
        d.setdefault(r["graph"], Counter())[r["classification"]] += 1
    return d


# ─── Group A — Structural ────────────────────────────────────────────


def test_per_graph_rollup_keys_match_per_graph_app_graphs(pgas):
    graphs_in_pga = {r["graph"] for r in pgas["per_graph_app"]}
    assert set(pgas["per_graph_rollup"].keys()) == graphs_in_pga


def test_per_graph_rollup_entries_have_locked_keys(pgas):
    expected = {"n_apps", "stable_unique", "regime_change", "partial"}
    bad = [
        (g, set(r.keys()))
        for g, r in pgas["per_graph_rollup"].items()
        if set(r.keys()) != expected
    ]
    assert not bad, bad


# ─── Group B — Per-graph count reproduction ──────────────────────────


def test_rollup_n_apps_matches_per_graph_app_count(pgas, by_graph_cls):
    mism = []
    for g, rollup in pgas["per_graph_rollup"].items():
        expected = sum(by_graph_cls[g].values())
        if rollup["n_apps"] != expected:
            mism.append((g, rollup["n_apps"], expected))
    assert not mism, mism


def test_rollup_stable_unique_merges_with_ties(pgas, by_graph_cls):
    mism = []
    for g, rollup in pgas["per_graph_rollup"].items():
        expected = sum(by_graph_cls[g].get(c, 0) for c in STABLE_UNIQUE_SET)
        if rollup["stable_unique"] != expected:
            mism.append((g, rollup["stable_unique"], expected))
    assert not mism, mism


def test_rollup_regime_change_matches(pgas, by_graph_cls):
    mism = []
    for g, rollup in pgas["per_graph_rollup"].items():
        expected = by_graph_cls[g].get("regime_change", 0)
        if rollup["regime_change"] != expected:
            mism.append((g, rollup["regime_change"], expected))
    assert not mism, mism


def test_rollup_partial_matches_stable_partial(pgas, by_graph_cls):
    mism = []
    for g, rollup in pgas["per_graph_rollup"].items():
        expected = by_graph_cls[g].get("stable_partial", 0)
        if rollup["partial"] != expected:
            mism.append((g, rollup["partial"], expected))
    assert not mism, mism


def test_rollup_buckets_bounded_by_n_apps(pgas):
    bad = []
    for g, rollup in pgas["per_graph_rollup"].items():
        s = rollup["stable_unique"] + rollup["regime_change"] + rollup["partial"]
        if s > rollup["n_apps"]:
            bad.append((g, s, rollup["n_apps"]))
    assert not bad, bad


def test_rollup_residual_equals_insufficient_count(pgas, by_graph_cls):
    """n_apps - (3 buckets) == count of insufficient_l3 for that graph."""
    mism = []
    for g, rollup in pgas["per_graph_rollup"].items():
        residual = rollup["n_apps"] - (
            rollup["stable_unique"] + rollup["regime_change"] + rollup["partial"]
        )
        expected = by_graph_cls[g].get("insufficient_l3", 0)
        if residual != expected:
            mism.append((g, residual, expected))
    assert not mism, mism


# ─── Group C — Corpus-level meta counts ──────────────────────────────


def test_meta_n_graph_app_pairs_matches_per_graph_app_len(pgas):
    assert pgas["meta"]["n_graph_app_pairs"] == len(pgas["per_graph_app"])


def test_meta_n_stable_unique_double_consistent(pgas, by_graph_cls):
    """meta.n_stable_unique == sum(rollup[g].stable_unique)
    AND == count of per_graph_app entries with cls ∈ stable_unique_set."""
    via_rollup = sum(r["stable_unique"] for r in pgas["per_graph_rollup"].values())
    via_pga = sum(
        by_graph_cls[g].get(c, 0)
        for g in by_graph_cls
        for c in STABLE_UNIQUE_SET
    )
    assert pgas["meta"]["n_stable_unique"] == via_rollup == via_pga


def test_meta_n_stable_partial_double_consistent(pgas, by_graph_cls):
    via_rollup = sum(r["partial"] for r in pgas["per_graph_rollup"].values())
    via_pga = sum(c.get("stable_partial", 0) for c in by_graph_cls.values())
    assert pgas["meta"]["n_stable_partial"] == via_rollup == via_pga


def test_meta_n_regime_change_double_consistent(pgas, by_graph_cls):
    via_rollup = sum(r["regime_change"] for r in pgas["per_graph_rollup"].values())
    via_pga = sum(c.get("regime_change", 0) for c in by_graph_cls.values())
    assert pgas["meta"]["n_regime_change"] == via_rollup == via_pga


def test_meta_n_insufficient_l3_matches_pga(pgas, by_graph_cls):
    via_pga = sum(c.get("insufficient_l3", 0) for c in by_graph_cls.values())
    assert pgas["meta"]["n_insufficient_l3"] == via_pga


def test_meta_four_counts_sum_to_total(pgas):
    total = (
        pgas["meta"]["n_stable_unique"]
        + pgas["meta"]["n_stable_partial"]
        + pgas["meta"]["n_regime_change"]
        + pgas["meta"]["n_insufficient_l3"]
    )
    assert total == pgas["meta"]["n_graph_app_pairs"]


# ─── Group D — stability_fraction_excl_insufficient ─────────────────


def test_stability_fraction_reproduces(pgas):
    n_stable = pgas["meta"]["n_stable_unique"]
    denom = max(
        pgas["meta"]["n_graph_app_pairs"] - pgas["meta"]["n_insufficient_l3"], 1
    )
    expected = round(n_stable / denom, 3)
    got = pgas["meta"]["stability_fraction_excl_insufficient"]
    assert abs(got - expected) <= 1e-3, (got, expected)


# ─── Group E — Headline list count parity ────────────────────────────


def test_stable_unique_cells_count_matches_meta(pgas):
    assert len(pgas["stable_unique_cells"]) == pgas["meta"]["n_stable_unique"]


def test_stable_partial_cells_count_matches_meta(pgas):
    assert len(pgas["stable_partial_cells"]) == pgas["meta"]["n_stable_partial"]


def test_regime_change_and_insufficient_cells_count_match_meta(pgas):
    assert len(pgas["regime_change_cells"]) == pgas["meta"]["n_regime_change"]
    assert len(pgas["insufficient_cells"]) == pgas["meta"]["n_insufficient_l3"]
