"""Pytest gate: WSS-relative L3 axis aggregator.

The paper's L3 axis is reported both in absolute bytes and in
WSS-relative units (L3 / WSS). This gate pins the WSS-relative
findings so a reviewer can't dismiss the absolute-byte tables as
unfairly comparing across graphs of wildly different sizes.

Load-bearing claim defended here:
  POPT has the smallest mean oracle gap in EVERY WSS regime
  (under_wss, near_wss, over_wss).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_JSON = REPO_ROOT / "wiki" / "data" / "wss_relative_l3.json"

REGIMES = ("under_wss", "near_wss", "over_wss")
POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")


@pytest.fixture(scope="module")
def doc() -> dict:
    if not DOC_JSON.exists():
        pytest.skip(f"{DOC_JSON} missing; run `make lit-wss-relative-l3`")
    return json.loads(DOC_JSON.read_text())


def test_top_level_schema(doc):
    expected = {"meta", "by_policy_regime", "per_regime_ranking",
                "per_regime_cell_count"}
    assert expected.issubset(doc.keys()), (
        f"missing keys: {expected - doc.keys()}"
    )


def test_all_twelve_buckets_present(doc):
    for pol in POLICIES:
        for regime in REGIMES:
            key = f"{pol}/{regime}"
            assert key in doc["by_policy_regime"], (
                f"missing bucket {key}"
            )


def test_each_regime_has_cells(doc):
    for regime in REGIMES:
        n = doc["per_regime_cell_count"].get(regime, 0)
        assert n >= 5, (
            f"regime `{regime}` has only {n} cells; below 5-cell floor "
            "(would make per-regime aggregates noise)"
        )


def test_cell_counts_sum_to_meta_total(doc):
    total = sum(doc["per_regime_cell_count"].values())
    assert total == doc["meta"]["n_cells_classified"], (
        f"cell-count books don't balance: "
        f"regime sum {total} != meta n_cells_classified "
        f"{doc['meta']['n_cells_classified']}"
    )


def test_popt_smallest_mean_gap_in_every_regime(doc):
    """Load-bearing paper claim: POPT has the smallest mean oracle
    gap in EVERY WSS regime (under, near, over).  If this ever
    breaks, the WSS-relative axis stops supporting the headline."""
    for regime in REGIMES:
        ranking = doc["per_regime_ranking"][regime]
        if not ranking:
            pytest.fail(f"empty ranking for regime {regime}")
        rank1 = ranking[0]
        assert rank1["policy"] == "POPT", (
            f"WSS regime `{regime}` rank-1 policy by mean gap = "
            f"{rank1['policy']!r} (mean {rank1['mean_gap_pp']} pp); "
            "expected POPT. The WSS-relative axis no longer supports "
            "the POPT-mean-smallest headline."
        )


def test_lru_is_never_rank_1(doc):
    """LRU should never beat all paper-grade policies on mean gap
    in any WSS regime."""
    for regime in REGIMES:
        rank1 = doc["per_regime_ranking"][regime][0]
        assert rank1["policy"] != "LRU", (
            f"WSS regime `{regime}` rank-1 = LRU — no paper-grade "
            "policy beats baseline; possible regression"
        )


def test_means_are_non_negative(doc):
    for k, v in doc["by_policy_regime"].items():
        m = v.get("mean_gap_pp")
        assert m is None or m >= -1e-6, (
            f"{k} mean gap {m} < 0 — oracle gap cannot be negative"
        )


def test_win_rates_in_unit_interval(doc):
    for k, v in doc["by_policy_regime"].items():
        wr = v.get("win_rate")
        if wr is None:
            continue
        assert 0.0 <= wr <= 1.0, (
            f"{k} win_rate {wr} outside [0, 1]"
        )


def test_per_graph_wss_proxies_present(doc):
    """Sanity: WSS proxies are recorded for every graph in the
    corpus, so a reviewer can audit the binning."""
    proxies = doc["meta"]["wss_proxies"]
    expected = {
        "email-Eu-core", "web-Google", "cit-Patents",
        "soc-pokec", "soc-LiveJournal1", "com-orkut",
        "roadNet-CA", "delaunay_n19",
    }
    missing = expected - set(proxies.keys())
    assert not missing, (
        f"WSS proxy missing for graphs: {missing}. "
        "Did corpus_diversity.json get truncated?"
    )


def test_unknown_graphs_empty(doc):
    """No oracle-gap rows should reference graphs absent from the
    WSS proxy map — would silently bias the regime aggregates."""
    unknown = doc["meta"].get("unknown_graphs", [])
    assert not unknown, (
        f"oracle-gap rows reference graphs without WSS proxy: "
        f"{unknown}. Add to corpus_diversity or filter the input."
    )
