"""Derivation parity gate for ``wiki/data/wss_relative_l3.json``.

Locks the WSS-relative L3 axis aggregator (gate 75) against its two
upstream sources so any silent drift in the per-graph WSS proxy
reducer, the L3/WSS regime classifier, the per-(policy, regime)
aggregator, or the ranking reducer trips a test before the dashboard
re-publishes the WSS-axis story:

    oracle_gap.json#rows                                  → per-cell gap rows
    corpus_diversity.json[*].features.working_set_ratio   → per-graph WSS proxy
                  │
            wss_relative_l3.py (_load_wss_map + _aggregate +
                                _per_regime_ranking)
                  │
                  ▼
    wiki/data/wss_relative_l3.json   ← gate target

This artifact is the headline upstream for gate 161
(winner_margin_by_regime) and the paper's defense against the
"absolute L3 bytes obscure cross-graph comparisons" reviewer
critique. The gated story: WSS-relative regimes (under_wss / near_wss
/ over_wss) produce ranked winner tables, with POPT/GRASP dominating
under_wss and the picture shifting in over_wss.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "wss_relative_l3.json"
ORACLE_PATH = WIKI_DATA / "oracle_gap.json"
CORPUS_PATH = WIKI_DATA / "corpus_diversity.json"

# Pinned mirror of generator constants.
POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")
REGIMES = ("under_wss", "near_wss", "over_wss")
WSS_REFERENCE_L3 = 1 * 1024 * 1024
L3_SIZE_BYTES = {
    "4kB":   4 * 1024,
    "8kB":   8 * 1024,
    "16kB":  16 * 1024,
    "32kB":  32 * 1024,
    "64kB":  64 * 1024,
    "128kB": 128 * 1024,
    "256kB": 256 * 1024,
    "512kB": 512 * 1024,
    "1MB":   1 * 1024 * 1024,
    "2MB":   2 * 1024 * 1024,
    "4MB":   4 * 1024 * 1024,
    "8MB":   8 * 1024 * 1024,
    "16MB": 16 * 1024 * 1024,
}


def _wss_regime(ratio: float) -> str:
    if ratio < 0.25:
        return "under_wss"
    if ratio > 4.0:
        return "over_wss"
    return "near_wss"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def oracle_doc() -> dict:
    if not ORACLE_PATH.exists():
        pytest.skip(f"missing {ORACLE_PATH}")
    return json.loads(ORACLE_PATH.read_text())


@pytest.fixture(scope="module")
def corpus_doc() -> list:
    if not CORPUS_PATH.exists():
        pytest.skip(f"missing {CORPUS_PATH}")
    raw = json.loads(CORPUS_PATH.read_text())
    if isinstance(raw, dict):
        raw = raw.get("graphs", []) or raw.get("rows", [])
    return raw


@pytest.fixture(scope="module")
def upstream_wss_map(corpus_doc) -> dict[str, float]:
    """Replicate ``_load_wss_map`` byte-exact."""
    out: dict[str, float] = {}
    for e in corpus_doc:
        graph = e.get("graph")
        wsr = (e.get("features") or {}).get("working_set_ratio")
        if graph is not None and wsr is not None:
            out[graph] = float(wsr) * WSS_REFERENCE_L3
    return out


@pytest.fixture(scope="module")
def reconstructed(oracle_doc, upstream_wss_map):
    """Replicate ``_aggregate`` over the same upstream inputs."""
    rows = oracle_doc.get("rows", [])
    cells: dict[tuple, list] = defaultdict(list)
    for r in rows:
        cells[(r["graph"], r["app"], r["l3_size"])].append(r)
    by_pr: dict[tuple, list[float]] = defaultdict(list)
    cells_with_winner: dict[str, set] = defaultdict(set)
    wins_by_pr: dict[tuple, int] = defaultdict(int)
    skipped = 0
    for (graph, app, l3), pr in cells.items():
        wss_b = upstream_wss_map.get(graph)
        if wss_b is None:
            skipped += len(pr)
            continue
        l3_b = L3_SIZE_BYTES.get(l3)
        if l3_b is None:
            skipped += len(pr)
            continue
        if not (wss_b > 0):
            skipped += len(pr)
            continue
        regime = _wss_regime(l3_b / wss_b)
        try:
            winner = min(pr, key=lambda r: float(r["gap_pp"]))["policy"]
        except Exception:
            winner = None
        cells_with_winner[regime].add((graph, app, l3))
        if winner:
            wins_by_pr[(winner, regime)] += 1
        for r in pr:
            pol = r["policy"]
            if pol not in POLICIES:
                continue
            try:
                gap = float(r["gap_pp"])
            except (TypeError, ValueError):
                continue
            by_pr[(pol, regime)].append(gap)
    return {
        "by_pr": by_pr,
        "cells_with_winner": cells_with_winner,
        "wins_by_pr": wins_by_pr,
        "skipped": skipped,
    }


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {
        "meta", "by_policy_regime", "per_regime_ranking",
        "per_regime_cell_count",
    }


def test_meta_carries_canonical_fields(artifact):
    expected = {
        "n_cells_classified", "n_cells_skipped",
        "wss_reference_bytes", "wss_proxies", "unknown_graphs",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing fields: {missing}"


def test_wss_reference_bytes_pinned(artifact):
    assert artifact["meta"]["wss_reference_bytes"] == WSS_REFERENCE_L3, (
        "WSS reference L3 drifted from 1 MiB — working_set_ratio is "
        "defined as WSS_at_1MB_PR / 1MB, so the reference scale is "
        "load-bearing."
    )


def test_by_policy_regime_keys_complete(artifact):
    expected = {f"{p}/{r}" for p in POLICIES for r in REGIMES}
    assert set(artifact["by_policy_regime"].keys()) == expected


def test_by_policy_regime_entry_shape(artifact):
    expected_keys = {
        "policy", "wss_regime", "n", "mean_gap_pp", "median_gap_pp",
        "p90_gap_pp", "n_cells_in_regime", "wins", "win_rate",
    }
    for k, e in artifact["by_policy_regime"].items():
        missing = expected_keys - set(e.keys())
        assert not missing, f"{k}: by_policy_regime entry missing {missing}"


def test_per_regime_ranking_keys_match_regimes(artifact):
    assert set(artifact["per_regime_ranking"].keys()) == set(REGIMES)
    for regime, ranking in artifact["per_regime_ranking"].items():
        assert len(ranking) == len(POLICIES), (
            f"per_regime_ranking.{regime} length {len(ranking)} ≠ "
            f"{len(POLICIES)} policies"
        )


def test_per_regime_cell_count_keys_match_regimes(artifact):
    assert set(artifact["per_regime_cell_count"].keys()) == set(REGIMES)


# ----------------------------------------------------------------------
# Group B: WSS-proxy cross-source parity
# ----------------------------------------------------------------------

def test_wss_proxies_match_upstream(artifact, upstream_wss_map):
    """Every wss_proxies[graph] equals round(working_set_ratio * 1MB, 2)."""
    for g, raw_bytes in upstream_wss_map.items():
        expected = round(raw_bytes, 2)
        got = artifact["meta"]["wss_proxies"].get(g)
        assert got == expected, (
            f"{g}: wss_proxies drift — recomputed {expected!r}, got {got!r}"
        )


def test_wss_proxies_count_matches_upstream(artifact, upstream_wss_map):
    assert len(artifact["meta"]["wss_proxies"]) == len(upstream_wss_map), (
        "wss_proxies count diverges from upstream working_set_ratio "
        "entries — silent corpus-diversity expansion would land here."
    )


def test_unknown_graphs_is_sorted_and_disjoint(artifact, upstream_wss_map):
    unknown = artifact["meta"]["unknown_graphs"]
    assert unknown == sorted(unknown), "unknown_graphs not sorted"
    for g in unknown:
        assert g not in upstream_wss_map, (
            f"{g} listed as unknown but is in upstream wss_map"
        )


# ----------------------------------------------------------------------
# Group C: cell aggregation cross-source parity
# ----------------------------------------------------------------------

def test_n_cells_classified_matches_per_regime_sum(artifact):
    expected = sum(artifact["per_regime_cell_count"].values())
    assert artifact["meta"]["n_cells_classified"] == expected, (
        "n_cells_classified inconsistent with per_regime_cell_count sum"
    )


def test_n_cells_classified_matches_recomputation(artifact, reconstructed):
    expected = sum(
        len(s) for s in reconstructed["cells_with_winner"].values()
    )
    assert artifact["meta"]["n_cells_classified"] == expected


def test_n_cells_skipped_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["n_cells_skipped"] == reconstructed["skipped"]


def test_per_regime_cell_count_matches_recomputation(artifact, reconstructed):
    for regime in REGIMES:
        expected = len(reconstructed["cells_with_winner"].get(regime, set()))
        assert artifact["per_regime_cell_count"][regime] == expected


# ----------------------------------------------------------------------
# Group D: per-(policy, regime) reducer cross-source parity
# ----------------------------------------------------------------------

def test_n_matches_upstream(artifact, reconstructed):
    for pol in POLICIES:
        for regime in REGIMES:
            vals = reconstructed["by_pr"].get((pol, regime), [])
            got = artifact["by_policy_regime"][f"{pol}/{regime}"]["n"]
            assert got == len(vals)


def test_mean_gap_pp_matches_upstream(artifact, reconstructed):
    for pol in POLICIES:
        for regime in REGIMES:
            vals = reconstructed["by_pr"].get((pol, regime), [])
            expected = round(statistics.fmean(vals), 4) if vals else None
            got = artifact["by_policy_regime"][f"{pol}/{regime}"]["mean_gap_pp"]
            assert got == expected, (
                f"{pol}/{regime}: mean_gap_pp drift — "
                f"recomputed {expected!r}, got {got!r}"
            )


def test_median_gap_pp_matches_upstream(artifact, reconstructed):
    for pol in POLICIES:
        for regime in REGIMES:
            vals = reconstructed["by_pr"].get((pol, regime), [])
            expected = round(statistics.median(vals), 4) if vals else None
            got = artifact["by_policy_regime"][f"{pol}/{regime}"]["median_gap_pp"]
            assert got == expected


def test_p90_gap_pp_matches_upstream(artifact, reconstructed):
    """Generator uses ``statistics.quantiles(vals, n=10)[-1]`` and
    requires n >= 10, else None.
    """
    for pol in POLICIES:
        for regime in REGIMES:
            vals = reconstructed["by_pr"].get((pol, regime), [])
            if len(vals) >= 10:
                expected = round(statistics.quantiles(vals, n=10)[-1], 4)
            else:
                expected = None
            got = artifact["by_policy_regime"][f"{pol}/{regime}"]["p90_gap_pp"]
            assert got == expected, (
                f"{pol}/{regime}: p90_gap_pp drift — "
                f"recomputed {expected!r}, got {got!r}"
            )


def test_wins_matches_upstream(artifact, reconstructed):
    for pol in POLICIES:
        for regime in REGIMES:
            expected = reconstructed["wins_by_pr"].get((pol, regime), 0)
            got = artifact["by_policy_regime"][f"{pol}/{regime}"]["wins"]
            assert got == expected


def test_n_cells_in_regime_matches_cell_count(artifact):
    """All four (policy, regime) entries inside a regime share the same
    n_cells_in_regime, which equals per_regime_cell_count[regime].
    """
    for regime in REGIMES:
        expected = artifact["per_regime_cell_count"][regime]
        for pol in POLICIES:
            got = artifact["by_policy_regime"][f"{pol}/{regime}"]["n_cells_in_regime"]
            assert got == expected, (
                f"{pol}/{regime}: n_cells_in_regime drift — "
                f"per_regime_cell_count={expected}, entry={got}"
            )


def test_win_rate_matches_wins_over_cells(artifact):
    """win_rate == round(wins / n_cells_in_regime, 4); None if no cells."""
    for k, e in artifact["by_policy_regime"].items():
        n_cells = e["n_cells_in_regime"]
        if n_cells == 0:
            assert e["win_rate"] is None
            continue
        expected = round(e["wins"] / n_cells, 4)
        assert e["win_rate"] == expected, (
            f"{k}: win_rate drift — recomputed {expected!r}, "
            f"got {e['win_rate']!r}"
        )


def test_wins_sum_per_regime_le_cell_count(artifact):
    """Sum of policy wins per regime ≤ regime cell count (= when every
    cell has exactly one winner; can be < if some cells in pr had no
    winner extractable).
    """
    for regime in REGIMES:
        total = sum(
            artifact["by_policy_regime"][f"{p}/{regime}"]["wins"]
            for p in POLICIES
        )
        cells = artifact["per_regime_cell_count"][regime]
        assert total <= cells, (
            f"{regime}: total wins {total} > cells {cells} — a cell "
            "produced more than one winner"
        )


# ----------------------------------------------------------------------
# Group E: per_regime_ranking sort + content parity
# ----------------------------------------------------------------------

def test_per_regime_ranking_entry_shape(artifact):
    expected = {"policy", "n", "mean_gap_pp", "win_rate", "wins"}
    for regime, ranking in artifact["per_regime_ranking"].items():
        for r in ranking:
            missing = expected - set(r.keys())
            assert not missing, f"{regime}: ranking entry missing {missing}"


def test_per_regime_ranking_includes_all_policies(artifact):
    for regime, ranking in artifact["per_regime_ranking"].items():
        assert {r["policy"] for r in ranking} == set(POLICIES)


def test_per_regime_ranking_sorted_ascending_mean_none_last(artifact):
    """Generator sorts by (mean is None, mean or 0) — i.e., ASC by
    mean_gap_pp with None entries pushed to the end.
    """
    for regime, ranking in artifact["per_regime_ranking"].items():
        keys = [(r["mean_gap_pp"] is None, r["mean_gap_pp"] or 0)
                for r in ranking]
        assert keys == sorted(keys), (
            f"{regime}: per_regime_ranking not sorted ASC by mean_gap_pp "
            f"(None last); got {[r['policy'] for r in ranking]}"
        )


def test_per_regime_ranking_fields_match_by_policy_regime(artifact):
    """Each entry in per_regime_ranking is a subset projection of the
    matching by_policy_regime entry.
    """
    for regime, ranking in artifact["per_regime_ranking"].items():
        for r in ranking:
            src = artifact["by_policy_regime"][f"{r['policy']}/{regime}"]
            assert r["n"] == src["n"]
            assert r["mean_gap_pp"] == src["mean_gap_pp"]
            assert r["win_rate"] == src["win_rate"]
            assert r["wins"] == src["wins"]
