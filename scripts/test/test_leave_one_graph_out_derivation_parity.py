"""Derivation parity gate for ``wiki/data/leave_one_graph_out.json``.

LOGO sibling of gate 176 (LOFO). Single upstream: ``oracle_gap.json#rows``
(no L3 scope filter — full corpus is used).

Locks the per-(app, dropped graph) re-rank pipeline so any drift in
the is_winner predicate (string ``"1"``), the tie-break sort key
(``(-wins, policy_name)``), the runner_up_wins default of 0 when only
one policy ever wins, the unique_top STRICT predicate, the graph-drop
loop (``[r for r in rows if r['graph'] != g]``), the same_winner_as_full
flag, the fragile_drops filter, the per-app is_logo_robust predicate
(empty fragile list), the n_robust_drops arithmetic, or the
robust/fragile partition trips a test before the dashboard re-publishes
"all top-line winners survive LOGO except sssp" headline.

Mirrors ``build_payload()`` from
``scripts/experiments/ecg/leave_one_graph_out.py`` verbatim.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "leave_one_graph_out.json"
UPSTREAM_PATH = WIKI_DATA / "oracle_gap.json"


def _top_policy_by_app(rows):
    win_counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r.get("is_winner") == "1":
            win_counts[r["app"]][r["policy"]] += 1
    out = {}
    for app, c in win_counts.items():
        ordered = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        top_policy, top_wins = ordered[0]
        runner_wins = ordered[1][1] if len(ordered) > 1 else 0
        out[app] = {
            "win_counts": dict(c),
            "top_policy": top_policy,
            "top_wins": top_wins,
            "runner_up_wins": runner_wins,
            "unique_top": top_wins > runner_wins,
            "margin": top_wins - runner_wins,
        }
    return out


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def rows():
    if not UPSTREAM_PATH.exists():
        pytest.skip(f"missing {UPSTREAM_PATH}")
    return json.loads(UPSTREAM_PATH.read_text())["rows"]


@pytest.fixture(scope="module")
def graphs(rows):
    return sorted({r["graph"] for r in rows})


@pytest.fixture(scope="module")
def apps(rows):
    return sorted({r["app"] for r in rows})


@pytest.fixture(scope="module")
def full_top(rows):
    return _top_policy_by_app(rows)


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_app"}


def test_meta_fields(artifact):
    expected = {
        "n_rows", "n_graphs", "graphs", "apps",
        "robust_apps", "fragile_apps",
        "n_robust_apps", "n_fragile_apps",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing: {missing}"


def test_per_app_entry_shape(artifact):
    expected = {
        "full_corpus", "drops", "n_drops",
        "n_robust_drops", "fragile_drops", "is_logo_robust",
    }
    for app, p in artifact["per_app"].items():
        assert set(p.keys()) == expected, f"{app}: per_app field drift"


def test_drops_entry_shape_non_missing(artifact):
    """Drops entries (non-missing) carry FIVE fields — NOT runner_up_wins
    (the LOGO generator omits runner_up_wins from drops)."""
    expected = {
        "top_policy", "top_wins", "margin",
        "unique_top", "same_winner_as_full",
    }
    for app, p in artifact["per_app"].items():
        for g, d in p["drops"].items():
            if d.get("missing"):
                assert set(d.keys()) == {"missing"}
            else:
                assert set(d.keys()) == expected, (
                    f"{app}/{g}: drops field drift"
                )


def test_full_corpus_entry_shape(artifact):
    expected = {
        "win_counts", "top_policy", "top_wins",
        "runner_up_wins", "unique_top", "margin",
    }
    for app, p in artifact["per_app"].items():
        if not p["full_corpus"]:
            continue
        assert set(p["full_corpus"].keys()) == expected, (
            f"{app}: full_corpus field drift"
        )


# ----------------------------------------------------------------------
# Group B: scope & meta counters
# ----------------------------------------------------------------------

def test_meta_n_rows_matches_upstream(artifact, rows):
    assert artifact["meta"]["n_rows"] == len(rows)


def test_meta_graphs_sorted_distinct(artifact, graphs):
    assert artifact["meta"]["graphs"] == graphs
    assert artifact["meta"]["n_graphs"] == len(graphs)


def test_meta_apps_sorted_distinct(artifact, apps):
    assert artifact["meta"]["apps"] == apps


def test_per_app_keys_match_meta_apps(artifact):
    assert sorted(artifact["per_app"].keys()) == sorted(artifact["meta"]["apps"])


# ----------------------------------------------------------------------
# Group C: full-corpus + drops byte-exact mirror
# ----------------------------------------------------------------------

def test_full_corpus_matches_top_policy_by_app(artifact, full_top):
    for app, p in artifact["per_app"].items():
        assert p["full_corpus"] == full_top.get(app, {})


def test_drops_present_for_every_graph(artifact, graphs):
    for app, p in artifact["per_app"].items():
        assert set(p["drops"].keys()) == set(graphs)


def test_n_drops_equals_n_graphs(artifact, graphs):
    n = len(graphs)
    for app, p in artifact["per_app"].items():
        assert p["n_drops"] == n


def test_drops_byte_exact_against_logo_recompute(artifact, rows, graphs, full_top):
    """For every (app, graph) the drop entry must match
    _top_policy_by_app on rows MINUS that graph. Note drops omit
    runner_up_wins (load-bearing — LOGO differs from LOFO here)."""
    by_graph_after = {}
    for g in graphs:
        drop_rows = [r for r in rows if r["graph"] != g]
        by_graph_after[g] = _top_policy_by_app(drop_rows)

    for app, p in artifact["per_app"].items():
        full_top_pol = full_top.get(app, {}).get("top_policy")
        for g, d in p["drops"].items():
            after = by_graph_after[g].get(app)
            if after is None:
                assert d.get("missing") is True
                continue
            # Post LOGO source fix: same_winner_as_full now requires both
            # top_policy match AND unique_top after graph drop.
            expected = {
                "top_policy":          after["top_policy"],
                "top_wins":            after["top_wins"],
                "margin":              after["margin"],
                "unique_top":          after["unique_top"],
                "same_winner_as_full": (
                    after["top_policy"] == full_top_pol
                    and after["unique_top"]
                ),
            }
            assert d == expected, (
                f"{app}/{g}: drops drift\n  art={d}\n  exp={expected}"
            )


def test_fragile_drops_match_non_same_winner(artifact):
    """fragile_drops appended in iteration-over-graphs order — generator
    iterates `for g in graphs` (sorted), so the list is in sorted graph
    order."""
    for app, p in artifact["per_app"].items():
        expected = [
            g for g, d in p["drops"].items()
            if not d.get("missing") and not d.get("same_winner_as_full", False)
        ]
        assert p["fragile_drops"] == expected, (
            f"{app}: fragile_drops drift\n  art={p['fragile_drops']}\n"
            f"  exp={expected}"
        )


def test_n_robust_drops_matches_arithmetic(artifact):
    for app, p in artifact["per_app"].items():
        assert p["n_robust_drops"] == p["n_drops"] - len(p["fragile_drops"])


def test_is_logo_robust_iff_empty_fragile(artifact):
    for app, p in artifact["per_app"].items():
        assert p["is_logo_robust"] is (len(p["fragile_drops"]) == 0)


# ----------------------------------------------------------------------
# Group D: robust/fragile partition
# ----------------------------------------------------------------------

def test_robust_apps_sorted_and_match_filter(artifact):
    expected = sorted(a for a, p in artifact["per_app"].items() if p["is_logo_robust"])
    assert artifact["meta"]["robust_apps"] == expected
    assert artifact["meta"]["n_robust_apps"] == len(expected)


def test_fragile_apps_sorted_and_match_filter(artifact):
    expected = sorted(a for a, p in artifact["per_app"].items() if not p["is_logo_robust"])
    assert artifact["meta"]["fragile_apps"] == expected
    assert artifact["meta"]["n_fragile_apps"] == len(expected)


def test_robust_and_fragile_partition_apps(artifact):
    r = set(artifact["meta"]["robust_apps"])
    f = set(artifact["meta"]["fragile_apps"])
    a = set(artifact["meta"]["apps"])
    assert r.isdisjoint(f)
    assert r | f == a


# ----------------------------------------------------------------------
# Group E: sort-key & is_winner sanity
# ----------------------------------------------------------------------

def test_top_policy_alphabetic_tie_break(artifact):
    for app, p in artifact["per_app"].items():
        full = p["full_corpus"]
        if not full:
            continue
        top_wins = full["top_wins"]
        tied = [pol for pol, w in full["win_counts"].items() if w == top_wins]
        if len(tied) > 1:
            assert full["top_policy"] == min(tied)


def test_unique_top_strict_inequality(artifact):
    for app, p in artifact["per_app"].items():
        if p["full_corpus"]:
            f = p["full_corpus"]
            assert f["unique_top"] is (f["top_wins"] > f["runner_up_wins"])
        for _, d in p["drops"].items():
            if d.get("missing"):
                continue
            # drops omit runner_up_wins, so unique_top can't be
            # re-derived here — sanity: must be a bool.
            assert isinstance(d["unique_top"], bool)


def test_margin_equals_top_minus_runner_in_full_corpus(artifact):
    for app, p in artifact["per_app"].items():
        f = p["full_corpus"]
        if f:
            assert f["margin"] == f["top_wins"] - f["runner_up_wins"]


def test_full_corpus_win_counts_sum_to_n_winner_rows(artifact, rows):
    by_app_winners = defaultdict(int)
    for r in rows:
        if r.get("is_winner") == "1":
            by_app_winners[r["app"]] += 1
    for app, p in artifact["per_app"].items():
        if p["full_corpus"]:
            assert sum(p["full_corpus"]["win_counts"].values()) == by_app_winners[app]
