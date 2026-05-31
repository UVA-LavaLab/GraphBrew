"""Derivation parity gate for ``wiki/data/lofo_robustness.json``.

Locks the leave-one-family-out winner-robustness reducer against
its single upstream — ``oracle_gap.json#rows`` filtered to the paper
L3 scope — so any drift in the row filter (must be in {1MB, 4MB,
8MB}), the is_winner predicate (string ``"1"`` — NOT integer 1, NOT
boolean truth), the per-app win counter (only winner rows), the
tie-break sort key (``(-wins, policy_name)`` — alphabetical
tie-break is load-bearing for reproducibility), the runner_up_wins
default of 0 when only one policy ever wins, the unique_top
predicate (STRICT top_wins > runner_wins), the family-drop loop
("drop_rows = rows with r['family'] != f"), the same_winner_as_full
flag, the fragile_family_drops filter, the per-app
is_lofo_robust predicate (empty fragile list), the n_drops counter
(== n_families regardless of family contents), or the
meta robustness_fraction (round 4 dp) trips a test before the
dashboard re-publishes the LOFO robustness headline.

Mirrors ``build_payload()`` from
``scripts/experiments/ecg/lofo_robustness.py`` verbatim.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "lofo_robustness.json"
UPSTREAM_PATH = WIKI_DATA / "oracle_gap.json"
PAPER_L3 = ("1MB", "4MB", "8MB")


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
def rows_in_scope():
    if not UPSTREAM_PATH.exists():
        pytest.skip(f"missing {UPSTREAM_PATH}")
    blob = json.loads(UPSTREAM_PATH.read_text())
    return [r for r in blob["rows"] if r["l3_size"] in PAPER_L3]


@pytest.fixture(scope="module")
def families(rows_in_scope):
    return sorted({r["family"] for r in rows_in_scope})


@pytest.fixture(scope="module")
def apps(rows_in_scope):
    return sorted({r["app"] for r in rows_in_scope})


@pytest.fixture(scope="module")
def full_top(rows_in_scope):
    return _top_policy_by_app(rows_in_scope)


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_app"}


def test_meta_fields(artifact):
    expected = {
        "source", "scope_l3_sizes", "n_rows_in_scope",
        "n_families", "families", "n_apps", "apps",
        "robust_apps", "fragile_apps",
        "n_robust_apps", "n_fragile_apps", "robustness_fraction",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing: {missing}"


def test_meta_scope_l3_sizes_match_constant(artifact):
    assert tuple(artifact["meta"]["scope_l3_sizes"]) == PAPER_L3


def test_per_app_entry_shape(artifact):
    expected = {
        "full_corpus", "drops", "n_drops",
        "n_robust_drops", "fragile_family_drops", "is_lofo_robust",
    }
    for app, p in artifact["per_app"].items():
        assert set(p.keys()) == expected, f"{app}: per_app field-set drift"


def test_drops_entry_shape_non_missing(artifact):
    expected = {
        "top_policy", "top_wins", "runner_up_wins",
        "margin", "unique_top", "same_winner_as_full",
    }
    for app, p in artifact["per_app"].items():
        for f, d in p["drops"].items():
            if d.get("missing"):
                assert set(d.keys()) == {"missing"}, (
                    f"{app}/{f}: missing entry has extra fields"
                )
            else:
                assert set(d.keys()) == expected, (
                    f"{app}/{f}: drops field-set drift"
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
            f"{app}: full_corpus field-set drift"
        )


# ----------------------------------------------------------------------
# Group B: scope & meta counters
# ----------------------------------------------------------------------

def test_meta_n_rows_in_scope_matches(artifact, rows_in_scope):
    assert artifact["meta"]["n_rows_in_scope"] == len(rows_in_scope)


def test_meta_families_sorted_distinct(artifact, families):
    assert artifact["meta"]["families"] == families
    assert artifact["meta"]["n_families"] == len(families)


def test_meta_apps_sorted_distinct(artifact, apps):
    assert artifact["meta"]["apps"] == apps
    assert artifact["meta"]["n_apps"] == len(apps)


def test_per_app_keys_match_meta_apps(artifact):
    assert sorted(artifact["per_app"].keys()) == sorted(artifact["meta"]["apps"])


# ----------------------------------------------------------------------
# Group C: full-corpus + drops byte-exact mirror
# ----------------------------------------------------------------------

def test_full_corpus_matches_top_policy_by_app(artifact, full_top):
    for app, p in artifact["per_app"].items():
        assert p["full_corpus"] == full_top.get(app, {}), (
            f"{app}: full_corpus drift"
        )


def test_drops_present_for_every_family(artifact, families):
    for app, p in artifact["per_app"].items():
        assert set(p["drops"].keys()) == set(families), (
            f"{app}: drops key set drift"
        )


def test_n_drops_equals_n_families(artifact, families):
    n = len(families)
    for app, p in artifact["per_app"].items():
        assert p["n_drops"] == n, (
            f"{app}: n_drops {p['n_drops']} ≠ {n}"
        )


def test_drops_top_policy_matches_lofo_recompute(artifact, rows_in_scope, families, full_top):
    """For every (app, family) the drop entry must match
    _top_policy_by_app on rows MINUS that family."""
    by_family_after = {}
    for f in families:
        drop_rows = [r for r in rows_in_scope if r["family"] != f]
        by_family_after[f] = _top_policy_by_app(drop_rows)

    for app, p in artifact["per_app"].items():
        full_top_pol = full_top.get(app, {}).get("top_policy")
        for f, d in p["drops"].items():
            after = by_family_after[f].get(app)
            if after is None:
                assert d.get("missing") is True, (
                    f"{app}/{f}: should be missing"
                )
                continue
            # Post lofo source fix: same_winner_as_full requires both
            # top_policy match AND unique_top after family drop.
            expected = {
                "top_policy":          after["top_policy"],
                "top_wins":            after["top_wins"],
                "runner_up_wins":      after["runner_up_wins"],
                "margin":              after["margin"],
                "unique_top":          after["unique_top"],
                "same_winner_as_full": (
                    after["top_policy"] == full_top_pol
                    and after["unique_top"]
                ),
            }
            assert d == expected, (
                f"{app}/{f}: drops entry drift\n  art={d}\n  exp={expected}"
            )


def test_fragile_family_drops_match_non_same_winner(artifact):
    for app, p in artifact["per_app"].items():
        expected = [
            f for f, d in p["drops"].items()
            if not d.get("missing") and not d.get("same_winner_as_full", False)
        ]
        # Generator iterates families in sorted order so order matches.
        assert p["fragile_family_drops"] == expected, (
            f"{app}: fragile_family_drops drift\n  art={p['fragile_family_drops']}\n"
            f"  exp={expected}"
        )


def test_n_robust_drops_matches_arithmetic(artifact):
    for app, p in artifact["per_app"].items():
        assert (
            p["n_robust_drops"] == p["n_drops"] - len(p["fragile_family_drops"])
        ), f"{app}: n_robust_drops drift"


def test_is_lofo_robust_iff_empty_fragile(artifact):
    for app, p in artifact["per_app"].items():
        assert p["is_lofo_robust"] is (len(p["fragile_family_drops"]) == 0), (
            f"{app}: is_lofo_robust drift"
        )


# ----------------------------------------------------------------------
# Group D: robust/fragile partition & robustness_fraction
# ----------------------------------------------------------------------

def test_robust_apps_sorted_and_match_filter(artifact):
    expected = sorted(a for a, p in artifact["per_app"].items() if p["is_lofo_robust"])
    assert artifact["meta"]["robust_apps"] == expected
    assert artifact["meta"]["n_robust_apps"] == len(expected)


def test_fragile_apps_sorted_and_match_filter(artifact):
    expected = sorted(a for a, p in artifact["per_app"].items() if not p["is_lofo_robust"])
    assert artifact["meta"]["fragile_apps"] == expected
    assert artifact["meta"]["n_fragile_apps"] == len(expected)


def test_robust_and_fragile_partition_apps(artifact):
    r = set(artifact["meta"]["robust_apps"])
    f = set(artifact["meta"]["fragile_apps"])
    a = set(artifact["meta"]["apps"])
    assert r.isdisjoint(f)
    assert r | f == a


def test_robustness_fraction_matches_round(artifact):
    apps = artifact["meta"]["apps"]
    expected = (
        round(artifact["meta"]["n_robust_apps"] / len(apps), 4) if apps else 0.0
    )
    assert artifact["meta"]["robustness_fraction"] == expected


# ----------------------------------------------------------------------
# Group E: sort-key & is_winner sanity
# ----------------------------------------------------------------------

def test_top_policy_sort_breaks_ties_alphabetically(artifact):
    """Tie-break: sorted by (-wins, policy_name). For full_corpus,
    if any policy has the same wins as top, top must be alphabetically
    smaller."""
    for app, p in artifact["per_app"].items():
        full = p["full_corpus"]
        if not full:
            continue
        top_wins = full["top_wins"]
        tied_with_top = [
            pol for pol, w in full["win_counts"].items() if w == top_wins
        ]
        if len(tied_with_top) > 1:
            assert full["top_policy"] == min(tied_with_top), (
                f"{app}: tie-break drift, expected {min(tied_with_top)} "
                f"got {full['top_policy']}"
            )


def test_unique_top_strict_inequality(artifact):
    for app, p in artifact["per_app"].items():
        if p["full_corpus"]:
            expected = p["full_corpus"]["top_wins"] > p["full_corpus"]["runner_up_wins"]
            assert p["full_corpus"]["unique_top"] is expected
        for f, d in p["drops"].items():
            if d.get("missing"):
                continue
            expected = d["top_wins"] > d["runner_up_wins"]
            assert d["unique_top"] is expected


def test_margin_equals_top_minus_runner(artifact):
    for app, p in artifact["per_app"].items():
        if p["full_corpus"]:
            f = p["full_corpus"]
            assert f["margin"] == f["top_wins"] - f["runner_up_wins"]
        for fam, d in p["drops"].items():
            if d.get("missing"):
                continue
            assert d["margin"] == d["top_wins"] - d["runner_up_wins"]


def test_full_corpus_win_counts_sum_to_n_winner_rows(artifact, rows_in_scope):
    """Sanity: per-app full_corpus.win_counts sums to the number of
    is_winner == '1' rows for that app in the paper L3 scope."""
    by_app_winners = defaultdict(int)
    for r in rows_in_scope:
        if r.get("is_winner") == "1":
            by_app_winners[r["app"]] += 1
    for app, p in artifact["per_app"].items():
        full = p["full_corpus"]
        if not full:
            continue
        assert sum(full["win_counts"].values()) == by_app_winners[app], (
            f"{app}: win_counts sum drift"
        )
