"""Derivation parity gate for ``wiki/data/cross_policy_asymmetry.json``.

Locks the cross-policy mean-margin asymmetry report (gate 64)
against its single upstream — ``oracle_gap.json#rows`` — so any
silent drift in the head-to-head winner partition (strict-less-than),
the per-pair mean-margin formulas (a_mean over A_wins of (mb−ma)*100,
symmetric for b), the asymmetry ratio (max/min with None on
degeneracy), or the 2-invariant AND verdict (every_pair_both_win
∧ max_ratio_under_ceiling) trips a test before the dashboard re-
publishes the "no policy pair has runaway loss profile" claim.

The gate fully mirrors `scripts/experiments/ecg/cross_policy_asymmetry.py`'s
`build()` against the same upstream JSON.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

UPSTREAM_PATH = WIKI_DATA / "oracle_gap.json"
ARTIFACT_PATH = WIKI_DATA / "cross_policy_asymmetry.json"

POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
ASYMMETRY_RATIO_CEILING = 20.0


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _expected_per_pair(rows):
    """Verbatim mirror of generator `build()`."""
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r["app"], r["graph"], r["l3_size"])][r["policy"]] = \
            float(r["miss_rate"])
    out = {}
    for a, b in itertools.combinations(POLICIES, 2):
        a_wins, b_wins, ties = [], [], 0
        for misses in by_cell.values():
            if a not in misses or b not in misses:
                continue
            ma, mb = misses[a], misses[b]
            if ma < mb:
                a_wins.append((mb - ma) * 100.0)
            elif mb < ma:
                b_wins.append((ma - mb) * 100.0)
            else:
                ties += 1
        a_mean = _mean(a_wins)
        b_mean = _mean(b_wins)
        ratio = (
            max(a_mean, b_mean) / min(a_mean, b_mean)
            if a_mean > 0 and b_mean > 0
            else float("inf")
        )
        out[f"{a}_vs_{b}"] = {
            "a_policy": a,
            "b_policy": b,
            "a_wins": len(a_wins),
            "b_wins": len(b_wins),
            "ties": ties,
            "a_mean_margin_pp": round(a_mean, 4),
            "b_mean_margin_pp": round(b_mean, 4),
            "asymmetry_ratio": (round(ratio, 4)
                                if ratio != float("inf") else None),
        }
    return out


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def upstream_rows() -> list:
    if not UPSTREAM_PATH.exists():
        pytest.skip(f"missing {UPSTREAM_PATH}")
    return json.loads(UPSTREAM_PATH.read_text())["rows"]


@pytest.fixture(scope="module")
def mirror(upstream_rows) -> dict:
    return _expected_per_pair(upstream_rows)


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_pair"}


def test_meta_fields(artifact):
    expected = {
        "policies", "pair_count", "every_pair_both_win",
        "max_asymmetry_ratio", "ratio_ceiling",
        "max_ratio_under_ceiling", "verdict", "verdict_invariant",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing fields: {missing}"


def test_policies_match_constant(artifact):
    assert tuple(artifact["meta"]["policies"]) == POLICIES


def test_ratio_ceiling_matches_constant(artifact):
    assert artifact["meta"]["ratio_ceiling"] == ASYMMETRY_RATIO_CEILING


def test_pair_count_is_six(artifact):
    """C(4, 2) == 6 unordered pairs from 4 policies."""
    assert artifact["meta"]["pair_count"] == 6


def test_pair_count_matches_per_pair_len(artifact):
    assert artifact["meta"]["pair_count"] == len(artifact["per_pair"])


def test_verdict_is_pass_or_fail(artifact):
    assert artifact["meta"]["verdict"] in {"PASS", "FAIL"}


def test_per_pair_entry_fields(artifact):
    expected = {
        "a_policy", "b_policy", "a_wins", "b_wins", "ties",
        "a_mean_margin_pp", "b_mean_margin_pp", "asymmetry_ratio",
    }
    for key, entry in artifact["per_pair"].items():
        missing = expected - set(entry.keys())
        assert not missing, f"per_pair[{key}] missing fields: {missing}"


def test_per_pair_keys_use_itertools_combinations_order(artifact):
    """Generator uses `itertools.combinations(POLICIES, 2)`, which
    preserves input order — so each key is `f'{a}_vs_{b}'` with a, b
    in POLICIES order."""
    expected_keys = [
        f"{a}_vs_{b}"
        for a, b in itertools.combinations(POLICIES, 2)
    ]
    assert sorted(artifact["per_pair"].keys()) == sorted(expected_keys)


def test_per_pair_a_and_b_policy_match_key(artifact):
    for key, entry in artifact["per_pair"].items():
        a, _, b = key.partition("_vs_")
        assert entry["a_policy"] == a
        assert entry["b_policy"] == b


# ----------------------------------------------------------------------
# Group B: per-pair cross-source parity
# ----------------------------------------------------------------------

def test_per_pair_keys_match_mirror(artifact, mirror):
    assert set(artifact["per_pair"].keys()) == set(mirror.keys())


def test_per_pair_a_wins_match_mirror(artifact, mirror):
    for key in artifact["per_pair"]:
        assert artifact["per_pair"][key]["a_wins"] == mirror[key]["a_wins"], (
            f"{key}: a_wins drift"
        )


def test_per_pair_b_wins_match_mirror(artifact, mirror):
    for key in artifact["per_pair"]:
        assert artifact["per_pair"][key]["b_wins"] == mirror[key]["b_wins"], (
            f"{key}: b_wins drift"
        )


def test_per_pair_ties_match_mirror(artifact, mirror):
    for key in artifact["per_pair"]:
        assert artifact["per_pair"][key]["ties"] == mirror[key]["ties"], (
            f"{key}: ties drift"
        )


def test_per_pair_a_mean_margin_matches_mirror(artifact, mirror):
    for key in artifact["per_pair"]:
        assert artifact["per_pair"][key]["a_mean_margin_pp"] == \
            mirror[key]["a_mean_margin_pp"], (
            f"{key}: a_mean_margin_pp drift"
        )


def test_per_pair_b_mean_margin_matches_mirror(artifact, mirror):
    for key in artifact["per_pair"]:
        assert artifact["per_pair"][key]["b_mean_margin_pp"] == \
            mirror[key]["b_mean_margin_pp"], (
            f"{key}: b_mean_margin_pp drift"
        )


def test_per_pair_asymmetry_ratio_matches_mirror(artifact, mirror):
    for key in artifact["per_pair"]:
        assert artifact["per_pair"][key]["asymmetry_ratio"] == \
            mirror[key]["asymmetry_ratio"], (
            f"{key}: asymmetry_ratio drift"
        )


def test_per_pair_total_cells_le_corpus_cell_count(artifact, upstream_rows):
    """a_wins + b_wins + ties cannot exceed the count of distinct
    (app, graph, l3) cells in the upstream rows."""
    cells = {(r["app"], r["graph"], r["l3_size"]) for r in upstream_rows}
    for key, entry in artifact["per_pair"].items():
        total = entry["a_wins"] + entry["b_wins"] + entry["ties"]
        assert total <= len(cells), (
            f"{key}: a+b+ties={total} > distinct cells={len(cells)}"
        )


def test_per_pair_asymmetry_ratio_ge_one_when_defined(artifact):
    """ratio = max/min with both positive ⇒ always ≥ 1.0 by definition."""
    for key, entry in artifact["per_pair"].items():
        r = entry["asymmetry_ratio"]
        if r is not None:
            assert r >= 1.0 - 1e-9, (
                f"{key}: asymmetry_ratio {r} < 1 (max/min must be ≥ 1)"
            )


def test_per_pair_mean_margins_non_negative(artifact):
    """Margins are computed as (loser_miss − winner_miss)*100, always ≥ 0."""
    for key, entry in artifact["per_pair"].items():
        assert entry["a_mean_margin_pp"] >= 0
        assert entry["b_mean_margin_pp"] >= 0


# ----------------------------------------------------------------------
# Group C: verdict reducer parity
# ----------------------------------------------------------------------

def test_every_pair_both_win_matches_recompute(artifact):
    expected = all(
        p["a_wins"] >= 1 and p["b_wins"] >= 1
        for p in artifact["per_pair"].values()
    )
    assert artifact["meta"]["every_pair_both_win"] == expected


def test_max_asymmetry_ratio_matches_recompute(artifact):
    finite = [
        p["asymmetry_ratio"]
        for p in artifact["per_pair"].values()
        if p["asymmetry_ratio"] is not None
    ]
    expected = round(max(finite) if finite else 0.0, 4)
    assert artifact["meta"]["max_asymmetry_ratio"] == expected


def test_max_ratio_under_ceiling_matches_recompute(artifact):
    """Generator uses strict less-than: max_ratio < ceiling."""
    max_r = artifact["meta"]["max_asymmetry_ratio"]
    ceil = artifact["meta"]["ratio_ceiling"]
    assert artifact["meta"]["max_ratio_under_ceiling"] == (max_r < ceil)


def test_verdict_pass_iff_both_checks_pass(artifact):
    m = artifact["meta"]
    expected = (
        "PASS"
        if (m["every_pair_both_win"] and m["max_ratio_under_ceiling"])
        else "FAIL"
    )
    assert m["verdict"] == expected


def test_verdict_invariant_mentions_ceiling(artifact):
    """Sanity check: documentation must reference the ceiling number,
    so a future ceiling change without doc-string update gets caught."""
    s = artifact["meta"]["verdict_invariant"]
    assert str(ASYMMETRY_RATIO_CEILING) in s, (
        f"verdict_invariant docstring missing ceiling {ASYMMETRY_RATIO_CEILING}"
    )


# ----------------------------------------------------------------------
# Group D: end-to-end sanity
# ----------------------------------------------------------------------

def test_oracle_pair_present(artifact):
    """GRASP_vs_POPT must be in per_pair (the headline oracle-vs-oracle
    comparison from gate 64's docstring)."""
    assert "GRASP_vs_POPT" in artifact["per_pair"]


def test_pass_verdict_means_no_pair_starved(artifact):
    if artifact["meta"]["verdict"] == "PASS":
        for key, entry in artifact["per_pair"].items():
            assert entry["a_wins"] >= 1 and entry["b_wins"] >= 1, (
                f"PASS but {key} starved: a_wins={entry['a_wins']}, "
                f"b_wins={entry['b_wins']}"
            )


def test_ratio_finite_only_when_both_means_positive(artifact):
    for key, entry in artifact["per_pair"].items():
        if entry["asymmetry_ratio"] is None:
            assert entry["a_mean_margin_pp"] == 0 or \
                entry["b_mean_margin_pp"] == 0, (
                f"{key}: ratio is None but both means are positive"
            )
        else:
            assert entry["a_mean_margin_pp"] > 0 and \
                entry["b_mean_margin_pp"] > 0, (
                f"{key}: ratio finite but a mean is zero"
            )
