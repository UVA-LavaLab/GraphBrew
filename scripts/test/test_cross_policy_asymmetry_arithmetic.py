"""Gate 123 — cross_policy_asymmetry.json head-to-head arithmetic.

For every unordered pair (A, B) of distinct policies, the artifact
partitions cache cells by who wins H2H (lower miss_rate) and records
margin means in pp. This gate reproduces every per-pair stat from the
upstream oracle_gap.json rows and locks the verdict invariant.

Source: wiki/data/oracle_gap.json (each row carries app/graph/l3_size/
policy/miss_rate as strings/numbers; miss_rate is a fraction so margins
are multiplied by 100 to get pp).

Per-pair arithmetic (itertools.combinations alphabetical order):
    a_wins = #cells where miss_rate[A] < miss_rate[B]
    b_wins = #cells where miss_rate[B] < miss_rate[A]
    ties   = #cells where miss_rate[A] == miss_rate[B]
    a_mean_margin_pp = mean over a_wins of (miss[B]-miss[A]) * 100.0
    b_mean_margin_pp = mean over b_wins of (miss[A]-miss[B]) * 100.0
    asymmetry_ratio  = max(a_mean,b_mean)/min(a_mean,b_mean)
                        (None when either mean is 0)

Verdict (PASS) requires both:
    every pair has at least one A-win and one B-win
    AND max observed asymmetry ratio < ratio_ceiling (20.0)

Invariants (15 tests, 4 groups):
- meta constants + structure (4): policies tuple, pair_count==6,
  ratio_ceiling==20.0, all 6 pair keys present in alphabetical order.
- per_pair counts + tie sum from source (3): a_wins, b_wins, ties
  recomputed from oracle_gap rows; a_wins+b_wins+ties = #cells with
  both policies present.
- per_pair margin means + ratio (3): means recomputed (× 100.0
  conversion to pp); ratio recomputed (or None when min==0).
- meta aggregates + verdict (5): every_pair_both_win, max_asymmetry_ratio,
  max_ratio_under_ceiling, verdict, verdict_invariant string.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/cross_policy_asymmetry.json")
SOURCE = Path("wiki/data/oracle_gap.json")

POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
EXPECTED_PAIR_COUNT = 6
RATIO_CEILING = 20.0
ROUND_TOL = 5e-4


@pytest.fixture(scope="module")
def data():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def by_cell():
    assert SOURCE.exists(), f"missing source: {SOURCE}"
    src = json.loads(SOURCE.read_text())
    cells: dict = defaultdict(dict)
    for r in src["rows"]:
        cells[(r["app"], r["graph"], r["l3_size"])][r["policy"]] = float(r["miss_rate"])
    return dict(cells)


def _pair_arith(by_cell, a, b):
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
    a_mean = sum(a_wins) / len(a_wins) if a_wins else 0.0
    b_mean = sum(b_wins) / len(b_wins) if b_wins else 0.0
    if a_mean > 0 and b_mean > 0:
        ratio = max(a_mean, b_mean) / min(a_mean, b_mean)
    else:
        ratio = None
    return len(a_wins), len(b_wins), ties, a_mean, b_mean, ratio


# ── group 1: meta + structure ────────────────────────────────────────────


def test_meta_policies_tuple(data):
    assert tuple(data["meta"]["policies"]) == POLICIES


def test_meta_pair_count(data):
    assert data["meta"]["pair_count"] == EXPECTED_PAIR_COUNT


def test_meta_ratio_ceiling(data):
    assert data["meta"]["ratio_ceiling"] == RATIO_CEILING


def test_per_pair_keys_alphabetical(data):
    expected = [f"{a}_vs_{b}" for a, b in itertools.combinations(POLICIES, 2)]
    assert set(data["per_pair"].keys()) == set(expected)
    assert len(data["per_pair"]) == EXPECTED_PAIR_COUNT


# ── group 2: per_pair counts from source ─────────────────────────────────


@pytest.mark.parametrize("a,b", list(itertools.combinations(POLICIES, 2)))
def test_per_pair_wins_and_ties_from_source(data, by_cell, a, b):
    entry = data["per_pair"][f"{a}_vs_{b}"]
    a_wins, b_wins, ties, _, _, _ = _pair_arith(by_cell, a, b)
    assert entry["a_policy"] == a and entry["b_policy"] == b
    assert entry["a_wins"] == a_wins, f"{a}_vs_{b}: a_wins"
    assert entry["b_wins"] == b_wins, f"{a}_vs_{b}: b_wins"
    assert entry["ties"] == ties, f"{a}_vs_{b}: ties"


def test_per_pair_count_sums_match_overlap(data, by_cell):
    for a, b in itertools.combinations(POLICIES, 2):
        entry = data["per_pair"][f"{a}_vs_{b}"]
        overlap = sum(1 for m in by_cell.values() if a in m and b in m)
        assert entry["a_wins"] + entry["b_wins"] + entry["ties"] == overlap, (
            f"{a}_vs_{b}: wins+ties != overlap ({overlap})"
        )


def test_per_pair_no_negative_counts(data):
    for key, entry in data["per_pair"].items():
        assert entry["a_wins"] >= 0
        assert entry["b_wins"] >= 0
        assert entry["ties"] >= 0


# ── group 3: per_pair means + ratio ──────────────────────────────────────


@pytest.mark.parametrize("a,b", list(itertools.combinations(POLICIES, 2)))
def test_per_pair_mean_margins(data, by_cell, a, b):
    entry = data["per_pair"][f"{a}_vs_{b}"]
    _, _, _, a_mean, b_mean, _ = _pair_arith(by_cell, a, b)
    assert math.isclose(entry["a_mean_margin_pp"], round(a_mean, 4), abs_tol=ROUND_TOL), (
        f"{a}_vs_{b}: a_mean={entry['a_mean_margin_pp']} expected={a_mean}"
    )
    assert math.isclose(entry["b_mean_margin_pp"], round(b_mean, 4), abs_tol=ROUND_TOL), (
        f"{a}_vs_{b}: b_mean={entry['b_mean_margin_pp']} expected={b_mean}"
    )


@pytest.mark.parametrize("a,b", list(itertools.combinations(POLICIES, 2)))
def test_per_pair_asymmetry_ratio(data, by_cell, a, b):
    entry = data["per_pair"][f"{a}_vs_{b}"]
    _, _, _, _, _, expected_ratio = _pair_arith(by_cell, a, b)
    if expected_ratio is None:
        assert entry["asymmetry_ratio"] is None
    else:
        assert entry["asymmetry_ratio"] is not None
        assert math.isclose(
            entry["asymmetry_ratio"], round(expected_ratio, 4), abs_tol=ROUND_TOL
        ), f"{a}_vs_{b}: ratio mismatch"


def test_ratio_consistent_with_means(data):
    for key, entry in data["per_pair"].items():
        a_mean, b_mean = entry["a_mean_margin_pp"], entry["b_mean_margin_pp"]
        if entry["asymmetry_ratio"] is None:
            assert a_mean == 0 or b_mean == 0
        else:
            assert a_mean > 0 and b_mean > 0
            recomputed = max(a_mean, b_mean) / min(a_mean, b_mean)
            assert math.isclose(entry["asymmetry_ratio"], recomputed, abs_tol=1e-3), key


# ── group 4: meta aggregates + verdict ───────────────────────────────────


def test_every_pair_both_win(data):
    expected = all(
        p["a_wins"] >= 1 and p["b_wins"] >= 1 for p in data["per_pair"].values()
    )
    assert data["meta"]["every_pair_both_win"] is expected


def test_max_asymmetry_ratio_is_max_finite(data):
    ratios = [
        p["asymmetry_ratio"]
        for p in data["per_pair"].values()
        if p["asymmetry_ratio"] is not None
    ]
    expected = max(ratios) if ratios else 0.0
    assert math.isclose(
        data["meta"]["max_asymmetry_ratio"], round(expected, 4), abs_tol=ROUND_TOL
    )


def test_max_ratio_under_ceiling_flag(data):
    expected = data["meta"]["max_asymmetry_ratio"] < RATIO_CEILING
    assert data["meta"]["max_ratio_under_ceiling"] is expected


def test_verdict_is_conjunction(data):
    expected = (
        "PASS"
        if data["meta"]["every_pair_both_win"] and data["meta"]["max_ratio_under_ceiling"]
        else "FAIL"
    )
    assert data["meta"]["verdict"] == expected


def test_verdict_invariant_string_mentions_ceiling(data):
    inv = data["meta"]["verdict_invariant"]
    assert "PASS iff" in inv
    assert "at least one cell" in inv
    assert str(RATIO_CEILING) in inv
