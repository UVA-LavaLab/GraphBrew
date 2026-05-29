"""Derivation parity gate for ``wiki/data/oracle_gap_effect_size.json``.

Locks the Cliff's-delta + Mann-Whitney U effect-size report against
its single upstream — ``oracle_gap.json#rows`` — so any silent drift
in the per-(app, policy) gap_pp grouping, the per-policy distribution
reducers ({n, median (sorted-list midpoint, NOT statistics.median),
mean (sum/len), min, max}), the O(n_a · n_b) Cliff's-delta core,
the magnitude classifier (negligible / small / medium / large),
the Mann-Whitney U with average-rank tie-breaking and asymptotic
two-sided normal p-value, the stochastically_smaller side-marker,
or the `large_negative_deltas` filter+sort trips a test before the
dashboard re-publishes the "POPT shows large nonparametric
dominance over LRU on pr" headline.

The gate fully mirrors `scripts/experiments/ecg/oracle_gap_effect_size.py`'s
`aggregate()` against the same upstream JSON.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

UPSTREAM_PATH = WIKI_DATA / "oracle_gap.json"
ARTIFACT_PATH = WIKI_DATA / "oracle_gap_effect_size.json"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")
APPS_ORDER = ("pr", "bc", "cc", "bfs", "sssp")
SMALL = 0.147
MEDIUM = 0.33
LARGE = 0.474


def _cliffs_delta(xs, ys):
    if not xs or not ys:
        return 0.0
    gt = lt = 0
    for x in xs:
        for y in ys:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    return (gt - lt) / (len(xs) * len(ys))


def _magnitude(d):
    ad = abs(d)
    if ad >= LARGE:
        return "large"
    if ad >= MEDIUM:
        return "medium"
    if ad >= SMALL:
        return "small"
    return "negligible"


def _mannwhitney_u(xs, ys):
    if not xs or not ys:
        return 0.0, 1.0
    combined = sorted([(v, "x") for v in xs] + [(v, "y") for v in ys])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_x = sum(r for r, (_, lab) in zip(ranks, combined) if lab == "x")
    n_x, n_y = len(xs), len(ys)
    u_x = rank_x - n_x * (n_x + 1) / 2.0
    mu = n_x * n_y / 2.0
    sigma = math.sqrt(n_x * n_y * (n_x + n_y + 1) / 12.0)
    if sigma == 0.0:
        return u_x, 1.0
    z = abs(u_x - mu) / sigma
    p = 2.0 * (0.5 * math.erfc(z / math.sqrt(2.0)))
    return u_x, p


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
    blob = json.loads(UPSTREAM_PATH.read_text())
    return blob["rows"] if isinstance(blob, dict) and "rows" in blob else blob


@pytest.fixture(scope="module")
def by_app_pol(upstream_rows):
    out = defaultdict(list)
    for r in upstream_rows:
        out[(r["app"], r["policy"])].append(float(r["gap_pp"]))
    return out


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {
        "large_negative_deltas", "meta", "per_app",
    }


def test_meta_fields(artifact):
    expected = {"source", "n_rows", "policies", "apps", "cliffs_thresholds"}
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing fields: {missing}"


def test_meta_policies_match_constant(artifact):
    assert tuple(artifact["meta"]["policies"]) == POLICIES


def test_meta_apps_match_canonical_order(artifact):
    assert tuple(artifact["meta"]["apps"]) == APPS_ORDER


def test_meta_thresholds_match_constants(artifact):
    assert artifact["meta"]["cliffs_thresholds"] == {
        "small": SMALL, "medium": MEDIUM, "large": LARGE,
    }


def test_meta_n_rows_matches_upstream(artifact, upstream_rows):
    assert artifact["meta"]["n_rows"] == len(upstream_rows)


def test_per_app_entry_shape(artifact):
    for app, entry in artifact["per_app"].items():
        assert set(entry.keys()) == {"per_policy", "comparisons"}


def test_per_policy_entry_shape(artifact):
    for app, entry in artifact["per_app"].items():
        for pol, block in entry["per_policy"].items():
            assert set(block.keys()) == {"n", "median", "mean", "min", "max"}


def test_comparison_entry_shape(artifact):
    expected = {
        "a", "b", "n_a", "n_b", "cliffs_delta_a_minus_b",
        "magnitude", "mannwhitney_u", "mannwhitney_p",
        "stochastically_smaller",
    }
    for app, entry in artifact["per_app"].items():
        for c in entry["comparisons"]:
            missing = expected - set(c.keys())
            assert not missing, (
                f"per_app[{app}].comparisons entry missing: {missing}"
            )


def test_per_app_keys_subset_of_canonical_apps(artifact):
    """Per-app keys must be a subset of the canonical apps tuple
    (sorted iteration over rows[].app, so order may differ — but
    membership must match)."""
    art_apps = set(artifact["per_app"].keys())
    assert art_apps <= set(APPS_ORDER), (
        f"per_app contains unknown apps: {art_apps - set(APPS_ORDER)}"
    )


# ----------------------------------------------------------------------
# Group B: per-policy distribution parity
# ----------------------------------------------------------------------

def test_per_app_apps_match_upstream_distinct(artifact, upstream_rows):
    expected = sorted({r["app"] for r in upstream_rows})
    assert sorted(artifact["per_app"].keys()) == expected


def test_per_policy_n_matches_upstream(artifact, by_app_pol):
    for app, entry in artifact["per_app"].items():
        for pol, block in entry["per_policy"].items():
            assert block["n"] == len(by_app_pol[(app, pol)]), (
                f"{app}/{pol}: n {block['n']} ≠ "
                f"{len(by_app_pol[(app, pol)])}"
            )


def test_per_policy_median_uses_sorted_list_midpoint(artifact, by_app_pol):
    """Per-policy median = sorted[n//2] — NOT statistics.median (does
    NOT average two middle elements for even n)."""
    for app, entry in artifact["per_app"].items():
        for pol, block in entry["per_policy"].items():
            vs = sorted(by_app_pol[(app, pol)])
            expected = round(vs[len(vs) // 2], 4)
            assert block["median"] == expected, (
                f"{app}/{pol}: median {block['median']} ≠ "
                f"round(sorted[{len(vs) // 2}], 4) = {expected}"
            )


def test_per_policy_mean_matches_sum_over_len(artifact, by_app_pol):
    for app, entry in artifact["per_app"].items():
        for pol, block in entry["per_policy"].items():
            vs = by_app_pol[(app, pol)]
            expected = round(sum(vs) / len(vs), 4)
            assert block["mean"] == expected, (
                f"{app}/{pol}: mean drift"
            )


def test_per_policy_min_max_match(artifact, by_app_pol):
    for app, entry in artifact["per_app"].items():
        for pol, block in entry["per_policy"].items():
            vs = by_app_pol[(app, pol)]
            assert block["min"] == round(min(vs), 4)
            assert block["max"] == round(max(vs), 4)


# ----------------------------------------------------------------------
# Group C: pairwise comparison parity
# ----------------------------------------------------------------------

def test_comparisons_use_ordered_pairs_a_ne_b(artifact):
    """Generator nests `for a in POLICIES: for b in POLICIES: if a == b
    or missing: continue` — every comparison has a ≠ b."""
    for app, entry in artifact["per_app"].items():
        for c in entry["comparisons"]:
            assert c["a"] != c["b"], (
                f"{app}: a==b in comparison entry"
            )


def test_comparisons_use_canonical_policies(artifact):
    for app, entry in artifact["per_app"].items():
        for c in entry["comparisons"]:
            assert c["a"] in POLICIES
            assert c["b"] in POLICIES


def test_comparisons_count_matches_ordered_pairs(artifact):
    """For each app with all 4 policies present, comparisons count is
    4 × 3 = 12 (ordered pairs a ≠ b)."""
    for app, entry in artifact["per_app"].items():
        present = {pol for pol in POLICIES if pol in entry["per_policy"]}
        expected = len(present) * (len(present) - 1)
        assert len(entry["comparisons"]) == expected, (
            f"{app}: comparisons count {len(entry['comparisons'])} ≠ "
            f"{expected} (n_present={len(present)})"
        )


def test_comparison_cliffs_delta_matches_recompute(artifact, by_app_pol):
    for app, entry in artifact["per_app"].items():
        for c in entry["comparisons"]:
            xs = by_app_pol[(app, c["a"])]
            ys = by_app_pol[(app, c["b"])]
            expected = round(_cliffs_delta(xs, ys), 4)
            assert c["cliffs_delta_a_minus_b"] == expected, (
                f"{app}/{c['a']}_vs_{c['b']}: Cliff's delta drift"
            )


def test_comparison_magnitude_matches_classifier(artifact):
    for app, entry in artifact["per_app"].items():
        for c in entry["comparisons"]:
            expected = _magnitude(c["cliffs_delta_a_minus_b"])
            assert c["magnitude"] == expected, (
                f"{app}/{c['a']}_vs_{c['b']}: magnitude drift"
            )


def test_comparison_n_a_and_n_b_match_per_policy(artifact):
    for app, entry in artifact["per_app"].items():
        per_pol = entry["per_policy"]
        for c in entry["comparisons"]:
            assert c["n_a"] == per_pol[c["a"]]["n"]
            assert c["n_b"] == per_pol[c["b"]]["n"]


def test_comparison_mannwhitney_u_and_p_match_recompute(artifact, by_app_pol):
    for app, entry in artifact["per_app"].items():
        for c in entry["comparisons"]:
            xs = by_app_pol[(app, c["a"])]
            ys = by_app_pol[(app, c["b"])]
            u, p = _mannwhitney_u(xs, ys)
            assert c["mannwhitney_u"] == round(u, 2), (
                f"{app}/{c['a']}_vs_{c['b']}: U drift "
                f"{c['mannwhitney_u']} ≠ {round(u, 2)}"
            )
            assert c["mannwhitney_p"] == round(p, 6), (
                f"{app}/{c['a']}_vs_{c['b']}: p drift "
                f"{c['mannwhitney_p']} ≠ {round(p, 6)}"
            )


def test_comparison_stochastically_smaller_matches_sign(artifact):
    """stochastically_smaller = a if d < 0, b if d > 0, 'tie' if d == 0."""
    for app, entry in artifact["per_app"].items():
        for c in entry["comparisons"]:
            d = c["cliffs_delta_a_minus_b"]
            if d < 0:
                expected = c["a"]
            elif d > 0:
                expected = c["b"]
            else:
                expected = "tie"
            assert c["stochastically_smaller"] == expected, (
                f"{app}/{c['a']}_vs_{c['b']}: smaller-side drift"
            )


# ----------------------------------------------------------------------
# Group D: large_negative_deltas filter + sort
# ----------------------------------------------------------------------

def test_large_negative_deltas_filter_matches(artifact):
    """Filter: magnitude == 'large' AND delta < 0, sorted ASC by delta."""
    expected = []
    for app, entry in artifact["per_app"].items():
        for c in entry["comparisons"]:
            if c["magnitude"] == "large" and c["cliffs_delta_a_minus_b"] < 0:
                expected.append({"app": app, **c})
    expected.sort(key=lambda r: r["cliffs_delta_a_minus_b"])
    assert artifact["large_negative_deltas"] == expected, (
        "large_negative_deltas drift — filter or sort changed"
    )


def test_large_negative_deltas_all_have_negative_delta(artifact):
    for r in artifact["large_negative_deltas"]:
        assert r["cliffs_delta_a_minus_b"] < 0, (
            f"large_negative_deltas entry has non-negative delta: {r}"
        )


def test_large_negative_deltas_all_large_magnitude(artifact):
    for r in artifact["large_negative_deltas"]:
        assert r["magnitude"] == "large", (
            f"large_negative_deltas entry has non-large magnitude: {r}"
        )


def test_large_negative_deltas_sorted_asc_by_delta(artifact):
    """Sort key is `lambda r: r['cliffs_delta_a_minus_b']` — most
    negative first."""
    deltas = [r["cliffs_delta_a_minus_b"]
              for r in artifact["large_negative_deltas"]]
    assert deltas == sorted(deltas), (
        f"large_negative_deltas not sorted ASC: {deltas}"
    )


# ----------------------------------------------------------------------
# Group E: end-to-end sanity
# ----------------------------------------------------------------------

def test_cliffs_delta_in_unit_band(artifact):
    """Cliff's delta is bounded in [−1, +1] by construction."""
    for app, entry in artifact["per_app"].items():
        for c in entry["comparisons"]:
            d = c["cliffs_delta_a_minus_b"]
            assert -1.0 - 1e-9 <= d <= 1.0 + 1e-9, (
                f"{app}/{c['a']}_vs_{c['b']}: delta {d} outside [−1, +1]"
            )


def test_mannwhitney_p_in_unit_band(artifact):
    for app, entry in artifact["per_app"].items():
        for c in entry["comparisons"]:
            p = c["mannwhitney_p"]
            assert 0.0 - 1e-9 <= p <= 1.0 + 1e-9, (
                f"{app}/{c['a']}_vs_{c['b']}: p {p} outside [0, 1]"
            )


def test_per_policy_min_le_median_le_max(artifact):
    for app, entry in artifact["per_app"].items():
        for pol, block in entry["per_policy"].items():
            assert block["min"] <= block["median"] <= block["max"], (
                f"{app}/{pol}: min/median/max ordering violated"
            )
