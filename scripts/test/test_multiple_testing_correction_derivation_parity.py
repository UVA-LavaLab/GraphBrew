"""Gate 183 — multiple_testing_correction derivation parity.

Reconstruct ``wiki/data/multiple_testing_correction.json`` from scratch by
walking its three upstream artifacts (oracle_gap_effect_size,
oracle_gap_by_app_bootstrap, popt_vs_grasp_by_family_app), collecting their
p-values with the same deduplication rules, and re-applying Holm-Bonferroni
+ Benjamini-Hochberg corrections from first principles.

Load-bearing rules being locked:

* ALPHA = 0.05 (per-test, also FDR level q).
* Mann-Whitney p-values come from per_app[app].comparisons[*].mannwhitney_p,
  ALREADY two-sided — NOT halved or doubled. Unordered pair dedup uses
  ``tuple(sorted([a, b]))``.
* Bootstrap p_a_lt_b is ONE-sided and converted via
  ``min(1.0, 2.0 * min(p, 1-p))`` — NOT just 2*p.
* popt_vs_grasp_by_family_app.p_popt_lt_grasp is also one-sided → two-sided
  via the same conversion; rows with ``None`` are skipped.
* Iteration order: dict iteration over per_app and per_family_app — preserves
  insertion order; rebuilder must mirror exactly.
* Holm-Bonferroni step-down: sort by p ASC; threshold = α / (n - rank + 1);
  rejection STOPS at the first p > threshold and ALL subsequent are
  non-survivors (rejected_so_far chains the survives flag — load-bearing).
* Benjamini-Hochberg step-up: largest k with p_(k) ≤ k/n · q; reject all
  H_(1)..H_(k); survives := rank ≤ largest_k (NOT ≤ k/n · q per-row).
* expected_false_positives_at_alpha = round(α · n_tests, 3) — 3dp.
* by_source aggregates the n_tests, naive_significant, hb_survivors,
  bh_survivors counters with int(bool) casting.
* JSON: sort_keys=True with trailing newline.

The whole gate runs offline against committed JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"
ARTIFACT = WIKI_DATA / "multiple_testing_correction.json"

EFFECT_SIZE = WIKI_DATA / "oracle_gap_effect_size.json"
BOOTSTRAP = WIKI_DATA / "oracle_gap_by_app_bootstrap.json"
FAMILY_APP = WIKI_DATA / "popt_vs_grasp_by_family_app.json"

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Reference rebuilders (mirror generator)
# ---------------------------------------------------------------------------


def _two_sided(p_one: float) -> float:
    return min(1.0, 2.0 * min(p_one, 1.0 - p_one))


def _collect_p_values() -> list[dict]:
    rows: list[dict] = []
    es = json.loads(EFFECT_SIZE.read_text())
    for app, app_block in es.get("per_app", {}).items():
        seen: set[tuple[str, str]] = set()
        for payload in app_block.get("comparisons", []):
            a = payload.get("a")
            b = payload.get("b")
            if a is None or b is None:
                continue
            ord_pair = tuple(sorted([a, b]))
            if ord_pair in seen:
                continue
            seen.add(ord_pair)
            rows.append({
                "source": "mannwhitney_gap",
                "scope": f"app={app}",
                "label": f"{ord_pair[0]} vs {ord_pair[1]}",
                "p_two_sided": float(payload["mannwhitney_p"]),
            })
    bs = json.loads(BOOTSTRAP.read_text())
    for app, pairs in bs.get("per_app_pairs", {}).items():
        seen = set()
        for pair_key, payload in pairs.items():
            if "_vs_" not in pair_key:
                continue
            a, b = pair_key.split("_vs_", 1)
            ord_pair = tuple(sorted([a, b]))
            if ord_pair in seen:
                continue
            seen.add(ord_pair)
            rows.append({
                "source": "bootstrap_paired_gap",
                "scope": f"app={app}",
                "label": f"{ord_pair[0]} vs {ord_pair[1]}",
                "p_two_sided": _two_sided(float(payload["p_a_lt_b"])),
            })
    fa = json.loads(FAMILY_APP.read_text())
    for fam_app, payload in fa.get("per_family_app", {}).items():
        p_raw = payload.get("p_popt_lt_grasp")
        if p_raw is None:
            continue
        rows.append({
            "source": "popt_vs_grasp_family_app",
            "scope": fam_app,
            "label": "POPT vs GRASP",
            "p_two_sided": _two_sided(float(p_raw)),
        })
    return rows


def _holm_bonferroni(p_values: list[float], alpha: float) -> list[dict]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    out: list[dict] = []
    rejected_so_far = True
    for rank, (orig_i, p) in enumerate(indexed, start=1):
        threshold = alpha / (n - rank + 1)
        survives = rejected_so_far and (p <= threshold)
        if not survives:
            rejected_so_far = False
        out.append({
            "rank": rank,
            "orig_index": orig_i,
            "p": p,
            "threshold": threshold,
            "survives": survives,
        })
    return out


def _benjamini_hochberg(p_values: list[float], q: float) -> list[dict]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    largest_k = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= rank / n * q:
            largest_k = rank
    out: list[dict] = []
    for rank, (orig_i, p) in enumerate(indexed, start=1):
        out.append({
            "rank": rank,
            "orig_index": orig_i,
            "p": p,
            "threshold": rank / n * q,
            "survives": rank <= largest_k,
        })
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def published() -> dict:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def expected_rows() -> list[dict]:
    return _collect_p_values()


@pytest.fixture(scope="module")
def expected_hb(expected_rows) -> list[dict]:
    return _holm_bonferroni([r["p_two_sided"] for r in expected_rows], ALPHA)


@pytest.fixture(scope="module")
def expected_bh(expected_rows) -> list[dict]:
    return _benjamini_hochberg([r["p_two_sided"] for r in expected_rows], ALPHA)


# ---------------------------------------------------------------------------
# Group 1 — Schema / meta
# ---------------------------------------------------------------------------


def test_top_keys(published):
    assert set(published.keys()) == {
        "meta",
        "by_source",
        "all_tests",
        "holm_bonferroni_ladder",
        "benjamini_hochberg_ladder",
    }


def test_meta_field_set(published):
    assert set(published["meta"].keys()) == {
        "alpha",
        "n_tests",
        "naive_significant_count",
        "holm_bonferroni_survivor_count",
        "benjamini_hochberg_survivor_count",
        "expected_false_positives_at_alpha",
    }


def test_meta_alpha_is_005(published):
    assert published["meta"]["alpha"] == 0.05


def test_n_tests_matches_collected(published, expected_rows):
    assert published["meta"]["n_tests"] == len(expected_rows)
    assert len(published["all_tests"]) == len(expected_rows)
    assert len(published["holm_bonferroni_ladder"]) == len(expected_rows)
    assert len(published["benjamini_hochberg_ladder"]) == len(expected_rows)


def test_expected_false_positives_formula(published):
    m = published["meta"]
    assert m["expected_false_positives_at_alpha"] == round(
        m["alpha"] * m["n_tests"], 3
    )


# ---------------------------------------------------------------------------
# Group 2 — P-value collection + two-sided conversion
# ---------------------------------------------------------------------------


def test_all_tests_match_rederive(published, expected_rows):
    # Strip the annotations and compare raw rows in order.
    raw = [
        {
            "source": r["source"],
            "scope": r["scope"],
            "label": r["label"],
            "p_two_sided": r["p_two_sided"],
        }
        for r in published["all_tests"]
    ]
    assert raw == expected_rows


def test_annotations_consistent_with_meta_counts(published):
    naive = sum(1 for r in published["all_tests"] if r["naive_significant_at_alpha"])
    hb = sum(1 for r in published["all_tests"] if r["holm_bonferroni_survives"])
    bh = sum(1 for r in published["all_tests"] if r["benjamini_hochberg_survives"])
    m = published["meta"]
    assert naive == m["naive_significant_count"]
    assert hb == m["holm_bonferroni_survivor_count"]
    assert bh == m["benjamini_hochberg_survivor_count"]


def test_naive_significant_uses_le_alpha(published):
    a = published["meta"]["alpha"]
    for r in published["all_tests"]:
        assert r["naive_significant_at_alpha"] == (r["p_two_sided"] <= a), r


def test_two_sided_conversion_bootstrap(published):
    bs = json.loads(BOOTSTRAP.read_text())
    expected: dict[tuple[str, str], float] = {}
    for app, pairs in bs.get("per_app_pairs", {}).items():
        seen: set[tuple[str, str]] = set()
        for pair_key, payload in pairs.items():
            if "_vs_" not in pair_key:
                continue
            a, b = pair_key.split("_vs_", 1)
            ord_pair = tuple(sorted([a, b]))
            if ord_pair in seen:
                continue
            seen.add(ord_pair)
            scope_label = (f"app={app}", f"{ord_pair[0]} vs {ord_pair[1]}")
            expected[scope_label] = _two_sided(float(payload["p_a_lt_b"]))

    for r in published["all_tests"]:
        if r["source"] != "bootstrap_paired_gap":
            continue
        key = (r["scope"], r["label"])
        assert r["p_two_sided"] == pytest.approx(expected[key], abs=1e-15), r


def test_mannwhitney_p_used_verbatim(published):
    es = json.loads(EFFECT_SIZE.read_text())
    expected: dict[tuple[str, str], float] = {}
    for app, app_block in es.get("per_app", {}).items():
        seen: set[tuple[str, str]] = set()
        for payload in app_block.get("comparisons", []):
            a = payload.get("a")
            b = payload.get("b")
            if a is None or b is None:
                continue
            ord_pair = tuple(sorted([a, b]))
            if ord_pair in seen:
                continue
            seen.add(ord_pair)
            scope_label = (f"app={app}", f"{ord_pair[0]} vs {ord_pair[1]}")
            expected[scope_label] = float(payload["mannwhitney_p"])

    for r in published["all_tests"]:
        if r["source"] != "mannwhitney_gap":
            continue
        key = (r["scope"], r["label"])
        assert r["p_two_sided"] == pytest.approx(expected[key], abs=1e-15), r


def test_family_app_skips_none_p(published):
    fa = json.loads(FAMILY_APP.read_text())
    n_published = sum(
        1 for r in published["all_tests"] if r["source"] == "popt_vs_grasp_family_app"
    )
    n_non_none = sum(
        1 for p in fa.get("per_family_app", {}).values()
        if p.get("p_popt_lt_grasp") is not None
    )
    assert n_published == n_non_none


def test_unordered_pair_dedup_in_each_source(published):
    by_source: dict[str, set] = {}
    for r in published["all_tests"]:
        if r["source"] == "popt_vs_grasp_family_app":
            key = (r["scope"], r["label"])
        else:
            key = (r["scope"], r["label"])
        by_source.setdefault(r["source"], set())
        assert key not in by_source[r["source"]], (r["source"], key)
        by_source[r["source"]].add(key)


# ---------------------------------------------------------------------------
# Group 3 — Holm-Bonferroni step-down semantics
# ---------------------------------------------------------------------------


def test_holm_bonferroni_ladder_byte_equivalent(published, expected_hb):
    assert published["holm_bonferroni_ladder"] == expected_hb


def test_holm_bonferroni_threshold_formula(published):
    n = published["meta"]["n_tests"]
    a = published["meta"]["alpha"]
    for entry in published["holm_bonferroni_ladder"]:
        assert entry["threshold"] == a / (n - entry["rank"] + 1)


def test_holm_bonferroni_sorted_by_p_asc(published):
    ps = [e["p"] for e in published["holm_bonferroni_ladder"]]
    assert ps == sorted(ps)


def test_holm_bonferroni_rejection_chains(published):
    # Once survives goes False, it never goes back to True.
    seen_false = False
    for entry in published["holm_bonferroni_ladder"]:
        if seen_false:
            assert entry["survives"] is False, entry
        if not entry["survives"]:
            seen_false = True


def test_holm_bonferroni_survivor_count_matches_meta(published):
    n_surv = sum(1 for e in published["holm_bonferroni_ladder"] if e["survives"])
    assert n_surv == published["meta"]["holm_bonferroni_survivor_count"]


def test_holm_bonferroni_orig_index_set_complete(published):
    n = published["meta"]["n_tests"]
    indices = {e["orig_index"] for e in published["holm_bonferroni_ladder"]}
    assert indices == set(range(n))


# ---------------------------------------------------------------------------
# Group 4 — Benjamini-Hochberg step-up semantics
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_ladder_byte_equivalent(published, expected_bh):
    assert published["benjamini_hochberg_ladder"] == expected_bh


def test_benjamini_hochberg_threshold_formula(published):
    n = published["meta"]["n_tests"]
    q = published["meta"]["alpha"]
    for entry in published["benjamini_hochberg_ladder"]:
        assert entry["threshold"] == entry["rank"] / n * q


def test_benjamini_hochberg_sorted_by_p_asc(published):
    ps = [e["p"] for e in published["benjamini_hochberg_ladder"]]
    assert ps == sorted(ps)


def test_benjamini_hochberg_largest_k_rule(published):
    # All survivors must have rank ≤ largest_k where largest_k is the
    # MAX rank with p ≤ threshold. Compute largest_k from the ladder.
    largest_k = 0
    for e in published["benjamini_hochberg_ladder"]:
        if e["p"] <= e["threshold"]:
            largest_k = e["rank"]
    for e in published["benjamini_hochberg_ladder"]:
        assert e["survives"] == (e["rank"] <= largest_k), e


def test_benjamini_hochberg_dominates_holm_bonferroni(published):
    # FDR-only is a less stringent correction; every HB survivor must also
    # be a BH survivor (over the same family).
    hb_set = {e["orig_index"] for e in published["holm_bonferroni_ladder"] if e["survives"]}
    bh_set = {e["orig_index"] for e in published["benjamini_hochberg_ladder"] if e["survives"]}
    assert hb_set.issubset(bh_set), (hb_set - bh_set)


def test_benjamini_hochberg_survivor_count_matches_meta(published):
    n_surv = sum(1 for e in published["benjamini_hochberg_ladder"] if e["survives"])
    assert n_surv == published["meta"]["benjamini_hochberg_survivor_count"]


# ---------------------------------------------------------------------------
# Group 5 — Per-source aggregation
# ---------------------------------------------------------------------------


def test_by_source_keys_match_observed_sources(published):
    sources_in_tests = {r["source"] for r in published["all_tests"]}
    assert set(published["by_source"].keys()) == sources_in_tests


def test_by_source_field_set(published):
    for src, s in published["by_source"].items():
        assert set(s.keys()) == {
            "n_tests",
            "naive_significant",
            "hb_survivors",
            "bh_survivors",
        }, src


def test_by_source_counts_match_rederive(published):
    expected: dict[str, dict] = {}
    for r in published["all_tests"]:
        src = r["source"]
        s = expected.setdefault(src, {
            "n_tests": 0,
            "naive_significant": 0,
            "hb_survivors": 0,
            "bh_survivors": 0,
        })
        s["n_tests"] += 1
        s["naive_significant"] += int(r["naive_significant_at_alpha"])
        s["hb_survivors"] += int(r["holm_bonferroni_survives"])
        s["bh_survivors"] += int(r["benjamini_hochberg_survives"])
    assert published["by_source"] == expected


def test_by_source_n_tests_sum_to_meta_total(published):
    total = sum(s["n_tests"] for s in published["by_source"].values())
    assert total == published["meta"]["n_tests"]


def test_by_source_hb_le_naive_le_n(published):
    for src, s in published["by_source"].items():
        assert s["hb_survivors"] <= s["bh_survivors"] <= s["naive_significant"] <= s["n_tests"], src
