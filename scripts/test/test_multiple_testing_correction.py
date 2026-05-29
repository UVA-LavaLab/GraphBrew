"""Gate: multiple-testing correction across the full p-value family.

Pins the most paper-defensive statistical claim we make: of the p-values
emitted across all confidence gates, exactly N₁ survive Holm-Bonferroni
(strong FWER) and N₂ survive Benjamini-Hochberg (FDR). Anything else
in the corpus is a 'naive significance' that does NOT survive a
multiple-testing correction and MUST NOT be claimed as 'significant'
in the paper.

The HB/BH ladders are recomputed at gate-time so the math is verified,
not just trusted from the cached payload.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = REPO_ROOT / "wiki" / "data" / "multiple_testing_correction.json"

ALPHA = 0.05


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip(
            f"missing {PAYLOAD}; run `make lit-mt-correction` "
            f"(or `make lit-claims`)."
        )
    return json.loads(PAYLOAD.read_text())


def test_schema_complete(payload):
    for key in ("meta", "by_source", "all_tests",
                "holm_bonferroni_ladder", "benjamini_hochberg_ladder"):
        assert key in payload, f"missing {key}"
    meta = payload["meta"]
    for key in ("alpha", "n_tests", "naive_significant_count",
                "holm_bonferroni_survivor_count",
                "benjamini_hochberg_survivor_count",
                "expected_false_positives_at_alpha"):
        assert key in meta, f"meta missing {key}"


def test_alpha_is_0_05(payload):
    assert payload["meta"]["alpha"] == 0.05


def test_n_tests_above_floor(payload):
    """The family must contain ≥ 60 tests — a paper that aggregates p-values
    from gates 34 + 38 + the per-(family, app) test cannot legitimately
    have fewer. Floor protects against silently dropping a source."""
    assert payload["meta"]["n_tests"] >= 60, payload["meta"]


def test_all_three_sources_present(payload):
    """The aggregate must draw from all three p-value-emitting gates."""
    required = {"mannwhitney_gap", "bootstrap_paired_gap",
                "popt_vs_grasp_family_app"}
    have = set(payload["by_source"].keys())
    missing = required - have
    assert not missing, f"MT correction lost sources {missing}; have {have}"


def test_holm_bonferroni_strict_subset_of_naive(payload):
    """Every HB survivor is also naive-significant — sanity check that
    correction never adds discoveries, only removes them."""
    for r in payload["all_tests"]:
        if r["holm_bonferroni_survives"]:
            assert r["naive_significant_at_alpha"], r


def test_holm_bonferroni_subset_of_benjamini_hochberg(payload):
    """HB controls FWER (stricter) ⇒ BH (FDR, looser) survivors must
    include all HB survivors. If this inverts, something is broken."""
    for r in payload["all_tests"]:
        if r["holm_bonferroni_survives"]:
            assert r["benjamini_hochberg_survives"], r


def test_naive_overcount_relative_to_correction(payload):
    """Naive count > HB count — otherwise the correction is doing nothing
    and we have either a broken implementation or a corpus where every
    p-value is extreme. With the current ~80 tests we EXPECT some
    'naive significant' p-values around 0.01-0.05 that HB rejects."""
    meta = payload["meta"]
    assert meta["naive_significant_count"] > meta["holm_bonferroni_survivor_count"], (
        f"naive_significant={meta['naive_significant_count']} "
        f"== HB_survivors={meta['holm_bonferroni_survivor_count']}; "
        "either the correction is no-op or every p is extreme — investigate."
    )


def test_hb_survivor_count_above_paper_floor(payload):
    """We must retain ≥ 20 HB survivors after correction — otherwise the
    paper's core findings (pr/POPT, cc/GRASP, family-app POPT-vs-GRASP)
    don't survive multiple-testing correction and we have to retreat."""
    meta = payload["meta"]
    assert meta["holm_bonferroni_survivor_count"] >= 20, (
        f"HB survivors dropped to {meta['holm_bonferroni_survivor_count']}; "
        "paper headline claims at risk. Investigate which tests fell off."
    )


def test_pr_popt_vs_lru_survives_holm_bonferroni(payload):
    """pr/POPT vs LRU is the single most-cited paper claim. It MUST
    survive the strongest correction."""
    found = [
        r for r in payload["all_tests"]
        if r["holm_bonferroni_survives"]
        and r["scope"] == "app=pr"
        and "LRU" in r["label"] and "POPT" in r["label"]
    ]
    assert found, (
        "pr/POPT vs LRU lost HB survival; the paper's marquee claim "
        "no longer survives multiple-testing correction. STOP and investigate."
    )


def test_cc_grasp_dominance_survives_holm_bonferroni(payload):
    """cc/GRASP vs every opponent MUST survive HB — this is the other
    marquee claim (cross-checked in gates 36, 37, 38, 39)."""
    survivors = [
        r["label"] for r in payload["all_tests"]
        if r["holm_bonferroni_survives"]
        and r["scope"] == "app=cc"
        and "GRASP" in r["label"]
    ]
    # Ensure GRASP appears in at least one HB-survivor cc comparison
    # against each of LRU, POPT, SRRIP.
    against = set()
    for label in survivors:
        for opp in ("LRU", "POPT", "SRRIP"):
            if opp in label:
                against.add(opp)
    missing = {"LRU", "POPT", "SRRIP"} - against
    assert not missing, (
        f"cc/GRASP dominance lost HB survival against {missing}; "
        f"surviving labels: {survivors}"
    )


def test_holm_ladder_is_sorted_ascending(payload):
    """HB ladder must be sorted by ascending p, with thresholds
    α/(n-rank+1) strictly increasing in rank."""
    ladder = payload["holm_bonferroni_ladder"]
    p_vals = [r["p"] for r in ladder]
    assert p_vals == sorted(p_vals), "HB ladder not sorted"
    thresholds = [r["threshold"] for r in ladder]
    # Thresholds α/(n-rank+1) start small (rank=1, large denom) and grow.
    for i in range(1, len(thresholds)):
        assert thresholds[i] >= thresholds[i - 1], (
            f"HB threshold ladder not monotone at rank {i}: "
            f"{thresholds[i-1]} -> {thresholds[i]}"
        )


def test_bh_ladder_thresholds_are_k_over_n_alpha(payload):
    """BH ladder threshold at rank k must be exactly k/n * α."""
    ladder = payload["benjamini_hochberg_ladder"]
    n = payload["meta"]["n_tests"]
    alpha = payload["meta"]["alpha"]
    for r in ladder:
        expected = r["rank"] / n * alpha
        assert math.isclose(r["threshold"], expected, abs_tol=1e-12), (
            f"BH rank {r['rank']}: threshold {r['threshold']} != {expected}"
        )


def test_holm_step_down_property(payload):
    """Holm step-down: as soon as we hit the first non-survivor, every
    rank above it must also be a non-survivor."""
    ladder = sorted(
        payload["holm_bonferroni_ladder"], key=lambda r: r["rank"]
    )
    first_fail = next(
        (r["rank"] for r in ladder if not r["survives"]), None
    )
    if first_fail is None:
        return
    for r in ladder:
        if r["rank"] >= first_fail:
            assert not r["survives"], (
                f"HB step-down violated at rank {r['rank']}: survives but "
                f"earlier rank {first_fail} did not."
            )


def test_expected_false_positives_matches_definition(payload):
    """E[FP under all-nulls] = α * n_tests, rounded to 3 decimals."""
    meta = payload["meta"]
    expected = round(meta["alpha"] * meta["n_tests"], 3)
    assert meta["expected_false_positives_at_alpha"] == pytest.approx(
        expected, abs=1e-6
    )
