"""
Confidence gate 117 — multiple_testing_correction internal arithmetic.

Locks the multi-test correction artifact that runs Holm-Bonferroni
(step-down family-wise error rate, Holm 1979) and Benjamini-Hochberg
(step-up false-discovery rate, Benjamini & Hochberg 1995) on the
combined family of 81 p-values drawn from three sources:
- bootstrap_paired_gap   (30 tests)
- mannwhitney_gap        (30 tests)
- popt_vs_grasp_family_app (21 tests)

The artifact reports survivor counts at α=0.05 (HB=28, BH=40, naive=44)
and two per-test ladders. This gate recomputes both ladders from
first principles:

- HB ladder: tests sorted by p ascending; rank k threshold = α/(n−k+1);
  survives iff p_k ≤ threshold_k AND all earlier ranks survived
  (step-down semantics).
- BH ladder: tests sorted by p ascending; rank k threshold = α·k/n;
  survives iff rank ≤ k_max where k_max is the largest rank with
  p_k ≤ threshold_k (step-up semantics).

The gate also cross-links all_tests[i] (the source-table view) to the
ladder entries with orig_index=i.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "wiki" / "data" / "multiple_testing_correction.json"

EXPECTED_ALPHA = 0.05
EXPECTED_N_TESTS = 81
EXPECTED_SOURCES = {"bootstrap_paired_gap", "mannwhitney_gap", "popt_vs_grasp_family_app"}
EXPECTED_SOURCE_COUNTS = {
    "bootstrap_paired_gap": 30,
    "mannwhitney_gap": 30,
    "popt_vs_grasp_family_app": 21,
}
THRESH_TOL = 1e-9


@pytest.fixture(scope="module")
def doc() -> dict:
    assert ARTIFACT_PATH.exists(), f"missing artifact: {ARTIFACT_PATH}"
    return json.loads(ARTIFACT_PATH.read_text())


# ---------------------------------------------------------------------------
# Group A — top-level structure + meta
# ---------------------------------------------------------------------------

def test_top_level_keys_and_meta(doc):
    assert set(doc) == {"all_tests", "benjamini_hochberg_ladder", "by_source",
                        "holm_bonferroni_ladder", "meta"}
    m = doc["meta"]
    assert m["alpha"] == EXPECTED_ALPHA
    assert m["n_tests"] == EXPECTED_N_TESTS
    assert len(doc["all_tests"]) == EXPECTED_N_TESTS
    assert len(doc["holm_bonferroni_ladder"]) == EXPECTED_N_TESTS
    assert len(doc["benjamini_hochberg_ladder"]) == EXPECTED_N_TESTS


def test_meta_survivor_counts_match_all_tests(doc):
    m = doc["meta"]
    naive_recomp = sum(1 for t in doc["all_tests"] if t["naive_significant_at_alpha"])
    hb_recomp = sum(1 for t in doc["all_tests"] if t["holm_bonferroni_survives"])
    bh_recomp = sum(1 for t in doc["all_tests"] if t["benjamini_hochberg_survives"])
    assert m["naive_significant_count"] == naive_recomp
    assert m["holm_bonferroni_survivor_count"] == hb_recomp
    assert m["benjamini_hochberg_survivor_count"] == bh_recomp


def test_meta_expected_false_positives_is_alpha_times_n(doc):
    m = doc["meta"]
    expected = m["alpha"] * m["n_tests"]
    assert abs(m["expected_false_positives_at_alpha"] - expected) < 1e-9


def test_by_source_partition_matches_all_tests(doc):
    counts = collections.Counter(t["source"] for t in doc["all_tests"])
    assert set(counts) == EXPECTED_SOURCES
    assert dict(counts) == EXPECTED_SOURCE_COUNTS

    for source, expected_n in EXPECTED_SOURCE_COUNTS.items():
        bd = doc["by_source"][source]
        assert bd["n_tests"] == expected_n, f"{source}: n_tests mismatch"

    # Survivor counts in by_source must match per-source filter of all_tests
    for source in EXPECTED_SOURCES:
        subset = [t for t in doc["all_tests"] if t["source"] == source]
        bd = doc["by_source"][source]
        assert bd["naive_significant"] == sum(1 for t in subset if t["naive_significant_at_alpha"])
        assert bd["hb_survivors"] == sum(1 for t in subset if t["holm_bonferroni_survives"])
        assert bd["bh_survivors"] == sum(1 for t in subset if t["benjamini_hochberg_survives"])


# ---------------------------------------------------------------------------
# Group B — Holm-Bonferroni step-down arithmetic
# ---------------------------------------------------------------------------

def test_hb_ladder_sorted_by_p_with_full_rank_set(doc):
    hb = doc["holm_bonferroni_ladder"]
    for i in range(len(hb) - 1):
        assert hb[i]["p"] <= hb[i + 1]["p"], f"hb ladder not ascending at rank {i}"
    assert [t["rank"] for t in hb] == list(range(1, EXPECTED_N_TESTS + 1))
    # orig_index covers 0..n-1 exactly once
    assert sorted(t["orig_index"] for t in hb) == list(range(EXPECTED_N_TESTS))


def test_hb_thresholds_follow_alpha_over_n_minus_rank_plus_one(doc):
    alpha = doc["meta"]["alpha"]
    n = doc["meta"]["n_tests"]
    for t in doc["holm_bonferroni_ladder"]:
        expected = alpha / (n - t["rank"] + 1)
        assert abs(t["threshold"] - expected) < THRESH_TOL, (
            f"hb rank {t['rank']}: threshold={t['threshold']} expected={expected}"
        )


def test_hb_step_down_survival_logic(doc):
    """Holm step-down: survives iff p_k ≤ threshold_k AND every earlier
    rank also survived. Once one fails, all subsequent ranks fail."""
    prev_all_survive = True
    for t in doc["holm_bonferroni_ladder"]:
        expected = prev_all_survive and (t["p"] <= t["threshold"])
        assert t["survives"] == expected, (
            f"hb rank {t['rank']}: survives={t['survives']} expected={expected} "
            f"(p={t['p']}, threshold={t['threshold']}, prev_all_survive={prev_all_survive})"
        )
        prev_all_survive = expected


# ---------------------------------------------------------------------------
# Group C — Benjamini-Hochberg step-up arithmetic
# ---------------------------------------------------------------------------

def test_bh_ladder_sorted_by_p_with_full_rank_set(doc):
    bh = doc["benjamini_hochberg_ladder"]
    for i in range(len(bh) - 1):
        assert bh[i]["p"] <= bh[i + 1]["p"], f"bh ladder not ascending at rank {i}"
    assert [t["rank"] for t in bh] == list(range(1, EXPECTED_N_TESTS + 1))
    assert sorted(t["orig_index"] for t in bh) == list(range(EXPECTED_N_TESTS))


def test_bh_thresholds_follow_alpha_times_rank_over_n(doc):
    alpha = doc["meta"]["alpha"]
    n = doc["meta"]["n_tests"]
    for t in doc["benjamini_hochberg_ladder"]:
        expected = alpha * t["rank"] / n
        assert abs(t["threshold"] - expected) < THRESH_TOL, (
            f"bh rank {t['rank']}: threshold={t['threshold']} expected={expected}"
        )


def test_bh_step_up_survival_logic(doc):
    """BH step-up: find k_max = max rank where p_k ≤ threshold_k.
    Then survives iff rank ≤ k_max (every test up to and including
    k_max survives, even if its own p > threshold)."""
    bh = doc["benjamini_hochberg_ladder"]
    k_max = 0
    for t in bh:
        if t["p"] <= t["threshold"]:
            k_max = t["rank"]
    for t in bh:
        expected = (t["rank"] <= k_max)
        assert t["survives"] == expected, (
            f"bh rank {t['rank']}: survives={t['survives']} expected={expected} "
            f"(k_max={k_max}, p={t['p']}, threshold={t['threshold']})"
        )


def test_bh_at_least_as_lenient_as_hb(doc):
    """BH always survives ≥ as many tests as HB at the same alpha (a
    standard guarantee of FDR vs FWER); this also catches direction
    swaps where the two ladders are accidentally crossed."""
    assert (doc["meta"]["benjamini_hochberg_survivor_count"]
            >= doc["meta"]["holm_bonferroni_survivor_count"])


# ---------------------------------------------------------------------------
# Group D — cross-link between all_tests and ladders
# ---------------------------------------------------------------------------

def test_all_tests_naive_flag_matches_p_le_alpha(doc):
    alpha = doc["meta"]["alpha"]
    for i, t in enumerate(doc["all_tests"]):
        expected = (t["p_two_sided"] <= alpha)
        assert t["naive_significant_at_alpha"] == expected, (
            f"all_tests[{i}] naive: got {t['naive_significant_at_alpha']} "
            f"but p={t['p_two_sided']} alpha={alpha} → expected {expected}"
        )


def test_all_tests_survives_flags_match_ladder_entries(doc):
    """The boolean fields on each all_tests[i] entry must agree with the
    survives flag of the ladder entry whose orig_index == i."""
    hb_by_idx = {t["orig_index"]: t for t in doc["holm_bonferroni_ladder"]}
    bh_by_idx = {t["orig_index"]: t for t in doc["benjamini_hochberg_ladder"]}
    for i, test in enumerate(doc["all_tests"]):
        hb_entry = hb_by_idx[i]
        bh_entry = bh_by_idx[i]
        assert test["holm_bonferroni_survives"] == hb_entry["survives"], (
            f"all_tests[{i}] HB: got {test['holm_bonferroni_survives']} but ladder says {hb_entry['survives']}"
        )
        assert test["benjamini_hochberg_survives"] == bh_entry["survives"], (
            f"all_tests[{i}] BH: got {test['benjamini_hochberg_survives']} but ladder says {bh_entry['survives']}"
        )
        # ladder.p must equal all_tests.p_two_sided
        assert abs(hb_entry["p"] - test["p_two_sided"]) < 1e-9
        assert abs(bh_entry["p"] - test["p_two_sided"]) < 1e-9
