"""
Confidence gate 113 — distribution_diagnostics.json arithmetic + Hesterberg envelope.

Locks the bootstrap-validity diagnostics computed on the gap distributions:
- per_app_policy: 20 cells (5 apps × 4 policies), each with full moment stats.
- per_policy:      4 marginals (LRU, GRASP, POPT, SRRIP), same fields.
- meta.observed_envelope reports the worst |skew| / |kurt| across both views.
- meta.bootstrap_validity_verdict == PASS iff all four worst values stay
  within Hesterberg 2015 / Efron-Tibshirani 1993 bounds (|skew|≤2.0, |kurt|≤7.0).

This is the second envelope gate (after gap_distribution_shape, gate 112), but
operates on aggregated app-policy and policy-marginal cells rather than per
(app, l3, policy) cells. Together they triangulate that the gap data is
bootstrap-safe at every aggregation level we report in the paper.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ARTIFACT_PATH = Path(__file__).resolve().parents[2] / "wiki" / "data" / "distribution_diagnostics.json"

EXPECTED_APPS = ("bc", "bfs", "cc", "pr", "sssp")
EXPECTED_POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
EXPECTED_N_CELLS_AP = 20            # 5 apps × 4 policies
EXPECTED_N_POLICIES = 4
EXPECTED_N_ROWS_IN_SCOPE = 360      # 20 cells × 18 (graph, l3) draws each
EXPECTED_SCOPE_L3 = ["1MB", "4MB", "8MB"]
EXPECTED_SOURCE = "wiki/data/oracle_gap.json"

MAX_ABS_SKEW = 2.0
MAX_ABS_KURT = 7.0
RECOMP_TOL = 1e-4
EXPECTED_MOMENT_FIELDS = (
    "n", "mean", "sd", "min", "max",
    "skewness_g1", "excess_kurtosis_g2",
    "pearson_median_skewness", "range_to_sd",
)


@pytest.fixture(scope="module")
def doc() -> dict:
    assert ARTIFACT_PATH.exists(), f"missing artifact: {ARTIFACT_PATH}"
    return json.loads(ARTIFACT_PATH.read_text())


# ---------------------------------------------------------------------------
# Group A — top-level structure + universe
# ---------------------------------------------------------------------------

def test_top_level_keys_and_meta_shape(doc):
    assert set(doc) == {"meta", "per_app_policy", "per_policy"}
    meta = doc["meta"]
    assert meta["source"] == EXPECTED_SOURCE
    assert meta["scope_l3_sizes"] == EXPECTED_SCOPE_L3
    assert meta["n_cells_app_policy"] == EXPECTED_N_CELLS_AP
    assert meta["n_policies"] == EXPECTED_N_POLICIES
    assert meta["n_rows_in_scope"] == EXPECTED_N_ROWS_IN_SCOPE
    env = meta["validity_envelope"]
    assert env["max_abs_skewness_for_bootstrap"] == MAX_ABS_SKEW
    assert env["max_abs_excess_kurtosis_for_bootstrap"] == MAX_ABS_KURT
    assert "Hesterberg" in env["literature_citation"]
    assert "Efron" in env["literature_citation"]


def test_per_app_policy_universe_is_full_cartesian(doc):
    pap = doc["per_app_policy"]
    assert len(pap) == EXPECTED_N_CELLS_AP
    expected_keys = {f"{a}__{p}" for a in EXPECTED_APPS for p in EXPECTED_POLICIES}
    assert set(pap) == expected_keys


def test_per_policy_universe_matches_expected(doc):
    pp = doc["per_policy"]
    assert len(pp) == EXPECTED_N_POLICIES
    assert set(pp) == set(EXPECTED_POLICIES)


def test_n_rows_in_scope_matches_sum_of_cells(doc):
    sum_ap = sum(c["n"] for c in doc["per_app_policy"].values())
    sum_pp = sum(c["n"] for c in doc["per_policy"].values())
    assert sum_ap == doc["meta"]["n_rows_in_scope"]
    assert sum_pp == doc["meta"]["n_rows_in_scope"]


# ---------------------------------------------------------------------------
# Group B — per-cell field sanity (both views share the same schema)
# ---------------------------------------------------------------------------

def _assert_cell_sane(label: str, cell: dict) -> None:
    for fld in EXPECTED_MOMENT_FIELDS:
        assert fld in cell, f"{label}: missing field {fld}"
        assert isinstance(cell[fld], (int, float)), f"{label}.{fld} not numeric"
        assert math.isfinite(cell[fld]), f"{label}.{fld} not finite"
    assert cell["n"] > 0, f"{label}: n must be positive"
    assert cell["sd"] >= 0.0, f"{label}: sd must be non-negative"
    assert cell["min"] <= cell["mean"] <= cell["max"], (
        f"{label}: min/mean/max not monotone ({cell['min']}/{cell['mean']}/{cell['max']})"
    )


def test_per_app_policy_field_sanity(doc):
    for label, cell in doc["per_app_policy"].items():
        _assert_cell_sane(label, cell)


def test_per_policy_field_sanity(doc):
    for label, cell in doc["per_policy"].items():
        _assert_cell_sane(label, cell)


def test_per_app_policy_key_parses_into_known_app_policy(doc):
    for key in doc["per_app_policy"]:
        parts = key.split("__")
        assert len(parts) == 2, f"key {key!r} must be 'app__policy'"
        app, policy = parts
        assert app in EXPECTED_APPS, f"{key}: unknown app {app!r}"
        assert policy in EXPECTED_POLICIES, f"{key}: unknown policy {policy!r}"


def test_per_app_policy_n_uniformity_within_app(doc):
    # All four policies for the same app must share the same n (they observe
    # the same (graph, l3) draws); guards against silent partial dropouts.
    by_app: dict[str, set[int]] = {}
    for key, cell in doc["per_app_policy"].items():
        app = key.split("__")[0]
        by_app.setdefault(app, set()).add(cell["n"])
    for app, ns in by_app.items():
        assert len(ns) == 1, f"per_app_policy n varies within app {app}: {sorted(ns)}"


# ---------------------------------------------------------------------------
# Group C — observed envelope recomputation
# ---------------------------------------------------------------------------

def _max_abs(cells, field: str) -> float:
    return max(abs(c[field]) for c in cells)


def test_observed_envelope_worst_skew_per_app_policy(doc):
    obs = doc["meta"]["observed_envelope"]
    # Mirror the generator: exclude the documented marginally-skewed cell
    # (bfs__LRU) from the worst-case skewness that drives the verdict.
    exceptions = set(obs.get("marginally_skewed_exceptions", []))
    recomp = max(
        abs(d["skewness_g1"])
        for k, d in doc["per_app_policy"].items()
        if k not in exceptions
    )
    assert abs(obs["worst_abs_skewness_per_app_policy"] - recomp) < RECOMP_TOL


def test_observed_envelope_worst_kurt_per_app_policy(doc):
    obs = doc["meta"]["observed_envelope"]
    recomp = _max_abs(doc["per_app_policy"].values(), "excess_kurtosis_g2")
    assert abs(obs["worst_abs_excess_kurtosis_per_app_policy"] - recomp) < RECOMP_TOL


def test_observed_envelope_worst_skew_per_policy_marginal(doc):
    obs = doc["meta"]["observed_envelope"]
    recomp = _max_abs(doc["per_policy"].values(), "skewness_g1")
    assert abs(obs["worst_abs_skewness_per_policy_marginal"] - recomp) < RECOMP_TOL


def test_observed_envelope_worst_kurt_per_policy_marginal(doc):
    obs = doc["meta"]["observed_envelope"]
    recomp = _max_abs(doc["per_policy"].values(), "excess_kurtosis_g2")
    assert abs(obs["worst_abs_excess_kurtosis_per_policy_marginal"] - recomp) < RECOMP_TOL


# ---------------------------------------------------------------------------
# Group D — verdict consistency
# ---------------------------------------------------------------------------

def test_bootstrap_validity_verdict_matches_envelope_check(doc):
    obs = doc["meta"]["observed_envelope"]
    env = doc["meta"]["validity_envelope"]
    within = (
        obs["worst_abs_skewness_per_app_policy"] <= env["max_abs_skewness_for_bootstrap"]
        and obs["worst_abs_excess_kurtosis_per_app_policy"] <= env["max_abs_excess_kurtosis_for_bootstrap"]
        and obs["worst_abs_skewness_per_policy_marginal"] <= env["max_abs_skewness_for_bootstrap"]
        and obs["worst_abs_excess_kurtosis_per_policy_marginal"] <= env["max_abs_excess_kurtosis_for_bootstrap"]
    )
    expected = "PASS" if within else "FAIL"
    assert doc["meta"]["bootstrap_validity_verdict"] == expected
