"""Gate 110: bootstrap_ci.json + oracle_gap_by_app_bootstrap.json parity & internal hygiene.

This gate locks the two bootstrap-confidence-interval artifacts that underpin
every "this difference is statistically real" claim in the paper:

  * ``bootstrap_ci.json`` enriches the per-(policy,family) and per-(policy,
    regime) oracle-gap aggregates with 95% bootstrap CIs (5000 resamples),
    and includes a POPT-vs-GRASP family-level paired-delta plus a 7-row
    sign-stability table.
  * ``oracle_gap_by_app_bootstrap.json`` adds per-app paired bootstrap CIs
    (2000 resamples) for every directional policy pair, indexed by app.

Both must agree exactly with their oracle_gap point-estimate parents
(mean / median / n recompute) and must satisfy strict internal hygiene
(CI ordering, anti-symmetric paired statistics, sign-vs-CI consistency).
Without this gate, a regression in the bootstrap pipeline (e.g. a wrong
resampling unit, a stale mean carried over, or a CI direction flip)
would slide into the paper undetected — and every confidence claim
downstream rests on the integrity of these numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_JSON = REPO_ROOT / "wiki" / "data" / "bootstrap_ci.json"
OG_APP_BOOT_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap_by_app_bootstrap.json"
OGAP_JSON = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"

EXPECTED_CI_LEVEL = 0.95
EXPECTED_SEED = 1729
EXPECTED_BOOTSTRAP_MIN_RESAMPLES = 1000
EXPECTED_BOOTSTRAP_N_RESAMPLES = 5000  # bootstrap_ci main
EXPECTED_OG_APP_BOOT_N_RESAMPLES = 2000  # oracle_gap_by_app_bootstrap
EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}
EXPECTED_FAMILIES = {"citation", "mesh", "road", "social", "web"}
EXPECTED_REGIMES = {"large", "small", "tiny"}
EXPECTED_POLICY_FAMILY_KEYS = 20  # 4 policies × 5 families
EXPECTED_POLICY_REGIME_KEYS = 12  # 4 policies × 3 regimes
EXPECTED_PAIRS_PER_APP = 12  # 4 × 3 (excluding self-pairs)
EXPECTED_N_TOTAL_ROWS = 456

POINT_TOL = 1e-4
WIDTH_TOL = 1e-3
ANTISYM_TOL = 1e-4
# Bootstrap p_a_lt_b is computed from independent draws on both sides; ties
# plus sampling noise mean p(a<b) + p(b<a) can exceed 1.0 by a small margin.
# Allow up to 0.05 above 1.0 (observed max in the current artifact: 0.022).
P_SUM_TOL = 0.05


# ---------- fixtures ----------


@pytest.fixture(scope="module")
def bootstrap():
    assert BOOTSTRAP_JSON.exists(), f"missing: {BOOTSTRAP_JSON}"
    return json.loads(BOOTSTRAP_JSON.read_text())


@pytest.fixture(scope="module")
def og_app_boot():
    assert OG_APP_BOOT_JSON.exists(), f"missing: {OG_APP_BOOT_JSON}"
    return json.loads(OG_APP_BOOT_JSON.read_text())


@pytest.fixture(scope="module")
def oracle_gap():
    assert OGAP_JSON.exists(), f"missing: {OGAP_JSON}"
    return json.loads(OGAP_JSON.read_text())


# ---------- Group A: bootstrap_ci ↔ oracle_gap point-estimate parity (4) ----------


def test_policy_family_mean_median_n_match_oracle_gap(bootstrap, oracle_gap):
    boot = bootstrap["oracle_gap_by_policy_family"]
    src = oracle_gap["summary"]["by_policy_family"]
    assert len(boot) == EXPECTED_POLICY_FAMILY_KEYS, len(boot)
    assert set(boot) == set(src), set(boot) ^ set(src)
    for k, v in boot.items():
        assert abs(v["mean"] - src[k]["mean"]) < POINT_TOL, (k, v["mean"], src[k]["mean"])
        assert abs(v["median"] - src[k]["median"]) < POINT_TOL, (
            k, v["median"], src[k]["median"]
        )
        assert v["n"] == src[k]["n"], (k, v["n"], src[k]["n"])


def test_policy_regime_mean_median_n_match_oracle_gap(bootstrap, oracle_gap):
    boot = bootstrap["oracle_gap_by_policy_regime"]
    src = oracle_gap["summary"]["by_policy_regime"]
    assert len(boot) == EXPECTED_POLICY_REGIME_KEYS, len(boot)
    assert set(boot) == set(src), set(boot) ^ set(src)
    for k, v in boot.items():
        assert abs(v["mean"] - src[k]["mean"]) < POINT_TOL, (k, v["mean"], src[k]["mean"])
        assert abs(v["median"] - src[k]["median"]) < POINT_TOL, (
            k, v["median"], src[k]["median"]
        )
        assert v["n"] == src[k]["n"], (k, v["n"], src[k]["n"])


def test_popt_minus_grasp_by_family_keys_exact(bootstrap):
    fam = bootstrap["popt_minus_grasp_by_family"]
    assert set(fam) == EXPECTED_FAMILIES, set(fam) ^ EXPECTED_FAMILIES


def test_bootstrap_meta_constants(bootstrap, og_app_boot):
    m = bootstrap["meta"]
    assert m["ci_level"] == EXPECTED_CI_LEVEL, m["ci_level"]
    assert m["seed"] == EXPECTED_SEED, m["seed"]
    assert m["n_resamples"] == EXPECTED_BOOTSTRAP_N_RESAMPLES, m["n_resamples"]
    om = og_app_boot["meta"]
    assert om["ci_level"] == EXPECTED_CI_LEVEL, om["ci_level"]
    assert om["seed"] == EXPECTED_SEED, om["seed"]
    assert om["n_resamples"] == EXPECTED_OG_APP_BOOT_N_RESAMPLES, om["n_resamples"]
    assert om["n_total_rows"] == EXPECTED_N_TOTAL_ROWS, om["n_total_rows"]
    assert set(om["apps"]) == EXPECTED_APPS, set(om["apps"]) ^ EXPECTED_APPS
    assert set(om["policies"]) == EXPECTED_POLICIES, set(om["policies"]) ^ EXPECTED_POLICIES


# ---------- Group B: bootstrap_ci internal CI hygiene (4) ----------


def _walk_ci_sources(bootstrap):
    for src_name, src in (
        ("policy_family", bootstrap["oracle_gap_by_policy_family"]),
        ("policy_regime", bootstrap["oracle_gap_by_policy_regime"]),
    ):
        for k, v in src.items():
            yield f"{src_name}/{k}", v


def test_ci_lo_le_hi_and_width_consistent(bootstrap):
    for tag, v in _walk_ci_sources(bootstrap):
        assert v["ci_lo"] <= v["ci_hi"], (tag, v)
        assert abs(v["ci_width"] - (v["ci_hi"] - v["ci_lo"])) < WIDTH_TOL, (tag, v)
        assert v["ci_level"] == EXPECTED_CI_LEVEL, (tag, v)


def test_popt_minus_grasp_sign_matches_ci(bootstrap):
    for fam, v in bootstrap["popt_minus_grasp_by_family"].items():
        if v["ci_lo"] > 0:
            expected = "+"
        elif v["ci_hi"] < 0:
            expected = "-"
        else:
            expected = "0"
        assert str(v["sign"]) == expected, (fam, v)
        # ci_excludes_zero must agree with sign-encoded directionality
        assert v["ci_excludes_zero"] == (expected != "0"), (fam, v)


def test_popt_minus_grasp_widths_and_means_consistent(bootstrap):
    for fam, v in bootstrap["popt_minus_grasp_by_family"].items():
        assert v["ci_lo"] <= v["ci_hi"], (fam, v)
        assert abs(v["ci_width"] - (v["ci_hi"] - v["ci_lo"])) < WIDTH_TOL, (fam, v)
        # mean_delta is a point estimate; it must lie inside (or very near) the CI
        # in normal cases. Allow ci_width slack on either side for skewed BCa.
        assert v["ci_lo"] - v["ci_width"] <= v["mean_delta"] <= v["ci_hi"] + v["ci_width"], (
            fam, v
        )
        assert isinstance(v["n"], int) and v["n"] > 0, (fam, v)


def test_sign_stability_shape_and_bounds(bootstrap):
    rows = bootstrap["sign_stability"]
    assert isinstance(rows, list) and rows, "sign_stability must be a non-empty list"
    for r in rows:
        assert r["n_a"] == r["n_b"], r  # paired stat
        assert r["n_a"] > 0, r
        assert 0.0 <= r["frac_a_lt_b"] <= 1.0, r
        assert r["policy_a"] != r["policy_b"], r
        assert r["family"] in EXPECTED_FAMILIES, r
        assert r["policy_a"] in EXPECTED_POLICIES, r
        assert r["policy_b"] in EXPECTED_POLICIES, r


# ---------- Group C: oracle_gap_by_app_bootstrap parity & anti-symmetry (4) ----------


def test_per_app_pairs_universe(og_app_boot):
    pap = og_app_boot["per_app_pairs"]
    assert set(pap) == EXPECTED_APPS, set(pap) ^ EXPECTED_APPS
    for app, pairs in pap.items():
        assert len(pairs) == EXPECTED_PAIRS_PER_APP, (app, len(pairs))
        for k in pairs:
            a, b = k.split("_vs_")
            assert a in EXPECTED_POLICIES, (app, k)
            assert b in EXPECTED_POLICIES, (app, k)
            assert a != b, (app, k)  # no self-pairs allowed


def test_per_app_n_paired_consistent_within_app(og_app_boot):
    for app, pairs in og_app_boot["per_app_pairs"].items():
        n_paired_set = {p["n_paired"] for p in pairs.values()}
        assert len(n_paired_set) == 1, (app, n_paired_set)
        n_paired = n_paired_set.pop()
        assert n_paired > 0, (app, n_paired)


def test_per_app_pairs_anti_symmetric_mean_delta(og_app_boot):
    for app, pairs in og_app_boot["per_app_pairs"].items():
        for k, v in pairs.items():
            a, b = k.split("_vs_")
            rev = f"{b}_vs_{a}"
            assert rev in pairs, (app, k)
            assert abs(v["mean_delta"] + pairs[rev]["mean_delta"]) < ANTISYM_TOL, (
                app, k, v["mean_delta"], pairs[rev]["mean_delta"]
            )


def test_per_app_pairs_p_a_lt_b_complement(og_app_boot):
    for app, pairs in og_app_boot["per_app_pairs"].items():
        for k, v in pairs.items():
            a, b = k.split("_vs_")
            rev = f"{b}_vs_{a}"
            s = v["p_a_lt_b"] + pairs[rev]["p_a_lt_b"]
            # Independent bootstrap draws on each side plus ties → s <= 1 + slack
            assert s <= 1.0 + P_SUM_TOL, (app, k, s)
            # Each probability is a valid fraction
            assert 0.0 <= v["p_a_lt_b"] <= 1.0, (app, k, v)
            # CI ordering on every pair
            assert v["ci_lo"] <= v["ci_hi"], (app, k, v)


# ---------- Group D: oracle_gap row total + bootstrap-meta agreement (1) ----------


def test_oracle_gap_n_rows_matches_bootstrap_meta(og_app_boot, oracle_gap):
    assert og_app_boot["meta"]["n_total_rows"] == oracle_gap["summary"]["n_rows"]
    assert og_app_boot["meta"]["n_total_rows"] == EXPECTED_N_TOTAL_ROWS
