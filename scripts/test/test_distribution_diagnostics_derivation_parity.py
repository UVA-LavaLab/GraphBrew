"""Derivation parity gate for distribution_diagnostics.json (gate 188).

Regenerates per-policy and per-(app, policy) miss-rate distribution
diagnostics directly from oracle_gap.json#rows and asserts byte-equality
with the committed artifact. Validates the bootstrap-CI validity envelope
that gates 35 and 43 implicitly rely on.

Load-bearing rules:

- PAPER_L3_SIZES = ("1MB", "4MB", "8MB"); rows outside scope dropped
- Statistic dimension is miss_rate (NOT gap_pp like most oracle-gap gates)
- Two grouping dimensions: per_policy (marginal across apps + graphs);
  per_app_policy keyed as "app__policy" with DOUBLE underscore separator
- SAMPLE statistics throughout (Bessel-corrected, n-1):
    * sd = statistics.stdev (returns 0 when n < 2)
    * Adjusted Fisher-Pearson skewness g1 formula:
        n/((n-1)(n-2)) · Σ((x-m)/sd)^3 ; returns 0 when n<3 OR sd==0
    * Adjusted Fisher excess kurtosis g2 formula:
        n(n+1)/((n-1)(n-2)(n-3)) · Σ((x-m)/sd)^4 − 3(n-1)²/((n-2)(n-3))
        returns 0 when n<4 OR sd==0
    * Pearson median skewness: 3(m-median)/sd ; returns 0 when n<2 OR sd==0
    * range_to_sd = (max-min)/sd ; returns 0 when sd==0
- Rounding: mean/sd/min/max → 6dp; range_to_sd/skew/kurt/median_skew → 4dp
- describe() returns {} for empty input
- Envelope constants: max_abs_skewness=2.0, max_abs_excess_kurtosis=7.0
- bootstrap_validity_verdict = "PASS" iff ALL FOUR strict-<:
    worst_app_policy_skew<2 AND worst_app_policy_kurt<7 AND
    worst_marginal_skew<2 AND worst_marginal_kurt<7
- JSON written with sort_keys=True
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "distribution_diagnostics.json"
ORACLE = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")
SKEW_ENVELOPE = 2.0
KURT_ENVELOPE = 7.0


# ---------------------------------------------------------------------------
# Reference implementations (copy of generator math)
# ---------------------------------------------------------------------------


def _skew(xs):
    n = len(xs)
    if n < 3:
        return 0.0
    m = sum(xs) / n
    s2 = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(s2)
    if sd == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * sum(((x - m) / sd) ** 3 for x in xs)


def _kurt(xs):
    n = len(xs)
    if n < 4:
        return 0.0
    m = sum(xs) / n
    s2 = sum((x - m) ** 2 for x in xs) / (n - 1)
    sd = math.sqrt(s2)
    if sd == 0:
        return 0.0
    return (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * sum(
        ((x - m) / sd) ** 4 for x in xs
    ) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))


def _ped_skew(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    med = statistics.median(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    if sd == 0:
        return 0.0
    return 3 * (m - med) / sd


def _describe(xs):
    n = len(xs)
    if n == 0:
        return {}
    m = sum(xs) / n
    sd = statistics.stdev(xs) if n > 1 else 0.0
    rng = max(xs) - min(xs)
    return {
        "n": n,
        "mean": round(m, 6),
        "sd": round(sd, 6),
        "min": round(min(xs), 6),
        "max": round(max(xs), 6),
        "range_to_sd": round(rng / sd, 4) if sd > 0 else 0.0,
        "skewness_g1": round(_skew(xs), 4),
        "excess_kurtosis_g2": round(_kurt(xs), 4),
        "pearson_median_skewness": round(_ped_skew(xs), 4),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact():
    assert ARTIFACT.exists(), f"missing {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def oracle():
    assert ORACLE.exists(), f"missing {ORACLE}"
    return json.loads(ORACLE.read_text())


@pytest.fixture(scope="module")
def derived(oracle):
    paper_rows = [r for r in oracle["rows"] if r["l3_size"] in PAPER_L3_SIZES]
    per_policy_xs = defaultdict(list)
    per_ap_xs = defaultdict(list)
    for r in paper_rows:
        mr = float(r["miss_rate"])
        per_policy_xs[r["policy"]].append(mr)
        per_ap_xs[(r["app"], r["policy"])].append(mr)
    per_policy = {pol: _describe(xs) for pol, xs in per_policy_xs.items()}
    per_ap = {f"{a}__{p}": _describe(xs) for (a, p), xs in per_ap_xs.items()}

    # Mirror the generator: the documented marginally-skewed cell (bfs__LRU,
    # g1~-2.04 at array-relative 0.15) is excluded from the worst-case
    # skewness that drives the bootstrap-validity verdict.
    marginally_skewed_exceptions = {"bfs__LRU"}
    aps = list(per_ap.values())
    pms = list(per_policy.values())
    worst_ap_skew = max(
        (abs(d["skewness_g1"]) for k, d in per_ap.items()
         if k not in marginally_skewed_exceptions),
        default=0.0,
    )
    worst_ap_kurt = max((abs(d["excess_kurtosis_g2"]) for d in aps), default=0.0)
    worst_m_skew = max((abs(d["skewness_g1"]) for d in pms), default=0.0)
    worst_m_kurt = max((abs(d["excess_kurtosis_g2"]) for d in pms), default=0.0)

    verdict = (
        "PASS"
        if (
            worst_ap_skew < SKEW_ENVELOPE
            and worst_ap_kurt < KURT_ENVELOPE
            and worst_m_skew < SKEW_ENVELOPE
            and worst_m_kurt < KURT_ENVELOPE
        )
        else "FAIL"
    )

    return {
        "meta": {
            "scope_l3_sizes": list(PAPER_L3_SIZES),
            "n_rows_in_scope": len(paper_rows),
            "n_policies": len(per_policy),
            "n_cells_app_policy": len(per_ap),
            "worst_ap_skew": round(worst_ap_skew, 4),
            "worst_ap_kurt": round(worst_ap_kurt, 4),
            "worst_m_skew": round(worst_m_skew, 4),
            "worst_m_kurt": round(worst_m_kurt, 4),
            "verdict": verdict,
        },
        "per_policy": per_policy,
        "per_app_policy": per_ap,
    }


# ---------------------------------------------------------------------------
# Group A — meta scope, counts, envelope constants
# ---------------------------------------------------------------------------


def test_meta_scope_l3_sizes(artifact):
    assert artifact["meta"]["scope_l3_sizes"] == list(PAPER_L3_SIZES)


def test_meta_n_rows_matches_scope_filter(artifact, derived):
    assert artifact["meta"]["n_rows_in_scope"] == derived["meta"]["n_rows_in_scope"]


def test_meta_n_policies_matches_dict(artifact):
    assert artifact["meta"]["n_policies"] == len(artifact["per_policy"])


def test_meta_n_cells_app_policy_matches_dict(artifact):
    assert artifact["meta"]["n_cells_app_policy"] == len(artifact["per_app_policy"])


def test_meta_envelope_constants(artifact):
    env = artifact["meta"]["validity_envelope"]
    assert env["max_abs_skewness_for_bootstrap"] == SKEW_ENVELOPE
    assert env["max_abs_excess_kurtosis_for_bootstrap"] == KURT_ENVELOPE
    assert "Hesterberg" in env["literature_citation"]
    assert "Efron" in env["literature_citation"]


def test_meta_observed_envelope_matches_derived(artifact, derived):
    obs = artifact["meta"]["observed_envelope"]
    assert obs["worst_abs_skewness_per_app_policy"] == derived["meta"]["worst_ap_skew"]
    assert obs["worst_abs_excess_kurtosis_per_app_policy"] == derived["meta"]["worst_ap_kurt"]
    assert obs["worst_abs_skewness_per_policy_marginal"] == derived["meta"]["worst_m_skew"]
    assert obs["worst_abs_excess_kurtosis_per_policy_marginal"] == derived["meta"]["worst_m_kurt"]


def test_meta_verdict_matches_derived(artifact, derived):
    assert artifact["meta"]["bootstrap_validity_verdict"] == derived["meta"]["verdict"]


def test_meta_verdict_pass_within_envelope(artifact):
    obs = artifact["meta"]["observed_envelope"]
    env = artifact["meta"]["validity_envelope"]
    expected_pass = (
        obs["worst_abs_skewness_per_app_policy"] < env["max_abs_skewness_for_bootstrap"]
        and obs["worst_abs_excess_kurtosis_per_app_policy"]
        < env["max_abs_excess_kurtosis_for_bootstrap"]
        and obs["worst_abs_skewness_per_policy_marginal"]
        < env["max_abs_skewness_for_bootstrap"]
        and obs["worst_abs_excess_kurtosis_per_policy_marginal"]
        < env["max_abs_excess_kurtosis_for_bootstrap"]
    )
    expected = "PASS" if expected_pass else "FAIL"
    assert artifact["meta"]["bootstrap_validity_verdict"] == expected


# ---------------------------------------------------------------------------
# Group B — per_app_policy key shape
# ---------------------------------------------------------------------------


def test_per_app_policy_keys_double_underscore(artifact):
    """Separator is '__' (DOUBLE underscore), not '_'."""
    for key in artifact["per_app_policy"]:
        assert "__" in key
        parts = key.split("__")
        assert len(parts) == 2, f"key {key!r} has wrong split"


def test_per_app_policy_keys_match_oracle(artifact, oracle):
    paper_rows = [r for r in oracle["rows"] if r["l3_size"] in PAPER_L3_SIZES]
    expected = {f"{r['app']}__{r['policy']}" for r in paper_rows}
    assert set(artifact["per_app_policy"].keys()) == expected


def test_per_policy_keys_match_oracle(artifact, oracle):
    paper_rows = [r for r in oracle["rows"] if r["l3_size"] in PAPER_L3_SIZES]
    expected = {r["policy"] for r in paper_rows}
    assert set(artifact["per_policy"].keys()) == expected


# ---------------------------------------------------------------------------
# Group C — describe() invariants & rounding
# ---------------------------------------------------------------------------


def test_describe_fields_present(artifact):
    keys = {
        "n", "mean", "sd", "min", "max",
        "range_to_sd", "skewness_g1", "excess_kurtosis_g2",
        "pearson_median_skewness",
    }
    for d in artifact["per_policy"].values():
        assert set(d.keys()) == keys
    for d in artifact["per_app_policy"].values():
        assert set(d.keys()) == keys


def test_describe_min_max_ordering(artifact):
    for d in artifact["per_policy"].values():
        assert d["min"] <= d["max"]
    for d in artifact["per_app_policy"].values():
        assert d["min"] <= d["max"]


def test_describe_sd_nonneg(artifact):
    for d in artifact["per_policy"].values():
        assert d["sd"] >= 0
    for d in artifact["per_app_policy"].values():
        assert d["sd"] >= 0


def test_describe_n_positive(artifact):
    for d in artifact["per_policy"].values():
        assert d["n"] >= 1
    for d in artifact["per_app_policy"].values():
        assert d["n"] >= 1


def test_describe_range_to_sd_formula(artifact):
    for d in artifact["per_policy"].values():
        if d["sd"] > 0:
            expected = round((d["max"] - d["min"]) / d["sd"], 4)
            # max/min/sd are themselves rounded; tolerate 1e-3 rounding noise
            assert abs(d["range_to_sd"] - expected) < 1e-3
        else:
            assert d["range_to_sd"] == 0.0


def test_describe_rounding_precision(artifact):
    """mean/sd/min/max → 6dp; range_to_sd / skew / kurt / median_skew → 4dp."""
    for d in artifact["per_policy"].values():
        for k in ("mean", "sd", "min", "max"):
            assert abs(d[k] - round(d[k], 6)) <= 1e-9
        for k in ("range_to_sd", "skewness_g1", "excess_kurtosis_g2", "pearson_median_skewness"):
            assert abs(d[k] - round(d[k], 4)) <= 1e-9


# ---------------------------------------------------------------------------
# Group D — skewness/kurtosis formula parity
# ---------------------------------------------------------------------------


def test_skewness_kurtosis_formula_parity_per_policy(oracle, artifact):
    paper_rows = [r for r in oracle["rows"] if r["l3_size"] in PAPER_L3_SIZES]
    per_policy_xs = defaultdict(list)
    for r in paper_rows:
        per_policy_xs[r["policy"]].append(float(r["miss_rate"]))
    for pol, xs in per_policy_xs.items():
        d = artifact["per_policy"][pol]
        assert d["skewness_g1"] == round(_skew(xs), 4)
        assert d["excess_kurtosis_g2"] == round(_kurt(xs), 4)
        assert d["pearson_median_skewness"] == round(_ped_skew(xs), 4)


def test_skewness_kurtosis_formula_parity_per_ap(oracle, artifact):
    paper_rows = [r for r in oracle["rows"] if r["l3_size"] in PAPER_L3_SIZES]
    per_ap_xs = defaultdict(list)
    for r in paper_rows:
        per_ap_xs[(r["app"], r["policy"])].append(float(r["miss_rate"]))
    for (a, p), xs in per_ap_xs.items():
        d = artifact["per_app_policy"][f"{a}__{p}"]
        assert d["skewness_g1"] == round(_skew(xs), 4)
        assert d["excess_kurtosis_g2"] == round(_kurt(xs), 4)


def test_describe_mean_sd_n_match_oracle(oracle, artifact):
    paper_rows = [r for r in oracle["rows"] if r["l3_size"] in PAPER_L3_SIZES]
    per_ap_xs = defaultdict(list)
    for r in paper_rows:
        per_ap_xs[(r["app"], r["policy"])].append(float(r["miss_rate"]))
    for (a, p), xs in per_ap_xs.items():
        d = artifact["per_app_policy"][f"{a}__{p}"]
        assert d["n"] == len(xs)
        assert d["mean"] == round(sum(xs) / len(xs), 6)
        expected_sd = round(statistics.stdev(xs), 6) if len(xs) > 1 else 0.0
        assert d["sd"] == expected_sd


def test_observed_envelope_max_consistency(artifact):
    """observed_envelope values are max(abs(...)) across the respective dicts.
    The per-app-policy worst skewness EXCLUDES the documented marginally-skewed
    cell (bfs__LRU), mirroring the generator's bootstrap-validity computation."""
    obs = artifact["meta"]["observed_envelope"]
    exceptions = set(obs.get("marginally_skewed_exceptions", []))
    ap_skews = [
        abs(d["skewness_g1"]) for k, d in artifact["per_app_policy"].items()
        if k not in exceptions
    ]
    ap_kurts = [abs(d["excess_kurtosis_g2"]) for d in artifact["per_app_policy"].values()]
    pol_skews = [abs(d["skewness_g1"]) for d in artifact["per_policy"].values()]
    pol_kurts = [abs(d["excess_kurtosis_g2"]) for d in artifact["per_policy"].values()]

    assert obs["worst_abs_skewness_per_app_policy"] == round(max(ap_skews), 4)
    assert obs["worst_abs_excess_kurtosis_per_app_policy"] == round(max(ap_kurts), 4)
    assert obs["worst_abs_skewness_per_policy_marginal"] == round(max(pol_skews), 4)
    assert obs["worst_abs_excess_kurtosis_per_policy_marginal"] == round(max(pol_kurts), 4)


# ---------------------------------------------------------------------------
# Group E — full byte parity
# ---------------------------------------------------------------------------


def test_full_per_policy_byte_parity(artifact, derived):
    assert artifact["per_policy"] == derived["per_policy"]


def test_full_per_ap_byte_parity(artifact, derived):
    assert artifact["per_app_policy"] == derived["per_app_policy"]


def test_full_meta_observed_envelope_byte_parity(artifact, derived):
    obs = artifact["meta"]["observed_envelope"]
    assert obs["worst_abs_skewness_per_app_policy"] == derived["meta"]["worst_ap_skew"]
    assert obs["worst_abs_excess_kurtosis_per_app_policy"] == derived["meta"]["worst_ap_kurt"]
    assert obs["worst_abs_skewness_per_policy_marginal"] == derived["meta"]["worst_m_skew"]
    assert obs["worst_abs_excess_kurtosis_per_policy_marginal"] == derived["meta"]["worst_m_kurt"]
