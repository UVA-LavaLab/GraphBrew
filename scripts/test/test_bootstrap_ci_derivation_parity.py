"""Gate 143+ — bootstrap_ci derivation parity.

bootstrap_ci.json is derived from oracle_gap.json + popt_vs_grasp_delta.json
via percentile bootstrap (5000 resamples, seed=1729) over:

  1. oracle_gap_by_policy_family — bucket by f"{policy}/{family}",
     bootstrap mean of gap_pp
  2. oracle_gap_by_policy_regime — bucket by f"{policy}/{regime}",
     bootstrap mean of gap_pp
  3. popt_minus_grasp_by_family  — paired bootstrap on Δ = POPT-GRASP
     delta_pp (negative = POPT wins)
  4. sign_stability              — fraction of resamples in which
     mean(a) < mean(b) for 7 headline (a, b, family) claims

This gate locks the parts that are EXACTLY reproducible from raw data
without re-running 5000 resamples × 4 sections (which would dominate
test runtime). Specifically:

  - aggregation correctness (n / mean / median exact)
  - field-derived invariants (ci_width = ci_hi - ci_lo)
  - sign / ci_excludes_zero logic from CI bounds
  - bracketing sanity (ci_lo <= mean <= ci_hi)
  - ci_level / meta consistency
  - sign_stability shape: 7 claims; fraction is exact ratio

The bootstrap CI BOUNDS themselves are pinned by the upstream byte-level
reproduce_smoke gate (same seed = byte-identical output). This test
adds a layer that catches mutations to the AGGREGATION step (which
would slip past byte-comparison because reproduce_smoke just regenerates
and compares — it cannot detect 'the formula is wrong AND the test was
also written wrong').

Invariants (19 tests, 5 groups):

Group A — Structural & scope
  1. Top-level keys
  2. meta: ci_level=0.95, n_resamples=5000, seed=1729
  3. oracle_gap_by_policy_family keys are f"{POL}/{FAM}" with valid POL and FAM
  4. oracle_gap_by_policy_regime keys are f"{POL}/{REGIME}"
  5. popt_minus_grasp_by_family keys are valid family strings

Group B — Per-bucket aggregation (oracle_gap → n / mean / median)
  6. n counts match oracle_gap aggregation per (policy, family)
  7. mean == round(fmean(gap_pp values), 4) per (policy, family) at 1e-6
  8. median == round(median(gap_pp values), 4) per (policy, family) at 1e-6
  9. n counts match per (policy, regime); mean/median match

Group C — Paired bootstrap aggregation (popt_vs_grasp_delta → mean_delta)
  10. n / mean_delta / median_delta match aggregation of delta_pp by family
      (cross-source: bootstrap_ci pulls from popt_vs_grasp_delta.json)

Group D — CI field invariants
  11. ci_width == round(ci_hi - ci_lo, 4) at 1e-3 (oracle_gap_by_policy_family)
  12. ci_width == round(ci_hi - ci_lo, 4) at 1e-3 (oracle_gap_by_policy_regime)
  13. ci_width == round(ci_hi - ci_lo, 4) at 1e-3 (popt_minus_grasp_by_family)
  14. ci_lo <= mean <= ci_hi (oracle_gap_by_policy_family + regime)
  15. ci_lo <= mean_delta <= ci_hi (popt_minus_grasp_by_family)

Group E — Sign / sign_stability semantics
  16. ci_excludes_zero == (ci_lo > 0 or ci_hi < 0) for delta buckets
  17. sign in {'+','-','0'} matching CI relationship to zero
  18. sign_stability has exactly 7 claims with required keys
  19. sign_stability frac_a_lt_b in [0,1] for non-empty buckets
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

VALID_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}
EPS_STAT = 1e-6
EPS_CI = 1e-3


@pytest.fixture(scope="module")
def og() -> dict:
    return json.loads((WIKI_DATA / "oracle_gap.json").read_text())


@pytest.fixture(scope="module")
def pvg() -> dict:
    return json.loads((WIKI_DATA / "popt_vs_grasp_delta.json").read_text())


@pytest.fixture(scope="module")
def bc() -> dict:
    return json.loads((WIKI_DATA / "bootstrap_ci.json").read_text())


@pytest.fixture(scope="module")
def by_policy_family(og) -> dict:
    """{f'{policy}/{family}': [gap_pp values]}"""
    d = defaultdict(list)
    for r in og["rows"]:
        if r.get("policy") and r.get("family"):
            d[f"{r['policy']}/{r['family']}"].append(float(r["gap_pp"]))
    return d


@pytest.fixture(scope="module")
def by_policy_regime(og) -> dict:
    d = defaultdict(list)
    for r in og["rows"]:
        if r.get("policy") and r.get("regime"):
            d[f"{r['policy']}/{r['regime']}"].append(float(r["gap_pp"]))
    return d


@pytest.fixture(scope="module")
def by_family_delta(pvg) -> dict:
    """{family: [delta_pp values]} from popt_vs_grasp_delta cells."""
    cells = pvg.get("cells", pvg.get("rows", []))
    d = defaultdict(list)
    for r in cells:
        fam = r.get("graph_family") or r.get("family")
        if fam:
            try:
                d[fam].append(float(r["delta_pp"]))
            except (KeyError, ValueError, TypeError):
                continue
    return d


# ─── Group A — Structural ────────────────────────────────────────────


def test_top_level_keys(bc):
    assert set(bc.keys()) >= {
        "meta",
        "oracle_gap_by_policy_family",
        "oracle_gap_by_policy_regime",
        "popt_minus_grasp_by_family",
        "sign_stability",
    }


def test_meta_pins_seed_resamples_level(bc):
    m = bc["meta"]
    assert m["ci_level"] == 0.95
    assert m["n_resamples"] == 5000
    assert m["seed"] == 1729


def test_policy_family_keys_shape(bc, by_policy_family):
    pat = re.compile(r"^([A-Z]+)/([a-z]+)$")
    bad = []
    for k in bc["oracle_gap_by_policy_family"]:
        m = pat.match(k)
        if not m or m.group(1) not in VALID_POLICIES:
            bad.append(k)
    assert not bad, bad
    assert set(bc["oracle_gap_by_policy_family"].keys()) == set(by_policy_family.keys())


def test_policy_regime_keys_shape(bc, by_policy_regime):
    pat = re.compile(r"^([A-Z]+)/(.+)$")
    bad = []
    for k in bc["oracle_gap_by_policy_regime"]:
        m = pat.match(k)
        if not m or m.group(1) not in VALID_POLICIES:
            bad.append(k)
    assert not bad, bad
    assert set(bc["oracle_gap_by_policy_regime"].keys()) == set(by_policy_regime.keys())


def test_family_delta_keys_match(bc, by_family_delta):
    assert set(bc["popt_minus_grasp_by_family"].keys()) == set(by_family_delta.keys())


# ─── Group B — Per-bucket aggregation (oracle_gap) ───────────────────


def test_oracle_policy_family_n_counts(bc, by_policy_family):
    mism = []
    for k, st in bc["oracle_gap_by_policy_family"].items():
        if st["n"] != len(by_policy_family[k]):
            mism.append((k, st["n"], len(by_policy_family[k])))
    assert not mism, mism


def test_oracle_policy_family_mean_matches(bc, by_policy_family):
    mism = []
    for k, st in bc["oracle_gap_by_policy_family"].items():
        exp = round(statistics.fmean(by_policy_family[k]), 4)
        if abs(st["mean"] - exp) > EPS_STAT:
            mism.append((k, st["mean"], exp))
    assert not mism, mism[:5]


def test_oracle_policy_family_median_matches(bc, by_policy_family):
    mism = []
    for k, st in bc["oracle_gap_by_policy_family"].items():
        exp = round(statistics.median(by_policy_family[k]), 4)
        if abs(st["median"] - exp) > EPS_STAT:
            mism.append((k, st["median"], exp))
    assert not mism, mism[:5]


def test_oracle_policy_regime_n_mean_median(bc, by_policy_regime):
    mism = []
    for k, st in bc["oracle_gap_by_policy_regime"].items():
        vals = by_policy_regime[k]
        if st["n"] != len(vals):
            mism.append(("n", k, st["n"], len(vals)))
        if abs(st["mean"] - round(statistics.fmean(vals), 4)) > EPS_STAT:
            mism.append(("mean", k, st["mean"]))
        if abs(st["median"] - round(statistics.median(vals), 4)) > EPS_STAT:
            mism.append(("median", k, st["median"]))
    assert not mism, mism[:5]


# ─── Group C — Paired bootstrap aggregation ─────────────────────────


def test_delta_family_n_mean_median(bc, by_family_delta):
    mism = []
    for k, st in bc["popt_minus_grasp_by_family"].items():
        vals = by_family_delta[k]
        if st["n"] != len(vals):
            mism.append(("n", k, st["n"], len(vals)))
        if abs(st["mean_delta"] - round(statistics.fmean(vals), 4)) > EPS_STAT:
            mism.append(("mean", k, st["mean_delta"]))
        if abs(st["median_delta"] - round(statistics.median(vals), 4)) > EPS_STAT:
            mism.append(("median", k, st["median_delta"]))
    assert not mism, mism[:5]


# ─── Group D — CI field invariants ──────────────────────────────────


def _check_width(scope: dict) -> list:
    mism = []
    for k, st in scope.items():
        exp = round(st["ci_hi"] - st["ci_lo"], 4)
        if abs(st["ci_width"] - exp) > EPS_CI:
            mism.append((k, st["ci_width"], exp))
    return mism


def test_ci_width_oracle_policy_family(bc):
    assert not _check_width(bc["oracle_gap_by_policy_family"])


def test_ci_width_oracle_policy_regime(bc):
    assert not _check_width(bc["oracle_gap_by_policy_regime"])


def test_ci_width_popt_minus_grasp(bc):
    assert not _check_width(bc["popt_minus_grasp_by_family"])


def test_ci_brackets_mean_for_oracle_scopes(bc):
    bad = []
    for scope_name in ("oracle_gap_by_policy_family", "oracle_gap_by_policy_regime"):
        for k, st in bc[scope_name].items():
            if not (st["ci_lo"] - EPS_CI <= st["mean"] <= st["ci_hi"] + EPS_CI):
                bad.append((scope_name, k, st))
    assert not bad, bad[:5]


def test_ci_brackets_mean_delta(bc):
    bad = []
    for k, st in bc["popt_minus_grasp_by_family"].items():
        if not (st["ci_lo"] - EPS_CI <= st["mean_delta"] <= st["ci_hi"] + EPS_CI):
            bad.append((k, st))
    assert not bad, bad[:5]


# ─── Group E — Sign / sign_stability semantics ──────────────────────


def test_ci_excludes_zero_logic(bc):
    mism = []
    for k, st in bc["popt_minus_grasp_by_family"].items():
        expected = st["ci_lo"] > 0 or st["ci_hi"] < 0
        if st["ci_excludes_zero"] != expected:
            mism.append((k, st["ci_excludes_zero"], expected))
    assert not mism, mism


def test_sign_field_matches_ci_relationship(bc):
    mism = []
    for k, st in bc["popt_minus_grasp_by_family"].items():
        if st["ci_lo"] > 0:
            expected = "+"
        elif st["ci_hi"] < 0:
            expected = "-"
        else:
            expected = "0"
        if st["sign"] != expected:
            mism.append((k, st["sign"], expected))
    assert not mism, mism


def test_sign_stability_has_seven_claims(bc):
    assert len(bc["sign_stability"]) == 7
    required = {"policy_a", "policy_b", "family", "n_a", "n_b"}
    bad = [s for s in bc["sign_stability"] if not required.issubset(s)]
    assert not bad, bad


def test_sign_stability_fraction_in_unit_interval(bc):
    bad = []
    for s in bc["sign_stability"]:
        f = s.get("frac_a_lt_b")
        if f is None:
            continue
        if not (0.0 <= f <= 1.0):
            bad.append(s)
    assert not bad, bad
