"""LIT-Sig — literature-faithfulness sign-mass concentration gate.

Locks per-bucket invariants for the lit-faith sign claims:

* ``expected_sign="-"`` × GRASP: GRASP beats LRU. Strict-sign concentration
  must be statistically distinguishable from 50/50 and the median observed
  delta_pct must be strongly negative.
* ``expected_sign="-"`` × POPT: POPT beats LRU. Same shape, smaller corpus.
* ``expected_sign="-"`` × POPT_GE_GRASP: POPT does not lose to GRASP. Ties
  are common (both saturate at the same value); ties count as half-credit
  toward the Wilson lower bound, but the binomial test still uses strict
  sign counts and must remain below alpha.

``~`` magnitude claims (SRRIP, POPT_NEAR_GRASP_IF_BIG_GAP, ``~`` × GRASP
which appears for a single edge-case graph) are not sign claims and the
gate does not lock their distributions — they are only reported in the
audit artifact for context.

The floors below are the smallest defensible values given today's corpus
plus a safety margin; they are tightened (never loosened) whenever new
data raises them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "lit_faith_signmass.json"

CLAIMS_TOTAL_FLOOR = 300
OK_ROWS_FLOOR = 250
BUCKETS_TOTAL_EXPECTED = 6
BUCKETS_WITH_OK_FLOOR = 6

# Per-bucket invariants — keys are (sign, policy) tuples.
# Each value pins (n_ok floor, wilson 95 % LB floor, binom p ceiling,
# median delta_pct ceiling, mean delta_pct ceiling) for the load-bearing
# sign claims. None = not locked.
LOAD_BEARING = {
    ("-", "GRASP"): {
        "n_ok_floor": 10,
        "wilson_lb_floor": 0.65,
        "binom_p_ceiling": 0.01,
        "median_delta_pct_ceiling": -2.0,
        "mean_delta_pct_ceiling": -2.0,
        "fraction_floor": 0.85,
    },
    ("-", "POPT"): {
        "n_ok_floor": 6,
        "wilson_lb_floor": 0.55,
        "binom_p_ceiling": 0.05,
        "median_delta_pct_ceiling": -5.0,
        "mean_delta_pct_ceiling": -5.0,
        "fraction_floor": 0.85,
    },
    ("-", "POPT_GE_GRASP"): {
        # POPT >= GRASP: delta = POPT − GRASP. Ties at 0 are legitimate
        # (both policies saturate), so floors are looser.
        "n_ok_floor": 60,
        "wilson_lb_floor": 0.50,
        "binom_p_ceiling": 0.05,
        # POPT_GE_GRASP often hovers near 0 — pin a slack upper bound that
        # still excludes "POPT loses to GRASP" regimes.
        "median_delta_pct_ceiling": 0.5,
        "mean_delta_pct_ceiling": 0.5,
        "fraction_floor": 0.55,
    },
}


@pytest.fixture(scope="module")
def signmass():
    assert ARTIFACT.exists(), (
        f"missing {ARTIFACT} — run `make lit-signmass`"
    )
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def buckets(signmass):
    return {
        (b["expected_sign"], b["policy"]): b for b in signmass["buckets"]
    }


# ────────────────────── shape / coverage ──────────────────────


def test_schema_version(signmass):
    assert signmass.get("schema_version") == 1


def test_claims_total_floor(signmass):
    assert signmass["summary"]["claims_total"] >= CLAIMS_TOTAL_FLOOR


def test_ok_rows_floor(signmass):
    assert signmass["summary"]["ok_rows_total"] >= OK_ROWS_FLOOR


def test_buckets_total(signmass):
    assert signmass["summary"]["buckets_total"] >= BUCKETS_TOTAL_EXPECTED


def test_buckets_with_ok_rows(signmass):
    assert (
        signmass["summary"]["buckets_with_ok_rows"]
        >= BUCKETS_WITH_OK_FLOOR
    )


def test_all_load_bearing_buckets_present(buckets):
    missing = [k for k in LOAD_BEARING if k not in buckets]
    assert not missing, f"missing load-bearing buckets: {missing}"


# ────────────────────── per-bucket sample-size floors ──────────────────────


@pytest.mark.parametrize("key", list(LOAD_BEARING))
def test_load_bearing_bucket_has_ok_rows(buckets, key):
    bucket = buckets[key]
    floor = LOAD_BEARING[key]["n_ok_floor"]
    assert bucket["n_ok"] >= floor, (
        f"{key} only has {bucket['n_ok']} ok rows; floor is {floor}"
    )


# ────────────────────── per-bucket sign concentration ──────────────────────


@pytest.mark.parametrize("key", list(LOAD_BEARING))
def test_correctly_signed_fraction_floor(buckets, key):
    bucket = buckets[key]
    floor = LOAD_BEARING[key]["fraction_floor"]
    frac = bucket["correctly_signed_fraction"]
    assert frac is not None, f"{key} fraction is None"
    assert frac >= floor, (
        f"{key} correctly_signed_fraction {frac:.3f} below floor {floor}"
    )


@pytest.mark.parametrize("key", list(LOAD_BEARING))
def test_wilson_lower_bound_floor(buckets, key):
    bucket = buckets[key]
    floor = LOAD_BEARING[key]["wilson_lb_floor"]
    lb = bucket["wilson_95_lower_bound"]
    assert lb is not None, f"{key} wilson_95_lower_bound is None"
    assert lb >= floor, (
        f"{key} wilson_95_lower_bound {lb:.3f} below floor {floor}"
    )


@pytest.mark.parametrize("key", list(LOAD_BEARING))
def test_binomial_sign_test_ceiling(buckets, key):
    bucket = buckets[key]
    ceiling = LOAD_BEARING[key]["binom_p_ceiling"]
    pv = bucket["binomial_sign_test_p"]
    assert pv is not None, f"{key} binomial_sign_test_p is None"
    assert pv <= ceiling, (
        f"{key} binomial_sign_test_p {pv:.4f} above ceiling {ceiling}"
    )


# ────────────────────── per-bucket effect size ──────────────────────


@pytest.mark.parametrize("key", list(LOAD_BEARING))
def test_median_delta_pct_ceiling(buckets, key):
    bucket = buckets[key]
    ceiling = LOAD_BEARING[key]["median_delta_pct_ceiling"]
    med = bucket["median_delta_pct"]
    assert med is not None, f"{key} median_delta_pct is None"
    assert med <= ceiling, (
        f"{key} median_delta_pct {med:.3f} above ceiling {ceiling}"
    )


@pytest.mark.parametrize("key", list(LOAD_BEARING))
def test_mean_delta_pct_ceiling(buckets, key):
    bucket = buckets[key]
    ceiling = LOAD_BEARING[key]["mean_delta_pct_ceiling"]
    mean = bucket["mean_delta_pct"]
    assert mean is not None, f"{key} mean_delta_pct is None"
    assert mean <= ceiling, (
        f"{key} mean_delta_pct {mean:.3f} above ceiling {ceiling}"
    )


# ────────────────────── no surprise wrong-sign outliers ──────────────────────


def test_grasp_no_wrong_sign_in_ok_rows(buckets):
    """GRASP sign claims must not contain a single wrong-signed ok row.

    A wrong-signed ok row would mean ``GRASP slower than LRU on a graph
    where the literature claims a speedup, AND the claim envelope was
    so loose it still passed`` — that's a corpus-design red flag.
    """
    bucket = buckets[("-", "GRASP")]
    assert bucket["wrong"] == 0, (
        f"GRASP has {bucket['wrong']} wrong-signed ok rows"
    )


def test_popt_no_wrong_sign_in_ok_rows(buckets):
    bucket = buckets[("-", "POPT")]
    assert bucket["wrong"] == 0, (
        f"POPT has {bucket['wrong']} wrong-signed ok rows"
    )


# ────────────────────── magnitude-only buckets reported (not locked) ──────────────────────


def test_magnitude_only_buckets_present(buckets):
    """`~` buckets should be reported in the audit even though they are
    not locked by sign-mass invariants (they are magnitude claims)."""
    magnitude_keys = [k for k in buckets if k[0] == "~"]
    assert len(magnitude_keys) >= 2, (
        f"expected >= 2 magnitude-only buckets, got {magnitude_keys}"
    )


def test_magnitude_buckets_are_all_ties(buckets):
    """Magnitude-only buckets must classify every ok row as 'tie' (sign
    is not asserted). If a `~` bucket reports correct/wrong counts, the
    classifier is mis-firing.
    """
    for key, bucket in buckets.items():
        if key[0] != "~":
            continue
        assert bucket["correctly_signed"] == 0 and bucket["wrong"] == 0, (
            f"magnitude-only bucket {key} reports "
            f"correct={bucket['correctly_signed']} / wrong={bucket['wrong']}"
        )


# ────────────────────── summary consistency ──────────────────────


def test_ok_rows_sum_matches_summary(signmass):
    """`summary.ok_rows_total` must equal the sum of `buckets[*].n_ok`."""
    bucket_sum = sum(b["n_ok"] for b in signmass["buckets"])
    assert bucket_sum == signmass["summary"]["ok_rows_total"]


def test_n_total_sum_matches_summary(signmass):
    bucket_sum = sum(b["n_total"] for b in signmass["buckets"])
    assert bucket_sum == signmass["summary"]["claims_total"]


def test_bucket_sign_classes_sum_matches_ok_rows(signmass):
    """For each bucket, correct + tie + wrong must equal n_ok."""
    bad = []
    for b in signmass["buckets"]:
        total = b["correctly_signed"] + b["tie"] + b["wrong"]
        if total != b["n_ok"]:
            bad.append(
                (b["expected_sign"], b["policy"], total, b["n_ok"])
            )
    assert not bad, f"sign-class totals don't match n_ok: {bad}"


def test_rows_payload_non_empty(signmass):
    """The detailed `rows` payload (one entry per cell) must be present
    for downstream re-aggregation."""
    assert len(signmass["rows"]) >= CLAIMS_TOTAL_FLOOR
