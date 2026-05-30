"""LIT-Stat (gate 230): pytest invariants for the statistical-sanity audit.

Locks the arithmetic and label-vocabulary invariants of the lit-faith
per_claim table:
  * No NaN/inf, no out-of-bounds miss rates, no missing-pair rows.
  * Re-derived delta_pct matches stored delta_pct exactly (modulo
    4-dp rounding noise).
  * No sign flips between stored and recomputed delta on the
    signed-delta row kinds.
  * signed_delta_pct on POPT_NEAR rows agrees with arithmetic, and
    |signed_delta_pct| == delta_pct (since POPT_NEAR stores magnitude).
  * status labels stay within the vocabulary the comparator emits.
  * Every 'ok'/'within_tolerance' row is genuinely within its
    magnitude bound (modulo the phase-transition-regime exception
    that POPT_NEAR rows carry).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "wiki" / "data" / "lit_faith_stat.json"


ALLOWED_STATUSES = {
    "ok", "within_tolerance", "disagree",
    "known_deviation", "missing", "insufficient_data",
}


@pytest.fixture(scope="module")
def audit() -> dict:
    assert AUDIT_PATH.exists(), (
        f"{AUDIT_PATH} missing — run `make lit-stat` (gate 230)"
    )
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_schema_keys(audit: dict) -> None:
    for key in ("schema_version", "summary",
                "nan_inf", "miss_rate_oob",
                "delta_mismatch", "sign_mismatch",
                "signed_delta_mismatch",
                "status_bad", "status_inconsistent",
                "unknown_kind", "missing_pair"):
        assert key in audit, f"missing top-level key {key!r}"
    assert audit["schema_version"] == 1


def test_tolerances_pinned(audit: dict) -> None:
    s = audit["summary"]
    assert s["delta_rounding_tol_pp"] == 0.001, (
        "delta rounding tolerance must remain 0.001 pp — the comparator "
        "rounds to 4 dp so the recomputation diff cannot exceed 5e-5 pp"
    )
    assert s["sign_noise_floor_pp"] == 0.01, (
        "sign-flip detection floor must remain 0.01 pp — below this both "
        "stored and recomputed deltas are dominated by rounding"
    )


def test_no_nan_inf(audit: dict) -> None:
    if audit["nan_inf"]:
        details = "; ".join(
            f"{r.get('graph')}/{r.get('app')}/{r.get('policy')}@{r.get('l3_size')}"
            for r in audit["nan_inf"][:5]
        )
        pytest.fail(f"{len(audit['nan_inf'])} rows with NaN/inf: {details}")


def test_no_miss_rate_out_of_bounds(audit: dict) -> None:
    if audit["miss_rate_oob"]:
        details = "; ".join(
            f"{r.get('graph')}/{r.get('app')}/{r.get('policy')}@{r.get('l3_size')}"
            for r in audit["miss_rate_oob"][:5]
        )
        pytest.fail(f"{len(audit['miss_rate_oob'])} rows with miss-rate "
                    f"outside [0,1]: {details}")


def test_no_unknown_row_kind(audit: dict) -> None:
    if audit["unknown_kind"]:
        pytest.fail(
            f"{len(audit['unknown_kind'])} rows do not match any known "
            f"schema branch (lru_vs_policy / popt_ge_grasp / popt_near_grasp)"
        )


def test_no_missing_pair_rows(audit: dict) -> None:
    """Every row must have its two miss-rate columns populated. Rows
    with `status in {missing, insufficient_data}` will still have
    None values, but those are filtered out by row_kind before
    landing in missing_pair (the audit treats them as either kind
    based on which keys ARE present)."""
    if audit["missing_pair"]:
        details = "; ".join(
            f"{r.get('graph')}/{r.get('app')}/{r.get('policy')}@{r.get('l3_size')}"
            for r in audit["missing_pair"][:5]
        )
        pytest.fail(f"{len(audit['missing_pair'])} rows missing miss-rate "
                    f"pair: {details}")


def test_no_delta_mismatch(audit: dict) -> None:
    if audit["delta_mismatch"]:
        details = "; ".join(
            f"{r['graph']}/{r['app']}/{r['policy']}@{r['l3_size']}: "
            f"stored={r['stored']:.4f} vs recomputed={r['recomputed']:.4f} "
            f"(diff={r['abs_diff_pp']} pp)"
            for r in audit["delta_mismatch"][:5]
        )
        pytest.fail(f"{len(audit['delta_mismatch'])} rows where stored "
                    f"delta_pct doesn't match recomputation: {details}")


def test_no_sign_mismatch(audit: dict) -> None:
    if audit["sign_mismatch"]:
        details = "; ".join(
            f"{r['graph']}/{r['app']}/{r['policy']}@{r['l3_size']}: "
            f"stored={r['stored']:+.4f} vs recomputed={r['recomputed']:+.4f}"
            for r in audit["sign_mismatch"][:5]
        )
        pytest.fail(f"{len(audit['sign_mismatch'])} sign-flipped rows: "
                    f"{details}")


def test_no_signed_delta_mismatch(audit: dict) -> None:
    """POPT_NEAR rows carry a `signed_delta_pct` for the sign-bearing
    direction and store the magnitude in `delta_pct`. Both must agree
    with the recomputed arithmetic."""
    if audit["signed_delta_mismatch"]:
        pytest.fail(
            f"{len(audit['signed_delta_mismatch'])} POPT_NEAR rows have "
            "inconsistent signed_delta_pct / delta_pct arithmetic"
        )


def test_no_bad_status_labels(audit: dict) -> None:
    if audit["status_bad"]:
        labels = sorted({r.get("status") for r in audit["status_bad"]})
        pytest.fail(f"{len(audit['status_bad'])} rows have status labels "
                    f"outside the allowed set: {labels}")


def test_status_label_vocabulary_locked(audit: dict) -> None:
    """All observed statuses today must be a subset of the locked
    vocabulary. Adding a new status requires lifting this gate."""
    observed = set(audit["summary"]["status_counts"].keys())
    unknown = observed - ALLOWED_STATUSES
    assert not unknown, (
        f"unexpected status label(s) in lit-faith corpus: {unknown}; "
        f"allowed = {sorted(ALLOWED_STATUSES)}"
    )


def test_no_status_inconsistencies(audit: dict) -> None:
    """Every 'ok'/'within_tolerance' row with a magnitude bound must
    actually be within that bound + tolerance (with the POPT_NEAR
    phase-transition exception folded in)."""
    if audit["status_inconsistent"]:
        details = "; ".join(
            f"{r['graph']}/{r['app']}/{r['policy']}@{r['l3_size']}: "
            f"abs={r['abs_delta']} > max_abs={r['max_abs']}+tol={r['tolerance']}"
            for r in audit["status_inconsistent"][:5]
        )
        pytest.fail(f"{len(audit['status_inconsistent'])} status "
                    f"inconsistencies: {details}")


def test_row_count_floor(audit: dict) -> None:
    """Lit-faith corpus must keep at least 250 per-claim rows. Today
    330. Falling below 250 means a graph/app/L3 axis was silently
    dropped."""
    assert audit["summary"]["total_rows"] >= 250, (
        f"only {audit['summary']['total_rows']} per_claim rows — "
        "corpus shrank"
    )


def test_kind_split_balanced(audit: dict) -> None:
    """All three row kinds must be represented (lru-vs-policy +
    popt-ge-grasp + popt-near-grasp). LRU rows should be >=80; each
    POPT kind should be >=80. Today: 102, 114, 114."""
    s = audit["summary"]
    assert s["rows_lru_vs_policy"]   >= 80, (
        f"LRU-vs-policy rows {s['rows_lru_vs_policy']} below floor")
    assert s["rows_popt_ge_grasp"]   >= 80, (
        f"POPT_GE_GRASP rows {s['rows_popt_ge_grasp']} below floor")
    assert s["rows_popt_near_grasp"] >= 80, (
        f"POPT_NEAR_GRASP rows {s['rows_popt_near_grasp']} below floor")


def test_all_apps_have_signal(audit: dict) -> None:
    """No app should be entirely flat (max abs delta ≤ 0.05 pp) —
    that would mean the policy comparison is meaningless for that app."""
    flat = audit["summary"]["apps_flat"]
    assert not flat, (
        f"apps with zero policy-vs-LRU signal: {flat} — these apps "
        "produce identical miss rates across policies"
    )


def test_signal_apps_cover_corpus(audit: dict) -> None:
    """At least 4 apps must show > 1 pp policy signal — guarantees the
    corpus exercises multiple algorithmic regimes."""
    signal = audit["summary"]["apps_with_pp_signal"]
    assert len(signal) >= 4, (
        f"only {len(signal)} apps with > 1 pp policy signal ({signal})"
    )


def test_status_counts_sum_matches_total(audit: dict) -> None:
    s = audit["summary"]
    total = sum(s["status_counts"].values())
    assert total == s["total_rows"], (
        f"status_counts sum {total} != total_rows {s['total_rows']}"
    )


def test_no_disagree_rows_today(audit: dict) -> None:
    """The lit-faith corpus is currently DISAGREE-free (with
    known_deviation absorbing the documented 30). Catches regressions
    where a new claim slips into outright DISAGREE without being
    explained in KNOWN_DEVIATIONS."""
    s = audit["summary"]
    assert s["status_counts"].get("disagree", 0) == 0, (
        f"{s['status_counts']['disagree']} disagree rows — either fix "
        "the underlying mismatch or add a KNOWN_DEVIATIONS entry"
    )


def test_known_deviation_count_ceiling(audit: dict) -> None:
    """Known-deviation count must not balloon — today 30. A ceiling
    of 50 catches silent claim drift where the comparator is
    increasingly relying on whitelisted exceptions."""
    s = audit["summary"]
    kd = s["status_counts"].get("known_deviation", 0)
    assert kd <= 50, f"known_deviation count {kd} exceeded ceiling of 50"


def test_ok_majority(audit: dict) -> None:
    """`ok` should remain the strict majority — today 298/330 = 90.3 %.
    Floor at 80 % catches large degradations."""
    s = audit["summary"]
    ok = s["status_counts"].get("ok", 0)
    total = s["total_rows"]
    assert ok / total >= 0.80, (
        f"ok fraction {ok}/{total} = {ok/total:.3f} below 0.80 floor"
    )
