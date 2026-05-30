"""LIT-Mar — literature-faithfulness margin distribution gate.

Locks the distance-to-disagree distribution of
``wiki/data/lit_faith_margin.json`` so the literature-faithfulness
corpus cannot silently drift into the fragile-cell regime where small
miss-rate jitter would flip ``ok`` rows into ``disagree`` rows.

Invariants:

* every claim has a computable, bounded margin (no ``unbounded`` rows).
* zero ``ok``-status rows have a negative margin
  (the audit and ``literature_faithfulness._classify`` must agree on
  which rows are inside the claim envelope).
* every ``known_deviation`` row has a non-positive margin
  (these *are* the disagree rows; whitelisting them by reason is the
  only thing keeping the gate green).
* median margin is comfortably positive (today: ~5.5 pp).
* per-family minimum and median margins do not collapse.
* the fragile-cell count (< 1 pp margin) stays below today's count.

The thresholds are pinned to today's observed minimums + a small
ceiling so a new contributor cannot add a wave of borderline claims
without explicitly bumping the floors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MARGIN_PATH = REPO_ROOT / "wiki" / "data" / "lit_faith_margin.json"

# --------- floors pinned to today's observed minimums ----------

CLAIMS_TOTAL_FLOOR = 300
MEDIAN_MARGIN_FLOOR_PP = 4.5
MEAN_MARGIN_FLOOR_PP = 4.5
P25_MARGIN_FLOOR_PP = 1.0
FRAGILE_CEILING = 60  # today: 47
NEGATIVE_OK_CEILING = 0  # strict invariant

EXPECTED_BINDING_BOUNDARIES = {
    "magnitude_max_abs",
    "near_grasp_upper",
    "sign_upper_min_abs",
    "sign_upper_tol",
    "trigger_headroom",
}

EXPECTED_FAMILIES = {"social", "citation", "web", "road", "mesh"}

# Per-family minimum medians (pinned to today's per-family medians ± headroom).
# A family-level collapse here means margin shrank for a whole graph class.
FAMILY_MEDIAN_FLOOR_PP = {
    "social": 3.5,    # today 4.94
    "citation": 3.0,  # today 4.28
    "web": 4.5,       # today 6.63
    "road": 7.0,      # today 9.80
    "mesh": 3.5,      # today 5.20
}


@pytest.fixture(scope="module")
def audit() -> dict:
    return json.loads(MARGIN_PATH.read_text())


# ---------- file-level integrity ----------

def test_file_exists_and_parses() -> None:
    assert MARGIN_PATH.exists(), MARGIN_PATH
    payload = json.loads(MARGIN_PATH.read_text())
    assert isinstance(payload, dict)
    assert payload.get("schema_version") == 1


def test_top_level_shape(audit: dict) -> None:
    required = {
        "schema_version",
        "fragile_threshold_pp",
        "summary",
        "per_status_stats",
        "per_family_stats",
        "binding_boundary_counts",
        "fragile_rows",
        "negative_ok_rows",
        "rows",
    }
    missing = required - set(audit.keys())
    assert not missing, f"missing top-level keys: {sorted(missing)}"


# ---------- summary invariants ----------

def test_claims_total_floor(audit: dict) -> None:
    total = audit["summary"]["claims_total"]
    assert (
        total >= CLAIMS_TOTAL_FLOOR
    ), f"corpus shrank: {total} < {CLAIMS_TOTAL_FLOOR}"


def test_every_claim_has_bounded_margin(audit: dict) -> None:
    """Every claim must produce a finite margin under the audit model."""
    s = audit["summary"]
    assert s["claims_unbounded"] == 0, (
        f"{s['claims_unbounded']} claims fell into unbounded; "
        "margin model is incomplete."
    )
    assert s["claims_with_margin"] == s["claims_total"], (
        f"{s['claims_with_margin']}/{s['claims_total']} bounded; "
        "every claim must be auditable."
    )


def test_no_ok_with_negative_margin(audit: dict) -> None:
    """`ok`-status rows must have non-negative margin.

    This is the *parity* invariant: the audit-side margin computation
    must agree with ``literature_faithfulness._classify`` on which
    rows are inside their claim envelope. A row classified ``ok`` but
    with negative audit margin means one of the two is wrong.
    """
    n = audit["summary"]["negative_ok_count"]
    assert n <= NEGATIVE_OK_CEILING, (
        f"{n} ok-status rows have negative audit margin "
        "(classifier/audit disagreement). Negative ok rows:\n"
        + "\n".join(
            f"  {r['graph']}/{r['app']}/{r['l3_size']}/{r['policy']}: "
            f"margin={r['margin_pp']:.3f}pp binding={r['binding_boundary']}"
            for r in audit["negative_ok_rows"]
        )
    )


def test_median_margin_floor(audit: dict) -> None:
    median = audit["summary"]["median"]
    assert (
        median is not None and median >= MEDIAN_MARGIN_FLOOR_PP
    ), f"median margin {median} below floor {MEDIAN_MARGIN_FLOOR_PP}"


def test_mean_margin_floor(audit: dict) -> None:
    mean = audit["summary"]["mean"]
    assert (
        mean is not None and mean >= MEAN_MARGIN_FLOOR_PP
    ), f"mean margin {mean} below floor {MEAN_MARGIN_FLOOR_PP}"


def test_p25_margin_floor(audit: dict) -> None:
    """Lower quartile of margin must stay positive enough.

    Locks distribution shape: even the 25th percentile cell should be
    > 1 pp from any disagree boundary.
    """
    p25 = audit["summary"]["p25"]
    assert (
        p25 is not None and p25 >= P25_MARGIN_FLOOR_PP
    ), f"25th percentile margin {p25} below floor {P25_MARGIN_FLOOR_PP}"


def test_fragile_ceiling(audit: dict) -> None:
    fragile = audit["summary"]["fragile_count"]
    threshold = audit["fragile_threshold_pp"]
    assert fragile <= FRAGILE_CEILING, (
        f"{fragile} cells now within {threshold}pp of disagree boundary "
        f"(ceiling {FRAGILE_CEILING})."
    )


# ---------- per-status invariants ----------

def test_per_status_ok_strictly_positive(audit: dict) -> None:
    ok_stats = audit["per_status_stats"].get("ok")
    assert ok_stats is not None, "missing per-status ok bucket"
    assert ok_stats["count"] > 0
    assert ok_stats["min"] is not None and ok_stats["min"] > 0, (
        f"ok-status min margin = {ok_stats['min']} (must be > 0)"
    )


def test_per_status_known_deviation_non_positive(audit: dict) -> None:
    """`known_deviation` rows are the cells that *would* disagree.

    They are whitelisted by `known_deviation_reason` in the comparator.
    By construction their audit-side margin must be non-positive
    (delta_pct sits outside the claim envelope).
    """
    kd = audit["per_status_stats"].get("known_deviation")
    if kd is None or kd["count"] == 0:
        pytest.skip("no known_deviation rows in corpus")
    assert kd["max"] <= 0, (
        f"known_deviation max margin {kd['max']} > 0 — these rows should "
        "all sit outside the claim envelope."
    )


def test_per_status_within_tolerance_inside_band(audit: dict) -> None:
    wt = audit["per_status_stats"].get("within_tolerance")
    if wt is None or wt["count"] == 0:
        pytest.skip("no within_tolerance rows in corpus")
    assert 0 <= wt["min"], (
        f"within_tolerance min margin {wt['min']} should be ≥ 0"
    )


# ---------- per-family invariants ----------

def test_every_expected_family_present(audit: dict) -> None:
    present = set(audit["per_family_stats"].keys())
    missing = EXPECTED_FAMILIES - present
    assert not missing, f"missing family in margin audit: {sorted(missing)}"


def test_no_unknown_family(audit: dict) -> None:
    extras = set(audit["per_family_stats"].keys()) - EXPECTED_FAMILIES
    assert not extras, (
        f"family map missed: {sorted(extras)} — extend GRAPH_FAMILY "
        "in lit_faith_margin.py."
    )


@pytest.mark.parametrize("family", sorted(EXPECTED_FAMILIES))
def test_per_family_median_floor(audit: dict, family: str) -> None:
    floor = FAMILY_MEDIAN_FLOOR_PP[family]
    fam_stats = audit["per_family_stats"][family]
    assert fam_stats["count"] > 0, f"family {family} has zero claims"
    assert (
        fam_stats["median"] is not None and fam_stats["median"] >= floor
    ), (
        f"family {family} median margin {fam_stats['median']} below "
        f"floor {floor}"
    )


# ---------- binding-boundary vocabulary ----------

def test_binding_boundary_vocabulary(audit: dict) -> None:
    seen = set(audit["binding_boundary_counts"].keys())
    extras = seen - EXPECTED_BINDING_BOUNDARIES
    assert not extras, (
        f"unexpected binding-boundary label(s) {sorted(extras)} — "
        "if this is intentional add to EXPECTED_BINDING_BOUNDARIES."
    )


def test_no_unbounded_binding(audit: dict) -> None:
    """No row should land in the ``unbounded`` bucket.

    All literature claims today are bounded (every magnitude-only
    ``~`` claim has a ``max_abs_delta_pct``; every POPT_NEAR claim
    falls into either trigger_headroom or near_grasp_upper). If this
    changes the margin audit can no longer guard the corpus on those
    rows — surface the regression here.
    """
    counts = audit["binding_boundary_counts"]
    assert counts.get("unbounded", 0) == 0, (
        f"{counts.get('unbounded')} rows fell into the unbounded "
        "binding bucket — extend lit_faith_margin.py to bound them."
    )


# ---------- rows-level invariants ----------

def test_rows_match_summary_total(audit: dict) -> None:
    assert len(audit["rows"]) == audit["summary"]["claims_total"]


def test_fragile_rows_listed_with_low_margin(audit: dict) -> None:
    threshold = audit["fragile_threshold_pp"]
    for row in audit["fragile_rows"]:
        assert row["margin_pp"] is not None and row["margin_pp"] < threshold, (
            f"fragile row {row} has margin {row['margin_pp']} >= {threshold}"
        )
        assert row["fragile"] is True
    # Conversely, rows tagged fragile=True in `rows` must appear in fragile_rows.
    rows_fragile_keys = {
        (r["graph"], r["app"], r["l3_size"], r["policy"])
        for r in audit["rows"]
        if r["fragile"]
    }
    listed_keys = {
        (r["graph"], r["app"], r["l3_size"], r["policy"])
        for r in audit["fragile_rows"]
    }
    assert rows_fragile_keys == listed_keys, (
        "fragile_rows list does not match fragile=True rows in `rows`"
    )
