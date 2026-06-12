"""Lock the LIT-DevExp invariants (gate 232).

Every known_deviation row's `known_deviation_reason` must name at least
one algorithmic mechanism, exceed a length floor, carry a citation,
resolve cross-references, and the same reason text may not cover more
than half the rows. See `scripts/experiments/ecg/lit_faith_devexp.py`
for the rule definitions (R1–R7).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT  = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "wiki" / "data" / "lit_faith_devexp.json"
MD    = ROOT / "wiki" / "data" / "lit_faith_devexp.md"
CSV   = ROOT / "wiki" / "data" / "lit_faith_devexp.csv"


@pytest.fixture(scope="module")
def audit() -> dict:
    if not AUDIT.exists():
        pytest.skip("lit_faith_devexp.json missing; run `make lit-devexp`")
    return json.loads(AUDIT.read_text(encoding="utf-8"))


# --- Artifacts present ------------------------------------------------------

def test_artifacts_present():
    assert AUDIT.exists(), f"missing {AUDIT}"
    assert MD.exists(),    f"missing {MD}"
    assert CSV.exists(),   f"missing {CSV}"


def test_schema_version(audit):
    assert audit["schema_version"] == 1


def test_top_level_keys(audit):
    for k in ("summary", "rows", "violations"):
        assert k in audit


# --- Tolerance pinning ------------------------------------------------------

def test_tolerances_pinned(audit):
    s = audit["summary"]
    assert s["min_reason_len"]       == 60
    assert s["min_mechanism_hits"]   == 1
    assert s["min_known_deviations"] == 15
    assert s["reuse_ceiling_frac"]   == pytest.approx(0.50)


# --- Zero-violation invariant -----------------------------------------------

def test_zero_violations(audit):
    assert audit["violations"] == [], (
        f"LIT-DevExp violations: {audit['violations'][:5]}"
    )


def test_summary_violation_count_matches(audit):
    assert audit["summary"]["violations"] == len(audit["violations"])


# --- Row-count floor + diversity --------------------------------------------

def test_known_deviation_row_floor(audit):
    n = audit["summary"]["known_deviation_rows"]
    assert n >= audit["summary"]["min_known_deviations"], (
        f"only {n} known_deviation rows (floor "
        f"{audit['summary']['min_known_deviations']})"
    )


def test_unique_reason_texts_majority(audit):
    n = audit["summary"]["known_deviation_rows"]
    uniq = audit["summary"]["unique_reason_texts"]
    if n == 0:
        pytest.skip("no rows")
    assert uniq / n >= 0.80, (
        f"unique reason ratio {uniq/n:.2f} < 0.80 — too much copy-paste"
    )


def test_max_reuse_count_under_ceiling(audit):
    n = audit["summary"]["known_deviation_rows"]
    if n == 0:
        pytest.skip("no rows")
    ceiling = int(audit["summary"]["reuse_ceiling_frac"] * n)
    assert audit["summary"]["max_reuse_count"] <= ceiling, (
        f"max reuse {audit['summary']['max_reuse_count']} > ceiling {ceiling}"
    )


# --- Per-row quality floors --------------------------------------------------

def test_every_row_has_length_above_floor(audit):
    floor = audit["summary"]["min_reason_len"]
    for r in audit["rows"]:
        assert r["reason_len"] >= floor, (
            f"row {r['graph']}/{r['app']}/{r['l3_size']}/{r['policy']} "
            f"reason length {r['reason_len']} < {floor}"
        )


def test_every_row_has_mechanism_hit(audit):
    floor = audit["summary"]["min_mechanism_hits"]
    for r in audit["rows"]:
        assert r["mechanism_hits"] >= floor, (
            f"row {r['graph']}/{r['app']}/{r['l3_size']}/{r['policy']} "
            f"has {r['mechanism_hits']} mechanism hits (floor {floor})"
        )


def test_every_row_has_citation(audit):
    for r in audit["rows"]:
        assert r["citation"], (
            f"row {r['graph']}/{r['app']}/{r['l3_size']}/{r['policy']} "
            f"missing citation"
        )


def test_median_reason_length_substantial(audit):
    med = audit["summary"]["median_reason_len"]
    assert med is None or med >= 120, (
        f"median reason length {med} < 120 — reasons are too terse to "
        f"explain a mechanism"
    )


def test_median_mechanism_hits_floor(audit):
    med = audit["summary"]["median_mechanism_hits"]
    assert med is None or med >= 2, (
        f"median mechanism hits {med} < 2"
    )


# --- Row-schema integrity ---------------------------------------------------

def test_row_keys_present(audit):
    required = {"graph", "app", "l3_size", "policy", "citation",
                "reason_len", "mechanism_hits", "cross_ref",
                "reason_excerpt"}
    for r in audit["rows"]:
        missing = required - set(r)
        assert not missing, f"row missing keys {missing}: {r}"


def test_rows_with_cross_ref_under_total(audit):
    s = audit["summary"]
    assert s["rows_with_cross_ref"] <= s["known_deviation_rows"]


# --- CSV / MD presence parity ------------------------------------------------

def test_csv_row_count_matches(audit):
    import csv as csvm
    with CSV.open(encoding="utf-8") as fh:
        rows = list(csvm.reader(fh))
    # rows[0] is header
    assert len(rows) - 1 == len(audit["rows"]), (
        f"CSV row count {len(rows) - 1} != JSON row count {len(audit['rows'])}"
    )


def test_markdown_artifact_nonempty():
    txt = MD.read_text(encoding="utf-8")
    assert "LIT-DevExp" in txt
    assert "Per-row detail" in txt
