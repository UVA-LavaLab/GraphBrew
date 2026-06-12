"""Tests for gate 237 — LIT-CitDate (literature-faithfulness citation/date audit)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_JSON = ROOT / "wiki" / "data" / "lit_faith_citdate.json"
ARTIFACT_MD   = ROOT / "wiki" / "data" / "lit_faith_citdate.md"
ARTIFACT_CSV  = ROOT / "wiki" / "data" / "lit_faith_citdate.csv"


@pytest.fixture(scope="module")
def audit() -> dict:
    assert ARTIFACT_JSON.exists(), (
        f"Missing {ARTIFACT_JSON}. Run `make lit-citdate`."
    )
    return json.loads(ARTIFACT_JSON.read_text())


# --- schema --------------------------------------------------------------

def test_schema_has_top_level_sections(audit):
    for key in ("rules", "constants", "totals", "by_policy", "violations"):
        assert key in audit, f"missing top-level section: {key}"


def test_seven_rules_present(audit):
    expected = {"D1", "D2", "D3", "D4", "D5", "D6", "D7"}
    assert set(audit["rules"].keys()) == expected


def test_constants_pinned(audit):
    c = audit["constants"]
    assert set(c["venue_whitelist"]) >= {"HPCA", "ISCA", "MICRO", "ASPLOS"}
    assert c["year_range"] == [2005, 2026]
    assert c["distinct_citation_floor"] == 10
    pol = c["policy_origin"]
    assert pol["GRASP"]              == {"author": "Faldu",  "venue": "HPCA", "year": 2020}
    assert pol["POPT"]               == {"author": "Balaji", "venue": "HPCA", "year": 2021}
    assert pol["POPT_GE_GRASP"]      == {"author": "Balaji", "venue": "HPCA", "year": 2021}
    assert pol["SRRIP"]              == {"author": "Jaleel", "venue": "ISCA", "year": 2010}


# --- core invariant ------------------------------------------------------

def test_zero_violations(audit):
    assert audit["violations"] == [], (
        f"unexpected citation audit violations: {audit['violations']}"
    )


def test_no_unparseable_citations(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "D2-unparseable-citation"]
    assert bad == []


def test_no_empty_citations(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "D1-empty-citation"]
    assert bad == []


def test_no_locator_omissions(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "D6-no-locator"]
    assert bad == []


def test_no_origin_mismatches(audit):
    bad = [v for v in audit["violations"] if v.get("rule") == "D5-policy-origin-mismatch"]
    assert bad == []


# --- floors / tallies ----------------------------------------------------

def test_all_rows_parsed(audit):
    t = audit["totals"]
    assert t["rows"] >= 270, f"per_claim row count regressed: {t['rows']}"
    assert t["rows_parsed_ok"] == t["rows"], (
        "every per_claim row must parse cleanly"
    )


def test_distinct_citation_floor(audit):
    t = audit["totals"]
    floor = audit["constants"]["distinct_citation_floor"]
    assert t["distinct_citation_count"] >= floor, (
        f"distinct citation strings dropped below floor "
        f"{floor}: got {t['distinct_citation_count']}"
    )


def test_venue_tally_only_whitelist(audit):
    t = audit["totals"]
    allowed = set(audit["constants"]["venue_whitelist"])
    assert set(t["venue_tally"]).issubset(allowed), (
        f"parsed venues outside whitelist: {set(t['venue_tally']) - allowed}"
    )


def test_year_tally_in_range(audit):
    t = audit["totals"]
    lo, hi = audit["constants"]["year_range"]
    for year_str in t["year_tally"]:
        y = int(year_str)
        assert lo <= y <= hi, f"year {y} outside [{lo}, {hi}]"


# --- per-policy coverage -------------------------------------------------

def test_grasp_cites_faldu_hpca_2020_or_cross_attribution(audit):
    by_policy = audit["by_policy"]
    grasp = by_policy["GRASP"]
    assert grasp["rows"] > 0, "GRASP policy is missing from per_claim"
    assert grasp["parses_ok"] == grasp["rows"], (
        "every GRASP row must parse"
    )


def test_popt_cites_balaji_hpca_2021(audit):
    by_policy = audit["by_policy"]
    popt = by_policy["POPT"]
    assert popt["rows"] > 0, "POPT policy is missing"
    has_balaji_2021 = any(
        "Balaji" in cit and "2021" in cit and "HPCA" in cit
        for cit in popt["distinct_citations"]
    )
    assert has_balaji_2021, (
        f"POPT must cite Balaji & Lucia HPCA 2021; got {popt['distinct_citations']}"
    )


def test_srrip_cites_jaleel_isca_2010(audit):
    by_policy = audit["by_policy"]
    srrip = by_policy["SRRIP"]
    assert srrip["rows"] > 0, "SRRIP policy is missing"
    has_jaleel = any(
        "Jaleel" in cit and ("2010" in cit or "ISCA" in cit)
        for cit in srrip["distinct_citations"]
    )
    assert has_jaleel, (
        f"SRRIP must cite Jaleel ISCA 2010; got {srrip['distinct_citations']}"
    )


def test_derived_policies_present(audit):
    by_policy = audit["by_policy"]
    for derived in ("POPT_GE_GRASP", "POPT_NEAR_GRASP_IF_BIG_GAP"):
        assert derived in by_policy, f"derived policy {derived} missing"
        assert by_policy[derived]["rows"] > 0


# --- companion artifact parity ------------------------------------------

def test_md_artifact_present_and_nonempty():
    assert ARTIFACT_MD.exists(), "Missing lit_faith_citdate.md"
    assert ARTIFACT_MD.stat().st_size > 200


def test_md_lists_all_seven_rules():
    text = ARTIFACT_MD.read_text()
    for rid in ("D1", "D2", "D3", "D4", "D5", "D6", "D7"):
        assert f"**{rid}**" in text, f"rule {rid} missing from md"


def test_csv_artifact_present_and_parses(audit):
    assert ARTIFACT_CSV.exists()
    text = ARTIFACT_CSV.read_text()
    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    assert len(rows) == len(audit["by_policy"]), (
        f"csv row count {len(rows)} != by_policy count {len(audit['by_policy'])}"
    )
    for r in rows:
        assert {"policy", "rows", "parses_ok", "distinct_citation_count"} <= set(r)
