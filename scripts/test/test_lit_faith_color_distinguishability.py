"""Tests for gate 243 — POLICY_COLORS perceptual distinguishability.

Companion to gate 242 (paper label-map integrity). Where gate 242
audits the policy *vocabulary*, this gate audits the *visual quality*
of the paper palette: can a reader (or a B&W printer) actually tell
the policies apart on the paper figures?

Like gate 242, this gate has NO scaffold-deferred mode: the
source-of-truth is the code itself (POLICY_COLORS / POLICY_HATCHES
in paper_pipeline.py). Every rule (C1-C6) has both a "no violations"
assertion and a "violation count == 0" assertion against the
rendered artifact.

The gate catches a real failure mode: someone tweaks a hex color or
adds a new policy whose color is visually confusable with an existing
one (exact-equal, ΔE-too-close, lightness-too-close-without-hatch,
or too-faint-on-white). Without the gate this only surfaces when a
reviewer squints at the print.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

ROOT          = Path(__file__).resolve().parents[2]
ARTIFACT_JSON = ROOT / "wiki" / "data" / "lit_faith_color_distinguishability.json"
ARTIFACT_MD   = ROOT / "wiki" / "data" / "lit_faith_color_distinguishability.md"
ARTIFACT_CSV  = ROOT / "wiki" / "data" / "lit_faith_color_distinguishability.csv"
PAPER_PIPELINE = ROOT / "scripts" / "experiments" / "ecg" / "paper_pipeline.py"

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


@pytest.fixture(scope="module")
def audit() -> dict:
    assert ARTIFACT_JSON.exists(), (
        f"Missing {ARTIFACT_JSON}. Run `make lit-color-distinguishability`."
    )
    return json.loads(ARTIFACT_JSON.read_text())


# --- schema --------------------------------------------------------------

def test_status_is_active(audit):
    assert audit["status"] == "active", (
        "Gate 243 has no deferred mode — source-of-truth is the code, "
        "not a curated fixture."
    )


def test_schema_has_top_level_sections(audit):
    for key in ("status", "rules", "thresholds", "totals",
                "acknowledged_bw_pairs", "palette", "pairs", "violations"):
        assert key in audit, f"missing top-level section: {key}"


def test_six_rules_present(audit):
    expected = {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert set(audit["rules"].keys()) == expected


def test_thresholds_present(audit):
    th = audit["thresholds"]
    for key in ("min_delta_e", "min_lightness_delta",
                "min_delta_e_from_white"):
        assert key in th, f"threshold {key} missing"
        assert isinstance(th[key], (int, float))
        assert th[key] > 0


# --- per-rule violation enforcement --------------------------------------

def test_zero_violations_total(audit):
    assert audit["totals"]["violations"] == 0, (
        "Color-distinguishability audit reports violations: "
        f"{audit['violations']}"
    )


def test_no_violations_in_list(audit):
    assert audit["violations"] == [], (
        "Color-distinguishability has open violations: "
        f"{audit['violations']}"
    )


@pytest.mark.parametrize("rule_id", ["C1", "C2", "C3", "C4", "C5", "C6"])
def test_no_violations_per_rule(audit, rule_id):
    hits = [v for v in audit["violations"] if v.get("rule") == rule_id]
    assert hits == [], f"rule {rule_id} has violations: {hits}"


# --- C1 (hex format) belt-and-suspenders ---------------------------------

def test_c1_every_palette_entry_has_well_formed_hex(audit):
    for row in audit["palette"]:
        assert row["color"], f"empty color for {row['policy_label']}"
        assert HEX_RE.match(row["color"]), (
            f"malformed hex for {row['policy_label']}: {row['color']!r}"
        )


# --- C2 (exact dedup) belt-and-suspenders --------------------------------

def test_c2_no_duplicate_hex_codes(audit):
    seen: dict[str, str] = {}
    for row in audit["palette"]:
        h = row["color"].upper()
        assert h not in seen, (
            f"duplicate color {h} on {seen[h]} and {row['policy_label']}"
        )
        seen[h] = row["policy_label"]


# --- C3 (color ΔE) -------------------------------------------------------

def test_c3_every_pair_meets_delta_e_floor(audit):
    th = audit["thresholds"]["min_delta_e"]
    fails = [p for p in audit["pairs"] if p["delta_e"] < th]
    assert fails == [], f"pairs below ΔE floor {th}: {fails}"


# --- C4 (B&W printability) -----------------------------------------------

def test_c4_acknowledged_pairs_each_have_reason(audit):
    for p in audit["acknowledged_bw_pairs"]:
        assert p["reason"].strip(), (
            f"acknowledged pair {p['a']}/{p['b']} has empty reason"
        )
        assert len(p["reason"]) >= 60, (
            f"acknowledged pair {p['a']}/{p['b']} reason too short "
            f"(< 60 chars): {p['reason']!r}"
        )


def test_c4_acknowledged_pairs_are_canonical_form(audit):
    """Pairs must be in (sorted) canonical order — no (b, a) duplicates."""
    for p in audit["acknowledged_bw_pairs"]:
        assert p["a"] < p["b"], (
            f"acknowledged pair not in canonical order: ({p['a']}, {p['b']})"
        )
    pair_keys = [(p["a"], p["b"]) for p in audit["acknowledged_bw_pairs"]]
    assert len(set(pair_keys)) == len(pair_keys), (
        "duplicate entries in acknowledged_bw_pairs"
    )


# --- C5 (against white) belt-and-suspenders ------------------------------

def test_c5_no_color_too_close_to_white(audit):
    """Direct re-check: rebuild ΔE-from-white in-test and ensure every
    color clears the floor."""
    from scripts.experiments.ecg.lit_faith_color_distinguishability import (
        _hex_to_lab, _delta_e76,
    )
    th = audit["thresholds"]["min_delta_e_from_white"]
    white_lab = _hex_to_lab("#FFFFFF")
    for row in audit["palette"]:
        de = _delta_e76(_hex_to_lab(row["color"]), white_lab)
        assert de >= th, (
            f"{row['policy_label']} ({row['color']}) too close to white: "
            f"ΔE={de:.2f} < {th}"
        )


# --- C6 (hatch hygiene) --------------------------------------------------

def test_c6_palette_includes_hatch_marker(audit):
    """Every palette entry must carry a has_hatch boolean."""
    for row in audit["palette"]:
        assert isinstance(row["has_hatch"], bool), (
            f"{row['policy_label']} has_hatch is not a bool"
        )


# --- canonical-policies-have-colors --------------------------------------

CANONICAL = {"LRU", "SRRIP", "GRASP", "POPT"}


def test_canonical_baselines_present_with_colors(audit):
    labels = {row["policy_label"] for row in audit["palette"]}
    missing = CANONICAL - labels
    assert not missing, f"canonical baselines missing from palette: {missing}"


def test_ecg_variants_present_with_colors(audit):
    labels = {row["policy_label"] for row in audit["palette"]}
    expected_ecg = {"ECG_DBG_ONLY", "ECG_DBG_PRIMARY",
                    "ECG_DBG_PRIMARY_CHARGED", "ECG_POPT_PRIMARY"}
    missing = expected_ecg - labels
    assert not missing, f"ECG variants missing from palette: {missing}"


def test_charged_variants_carry_hatch(audit):
    """The two ``*_CHARGED`` policies must use a hatch pattern so that
    a B&W reader can distinguish them from their non-charged sibling."""
    hatched = {row["policy_label"] for row in audit["palette"]
               if row["has_hatch"]}
    for k in ("POPT_CHARGED", "ECG_DBG_PRIMARY_CHARGED"):
        assert k in hatched, f"{k} should carry a hatch pattern"


# --- artifact byte-stability ---------------------------------------------

def test_csv_round_trips(audit):
    """CSV must parse cleanly and have one header + 9 palette rows."""
    rows = list(csv.DictReader(ARTIFACT_CSV.open()))
    assert len(rows) == len(audit["palette"]), (
        f"CSV row count ({len(rows)}) != palette length "
        f"({len(audit['palette'])})"
    )
    for r in rows:
        for col in ("policy_label", "figure_label", "color", "lab_L",
                    "lab_a", "lab_b", "has_hatch", "hatch"):
            assert col in r, f"CSV missing column {col}"


def test_md_starts_with_gate_header(audit):
    text = ARTIFACT_MD.read_text()
    assert text.startswith("# POLICY_COLORS perceptual distinguishability"), (
        "MD artifact must start with gate-243 header"
    )
    assert "Rules" in text and "Palette" in text
    assert "Pairwise distances" in text


# --- totals sanity -------------------------------------------------------

def test_pair_count_matches_palette(audit):
    n = audit["totals"]["policy_colors_count"]
    # all pairs from n distinct colors = n choose 2
    expected = n * (n - 1) // 2
    assert audit["totals"]["pairs_checked"] == expected, (
        f"expected {expected} pairs from {n} colors, got "
        f"{audit['totals']['pairs_checked']}"
    )


def test_palette_length_matches_label_count(audit):
    assert len(audit["palette"]) == audit["totals"]["policy_labels_count"]


# --- code-source sanity --------------------------------------------------

def test_paper_pipeline_exports_color_dict():
    """The audit imports POLICY_COLORS from paper_pipeline.py — that
    import must continue to work to keep the gate honest."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("paper_pipeline_test",
                                                  PAPER_PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "POLICY_COLORS"), "POLICY_COLORS missing"
    assert hasattr(mod, "POLICY_HATCHES"), "POLICY_HATCHES missing"
    assert isinstance(mod.POLICY_COLORS, dict)
    assert isinstance(mod.POLICY_HATCHES, dict)
