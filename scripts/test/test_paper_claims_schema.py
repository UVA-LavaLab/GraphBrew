"""Confidence gate 214 — paper claims schema integrity (PCS-Sch).

paper_claims.json is the single registry the paper introduction,
abstract, and discussion cite by ID. Its rows MUST conform to a strict
schema so that:

* downstream rendering (paper_claims.md, paper-text quoting) can rely
  on every claim having every field;
* a single typo in a units string or category name doesn't silently
  pollute the schema by inventing a new bucket;
* claims that reference deleted artifacts or non-existent gates fail
  loudly here, not at paper-rendering time.

Complements:
* PCV-Src (gate 213): every claim.value matches its source-derivation
  (value parity).
* XAI-Int (gate 209): every claim.source and claim.gate path appears
  in CATALOG/PYTEST_SUITES (cross-registry graph).
* PCS-Sch (this gate): every claim's schema fields are present, of
  the right type, and from the controlled vocabularies (schema parity).

Six groups (19 tests):

  A. Registry shape (1):
       1. claims is a non-empty list

  B. Per-claim required fields (1):
       2. every claim has all required fields: id, category, text,
          value, units, source, gate (parametric across REQUIRED_FIELDS)

  C. Per-claim field types (6):
       3. id is a dotted-string (str, contains '.', no whitespace)
       4. category is in KNOWN_CATEGORIES
       5. units is in KNOWN_UNITS
       6. value is int or float (NOT str or bool)
       7. text is a non-empty string
       8. source/gate are non-empty strings

  D. ID uniqueness + naming (2):
       9. claim IDs are unique across the registry
      10. claim IDs are lowercase ASCII + dot + underscore only
          (no spaces, no uppercase, no special chars)

  E. Cross-field path resolution (4):
      11. every source path starts with 'wiki/data/'
      12. every source file exists on disk
      13. every gate path starts with 'scripts/'
      14. every gate file exists on disk

  F. Vocabulary closure + sanity (5):
      15. all observed categories ⊆ KNOWN_CATEGORIES
      16. all observed units ⊆ KNOWN_UNITS
      17. KNOWN_CATEGORIES is non-empty
      18. KNOWN_UNITS is non-empty
      19. text length >= 10 chars (claims must be human-readable)

Load-bearing rules:

* KNOWN_CATEGORIES and KNOWN_UNITS are controlled vocabularies — when
  a new claim genuinely needs a new category/unit, expand the
  vocabulary here AND document the addition. Forbidding open-ended
  expansion catches typos like 'percentage' vs 'percent' or
  'cross_tool' vs 'cross-tool'.
* claim.id and claim.category are independent — id can be e.g.
  'winner.grasp_share' while category is 'winner_table' (gate 213
  documented the legitimate cases). The id prefix is NOT required to
  match the category.
* meta-claims (category='meta') still must follow schema — but their
  source/gate may legitimately be the dashboard generator itself.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PAPER_CLAIMS = json.loads((WIKI_DATA / "paper_claims.json").read_text())["claims"]

REQUIRED_FIELDS = ("id", "category", "text", "value", "units", "source", "gate")

KNOWN_CATEGORIES = frozenset(
    {
        "corpus",
        "cross_tool",
        "deviations",
        "lit_faith",
        "meta",
        "popt_vs_grasp",
        "reproduction",
        "thrash",
        "winner_table",
    }
)

KNOWN_UNITS = frozenset(
    {
        "cells",
        "claims",
        "disagreements",
        "gates",
        "graphs",
        "percent",
        "pp",
    }
)

_ID_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")


# ---------------------------------------------------------------------------
# Group A — Registry shape
# ---------------------------------------------------------------------------


def test_claims_is_non_empty_list():
    assert isinstance(PAPER_CLAIMS, list), "paper_claims.json 'claims' must be a list"
    assert len(PAPER_CLAIMS) > 0, "paper_claims.json 'claims' must be non-empty"


# ---------------------------------------------------------------------------
# Group B — Per-claim required fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_claim_has_required_field(field: str):
    missing = [c.get("id", "<no-id>") for c in PAPER_CLAIMS if field not in c]
    assert not missing, f"Claims missing required field '{field}': {missing}"


# ---------------------------------------------------------------------------
# Group C — Per-claim field types
# ---------------------------------------------------------------------------


def test_every_id_is_dotted_string():
    bad = [c["id"] for c in PAPER_CLAIMS if not (isinstance(c["id"], str) and "." in c["id"])]
    assert not bad, f"Claim IDs not dotted-strings: {bad}"


def test_every_category_is_known():
    bad = [(c["id"], c["category"]) for c in PAPER_CLAIMS if c["category"] not in KNOWN_CATEGORIES]
    assert not bad, (
        f"Claims with unknown category (typo or new vocabulary): {bad}. "
        f"Known: {sorted(KNOWN_CATEGORIES)}"
    )


def test_every_units_is_known():
    bad = [(c["id"], c["units"]) for c in PAPER_CLAIMS if c["units"] not in KNOWN_UNITS]
    assert not bad, (
        f"Claims with unknown units (typo or new vocabulary): {bad}. "
        f"Known: {sorted(KNOWN_UNITS)}"
    )


def test_every_value_is_numeric():
    bad = []
    for c in PAPER_CLAIMS:
        v = c["value"]
        # NOTE: bool is a subclass of int in Python — exclude explicitly
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            bad.append((c["id"], type(v).__name__, v))
    assert not bad, f"Claim values that are not int/float (bool excluded): {bad}"


def test_every_text_is_non_empty_string():
    bad = [c["id"] for c in PAPER_CLAIMS if not (isinstance(c["text"], str) and c["text"].strip())]
    assert not bad, f"Claims with empty/non-string text: {bad}"


def test_every_source_and_gate_are_strings():
    bad = []
    for c in PAPER_CLAIMS:
        if not (isinstance(c["source"], str) and c["source"]):
            bad.append((c["id"], "source", c["source"]))
        if not (isinstance(c["gate"], str) and c["gate"]):
            bad.append((c["id"], "gate", c["gate"]))
    assert not bad, f"Claims with empty/non-string source or gate: {bad}"


# ---------------------------------------------------------------------------
# Group D — ID uniqueness + naming
# ---------------------------------------------------------------------------


def test_claim_ids_unique():
    ids = [c["id"] for c in PAPER_CLAIMS]
    seen = {}
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    assert not dupes, f"Duplicate claim IDs: {dupes}"


def test_claim_ids_match_naming_convention():
    bad = [c["id"] for c in PAPER_CLAIMS if not _ID_RE.match(c["id"])]
    assert not bad, (
        f"Claim IDs not matching [a-z0-9_]+(\\.[a-z0-9_]+)+: {bad}. "
        f"Lowercase ASCII + dot + underscore only; no spaces; no uppercase."
    )


# ---------------------------------------------------------------------------
# Group E — Cross-field path resolution
# ---------------------------------------------------------------------------


def test_every_source_under_wiki_data():
    bad = [(c["id"], c["source"]) for c in PAPER_CLAIMS if not c["source"].startswith("wiki/data/")]
    assert not bad, f"Claims with source not under wiki/data/: {bad}"


def test_every_source_file_exists():
    missing = [(c["id"], c["source"]) for c in PAPER_CLAIMS if not (REPO_ROOT / c["source"]).is_file()]
    assert not missing, f"Claims with non-existent source file: {missing}"


def test_every_gate_under_scripts():
    bad = [(c["id"], c["gate"]) for c in PAPER_CLAIMS if not c["gate"].startswith("scripts/")]
    assert not bad, f"Claims with gate path not under scripts/: {bad}"


def test_every_gate_file_exists():
    missing = [(c["id"], c["gate"]) for c in PAPER_CLAIMS if not (REPO_ROOT / c["gate"]).is_file()]
    assert not missing, f"Claims with non-existent gate file: {missing}"


# ---------------------------------------------------------------------------
# Group F — Vocabulary closure + sanity
# ---------------------------------------------------------------------------


def test_categories_are_closed_vocabulary():
    observed = {c["category"] for c in PAPER_CLAIMS}
    extras = observed - KNOWN_CATEGORIES
    assert not extras, (
        f"Observed categories not in controlled vocabulary: {sorted(extras)}. "
        f"If genuinely new, expand KNOWN_CATEGORIES and document."
    )


def test_units_are_closed_vocabulary():
    observed = {c["units"] for c in PAPER_CLAIMS}
    extras = observed - KNOWN_UNITS
    assert not extras, (
        f"Observed units not in controlled vocabulary: {sorted(extras)}. "
        f"If genuinely new, expand KNOWN_UNITS and document."
    )


def test_known_categories_non_empty():
    assert KNOWN_CATEGORIES, "KNOWN_CATEGORIES is empty — vocabulary mechanism broken"


def test_known_units_non_empty():
    assert KNOWN_UNITS, "KNOWN_UNITS is empty — vocabulary mechanism broken"


def test_text_has_minimum_length():
    bad = [(c["id"], len(c["text"])) for c in PAPER_CLAIMS if len(c["text"]) < 10]
    assert not bad, (
        f"Claims with text shorter than 10 chars (not human-readable): {bad}"
    )
