"""Pytest gate: paper-artifact catalog completeness.

The catalog is the single canonical index from each paper claim to
its source artifact + governing pytest gate. This gate enforces:

* every generator script referenced by the catalog exists;
* every pytest gate referenced by the catalog exists;
* every JSON artifact referenced by the catalog exists;
* the catalog ids are unique and use a stable kebab/snake naming;
* the catalog grows monotonically — the entry count must stay above
  a floor so that future PRs cannot silently drop an aggregator.

If this gate ever goes red, the paper's evidence chain has a
dangling link.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_JSON = REPO_ROOT / "wiki" / "data" / "artifact_catalog.json"
CATALOG_MD = REPO_ROOT / "wiki" / "data" / "artifact_catalog.md"

# Bump this whenever a new aggregator is folded in. The dashboard
# gate count and this floor should be the only two places that
# track aggregator growth.
ENTRY_COUNT_FLOOR = 72

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@pytest.fixture(scope="module")
def catalog() -> dict:
    if not CATALOG_JSON.exists():
        pytest.skip(f"{CATALOG_JSON} not generated; run `make lit-catalog`")
    return json.loads(CATALOG_JSON.read_text())


def test_catalog_top_level_schema(catalog):
    assert {"summary", "entries"}.issubset(catalog.keys())
    assert "n_entries" in catalog["summary"]


def test_catalog_has_paper_md_output():
    assert CATALOG_MD.exists(), (
        "wiki/data/artifact_catalog.md missing — regenerate with "
        "`make lit-catalog`"
    )


def test_entry_count_above_floor(catalog):
    n = catalog["summary"]["n_entries"]
    assert n >= ENTRY_COUNT_FLOOR, (
        f"catalog shrank below the {ENTRY_COUNT_FLOOR}-entry floor "
        f"(now {n}) — was an aggregator removed without bumping the floor?"
    )


def test_no_missing_generators(catalog):
    missing = catalog["summary"]["missing_generators"]
    assert not missing, (
        f"catalog lists generators that don't exist on disk: {missing}"
    )


def test_no_missing_gates(catalog):
    missing = catalog["summary"]["missing_gates"]
    assert not missing, (
        f"catalog lists pytest gates that don't exist on disk: {missing}"
    )


def test_no_missing_artifacts(catalog):
    missing = catalog["summary"]["missing_artifacts"]
    assert not missing, (
        f"catalog lists JSON artifacts that don't exist on disk: {missing} — "
        "regenerate with `make confidence`"
    )


def test_ids_are_unique(catalog):
    ids = [e["id"] for e in catalog["entries"]]
    dups = {i for i in ids if ids.count(i) > 1}
    assert not dups, f"duplicate catalog ids: {sorted(dups)}"


def test_ids_use_snake_lower(catalog):
    bad = [e["id"] for e in catalog["entries"] if not ID_PATTERN.match(e["id"])]
    assert not bad, f"catalog ids that don't match [a-z][a-z0-9_]*: {bad}"


def test_each_entry_has_summary(catalog):
    for e in catalog["entries"]:
        assert e.get("summary"), f"catalog entry {e['id']} has no summary"
        assert len(e["summary"]) >= 30, (
            f"catalog entry {e['id']} summary too short ({len(e['summary'])} chars) — "
            "needs to fit a one-line headline finding"
        )


def test_each_entry_has_required_fields(catalog):
    required = {"id", "label", "generator", "gate", "artifact", "summary"}
    for e in catalog["entries"]:
        missing = required - set(e)
        assert not missing, f"catalog entry {e.get('id')} missing fields {missing}"
