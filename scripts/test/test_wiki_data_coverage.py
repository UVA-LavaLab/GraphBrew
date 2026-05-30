"""Confidence gate 212 — wiki/data/ coverage (WDC-Cov).

Closes the loop between on-disk wiki/data/ files and the reproduce_smoke
byte-parity audit. Every file under wiki/data/ with a recognised
artifact extension (.json, .md, .csv) must be either in
TRACKED_ARTIFACTS or in a documented WIKI_UNTRACKED_EXEMPT allow-list.
This catches the failure mode of: a generator drops a new artifact in
wiki/data/ and nobody remembers to add it to TRACKED_ARTIFACTS, so its
byte stability is silently un-audited (regen could quietly drift and
nobody would know).

Complements:
* RSC-Cov (gate 210): every CATALOG.artifact is in TRACKED_ARTIFACTS.
  CATALOG is generator-centric — every entry has a generator script.
* MFC-Int (gate 211): every CATALOG.generator is invoked by Makefile
  (or on documented allow-list).
* WDC-Cov (this gate): every wiki/data/*.{json,md,csv} ON DISK is in
  TRACKED_ARTIFACTS (or allow-list). The .md and .csv siblings of
  catalogued JSONs may not have their OWN catalog entry, so RSC-Cov
  alone would miss them.

Together: any new wiki/data/ artifact must either land in CATALOG
(audited by RSC-Cov + reproduce_smoke), or live as a sibling of one
(audited by WDC-Cov + reproduce_smoke), or be in this gate's
WIKI_UNTRACKED_EXEMPT (audited only by RSC-Cov SELF_REF_EXEMPT — for
the audit output itself, which is chicken-and-egg).

Six groups (14 tests):

  A. On-disk inventory (3):
       1. wiki/data/ contains >= 50 .json files (sanity floor)
       2. wiki/data/ contains >= 50 .md files (sanity floor)
       3. wiki/data/ contains >= 10 .csv files (sanity floor)

  B. Tracked coverage (3):
       4. every on-disk .json is in TRACKED_ARTIFACTS or
          WIKI_UNTRACKED_EXEMPT
       5. every on-disk .md is in TRACKED_ARTIFACTS or
          WIKI_UNTRACKED_EXEMPT
       6. every on-disk .csv is in TRACKED_ARTIFACTS or
          WIKI_UNTRACKED_EXEMPT

  C. Allow-list well-formedness (3):
       7. WIKI_UNTRACKED_EXEMPT is non-empty
       8. every WIKI_UNTRACKED_EXEMPT entry exists on disk
       9. WIKI_UNTRACKED_EXEMPT and TRACKED_ARTIFACTS are disjoint
          (an entry is either tracked OR exempt, never both)

  D. TRACKED_ARTIFACTS soundness (2):
      10. every TRACKED_ARTIFACTS entry exists on disk
      11. every TRACKED_ARTIFACTS entry's extension ∈ {.json, .md, .csv}

  E. Extension-coverage parity (2):
      12. CSV companions: every .csv on disk that has a sibling .json
          OR .md on disk is itself tracked (no silent CSV drift)
      13. MD companions: every .json on disk that has a sibling .md on
          disk has BOTH tracked together (catches half-tracked pairs)

  F. Self-consistency (1):
      14. |on-disk recognised| == |TRACKED_ARTIFACTS ∩ on-disk| +
          |WIKI_UNTRACKED_EXEMPT ∩ on-disk|

Load-bearing rules:

* WIKI_UNTRACKED_EXEMPT covers the audit's own output: reproduce_smoke
  produces reproduce_smoke.json + reproduce_smoke.md. Tracking them in
  TRACKED_ARTIFACTS would create a chicken-and-egg loop where each
  audit run would change its own output and detect drift in itself.
* CSV siblings of catalogued JSONs were silently un-audited until this
  gate; gate 212 added all 15 to TRACKED_ARTIFACTS after verifying each
  is byte-stable across `make lit-claims lit-catalog`.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.experiments.ecg.reproduce_smoke import TRACKED_ARTIFACTS  # noqa: E402

RECOGNISED_EXTENSIONS = (".json", ".md", ".csv")

# Allow-list — documented exceptions ONLY. Adding an entry here requires
# documenting why the file is intentionally NOT byte-audited by
# reproduce_smoke.
WIKI_UNTRACKED_EXEMPT: frozenset[str] = frozenset(
    {
        # The audit output itself. reproduce_smoke writes these on
        # every run; including them in TRACKED_ARTIFACTS would form
        # a chicken-and-egg loop where the audit drifts on itself.
        "reproduce_smoke.json",
        "reproduce_smoke.md",
        # Source-of-truth postfix JSON for the ECG substrate-parity
        # gate (gate 238). This file is a hand-curated snapshot of
        # `scripts/experiments/ecg/proof_matrix.py` output; it is the
        # *input* to `lit-ecg-parity`, not a regenerated output, so
        # the reproduce_smoke chain (which exists to detect drift in
        # *generated* artifacts) does not own it.
        "ecg_substrate_parity_postfix.json",
        # Source-of-truth postfix JSON for the ECG gem5 substrate-parity
        # gate (gate 239). Hand-curated snapshot of the gem5 bracket-
        # sweep `roi_matrix.csv`; INPUT to `lit-ecg-gem5-parity`, not
        # a regenerated output. Same pattern as the cache_sim postfix
        # one entry above.
        "ecg_gem5_parity_postfix.json",
    }
)


def _on_disk(ext: str) -> set[str]:
    return {p.name for p in WIKI_DATA.glob(f"*{ext}")}


_TRACKED_SET = set(TRACKED_ARTIFACTS)
_ALL_ON_DISK = (
    _on_disk(".json") | _on_disk(".md") | _on_disk(".csv")
)


# ---------------------------------------------------------------------------
# Group A — On-disk inventory
# ---------------------------------------------------------------------------


def test_min_json_count():
    n = len(_on_disk(".json"))
    assert n >= 50, f"wiki/data/*.json count dropped below floor of 50 (got {n}); corpus collapse?"


def test_min_md_count():
    n = len(_on_disk(".md"))
    assert n >= 50, f"wiki/data/*.md count dropped below floor of 50 (got {n})"


def test_min_csv_count():
    n = len(_on_disk(".csv"))
    assert n >= 10, f"wiki/data/*.csv count dropped below floor of 10 (got {n})"


# ---------------------------------------------------------------------------
# Group B — Tracked coverage
# ---------------------------------------------------------------------------


def _untracked(ext: str) -> list[str]:
    return sorted(
        n for n in _on_disk(ext)
        if n not in _TRACKED_SET and n not in WIKI_UNTRACKED_EXEMPT
    )


def test_all_json_files_tracked():
    bad = _untracked(".json")
    assert not bad, (
        f"wiki/data/*.json files NOT in TRACKED_ARTIFACTS and NOT in "
        f"WIKI_UNTRACKED_EXEMPT: {bad}. Either add to TRACKED_ARTIFACTS "
        f"or to WIKI_UNTRACKED_EXEMPT with a documented reason."
    )


def test_all_md_files_tracked():
    bad = _untracked(".md")
    assert not bad, (
        f"wiki/data/*.md files NOT in TRACKED_ARTIFACTS and NOT in "
        f"WIKI_UNTRACKED_EXEMPT: {bad}."
    )


def test_all_csv_files_tracked():
    bad = _untracked(".csv")
    assert not bad, (
        f"wiki/data/*.csv files NOT in TRACKED_ARTIFACTS and NOT in "
        f"WIKI_UNTRACKED_EXEMPT: {bad}."
    )


# ---------------------------------------------------------------------------
# Group C — Allow-list well-formedness
# ---------------------------------------------------------------------------


def test_allow_list_non_empty():
    assert WIKI_UNTRACKED_EXEMPT, (
        "WIKI_UNTRACKED_EXEMPT is empty — if no exemptions are needed, "
        "this gate should not have an allow-list mechanism"
    )


def test_allow_list_entries_on_disk():
    missing = sorted(n for n in WIKI_UNTRACKED_EXEMPT if not (WIKI_DATA / n).is_file())
    assert not missing, (
        f"WIKI_UNTRACKED_EXEMPT entries missing on disk: {missing}. "
        f"Either restore the file or remove from the allow-list."
    )


def test_allow_list_disjoint_from_tracked():
    overlap = WIKI_UNTRACKED_EXEMPT & _TRACKED_SET
    assert not overlap, (
        f"Files appear in BOTH TRACKED_ARTIFACTS and WIKI_UNTRACKED_EXEMPT: {sorted(overlap)}. "
        f"Pick one — either audit it OR exempt it, never both."
    )


# ---------------------------------------------------------------------------
# Group D — TRACKED_ARTIFACTS soundness
# ---------------------------------------------------------------------------


def test_tracked_artifacts_all_on_disk():
    missing = sorted(n for n in TRACKED_ARTIFACTS if not (WIKI_DATA / n).is_file())
    assert not missing, f"TRACKED_ARTIFACTS entries missing on disk: {missing}"


def test_tracked_artifacts_extensions():
    bad = sorted(n for n in TRACKED_ARTIFACTS if not n.endswith(RECOGNISED_EXTENSIONS))
    assert not bad, (
        f"TRACKED_ARTIFACTS entries with unrecognised extension "
        f"(expected {RECOGNISED_EXTENSIONS}): {bad}"
    )


# ---------------------------------------------------------------------------
# Group E — Extension-coverage parity
# ---------------------------------------------------------------------------


def test_csv_with_sibling_is_tracked():
    """If foo.csv exists AND foo.json or foo.md exists, foo.csv must
    be tracked (or exempt). Catches the half-tracked-trio failure mode.
    """
    on_disk_json = _on_disk(".json")
    on_disk_md = _on_disk(".md")
    bad = []
    for csv_name in _on_disk(".csv"):
        stem = csv_name[: -len(".csv")]
        json_name = f"{stem}.json"
        md_name = f"{stem}.md"
        has_sibling = json_name in on_disk_json or md_name in on_disk_md
        if has_sibling and csv_name not in _TRACKED_SET and csv_name not in WIKI_UNTRACKED_EXEMPT:
            bad.append(csv_name)
    assert not bad, (
        f".csv files with sibling .json/.md on disk that are NOT tracked: {sorted(bad)}. "
        f"If the .json/.md is tracked, the .csv must be too (or exempt)."
    )


def test_json_md_pairs_both_tracked():
    """If foo.json AND foo.md both exist on disk, both must be tracked
    (or exempt) — catches half-tracked .json/.md pairs.
    """
    on_disk_json = _on_disk(".json")
    on_disk_md = _on_disk(".md")
    bad = []
    for json_name in on_disk_json:
        stem = json_name[: -len(".json")]
        md_name = f"{stem}.md"
        if md_name not in on_disk_md:
            continue
        json_status = json_name in _TRACKED_SET or json_name in WIKI_UNTRACKED_EXEMPT
        md_status = md_name in _TRACKED_SET or md_name in WIKI_UNTRACKED_EXEMPT
        if json_status != md_status:
            bad.append((json_name, json_status, md_name, md_status))
    assert not bad, (
        f"Half-tracked .json/.md pairs (one tracked/exempt but not the other): {bad}. "
        f"Either track both or exempt both."
    )


# ---------------------------------------------------------------------------
# Group F — Self-consistency
# ---------------------------------------------------------------------------


def test_recognised_files_fully_partition():
    """Every on-disk recognised file is in exactly one of {tracked, exempt}.
    Equivalent to: |on-disk recognised| == |tracked ∩ on-disk| + |exempt ∩ on-disk|.
    """
    tracked_on_disk = _TRACKED_SET & _ALL_ON_DISK
    exempt_on_disk = WIKI_UNTRACKED_EXEMPT & _ALL_ON_DISK
    overlap = tracked_on_disk & exempt_on_disk
    assert not overlap, f"On-disk files in BOTH tracked and exempt: {sorted(overlap)}"
    partition_size = len(tracked_on_disk) + len(exempt_on_disk)
    assert partition_size == len(_ALL_ON_DISK), (
        f"On-disk recognised files ({len(_ALL_ON_DISK)}) != "
        f"|tracked ∩ on-disk| ({len(tracked_on_disk)}) + "
        f"|exempt ∩ on-disk| ({len(exempt_on_disk)}) = {partition_size}. "
        f"Missing files (on disk, neither tracked nor exempt): "
        f"{sorted(_ALL_ON_DISK - tracked_on_disk - exempt_on_disk)}"
    )
