"""Cross-tool parity gate: the three lit/baseline emitters must agree.

Three independent scripts derive their tables from the same
``literature_baselines.py`` + sweep root:

* ``literature_faithfulness.py``  → ``literature_faithfulness_postfix.csv``
   (one row per (graph, app, l3, policy) claim)
* ``literature_reproduction_summary.py`` → ``literature_reproduction_summary.csv``
   (one row per (citation, graph, app, l3, policy) claim, with status)
* ``paper_baseline_table.py``      → ``paper_baseline_table.csv``
   (one row per (graph, app, l3), all four policy miss-rates in columns)

If those three views drift — e.g. one emitter silently drops cells, or
fields go stale relative to the JSON schema — published wiki tables will
contradict each other. This module asserts:

1. The (graph, app, l3) cell set in ``paper_baseline_table.csv`` is
   exactly the set of (graph, app, l3) groups in the lit-faith CSV.
2. The (graph, app, l3) cell set in the reproduction summary is a
   subset of the cell set in the lit-faith CSV.
3. For cache-policy claims (predicate ∈ {SRRIP, GRASP, POPT}) emitted by
   the reproduction summary, the (graph, app, l3, policy) row also
   appears in the lit-faith CSV.
4. The status vocabularies are consistent: every status present in the
   reproduction summary appears in lit-faith, modulo statuses that one
   emitter computes after the fact (lit-faith adds ``no_claim`` for un-
   cited rows; repro adds ``known_deviation`` for cells registered in
   ``literature_baselines.KNOWN_DEVIATIONS``).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

LIT_FAITH_CSV = WIKI_DATA / "literature_faithfulness_postfix.csv"
REPRO_CSV = WIKI_DATA / "literature_reproduction_summary.csv"
TABLE_CSV = WIKI_DATA / "paper_baseline_table.csv"


def _read_csv(p: Path) -> list[dict]:
    if not p.exists():
        pytest.skip(f"{p.relative_to(REPO_ROOT)} not on disk")
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _cell(row: dict) -> tuple[str, str, str]:
    return (row["graph"], row["app"], row["l3_size"])


def _policy_cell(row: dict) -> tuple[str, str, str, str]:
    return (row["graph"], row["app"], row["l3_size"], row["policy"])


def test_paper_table_cells_match_lit_faith_cells() -> None:
    lit = _read_csv(LIT_FAITH_CSV)
    table = _read_csv(TABLE_CSV)
    lit_cells = {_cell(r) for r in lit}
    table_cells = {_cell(r) for r in table}
    missing_in_table = lit_cells - table_cells
    extra_in_table = table_cells - lit_cells
    assert not missing_in_table, (
        f"{len(missing_in_table)} (graph,app,l3) cell(s) are in "
        f"lit-faith CSV but absent from paper_baseline_table.csv "
        f"(table emitter dropped rows). Examples: "
        f"{sorted(missing_in_table)[:5]}"
    )
    assert not extra_in_table, (
        f"{len(extra_in_table)} (graph,app,l3) cell(s) appear in "
        f"paper_baseline_table.csv but not in lit-faith CSV "
        f"(stale rows in table). Examples: "
        f"{sorted(extra_in_table)[:5]}"
    )


def test_repro_summary_cells_are_subset_of_lit_faith_cells() -> None:
    """Every (graph,app,l3) row in the reproduction summary must appear
    in the lit-faith CSV. The reproduction summary expands each cell
    into multiple rows (one per claim predicate), so we compare at the
    cell granularity only.
    """
    lit = _read_csv(LIT_FAITH_CSV)
    repro = _read_csv(REPRO_CSV)
    lit_cells = {_cell(r) for r in lit}
    repro_cells = {_cell(r) for r in repro}
    extra = repro_cells - lit_cells
    assert not extra, (
        f"{len(extra)} (graph,app,l3) cell(s) appear in the "
        f"reproduction summary but NOT in the lit-faith CSV — the two "
        f"emitters disagree on the live cell set. Examples: "
        f"{sorted(extra)[:5]}"
    )


# Predicate names in the reproduction summary that map directly to a
# cache-policy row in the lit-faith CSV. Multi-policy comparison
# predicates (e.g. POPT_GE_GRASP) don't have a single cache-policy
# counterpart so they're excluded from the strict row check below.
CACHE_POLICY_PREDICATES = {"LRU", "SRRIP", "GRASP", "POPT"}


def test_cache_policy_predicate_rows_align_with_lit_faith() -> None:
    """Within the reproduction summary, rows whose ``policy`` is a
    plain cache-policy name (SRRIP/GRASP/POPT) must have a matching
    (graph,app,l3,policy) row in the lit-faith CSV.
    """
    lit = _read_csv(LIT_FAITH_CSV)
    repro = _read_csv(REPRO_CSV)
    lit_keys = {_policy_cell(r) for r in lit}
    repro_cache_keys = {
        _policy_cell(r)
        for r in repro
        if r.get("policy") in CACHE_POLICY_PREDICATES
    }
    extra = repro_cache_keys - lit_keys
    assert not extra, (
        f"{len(extra)} cache-policy row(s) in the reproduction summary "
        f"don't exist in the lit-faith CSV — emitters dropped cells. "
        f"Examples: {sorted(extra)[:5]}"
    )


def test_repro_summary_covers_all_cited_lit_faith_rows() -> None:
    """Every lit-faith row with a citation that names a cache policy
    must appear in the reproduction summary.

    Lit-faith carries an empty ``claim_citation`` for un-cited (LRU
    reference, sensitivity probe) rows; those are legitimately dropped
    by the reproduction summary. We also restrict to plain
    cache-policy rows because the reproduction summary's other rows
    (POPT_GE_GRASP, …) don't have lit-faith counterparts.
    """
    lit = _read_csv(LIT_FAITH_CSV)
    repro = _read_csv(REPRO_CSV)
    cited_lit = {
        _policy_cell(r)
        for r in lit
        if (r.get("claim_citation") or "").strip()
        and r.get("policy") in CACHE_POLICY_PREDICATES
    }
    repro_keys = {
        _policy_cell(r)
        for r in repro
        if r.get("policy") in CACHE_POLICY_PREDICATES
    }
    missing = cited_lit - repro_keys
    assert not missing, (
        f"{len(missing)} cited lit-faith cache-policy row(s) are "
        f"missing from the reproduction summary. The summary is "
        f"supposed to be the per-citation rollup of every claim. "
        f"Examples: {sorted(missing)[:5]}"
    )


def test_lit_faith_and_repro_share_core_status_vocabulary() -> None:
    """Both emitters must agree on the *core* status vocabulary even
    though each adds emitter-specific sentinels.

    The lit-faith CSV emits ``no_claim`` for cache-policy rows that
    have no citation (an emitter detail). The reproduction summary
    emits ``known_deviation`` for rows whose disagreement is registered
    in ``literature_baselines.KNOWN_DEVIATIONS`` (also an emitter
    detail). Outside those two emitter-specific labels, the two CSVs
    must use the same status names — otherwise a future status
    introduced in one place will silently drift.
    """
    lit = _read_csv(LIT_FAITH_CSV)
    repro = _read_csv(REPRO_CSV)
    lit_statuses = {(r.get("claim_status") or "").strip() for r in lit}
    repro_statuses = {(r.get("status") or "").strip() for r in repro}
    lit_only_ok = {"no_claim", ""}
    repro_only_ok = {"known_deviation", ""}
    core_lit = lit_statuses - lit_only_ok
    core_repro = repro_statuses - repro_only_ok
    assert core_lit == core_repro, (
        f"core status vocabularies disagree:\n"
        f"  lit-faith core: {sorted(core_lit)}\n"
        f"  repro    core: {sorted(core_repro)}\n"
        f"If you added a status to one emitter, add it (or its "
        f"mapping) to the other, then update the allowlists in this "
        f"test."
    )


def test_no_cell_is_completely_blank() -> None:
    """Every (graph,app,l3) row in paper_baseline_table has at least
    one numeric policy miss_rate filled in.
    """
    table = _read_csv(TABLE_CSV)
    fully_blank: list[tuple[str, str, str]] = []
    for row in table:
        cells = [
            row.get(k) for k in (
                "lru_miss_rate",
                "srrip_miss_rate",
                "grasp_miss_rate",
                "popt_miss_rate",
            )
        ]
        if not any((c or "").strip() for c in cells):
            fully_blank.append(_cell(row))
    assert not fully_blank, (
        f"{len(fully_blank)} (graph,app,l3) row(s) in paper_baseline_table "
        f"have no policy miss_rate at all — emitter wrote shell rows. "
        f"Examples: {fully_blank[:5]}"
    )


# ----------------------------------------------------------------------
# Citation dead-code / coverage gate
# ----------------------------------------------------------------------

import re

LIT_BASELINES_PY = REPO_ROOT / "scripts" / "experiments" / "ecg" / "literature_baselines.py"
_CITATION_RE = re.compile(r'citation="([^"]+)"')


def _source_citations() -> set[str]:
    if not LIT_BASELINES_PY.exists():
        pytest.skip(f"{LIT_BASELINES_PY.relative_to(REPO_ROOT)} not on disk")
    return set(_CITATION_RE.findall(LIT_BASELINES_PY.read_text()))


def _repro_csv_citations() -> set[str]:
    repro = _read_csv(REPRO_CSV)
    return {(r.get("citation") or "").strip() for r in repro if (r.get("citation") or "").strip()}


def test_every_source_citation_appears_in_repro_summary() -> None:
    """Catches dead-code citations: any ``citation="…"`` in
    ``literature_baselines.py`` whose claim never produces a sweep row
    (typo, stale graph name, etc.) won't appear in the reproduction
    summary. The summary is then incomplete and a paper figure quietly
    goes unreproduced.
    """
    src = _source_citations()
    # The geomean POPT-vs-GRASP claim (POPT_GE_GRASP_GEOMEAN) is a
    # SUMMARY-LEVEL aggregate over all power-law cells — it is reproduced by
    # the geomean gate, not by any single per-cell sweep row, so its citation
    # legitimately does not appear in the per-cell reproduction summary.
    summary_level = {
        "Balaji & Lucia HPCA 2021 §6.3 (geomean LLC miss reduction vs GRASP)",
    }
    csv_cites = _repro_csv_citations()
    missing = (src - summary_level) - csv_cites
    assert not missing, (
        f"{len(missing)} citation(s) declared in literature_baselines.py "
        f"never appear in the reproduction summary — the corresponding "
        f"claim(s) never produced a sweep row (typo, stale graph name, "
        f"or app mismatch). Examples: {sorted(missing)[:5]}"
    )


def test_no_phantom_citations_in_repro_summary() -> None:
    """Conversely, every citation in the CSV must trace back to a
    ``citation="…"`` literal in the source. Otherwise the summary
    references a citation that no longer exists in the baselines
    table — i.e. someone deleted the claim but the cached CSV is stale.
    """
    src = _source_citations()
    csv_cites = _repro_csv_citations()
    phantoms = csv_cites - src
    assert not phantoms, (
        f"{len(phantoms)} citation(s) appear in the reproduction "
        f"summary but are not declared in literature_baselines.py — "
        f"summary is stale, run `make lit-repro` to refresh. "
        f"Examples: {sorted(phantoms)[:5]}"
    )
