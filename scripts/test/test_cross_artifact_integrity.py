"""Cross-artifact integrity gate.

Why this gate exists
--------------------
The paper-grade evidence chain is a graph of relationships:

* ``artifact_catalog.CATALOG`` lists every (generator, gate, artifact)
  triple — the canonical evidence chain.
* ``confidence_dashboard.PYTEST_SUITES`` lists every pytest module the
  dashboard exercises and its short ticker label.
* ``paper_claims_registry`` lists every numerical claim the paper makes,
  each linked to a ``source`` artifact and a ``gate``.

Each of these registries can drift independently — a generator added
without a gate, a gate added without a PYTEST_SUITES entry, a paper
claim pointing at a moved artifact, two suites sharing the same short
ticker, etc. A single integrity gate that asserts the graph is
consistent end-to-end means *any* such drift fails the dashboard before
it can hide a missing/dangling reference downstream.

What it verifies
----------------
Group 1 — ``CATALOG`` self-integrity: IDs unique; every generator,
          gate, and artifact path resolves on disk.
Group 2 — ``CATALOG`` cross-row uniqueness: generator paths unique;
          gate paths may repeat (some gates govern multiple entries);
          artifact paths unique.
Group 3 — ``PYTEST_SUITES`` self-integrity: long labels unique;
          short tickers unique; every test file path resolves on disk.
Group 4 — ``paper_claims_registry`` integrity: claim IDs unique;
          every ``source`` and ``gate`` path resolves on disk.
Group 5 — Cross-registry consistency: every paper claim's gate path
          appears either in CATALOG.gate values or in PYTEST_SUITES
          paths (no orphan gates referenced from claims that don't
          appear in either index).
Group 6 — Cross-registry uniqueness: CATALOG IDs and paper_claims IDs
          don't accidentally collide on the same id; PYTEST_SUITES
          short tickers don't collide with any CATALOG id.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"
sys.path.insert(0, str(ECG_DIR))

import artifact_catalog as _cat  # noqa: E402
import confidence_dashboard as _dash  # noqa: E402

PAPER_CLAIMS_JSON = REPO_ROOT / "wiki" / "data" / "paper_claims.json"


def _load_claims():
    if not PAPER_CLAIMS_JSON.exists():
        pytest.skip(f"{PAPER_CLAIMS_JSON} not generated yet")
    return json.loads(PAPER_CLAIMS_JSON.read_text())["claims"]


# ---------------------------------------------------------------------------
# Group 1 — CATALOG self-integrity
# ---------------------------------------------------------------------------


def test_catalog_ids_unique():
    """No two CATALOG entries share an id. id collisions silently
    overwrite each other in any id-keyed lookup downstream."""
    ids = [e["id"] for e in _cat.CATALOG]
    duplicates = [i for i in set(ids) if ids.count(i) > 1]
    assert not duplicates, f"duplicate catalog ids: {duplicates}"


def test_catalog_generators_all_exist():
    """Every generator path must resolve. A dangling generator means
    the artifact can't be regenerated when the source changes."""
    missing = [
        e["id"] for e in _cat.CATALOG
        if not (REPO_ROOT / e["generator"]).exists()
    ]
    assert not missing, f"catalog entries with missing generators: {missing}"


def test_catalog_gates_all_exist():
    """Every gate path must resolve. A dangling gate means the
    artifact is unguarded — drift will land unnoticed."""
    missing = [
        e["id"] for e in _cat.CATALOG
        if not (REPO_ROOT / e["gate"]).exists()
    ]
    assert not missing, f"catalog entries with missing gates: {missing}"


def test_catalog_artifacts_all_exist():
    """Every artifact path must resolve. A dangling artifact means
    the catalog references a paper-grade output that was never
    generated (or was renamed without a catalog update)."""
    missing = [
        e["id"] for e in _cat.CATALOG
        if not (REPO_ROOT / e["artifact"]).exists()
    ]
    assert not missing, f"catalog entries with missing artifacts: {missing}"


def test_catalog_has_required_fields():
    """Every entry must have id, label, generator, gate, artifact,
    summary. Defensive — silent .get() lookups in the _audit helper
    would otherwise emit empty cells in the rendered table."""
    required = {"id", "label", "generator", "gate", "artifact", "summary"}
    for e in _cat.CATALOG:
        missing = required - set(e.keys())
        assert not missing, f"entry {e.get('id')} missing fields: {missing}"


# ---------------------------------------------------------------------------
# Group 2 — CATALOG cross-row uniqueness
# ---------------------------------------------------------------------------


def test_catalog_generator_paths_unique():
    """No two CATALOG entries share the same generator script EXCEPT
    for documented multi-artifact generators. ``gem5_anchor_summary.py``
    is the canonical multi-artifact case: it generates both
    ``gem5_anchor.json`` and ``sniper_anchor.json`` from a single script
    by CLI flag (the script docstring documents this). Any new shared
    generator must be added to ALLOWED_SHARED_GENERATORS below with a
    comment explaining why."""
    ALLOWED_SHARED_GENERATORS = {
        # gem5_anchor_summary.py generates both gem5_anchor.json and
        # sniper_anchor.json — single generator, two artifacts, shared
        # invariant-evaluator. Documented in the script's docstring.
        "scripts/experiments/ecg/gem5_anchor_summary.py",
        # paper_table_prefetcher.py generates both
        # paper_table_prefetcher.json (literature corpus, 16 cells from
        # /tmp/graphbrew-ecg-pfx-cache_sim-scale) and
        # paper_table_prefetcher_kronecker.json (synthetic Kronecker
        # extrapolation, 3 cells from /tmp/graphbrew-ecg-pfx-cache_sim-
        # kronecker) — same emit code, different --sweep-root.
        "scripts/experiments/ecg/paper_table_prefetcher.py",
    }
    gens = [e["generator"] for e in _cat.CATALOG]
    duplicates = [
        g for g in set(gens)
        if gens.count(g) > 1 and g not in ALLOWED_SHARED_GENERATORS
    ]
    assert not duplicates, (
        f"undocumented catalog generators reused: {duplicates} "
        f"(add to ALLOWED_SHARED_GENERATORS with rationale if intentional)"
    )


def test_catalog_artifact_paths_unique():
    """No two CATALOG entries share the same artifact path. Sharing
    would mean the same file is claimed by two evidence chains —
    breaks the source-of-truth invariant for that artifact."""
    arts = [e["artifact"] for e in _cat.CATALOG]
    duplicates = [a for a in set(arts) if arts.count(a) > 1]
    assert not duplicates, f"catalog artifacts reused: {duplicates}"


def test_catalog_gate_paths_may_repeat():
    """Gate paths MAY repeat: a single pytest module can guard
    multiple artifacts. This test pins the absence of any uniqueness
    constraint on the gate column so future tightening is deliberate."""
    gates = [e["gate"] for e in _cat.CATALOG]
    # No assertion on uniqueness — repeats are explicitly allowed.
    # The test exists to document the contract.
    assert isinstance(gates, list)


# ---------------------------------------------------------------------------
# Group 3 — PYTEST_SUITES self-integrity
# ---------------------------------------------------------------------------


def test_pytest_suites_long_labels_unique():
    """The dict-key (long label) of PYTEST_SUITES must be unique by
    construction (Python dict semantics), but a copy-paste duplicate
    in the source overwrites silently. Length-check pins it."""
    # dict keys are by definition unique — but re-confirm via the
    # parsed dict to catch any future refactor that converts to a
    # list of tuples.
    assert len(_dash.PYTEST_SUITES) == len(set(_dash.PYTEST_SUITES.keys()))


def test_pytest_suites_short_tickers_unique():
    """Short tickers (e.g. 'LCS-Der') are the dashboard column
    headers; collisions hide one suite's results behind another's
    cell in the rendered table."""
    shorts = [short for _path, short in _dash.PYTEST_SUITES.values()]
    duplicates = [s for s in set(shorts) if shorts.count(s) > 1]
    assert not duplicates, f"PYTEST_SUITES short tickers collide: {duplicates}"


def test_pytest_suites_paths_all_exist():
    """Every test file referenced by the dashboard must exist on
    disk; otherwise the dashboard would silently report 0 passed /
    0 failed for the missing suite (worst-case: GREEN despite
    coverage being absent)."""
    missing = [
        (label, path)
        for label, (path, _short) in _dash.PYTEST_SUITES.items()
        if not (REPO_ROOT / path).exists()
    ]
    assert not missing, f"PYTEST_SUITES paths missing on disk: {missing}"


def test_pytest_suites_paths_unique():
    """No two PYTEST_SUITES entries share the same test file path.
    A shared path would mean the same suite is run twice with two
    different tickers — wastes time and double-counts pass/fail."""
    paths = [path for path, _short in _dash.PYTEST_SUITES.values()]
    duplicates = [p for p in set(paths) if paths.count(p) > 1]
    assert not duplicates, f"PYTEST_SUITES paths reused: {duplicates}"


# ---------------------------------------------------------------------------
# Group 4 — paper_claims_registry integrity
# ---------------------------------------------------------------------------


def test_paper_claims_ids_unique():
    claims = _load_claims()
    ids = [c["id"] for c in claims]
    duplicates = [i for i in set(ids) if ids.count(i) > 1]
    assert not duplicates, f"duplicate paper-claim ids: {duplicates}"


def test_paper_claims_source_paths_resolve():
    """Every claim's ``source`` (the artifact backing it) must
    resolve on disk. A claim pointing at a moved/renamed artifact
    cannot be re-verified by reviewers."""
    claims = _load_claims()
    missing = [
        c["id"] for c in claims
        if not (REPO_ROOT / c["source"]).exists()
    ]
    assert not missing, f"paper-claim source artifacts missing: {missing}"


def test_paper_claims_gate_paths_resolve():
    """Every claim's ``gate`` (the pytest module enforcing it) must
    resolve on disk. A claim citing a deleted gate has no automated
    enforcement."""
    claims = _load_claims()
    missing = [
        c["id"] for c in claims
        if not (REPO_ROOT / c["gate"]).exists()
    ]
    assert not missing, f"paper-claim gates missing: {missing}"


def test_paper_claims_have_required_fields():
    required = {"id", "category", "text", "value", "units", "source", "gate"}
    claims = _load_claims()
    for c in claims:
        missing = required - set(c.keys())
        assert not missing, f"paper-claim {c.get('id')} missing fields: {missing}"


# ---------------------------------------------------------------------------
# Group 5 — Cross-registry consistency
# ---------------------------------------------------------------------------


def test_every_paper_claim_gate_is_indexed():
    """Every paper-claim gate must appear either in CATALOG.gate
    values OR in PYTEST_SUITES paths, with one documented exception:
    the ``confidence.green_gate_count`` meta-claim has its gate set to
    the dashboard *generator* itself (``confidence_dashboard.py``),
    since it's the self-recomputed gate count — there is no separate
    pytest module that re-derives the count.

    Orphan gates (cited by a claim but not indexed by either registry
    and not in this allow-list) escape the dashboard summary and the
    catalog enumeration, so they fail the gate."""
    ALLOWED_SELF_GATED_CLAIMS = {
        # confidence.green_gate_count's "gate" is the dashboard generator
        # itself; it's a meta-claim about the dashboard's own coverage.
        "confidence.green_gate_count",
    }
    claims = _load_claims()
    catalog_gates = {e["gate"] for e in _cat.CATALOG}
    dashboard_paths = {path for path, _short in _dash.PYTEST_SUITES.values()}
    indexed = catalog_gates | dashboard_paths
    orphans = [
        c["id"] for c in claims
        if c["gate"] not in indexed
        and c["id"] not in ALLOWED_SELF_GATED_CLAIMS
    ]
    assert not orphans, (
        f"paper claims with orphan gates: {orphans} "
        f"(add to ALLOWED_SELF_GATED_CLAIMS with rationale if intentional)"
    )


def test_every_paper_claim_source_is_catalogued():
    """Every paper-claim source artifact must appear in CATALOG.artifact
    values. A claim citing a wiki/data file that isn't catalogued
    means the artifact isn't tracked in the canonical evidence
    chain — invisible to reviewers consulting the catalog."""
    claims = _load_claims()
    catalog_arts = {e["artifact"] for e in _cat.CATALOG}
    uncatalogued = [
        c["id"] for c in claims if c["source"] not in catalog_arts
    ]
    assert not uncatalogued, (
        f"paper claims with un-catalogued sources: {uncatalogued}"
    )


# ---------------------------------------------------------------------------
# Group 6 — Cross-registry uniqueness
# ---------------------------------------------------------------------------


def test_catalog_ids_disjoint_from_pytest_suite_tickers():
    """CATALOG ids (e.g. 'corpus_diversity') and PYTEST_SUITES short
    tickers (e.g. 'CDV-Der') live in distinct id spaces but a stray
    overlap would mean a ticker and an artifact id resolve to the
    same logical entity — confusing for grep/code-search."""
    catalog_ids = {e["id"] for e in _cat.CATALOG}
    short_tickers = {short for _p, short in _dash.PYTEST_SUITES.values()}
    overlap = catalog_ids & short_tickers
    assert not overlap, (
        f"CATALOG ids accidentally collide with PYTEST_SUITES tickers: {overlap}"
    )


def test_paper_claim_ids_disjoint_from_catalog_ids():
    """Paper-claim IDs (e.g. 'corpus.graph_count') are dotted; CATALOG
    IDs (e.g. 'corpus_diversity') are flat snake-case. A collision
    here would mean a claim and an artifact share an ID, breaking
    any id-keyed cross-walk."""
    claims = _load_claims()
    claim_ids = {c["id"] for c in claims}
    catalog_ids = {e["id"] for e in _cat.CATALOG}
    overlap = claim_ids & catalog_ids
    assert not overlap, (
        f"paper-claim ids collide with CATALOG ids: {overlap}"
    )


def test_catalog_summary_count_matches_pytest_suite_count():
    """``confidence_dashboard`` CATALOG entry's summary line cites the
    current gate count (e.g. '208 gates today'). This must match
    the actual len(PYTEST_SUITES) — otherwise the catalog summary
    silently lies about the dashboard's coverage."""
    dash_entry = next(
        (e for e in _cat.CATALOG if e["id"] == "confidence_dashboard"),
        None,
    )
    assert dash_entry is not None, "confidence_dashboard catalog entry missing"
    summary = dash_entry["summary"]
    actual_count = len(_dash.PYTEST_SUITES)
    # Extract the number before " gates today"
    import re
    m = re.search(r"\((\d+) gates today", summary)
    assert m is not None, (
        f"confidence_dashboard summary missing 'N gates today' marker: {summary!r}"
    )
    cited = int(m.group(1))
    assert cited == actual_count, (
        f"catalog summary cites {cited} gates but PYTEST_SUITES has {actual_count}"
    )
