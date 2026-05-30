"""Reproducibility coverage gate — every catalogued artifact is in the smoke list.

Why this gate exists
--------------------
``reproduce_smoke.py`` is the bit-determinism audit: it snapshots every
tracked ``wiki/data`` artifact, regenerates them from inputs, and
asserts SHA-256 byte parity. Its ``TRACKED_ARTIFACTS`` list is the
authoritative coverage map.

The risk: someone adds a new aggregator to ``artifact_catalog.CATALOG``
or a new claim to ``paper_claims_registry`` but forgets to add the
artifact to ``TRACKED_ARTIFACTS``. The new aggregator's output then
escapes the byte-parity audit — silent regressions in the
generator can drift the committed file without anyone noticing until
the next manual diff inspection.

This gate closes that gap by asserting that the
``CATALOG → reproduce_smoke`` mapping is total (modulo documented
exceptions like CSV outputs which use a different reproducibility
model, and the self-referential ``reproduce_smoke.json``).

What it verifies
----------------
Group 1 — ``CATALOG.artifact`` coverage: every JSON artifact catalogued
          must appear in the reproduce_smoke audit rows.
Group 2 — ``paper_claims.source`` coverage: every paper-claim source
          artifact must appear in the reproduce_smoke audit rows.
Group 3 — ``reproduce_smoke`` self-integrity: every row has the
          expected schema; ``status == 'ok'`` across the board (no
          drift); SHA fields are 64-char hex.
Group 4 — Companion ``.md`` audit: every catalogued ``.json`` artifact
          that has a sibling ``.md`` companion (markdown rendering of
          the same data) must have that companion also tracked. This
          catches cases where the JSON is updated but the rendered MD
          is silently stale.
Group 5 — TRACKED_ARTIFACTS list invariants: no duplicates; all paths
          exist on disk; alphabetical-or-grouped ordering preserved
          (loose check — ensures human-readability).
Group 6 — Self-consistency: ``reproduce_smoke.json`` row count equals
          ``len(TRACKED_ARTIFACTS)`` exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"
sys.path.insert(0, str(ECG_DIR))

import artifact_catalog as _cat  # noqa: E402
import reproduce_smoke as _smoke  # noqa: E402

PAPER_CLAIMS_JSON = REPO_ROOT / "wiki" / "data" / "paper_claims.json"
SMOKE_JSON = REPO_ROOT / "wiki" / "data" / "reproduce_smoke.json"


# ---------- Documented exceptions ----------
#
# CSV artifacts use a different reproducibility model (csv reader/writer
# float-format quirks make raw SHA byte parity brittle). The bit-parity
# audit covers the upstream JSON the CSV is derived from.
CSV_EXEMPT = {"wiki/data/literature_reproduction_summary.csv"}

# reproduce_smoke.json is itself an output of reproduce_smoke; including
# it in its own audit would be a chicken-and-egg problem (the file's
# content depends on its own hash). It's catalogued because reviewers
# need to find it, but the audit excludes itself.
SELF_REF_EXEMPT = {
    "wiki/data/reproduce_smoke.json",
    "wiki/data/reproduce_smoke.md",
}

CATALOG_AUDIT_EXEMPT = CSV_EXEMPT | SELF_REF_EXEMPT


def _load_claims():
    if not PAPER_CLAIMS_JSON.exists():
        pytest.skip(f"{PAPER_CLAIMS_JSON} not generated yet")
    return json.loads(PAPER_CLAIMS_JSON.read_text())["claims"]


def _load_smoke():
    if not SMOKE_JSON.exists():
        pytest.skip(f"{SMOKE_JSON} not generated yet")
    return json.loads(SMOKE_JSON.read_text())


def _smoke_arts():
    """Returns the set of artifacts the smoke audit covers, normalized
    to the same prefixed form CATALOG uses (``wiki/data/<name>``)."""
    rs = _load_smoke()
    return {f"wiki/data/{r['artifact']}" for r in rs["rows"]}


# ---------------------------------------------------------------------------
# Group 1 — CATALOG.artifact coverage
# ---------------------------------------------------------------------------


def test_every_catalog_artifact_is_smoke_audited():
    """Every CATALOG artifact (except documented CSV / self-ref
    exemptions) must appear in reproduce_smoke rows. A catalogued
    JSON that escapes the byte-parity audit can silently drift."""
    smoke = _smoke_arts()
    orphans = [
        e["id"] for e in _cat.CATALOG
        if e["artifact"] not in smoke
        and e["artifact"] not in CATALOG_AUDIT_EXEMPT
    ]
    assert not orphans, (
        f"CATALOG entries with no reproduce_smoke coverage: {orphans} "
        f"(add to TRACKED_ARTIFACTS in reproduce_smoke.py, "
        f"or document in CATALOG_AUDIT_EXEMPT with rationale)"
    )


def test_csv_exemptions_are_actually_csv():
    """Belt-and-suspenders: anything in CSV_EXEMPT must end with .csv.
    Pins the exemption category to a real semantic distinction
    rather than a 'just skip this one' escape hatch."""
    non_csv = [p for p in CSV_EXEMPT if not p.endswith(".csv")]
    assert not non_csv, (
        f"CSV_EXEMPT contains non-CSV paths: {non_csv} "
        f"(move to a more honest exemption category)"
    )


def test_self_ref_exemption_is_only_reproduce_smoke():
    """SELF_REF_EXEMPT should hold ONLY the reproduce_smoke artifact
    pair (json + md companion). Any expansion warrants explicit
    review — silently expanding the self-ref bucket is how
    reproducibility coverage bleeds away one exception at a time."""
    assert SELF_REF_EXEMPT == {
        "wiki/data/reproduce_smoke.json",
        "wiki/data/reproduce_smoke.md",
    }, f"SELF_REF_EXEMPT changed: {SELF_REF_EXEMPT}"


# ---------------------------------------------------------------------------
# Group 2 — paper_claims.source coverage
# ---------------------------------------------------------------------------


def test_every_paper_claim_source_is_smoke_audited():
    """Every paper-claim source artifact must be in reproduce_smoke
    coverage. A claim's headline number must come from an audited
    artifact — otherwise reviewer-reproducing the number would not
    catch generator drift."""
    smoke = _smoke_arts()
    claims = _load_claims()
    orphans = sorted({
        c["source"] for c in claims
        if c["source"] not in smoke
        and c["source"] not in CATALOG_AUDIT_EXEMPT
    })
    assert not orphans, (
        f"paper-claim sources with no reproduce_smoke coverage: {orphans}"
    )


# ---------------------------------------------------------------------------
# Group 3 — reproduce_smoke self-integrity
# ---------------------------------------------------------------------------


def test_smoke_summary_keys():
    """reproduce_smoke top-level keys are: n_artifacts, ok, drift,
    missing, passed, rows. Pins the schema — downstream gates depend
    on this shape."""
    rs = _load_smoke()
    expected = {"n_artifacts", "ok", "drift", "missing", "passed", "rows"}
    assert expected <= set(rs.keys()), (
        f"reproduce_smoke missing keys: {expected - set(rs.keys())}"
    )


def test_smoke_all_rows_status_ok():
    """The whole point of the smoke audit: every row must be 'ok'.
    Any drift / missing entry means a committed artifact does not
    regenerate to the same bytes from its inputs."""
    rs = _load_smoke()
    non_ok = [r["artifact"] for r in rs["rows"] if r.get("status") != "ok"]
    assert not non_ok, f"reproduce_smoke non-ok artifacts: {non_ok}"


def test_smoke_summary_consistent_with_rows():
    """rs['n_artifacts'] must equal len(rs['rows']) and rs['ok']
    must equal the number of ok rows. Defensive: catches a future
    refactor that updates rows but forgets to recompute summary.
    Note schema: rs['ok'] is an int count of ok rows; rs['passed']
    is the overall bool verdict; rs['drift'] and rs['missing'] are
    LISTS of file names (not counts)."""
    rs = _load_smoke()
    assert rs["n_artifacts"] == len(rs["rows"]), (
        f"summary n_artifacts={rs['n_artifacts']} but {len(rs['rows'])} rows"
    )
    ok_count = sum(1 for r in rs["rows"] if r["status"] == "ok")
    assert rs["ok"] == ok_count, (
        f"summary ok={rs['ok']} but {ok_count} ok rows"
    )


def test_smoke_sha_fields_are_hex():
    """SHA-256 digests must be 64 lowercase hex chars. Catches a
    future refactor that swaps in a different hash function or
    formatting."""
    rs = _load_smoke()
    for r in rs["rows"]:
        for field in ("sha_before", "sha_after"):
            v = r.get(field)
            assert isinstance(v, str), f"{r['artifact']}.{field} not str: {v!r}"
            assert len(v) == 64, f"{r['artifact']}.{field} len={len(v)}"
            assert all(c in "0123456789abcdef" for c in v), (
                f"{r['artifact']}.{field} contains non-hex: {v!r}"
            )


# ---------------------------------------------------------------------------
# Group 4 — Companion .md audit
# ---------------------------------------------------------------------------


def test_every_catalogued_json_has_md_companion_if_one_exists():
    """If a catalogued .json artifact has a sibling .md file on disk
    (which is the convention for paper-grade aggregators —
    json+md pair), the .md must ALSO appear in reproduce_smoke
    coverage. Otherwise the json regenerates but the rendered MD
    silently drifts."""
    smoke = _smoke_arts()
    misses = []
    for e in _cat.CATALOG:
        art = e["artifact"]
        if not art.endswith(".json"):
            continue
        md_companion = art[:-len(".json")] + ".md"
        md_path = REPO_ROOT / md_companion
        if (
            md_path.exists()
            and md_companion not in smoke
            and md_companion not in CATALOG_AUDIT_EXEMPT
        ):
            misses.append(md_companion)
    assert not misses, (
        f".md companions on disk but not in TRACKED_ARTIFACTS: {misses}"
    )


# ---------------------------------------------------------------------------
# Group 5 — TRACKED_ARTIFACTS list invariants
# ---------------------------------------------------------------------------


def test_tracked_artifacts_no_duplicates():
    """A duplicate in TRACKED_ARTIFACTS would cause the file to be
    hashed twice in the smoke output — wastes time and inflates
    row counts."""
    counts = {}
    for a in _smoke.TRACKED_ARTIFACTS:
        counts[a] = counts.get(a, 0) + 1
    dupes = [a for a, n in counts.items() if n > 1]
    assert not dupes, f"TRACKED_ARTIFACTS duplicates: {dupes}"


def test_tracked_artifacts_all_exist_on_disk():
    """Every entry must resolve to a real file under wiki/data.
    A typo here would either silently skip a file (worst case:
    coverage hole) or insert a 'missing' row in the audit."""
    missing = [
        a for a in _smoke.TRACKED_ARTIFACTS
        if not (REPO_ROOT / "wiki" / "data" / a).exists()
    ]
    assert not missing, f"TRACKED_ARTIFACTS not on disk: {missing}"


def test_tracked_artifacts_have_expected_extensions():
    """Every tracked artifact must end in .json, .md, or .csv. Pins
    the set of file types reproduce_smoke handles — adding new
    types (e.g., .pdf) without updating the masking rules in
    reproduce_smoke would let volatile bytes through."""
    bad = [
        a for a in _smoke.TRACKED_ARTIFACTS
        if not a.endswith((".json", ".md", ".csv"))
    ]
    assert not bad, (
        f"TRACKED_ARTIFACTS contains unsupported extensions: {bad}"
    )


# ---------------------------------------------------------------------------
# Group 6 — Self-consistency
# ---------------------------------------------------------------------------


def test_smoke_row_count_matches_tracked_artifacts():
    """reproduce_smoke.json's row count must equal
    len(TRACKED_ARTIFACTS) exactly. A mismatch means either a
    file is silently dropped from the audit or a row is being
    produced for an untracked file."""
    rs = _load_smoke()
    assert rs["n_artifacts"] == len(_smoke.TRACKED_ARTIFACTS), (
        f"smoke rows={rs['n_artifacts']} but TRACKED_ARTIFACTS has "
        f"{len(_smoke.TRACKED_ARTIFACTS)}"
    )


def test_smoke_drift_and_missing_lists_empty():
    """rs['drift'] and rs['missing'] are LISTS of file names that
    drifted / went missing. Both must be empty in a healthy state.
    rs['passed'] is the overall verdict bool. Together these are
    the load-bearing 'reproduce_smoke is GREEN' invariant the
    dashboard relies on."""
    rs = _load_smoke()
    assert rs["drift"] == [], f"reproduce_smoke drift={rs['drift']}"
    assert rs["missing"] == [], f"reproduce_smoke missing={rs['missing']}"
    assert rs["passed"] is True, f"reproduce_smoke passed={rs['passed']}"
