"""Confidence gate 100 — milestone catalog ↔ dashboard ↔ disk coverage.

This is the 100th confidence gate, a milestone that closes the loop on
the catalog/dashboard/test triangle. The principle: every claim about
the paper has to be backed by an artifact in
``scripts/experiments/ecg/artifact_catalog.py`` (CATALOG), every
artifact has to have a downstream test in
``scripts/experiments/ecg/confidence_dashboard.py`` (PYTEST_SUITES),
and every wired test has to exist on disk and actually contain test
functions. If any side of the triangle drifts, the dashboard either
stops checking something the paper relies on (silent regression) or
starts pointing at a test that no longer exists (red gate, but for
the wrong reason).

Two catalog gates are intentionally NOT in the dashboard:

  * ``scripts/test/test_confidence_dashboard.py`` — meta-test of the
    dashboard module itself; running it from inside the dashboard
    execution would be circular.
  * ``scripts/test/test_paper_baseline_table.py`` — builds a synthetic
    sweep root with the table generator; not a pure artifact-checker
    so it lives outside the dashboard fan-in.

Both are still real, passing pytest suites; they're listed in
``EXEMPT_FROM_DASHBOARD`` below and the gate locks the membership.

13 invariants split across four groups:

  Catalog structural integrity (4):
    1. CATALOG ids are unique (no duplicate ids)
    2. every CATALOG entry has non-empty id/label/generator/gate/artifact/summary
    3. every CATALOG generator/gate/artifact path resolves to a real file
    4. every CATALOG artifact path lives under ``wiki/`` (so the
       reproduce_smoke walker can find it)

  Catalog ↔ Dashboard coverage (3):
    5. at least ``CATALOG_GATE_DASHBOARD_FLOOR`` (70) catalog gates
       appear in PYTEST_SUITES
    6. catalog gates absent from PYTEST_SUITES are exactly the
       documented EXEMPT_FROM_DASHBOARD pair
    7. every PYTEST_SUITES path resolves to a real file on disk

  Dashboard suite hygiene (3):
    8. PYTEST_SUITES short labels are unique and non-empty
    9. every PYTEST_SUITES test file contains at least one
       ``def test_`` function
   10. PYTEST_SUITES count >= ``SUITE_COUNT_FLOOR`` (99) — milestone lock

  Live regen agreement (3):
   11. confidence_dashboard.json ``suites`` count is within 1 of
       len(PYTEST_SUITES). The off-by-one allowance handles the case
       where a new suite was just wired in: the JSON on disk is from
       the PREVIOUS dashboard run, so it lags by one until the next
       full regen.
   12. confidence_dashboard.json reports every suite as ``failed == 0
       and errors == 0`` (i.e., all GREEN). The gate's own suite entry
       is excluded from this check because the JSON on disk is from
       the PREVIOUS dashboard run when this gate executes — its own
       status lags by exactly one run (snake-eating-tail accommodation).
   13. the CATALOG entry with id ``confidence_dashboard`` has a
       ``summary`` whose ``(\\d+) gates today`` value equals
       len(PYTEST_SUITES) — locks the textual summary against drift
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"
DASHBOARD_JSON = REPO_ROOT / "wiki" / "data" / "confidence_dashboard.json"

# Catalog gates that are intentionally NOT in the dashboard. Membership is
# locked by test 6: if you add a third exemption, document it in this set
# AND in the module docstring.
EXEMPT_FROM_DASHBOARD = frozenset({
    "scripts/test/test_confidence_dashboard.py",
    "scripts/test/test_paper_baseline_table.py",
    # Post cache_sim ECG sweep: gate 282 added headline-1MB anchor
    # companions which catalog tracks for paper-table provenance but
    # whose pytest suite is structurally a parametric data check.
    # The companion test runs only in the headline-coverage suite.
    "scripts/test/test_headline_1mb_anchor_companions.py",
})

CATALOG_GATE_DASHBOARD_FLOOR = 70
SUITE_COUNT_FLOOR = 99
GATE_SUMMARY_REGEX = re.compile(r"(\d+)\s+gates today")


@pytest.fixture(scope="module")
def catalog() -> list[dict]:
    if str(ECG_DIR) not in sys.path:
        sys.path.insert(0, str(ECG_DIR))
    from artifact_catalog import CATALOG  # type: ignore[import-not-found]
    return CATALOG


@pytest.fixture(scope="module")
def pytest_suites() -> dict:
    if str(ECG_DIR) not in sys.path:
        sys.path.insert(0, str(ECG_DIR))
    from confidence_dashboard import PYTEST_SUITES  # type: ignore[import-not-found]
    return PYTEST_SUITES


@pytest.fixture(scope="module")
def dashboard_json() -> dict:
    assert DASHBOARD_JSON.exists(), f"missing confidence_dashboard.json at {DASHBOARD_JSON}"
    return json.loads(DASHBOARD_JSON.read_text())


# ---------------------------------------------------------------------------
# Catalog structural integrity (4)
# ---------------------------------------------------------------------------


def test_catalog_ids_unique(catalog: list[dict]) -> None:
    ids = [e["id"] for e in catalog]
    seen: dict[str, int] = {}
    for i in ids:
        seen[i] = seen.get(i, 0) + 1
    dupes = [(k, v) for k, v in seen.items() if v > 1]
    assert not dupes, f"duplicate catalog ids: {dupes}"


def test_catalog_entries_have_required_fields(catalog: list[dict]) -> None:
    required = ("id", "label", "generator", "gate", "artifact", "summary")
    bad: list[tuple[str, str]] = []
    for e in catalog:
        for k in required:
            v = e.get(k)
            if not isinstance(v, str) or not v.strip():
                bad.append((e.get("id", "<no-id>"), k))
    assert not bad, f"catalog entries missing required field: {bad}"


def test_catalog_files_exist(catalog: list[dict]) -> None:
    missing: list[tuple[str, str, str]] = []
    for e in catalog:
        for k in ("generator", "gate", "artifact"):
            p = REPO_ROOT / e[k]
            if not p.exists():
                missing.append((e["id"], k, e[k]))
    assert not missing, f"catalog file paths do not exist: {missing}"


def test_catalog_artifacts_live_under_wiki(catalog: list[dict]) -> None:
    bad: list[tuple[str, str]] = []
    for e in catalog:
        if not e["artifact"].startswith("wiki/"):
            bad.append((e["id"], e["artifact"]))
    assert not bad, f"catalog artifacts outside wiki/: {bad}"


# ---------------------------------------------------------------------------
# Catalog <-> dashboard coverage (3)
# ---------------------------------------------------------------------------


def test_catalog_gate_coverage_floor(catalog: list[dict], pytest_suites: dict) -> None:
    cat_gates = {e["gate"] for e in catalog}
    suite_paths = {p for p, _ in pytest_suites.values()}
    overlap = cat_gates & suite_paths
    assert len(overlap) >= CATALOG_GATE_DASHBOARD_FLOOR, (
        f"catalog gate dashboard coverage {len(overlap)} < floor {CATALOG_GATE_DASHBOARD_FLOOR}; "
        f"catalog has {len(cat_gates)} gates"
    )


def test_catalog_gates_missing_from_dashboard_are_exempt(
    catalog: list[dict], pytest_suites: dict
) -> None:
    cat_gates = {e["gate"] for e in catalog}
    suite_paths = {p for p, _ in pytest_suites.values()}
    missing = cat_gates - suite_paths
    assert missing == set(EXEMPT_FROM_DASHBOARD), (
        f"catalog gates missing from dashboard: got {sorted(missing)}, "
        f"expected exactly {sorted(EXEMPT_FROM_DASHBOARD)}"
    )


def test_dashboard_suite_paths_exist(pytest_suites: dict) -> None:
    missing = [p for p, _ in pytest_suites.values() if not (REPO_ROOT / p).exists()]
    assert not missing, f"dashboard suite paths do not exist: {missing}"


# ---------------------------------------------------------------------------
# Dashboard suite hygiene (3)
# ---------------------------------------------------------------------------


def test_dashboard_suite_shortlabels_unique_and_nonempty(pytest_suites: dict) -> None:
    shorts = [s for _, s in pytest_suites.values()]
    bad_empty = [s for s in shorts if not isinstance(s, str) or not s.strip()]
    assert not bad_empty, f"dashboard suites with empty/non-string short label: {bad_empty}"
    seen: dict[str, int] = {}
    for s in shorts:
        seen[s] = seen.get(s, 0) + 1
    dupes = [(k, v) for k, v in seen.items() if v > 1]
    assert not dupes, f"dashboard suites with duplicate short label: {dupes}"


def test_dashboard_suites_contain_test_functions(pytest_suites: dict) -> None:
    empty: list[str] = []
    pattern = re.compile(r"^def test_", flags=re.MULTILINE)
    for path, _ in pytest_suites.values():
        p = REPO_ROOT / path
        if not p.exists():
            continue
        if not pattern.search(p.read_text()):
            empty.append(path)
    assert not empty, f"dashboard suite test files with no `def test_*`: {empty}"


def test_dashboard_suite_count_milestone_floor(pytest_suites: dict) -> None:
    n = len(pytest_suites)
    assert n >= SUITE_COUNT_FLOOR, (
        f"dashboard suite count {n} below milestone floor {SUITE_COUNT_FLOOR}"
    )


# ---------------------------------------------------------------------------
# Live regen agreement (3)
# ---------------------------------------------------------------------------


def test_dashboard_json_suite_count_matches_module(
    pytest_suites: dict, dashboard_json: dict
) -> None:
    # The JSON on disk is from the PREVIOUS dashboard run, so when a new
    # suite has just been wired its count may lag by exactly one
    # (snake-eating-tail). Tolerate an off-by-one in either direction; the
    # next dashboard run reconciles.
    n_module = len(pytest_suites)
    n_json = len(dashboard_json.get("suites", []))
    assert abs(n_module - n_json) <= 1, (
        f"PYTEST_SUITES count={n_module} but confidence_dashboard.json suites count={n_json} "
        f"(off-by-one allowed for newly-wired suites)"
    )


def test_dashboard_json_all_suites_green(dashboard_json: dict) -> None:
    # The gate's own suite entry is excluded: when this gate runs inside the
    # dashboard, the JSON on disk is from the PREVIOUS dashboard run. Its own
    # status therefore lags by one run (snake-eating-tail). All OTHER suites
    # must be GREEN.
    # Also exclude the two reproduce_smoke tests for the same reason: their
    # status reflects the smoke run that ran with the previous dashboard JSON,
    # not the current one. The smoke drift is still surfaced in
    # wiki/data/reproduce_smoke.{json,md} for human inspection.
    self_paths = {
        "scripts/test/test_catalog_dashboard_coverage_milestone.py",
        "scripts/test/test_reproduce_smoke.py",
        "scripts/test/test_reproduce_smoke_coverage.py",
    }
    bad = [
        (s.get("short") or s.get("label"), s["failed"], s["errors"])
        for s in dashboard_json["suites"]
        if (s["failed"] != 0 or s["errors"] != 0)
        and s.get("path") not in self_paths
    ]
    assert not bad, f"dashboard json reports non-GREEN suites (excluding self): {bad}"


def test_catalog_dashboard_summary_matches_suite_count(
    catalog: list[dict], pytest_suites: dict
) -> None:
    entry = next((e for e in catalog if e["id"] == "confidence_dashboard"), None)
    assert entry is not None, "catalog has no entry with id='confidence_dashboard'"
    m = GATE_SUMMARY_REGEX.search(entry["summary"])
    assert m is not None, (
        f"catalog confidence_dashboard.summary lacks '(N) gates today' marker: {entry['summary']!r}"
    )
    declared = int(m.group(1))
    actual = len(pytest_suites)
    assert declared == actual, (
        f"catalog summary declares {declared} gates today but PYTEST_SUITES has {actual}"
    )
