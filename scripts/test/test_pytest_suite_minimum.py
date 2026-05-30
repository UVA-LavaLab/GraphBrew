"""Per-suite test-count floor (PST-Min) — gate 215.

Every PYTEST_SUITES entry in confidence_dashboard.py must:
  (a) point to an existing, parseable Python file,
  (b) define at least one module-level ``def test_*`` function,
  (c) produce at least one PASSED test in the dashboard,
  (d) not exceed its AST-counted test floor (parametric tests can expand
      the runtime count above the AST count, but never collapse below it),
  (e) report zero collection errors.

This catches three real failure modes that the existing 214-gate stack
does NOT catch:
  1. **Empty test file** — somebody creates ``test_foo.py`` with only a
     module docstring and forgets the actual tests; pytest reports 0 tests,
     the suite "passes vacuously" but covers nothing.
  2. **Collection failure** — import error or syntax error; pytest reports
     errors>0 and 0 passed; the gate would silently provide no coverage.
  3. **All-xfail / all-skip suite** — every test is marked xfail or skip;
     the suite turns green via avoidance, not via verification.

The PASSED-count floor (rule c) is the strongest check: a suite must have
*verified* at least one positive assertion. Skips and xfails do not count.

Self-consistency cross-checks:
  - len(dashboard.suites) == len(PYTEST_SUITES) (no orphaned dashboard rows)
  - every dashboard.suite.short is in PYTEST_SUITES.values()[1] (short codes)
  - every PYTEST_SUITES short is in dashboard.suites (no missing rows)
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"
DASHBOARD_PATH = REPO_ROOT / "wiki" / "data" / "confidence_dashboard.json"

sys.path.insert(0, str(ECG_DIR))
from confidence_dashboard import PYTEST_SUITES  # noqa: E402


def _ast_test_count(path: Path) -> int:
    """Count module-level ``def test_*`` functions via AST."""
    src = path.read_text()
    tree = ast.parse(src)
    return sum(
        1
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )


def _load_dashboard() -> dict:
    return json.loads(DASHBOARD_PATH.read_text())


# ---------------------------------------------------------------------------
# Group 1: PYTEST_SUITES shape and file existence
# ---------------------------------------------------------------------------


def test_pytest_suites_is_non_empty_dict():
    assert isinstance(PYTEST_SUITES, dict)
    assert len(PYTEST_SUITES) >= 200, (
        f"expected >=200 suites in PYTEST_SUITES, got {len(PYTEST_SUITES)}"
    )


@pytest.mark.parametrize(
    "label,path,short",
    [(label, path, short) for label, (path, short) in PYTEST_SUITES.items()],
    ids=[short for label, (path, short) in PYTEST_SUITES.items()],
)
def test_every_suite_path_exists(label: str, path: str, short: str):
    full = REPO_ROOT / path
    assert full.is_file(), f"suite {short!r} ({label}) path missing: {path}"


@pytest.mark.parametrize(
    "label,path,short",
    [(label, path, short) for label, (path, short) in PYTEST_SUITES.items()],
    ids=[short for label, (path, short) in PYTEST_SUITES.items()],
)
def test_every_suite_path_is_parseable_python(label: str, path: str, short: str):
    full = REPO_ROOT / path
    try:
        ast.parse(full.read_text())
    except SyntaxError as e:
        pytest.fail(f"suite {short!r} has SyntaxError: {e}")


# ---------------------------------------------------------------------------
# Group 2: AST test-function floor (catches empty test files)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,path,short",
    [(label, path, short) for label, (path, short) in PYTEST_SUITES.items()],
    ids=[short for label, (path, short) in PYTEST_SUITES.items()],
)
def test_every_suite_has_at_least_one_ast_test_function(
    label: str, path: str, short: str
):
    full = REPO_ROOT / path
    n = _ast_test_count(full)
    assert n >= 1, (
        f"suite {short!r} ({label}) has 0 module-level def test_* functions "
        f"(empty test file): {path}"
    )


def test_minimum_ast_test_count_across_all_suites_is_positive():
    counts = {short: _ast_test_count(REPO_ROOT / path) for _, (path, short) in PYTEST_SUITES.items()}
    floor = min(counts.values())
    assert floor >= 1, f"some suite has 0 tests: {[s for s, c in counts.items() if c < 1]}"


# ---------------------------------------------------------------------------
# Group 3: Dashboard PASSED-count floor (catches all-skip/all-xfail suites)
# ---------------------------------------------------------------------------


def test_dashboard_artifact_exists():
    assert DASHBOARD_PATH.is_file(), f"dashboard json missing: {DASHBOARD_PATH}"


def test_every_suite_has_at_least_one_passed_test():
    d = _load_dashboard()
    by_short = {s["short"]: s for s in d["suites"]}
    failures = []
    for _, (path, short) in PYTEST_SUITES.items():
        s = by_short.get(short)
        if s is None:
            failures.append(f"{short}: not in dashboard")
            continue
        if s.get("passed", 0) < 1:
            failures.append(
                f"{short}: passed={s.get('passed', 0)} skipped={s.get('skipped', 0)} "
                f"xfailed={s.get('xfailed', 0)} failed={s.get('failed', 0)} "
                f"errors={s.get('errors', 0)} (no positive verification)"
            )
    assert not failures, "\n".join(failures)


def test_no_suite_has_collection_errors():
    d = _load_dashboard()
    failures = [
        f"{s['short']}: errors={s['errors']}"
        for s in d["suites"]
        if s.get("errors", 0) > 0
    ]
    assert not failures, "suites with collection errors:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Group 4: AST count <= runtime PASSED count (parametric expansion sanity)
# ---------------------------------------------------------------------------


def test_dashboard_passed_count_dominates_ast_count():
    """Runtime passed should never be LESS than AST test_* count, because
    parametric expansions, fixture-driven variants, and class methods only
    add tests; they never remove plain ``def test_*`` from the count.

    Exception: a test can be skipped or xfail-marked, reducing passed.
    So the predicate is: passed + skipped + xfailed + failed + errors >=
    AST count (every static test is accounted for at runtime).
    """
    d = _load_dashboard()
    by_short = {s["short"]: s for s in d["suites"]}
    failures = []
    for _, (path, short) in PYTEST_SUITES.items():
        ast_n = _ast_test_count(REPO_ROOT / path)
        s = by_short.get(short)
        if s is None:
            continue
        accounted = (
            s.get("passed", 0)
            + s.get("skipped", 0)
            + s.get("xfailed", 0)
            + s.get("xpassed", 0)
            + s.get("failed", 0)
            + s.get("errors", 0)
        )
        if accounted < ast_n:
            failures.append(
                f"{short}: AST={ast_n} > runtime-accounted={accounted} "
                f"(some static tests not collected)"
            )
    assert not failures, "AST/runtime accounting drift:\n" + "\n".join(failures)


# ---------------------------------------------------------------------------
# Group 5: Self-consistency dashboard <-> PYTEST_SUITES
# ---------------------------------------------------------------------------


def test_dashboard_suite_count_matches_pytest_suites():
    d = _load_dashboard()
    assert len(d["suites"]) == len(PYTEST_SUITES), (
        f"dashboard has {len(d['suites'])} suites but "
        f"PYTEST_SUITES has {len(PYTEST_SUITES)}"
    )


def test_every_dashboard_short_is_in_pytest_suites():
    d = _load_dashboard()
    suite_shorts = {short for _, (_, short) in PYTEST_SUITES.items()}
    dash_shorts = {s["short"] for s in d["suites"]}
    orphan = dash_shorts - suite_shorts
    assert not orphan, f"dashboard rows not in PYTEST_SUITES: {sorted(orphan)}"


def test_every_pytest_suite_short_is_in_dashboard():
    d = _load_dashboard()
    suite_shorts = {short for _, (_, short) in PYTEST_SUITES.items()}
    dash_shorts = {s["short"] for s in d["suites"]}
    missing = suite_shorts - dash_shorts
    assert not missing, f"PYTEST_SUITES entries missing from dashboard: {sorted(missing)}"


def test_short_codes_are_unique_in_pytest_suites():
    shorts = [short for _, (_, short) in PYTEST_SUITES.items()]
    dupes = sorted({s for s in shorts if shorts.count(s) > 1})
    assert not dupes, f"duplicate short codes in PYTEST_SUITES: {dupes}"


def test_short_codes_are_unique_in_dashboard():
    d = _load_dashboard()
    shorts = [s["short"] for s in d["suites"]]
    dupes = sorted({s for s in shorts if shorts.count(s) > 1})
    assert not dupes, f"duplicate short codes in dashboard: {dupes}"


# ---------------------------------------------------------------------------
# Group 6: Test-count distribution sanity (catches mass-stub regression)
# ---------------------------------------------------------------------------


def test_at_least_half_of_suites_have_3_or_more_tests():
    counts = {short: _ast_test_count(REPO_ROOT / path) for _, (path, short) in PYTEST_SUITES.items()}
    rich = sum(1 for n in counts.values() if n >= 3)
    assert rich >= len(counts) // 2, (
        f"only {rich}/{len(counts)} suites have >=3 tests; "
        f"suspicious mass-stub regression"
    )


def test_aggregate_test_function_count_floor():
    """The full PYTEST_SUITES collection should expose at least 500 static
    test functions in total (current baseline ~1000+)."""
    total = sum(_ast_test_count(REPO_ROOT / path) for _, (path, _) in PYTEST_SUITES.items())
    assert total >= 500, (
        f"aggregate static test count = {total}; expected >=500 across "
        f"{len(PYTEST_SUITES)} suites"
    )
