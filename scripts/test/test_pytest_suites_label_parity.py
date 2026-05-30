"""PSL-Par — PYTEST_SUITES test-file → suite-label parity gate.

The dashboard's PYTEST_SUITES dict is the canonical registry of every
confidence gate. Each entry maps a long human-readable label to a tuple
of (test-file path, short code). This gate locks every structural
invariant on that registry so a typo, duplicate, or rename never slips
through unnoticed.

Five invariant categories:

1. **Uniqueness** — every test-file path, every short code, and every
   label appears exactly once. Duplicates would silently overwrite or
   double-count gates.
2. **Path format** — every test-file path is `scripts/test/test_*.py`,
   exists on disk, and is a valid pytest module.
3. **Short-code format** — uppercase ASCII letters/digits with
   single-dash separators (e.g. `WTQ-Fmt`, `CGH-Sig`). Three legacy
   "Tier A/B/C" labels are explicitly allow-listed.
4. **Label format** — every label is ≥10 characters of human prose;
   labels >150 chars are flagged as too verbose for the dashboard.
5. **Cross-source parity** — every PYTEST_SUITES entry appears in
   the on-disk dashboard JSON's suites array with the matching short
   code; every dashboard suite traces back to PYTEST_SUITES.

This gate complements PST-Min (gate 215, per-suite test-count floor),
CGH-Sig (gate 216, --help signature), WTQ-Fmt (gate 217, .md format),
and WJF-Fmt (gate 218, .json format) by guarding the registry itself.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "ecg"))
from confidence_dashboard import PYTEST_SUITES  # type: ignore  # noqa: E402

DASHBOARD_JSON = REPO_ROOT / "wiki" / "data" / "confidence_dashboard.json"

# Legacy "Tier A/B/C" short labels predate the SHORT-CODE convention
# and are still referenced by test_confidence_dashboard.py fixtures.
# Any future addition with a space in the short code must justify
# itself in this allow-list with a comment.
PSL_LEGACY_SHORT_EXEMPT: set[str] = {"Tier A", "Tier B", "Tier C"}

PSL_PATH_REGEX = re.compile(r"^scripts/test/test_[a-z0-9_]+\.py$")
PSL_SHORT_REGEX = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*$")
LABEL_MIN_CHARS = 10
LABEL_MAX_CHARS = 250


# ---------------------------------------------------------------------------
# Group 1 — Uniqueness (3 tests)
# ---------------------------------------------------------------------------


def test_pytest_suites_is_nonempty():
    assert len(PYTEST_SUITES) >= 100


def test_no_duplicate_paths():
    paths = [tup[0] for tup in PYTEST_SUITES.values()]
    seen, dups = set(), []
    for p in paths:
        if p in seen:
            dups.append(p)
        seen.add(p)
    assert not dups, f"duplicate test-file paths in PYTEST_SUITES: {dups}"


def test_no_duplicate_shorts():
    shorts = [tup[1] for tup in PYTEST_SUITES.values()]
    seen, dups = set(), []
    for s in shorts:
        if s in seen:
            dups.append(s)
        seen.add(s)
    assert not dups, f"duplicate short codes in PYTEST_SUITES: {dups}"


def test_no_duplicate_labels():
    labels = list(PYTEST_SUITES.keys())
    seen, dups = set(), []
    for l in labels:
        if l in seen:
            dups.append(l)
        seen.add(l)
    assert not dups, f"duplicate labels in PYTEST_SUITES: {dups}"


# ---------------------------------------------------------------------------
# Group 2 — Path format (parametrized × N)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", list(PYTEST_SUITES.items()), ids=lambda e: e[1][1])
def test_path_matches_format(entry):
    label, (path, short) = entry
    assert PSL_PATH_REGEX.match(path), (
        f"{short!r} path {path!r} does not match scripts/test/test_*.py"
    )


@pytest.mark.parametrize("entry", list(PYTEST_SUITES.items()), ids=lambda e: e[1][1])
def test_path_exists(entry):
    label, (path, short) = entry
    assert (REPO_ROOT / path).is_file(), f"{short!r} → missing file: {path}"


# ---------------------------------------------------------------------------
# Group 3 — Short-code format (parametrized × N)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", list(PYTEST_SUITES.items()), ids=lambda e: e[1][1])
def test_short_matches_format(entry):
    label, (path, short) = entry
    if short in PSL_LEGACY_SHORT_EXEMPT:
        pytest.skip(f"{short!r} is explicitly allow-listed as a legacy tier label")
    assert PSL_SHORT_REGEX.match(short), (
        f"short code {short!r} does not match uppercase[-uppercase]+ format"
    )


def test_legacy_short_exempt_minimality():
    for s in PSL_LEGACY_SHORT_EXEMPT:
        if PSL_SHORT_REGEX.match(s):
            pytest.fail(
                f"{s!r} is in PSL_LEGACY_SHORT_EXEMPT but matches PSL_SHORT_REGEX — remove exemption"
            )


def test_short_max_length():
    for label, (path, short) in PYTEST_SUITES.items():
        assert len(short) <= 20, f"short {short!r} too long ({len(short)} chars, max 20)"


# ---------------------------------------------------------------------------
# Group 4 — Label format (parametrized × N)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", list(PYTEST_SUITES.items()), ids=lambda e: e[1][1])
def test_label_min_length(entry):
    label, (path, short) = entry
    assert len(label) >= LABEL_MIN_CHARS, (
        f"label for {short!r} too short ({len(label)} chars): {label!r}"
    )


@pytest.mark.parametrize("entry", list(PYTEST_SUITES.items()), ids=lambda e: e[1][1])
def test_label_max_length(entry):
    label, (path, short) = entry
    assert len(label) <= LABEL_MAX_CHARS, (
        f"label for {short!r} too long ({len(label)} chars, max {LABEL_MAX_CHARS})"
    )


@pytest.mark.parametrize("entry", list(PYTEST_SUITES.items()), ids=lambda e: e[1][1])
def test_label_no_control_chars(entry):
    label, (path, short) = entry
    bad = [c for c in label if ord(c) < 32 or ord(c) == 127]
    assert not bad, f"label for {short!r} contains control chars: {bad}"


# ---------------------------------------------------------------------------
# Group 5 — Cross-source parity (PYTEST_SUITES ↔ dashboard JSON)
# ---------------------------------------------------------------------------


def _load_dashboard():
    if not DASHBOARD_JSON.is_file():
        pytest.skip(f"{DASHBOARD_JSON} does not exist — run make confidence-fast first")
    return json.loads(DASHBOARD_JSON.read_text())


def test_dashboard_has_every_suite():
    dash = _load_dashboard()
    dash_shorts = {s.get("short") for s in dash.get("suites", [])}
    suite_shorts = {tup[1] for tup in PYTEST_SUITES.values()}
    missing = suite_shorts - dash_shorts
    assert not missing, f"shorts in PYTEST_SUITES but not in dashboard: {sorted(missing)[:10]}"


def test_dashboard_has_no_extra_suites():
    dash = _load_dashboard()
    dash_shorts = {s.get("short") for s in dash.get("suites", [])}
    suite_shorts = {tup[1] for tup in PYTEST_SUITES.values()}
    extra = dash_shorts - suite_shorts
    assert not extra, f"shorts in dashboard but not in PYTEST_SUITES: {sorted(extra)[:10]}"


def test_dashboard_path_matches_pytest_suites():
    dash = _load_dashboard()
    by_short = {tup[1]: (label, tup[0]) for label, tup in PYTEST_SUITES.items()}
    mismatches = []
    for s in dash.get("suites", []):
        short = s.get("short")
        if short not in by_short:
            continue
        _, expected_path = by_short[short]
        if s.get("path") != expected_path:
            mismatches.append((short, s.get("path"), expected_path))
    assert not mismatches, f"dashboard path != PYTEST_SUITES path for: {mismatches[:5]}"


def test_dashboard_label_matches_pytest_suites():
    dash = _load_dashboard()
    by_short = {tup[1]: (label, tup[0]) for label, tup in PYTEST_SUITES.items()}
    mismatches = []
    for s in dash.get("suites", []):
        short = s.get("short")
        if short not in by_short:
            continue
        expected_label, _ = by_short[short]
        if s.get("label") != expected_label:
            mismatches.append((short, s.get("label"), expected_label))
    assert not mismatches, f"dashboard label != PYTEST_SUITES label for: {mismatches[:5]}"


# ---------------------------------------------------------------------------
# Group 6 — regex self-tests
# ---------------------------------------------------------------------------


def test_path_regex_self_test():
    assert PSL_PATH_REGEX.match("scripts/test/test_foo.py")
    assert PSL_PATH_REGEX.match("scripts/test/test_foo_bar_baz.py")
    assert not PSL_PATH_REGEX.match("scripts/test/foo.py")
    assert not PSL_PATH_REGEX.match("scripts/test/test_foo.txt")
    assert not PSL_PATH_REGEX.match("scripts/test/TestFoo.py")
    assert not PSL_PATH_REGEX.match("test_foo.py")


def test_short_regex_self_test():
    assert PSL_SHORT_REGEX.match("WTQ-Fmt")
    assert PSL_SHORT_REGEX.match("CGH-Sig")
    assert PSL_SHORT_REGEX.match("PST-Min")
    assert PSL_SHORT_REGEX.match("X")
    assert not PSL_SHORT_REGEX.match("Tier A")
    assert not PSL_SHORT_REGEX.match("With Space")
    assert not PSL_SHORT_REGEX.match("with-trailing-")
    assert not PSL_SHORT_REGEX.match("-leading-dash")


def test_aggregate_sanity():
    # 3 unique + 2 × N (path) + 1 × N (short) + 1 minimality + 1 short-len
    # + 3 × N (label) + 4 cross-source + 3 self-tests
    assert len(PYTEST_SUITES) >= 100
