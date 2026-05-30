"""Catalog generator --help signature (CGH-Sig) — gate 216.

Every generator script registered in ``artifact_catalog.CATALOG`` must
respond to ``--help`` with a zero exit code. This catches two distinct
classes of regression that the existing test stack does not catch:

  1. **Import-time breakage** — a generator that crashes on import (e.g.
     a stale `from X import Y` after a refactor, a missing dependency,
     a top-level `assert sys.version_info >= (3, 13)`) is silently
     broken for everyone who tries to run it ad hoc; ``make`` only
     catches it when that specific target is invoked, which may not be
     for hours of CI time.
  2. **Argparse contract drift** — a generator without ``--help``
     support (e.g. argv parsed by hand without argparse, or argparse
     misconfigured to consume `--help` as a positional) has no
     discoverable interface; new contributors cannot ask "what flags
     does this take?" without reading source.

The gate runs each unique ``generator`` field from CATALOG with the
``--help`` flag and asserts:

  - the subprocess returns exit code 0,
  - the subprocess returns within a generous timeout (30s),
  - the stdout contains either ``usage:`` (argparse default) or
    ``Usage:`` / ``USAGE:`` (some hand-rolled helpers), establishing
    minimal discoverability.

Per-generator parametric expansion gives a fail-loud signal naming the
exact script that broke.

Allow-list discipline: there is currently no allow-list. Every
catalogued generator passes today. If a future generator is genuinely
non-CLI (library-style entry point), add it to ``CGH_NON_CLI_EXEMPT``
with a module-level comment explaining why it cannot have ``--help``,
and add it to ``test_exempt_entries_are_in_catalog`` so the exempt
list cannot drift away from reality.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"

sys.path.insert(0, str(ECG_DIR))
from artifact_catalog import CATALOG  # noqa: E402

# Allow-list for generators that legitimately do NOT expose --help.
# Empty today; if you add an entry, leave a one-line comment naming
# the reason (e.g. "library-style entry point, no main()").
CGH_NON_CLI_EXEMPT: set[str] = set()

_GENERATORS: list[str] = sorted({entry["generator"] for entry in CATALOG if "generator" in entry})
_TESTABLE: list[str] = [g for g in _GENERATORS if g not in CGH_NON_CLI_EXEMPT]

_USAGE_PATTERN = re.compile(r"\b(?:[Uu]sage|USAGE)\s*:", re.MULTILINE)

# Subprocess timeout per script — argparse --help should return instantly
# but some generators import heavy modules at top of file (pandas, gem5
# stats parsers, etc.). 30s is comfortable headroom.
HELP_TIMEOUT_SEC = 30


def _run_help(script: str) -> subprocess.CompletedProcess:
    """Invoke ``python3 <script> --help`` from the repo root."""
    return subprocess.run(
        ["python3", script, "--help"],
        capture_output=True,
        text=True,
        timeout=HELP_TIMEOUT_SEC,
        cwd=str(REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# Group 1: CATALOG shape pre-conditions
# ---------------------------------------------------------------------------


def test_catalog_is_non_empty_list():
    assert isinstance(CATALOG, list)
    assert len(CATALOG) >= 60, f"CATALOG shrank? len={len(CATALOG)}"


def test_at_least_60_unique_generators():
    assert len(_GENERATORS) >= 60, (
        f"only {len(_GENERATORS)} unique generators in CATALOG"
    )


def test_every_generator_file_exists():
    missing = [g for g in _GENERATORS if not (REPO_ROOT / g).is_file()]
    assert not missing, f"generator scripts missing on disk: {missing}"


# ---------------------------------------------------------------------------
# Group 2: --help returns 0 (parametric, one per generator)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "generator",
    _TESTABLE,
    ids=[Path(g).stem for g in _TESTABLE],
)
def test_generator_help_returns_zero(generator: str):
    """Every catalog generator must respond to --help with exit 0."""
    try:
        r = _run_help(generator)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"{generator} --help timed out after {HELP_TIMEOUT_SEC}s "
            f"(probably an import-time long-running call)"
        )
    assert r.returncode == 0, (
        f"{generator} --help exited {r.returncode}\n"
        f"stderr: {r.stderr[-500:]}\n"
        f"stdout: {r.stdout[-500:]}"
    )


# ---------------------------------------------------------------------------
# Group 3: --help output is discoverable (contains usage: marker)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "generator",
    _TESTABLE,
    ids=[Path(g).stem for g in _TESTABLE],
)
def test_generator_help_contains_usage_marker(generator: str):
    """--help must produce something resembling argparse usage output.

    Accepts ``usage:``, ``Usage:``, or ``USAGE:`` to allow hand-rolled
    helpers; argparse default is lowercase. Empty stdout is rejected.
    """
    try:
        r = _run_help(generator)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{generator} --help timed out")
    # Check stdout first (where argparse writes), fall back to stderr.
    haystack = (r.stdout or "") + "\n" + (r.stderr or "")
    assert _USAGE_PATTERN.search(haystack), (
        f"{generator} --help produced no 'usage:' marker.\n"
        f"stdout[-400:]: {r.stdout[-400:]}\n"
        f"stderr[-400:]: {r.stderr[-400:]}"
    )


# ---------------------------------------------------------------------------
# Group 4: --help completes promptly (sanity floor — most return instantly)
# ---------------------------------------------------------------------------


def test_aggregate_help_runtime_under_5_minutes():
    """Sanity floor: the full --help sweep should comfortably fit under
    5 wall-clock minutes (typically ~30s on this corpus). Catches a
    generator that accidentally does heavy work at import time."""
    import time
    t0 = time.time()
    for g in _TESTABLE:
        try:
            _run_help(g)
        except subprocess.TimeoutExpired:
            pytest.fail(f"{g} --help timed out during aggregate sweep")
    elapsed = time.time() - t0
    assert elapsed < 300, (
        f"aggregate --help sweep took {elapsed:.1f}s; "
        f"expected <300s. Likely a generator with heavy import-time work."
    )


# ---------------------------------------------------------------------------
# Group 5: Allow-list discipline (CGH_NON_CLI_EXEMPT minimality)
# ---------------------------------------------------------------------------


def test_exempt_entries_are_in_catalog():
    """If a generator is in CGH_NON_CLI_EXEMPT it must still be a
    catalogued generator; otherwise the allow-list has drifted away
    from reality."""
    catalog_gens = set(_GENERATORS)
    orphan = CGH_NON_CLI_EXEMPT - catalog_gens
    assert not orphan, (
        f"CGH_NON_CLI_EXEMPT lists generators that are NOT in CATALOG: "
        f"{sorted(orphan)} — remove them from the allow-list"
    )


def test_exempt_entries_genuinely_lack_help():
    """If an entry is on the allow-list, calling --help should NOT
    return 0 — otherwise the allow-list entry is invalid (the script
    works fine and shouldn't be exempt)."""
    spurious = []
    for g in CGH_NON_CLI_EXEMPT:
        if not (REPO_ROOT / g).is_file():
            continue
        try:
            r = _run_help(g)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0 and _USAGE_PATTERN.search((r.stdout or "") + (r.stderr or "")):
            spurious.append(g)
    assert not spurious, (
        f"CGH_NON_CLI_EXEMPT entries that ACTUALLY support --help: "
        f"{spurious} — they should be removed from the allow-list"
    )


# ---------------------------------------------------------------------------
# Group 6: Aggregate sanity
# ---------------------------------------------------------------------------


def test_testable_set_covers_almost_all_catalog_generators():
    """The exempt set should be tiny relative to the catalog — if more
    than 10% of generators are exempt, something is wrong with the gate."""
    if not _GENERATORS:
        pytest.skip("empty catalog")
    exempt_frac = len(CGH_NON_CLI_EXEMPT) / len(_GENERATORS)
    assert exempt_frac < 0.10, (
        f"CGH_NON_CLI_EXEMPT covers {exempt_frac:.0%} of generators; "
        f"that's too many exempt — gate is meaningless"
    )


def test_pattern_compiles_and_matches_argparse_default():
    """Self-test the usage-marker regex against a known argparse output."""
    sample = "usage: foo.py [-h] [--bar BAR]\n\nA tool."
    assert _USAGE_PATTERN.search(sample)
    sample_upper = "USAGE: foo.py [-h]\n"
    assert _USAGE_PATTERN.search(sample_upper)
    sample_caps = "Usage: foo.py\n"
    assert _USAGE_PATTERN.search(sample_caps)
    sample_no_usage = "Just a script.\n"
    assert not _USAGE_PATTERN.search(sample_no_usage)
