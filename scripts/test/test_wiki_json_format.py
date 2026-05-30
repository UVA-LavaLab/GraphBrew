"""WJF-Fmt — wiki/data/*.json formatting stability gate.

Every tracked wiki/data/*.json artifact must satisfy:

1. Valid UTF-8.
2. Parses as JSON (no syntax errors).
3. No CRLF line endings.
4. Ends with exactly one trailing newline (no missing, no double).
5. No literal trailing whitespace on any line (tabs or spaces before EOL).

Each rule has an empty allow-list with a minimality self-test asserting
any exempt entry would actually violate its rule.

This is the JSON counterpart to WTQ-Fmt (gate 217). It exists because we
discovered 28/72 tracked JSON files were missing trailing newlines —
caused by 27 generators calling `write_text(json.dumps(...))` without
appending `"\n"`. All 27 generators have been normalized to the
canonical pattern: `write_text(json.dumps(..., indent=2, sort_keys=True) + "\n")`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA_DIR = REPO_ROOT / "wiki" / "data"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "ecg"))
import reproduce_smoke  # type: ignore  # noqa: E402

TRACKED_JSON = sorted(
    name for name in reproduce_smoke.TRACKED_ARTIFACTS if name.endswith(".json")
)

# Per-rule allow-lists (must remain empty; minimality tests enforce that
# any exempt entry actually violates its rule).
WJF_VALID_JSON_EXEMPT: set[str] = set()
WJF_CRLF_EXEMPT: set[str] = set()
WJF_FINAL_NEWLINE_EXEMPT: set[str] = set()
WJF_TRAILING_WS_EXEMPT: set[str] = set()
WJF_UTF8_EXEMPT: set[str] = set()


# ---------------------------------------------------------------------------
# Group 1: WIKI_DATA_DIR + TRACKED_JSON coverage sanity
# ---------------------------------------------------------------------------


def test_wiki_data_dir_exists():
    assert WIKI_DATA_DIR.is_dir(), f"missing dir: {WIKI_DATA_DIR}"


def test_tracked_json_nonempty():
    assert len(TRACKED_JSON) >= 50, f"too few tracked JSON artifacts: {len(TRACKED_JSON)}"


def test_every_tracked_json_exists():
    missing = [n for n in TRACKED_JSON if not (WIKI_DATA_DIR / n).is_file()]
    assert not missing, f"tracked JSON missing on disk: {missing[:10]}"


# ---------------------------------------------------------------------------
# Group 2: UTF-8 decode (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TRACKED_JSON)
def test_json_is_utf8(name: str):
    if name in WJF_UTF8_EXEMPT:
        pytest.skip(f"{name} explicitly exempt")
    raw = (WIKI_DATA_DIR / name).read_bytes()
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as e:
        pytest.fail(f"{name} is not valid UTF-8: {e}")


# ---------------------------------------------------------------------------
# Group 3: valid JSON (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TRACKED_JSON)
def test_json_parses(name: str):
    if name in WJF_VALID_JSON_EXEMPT:
        pytest.skip(f"{name} explicitly exempt")
    p = WIKI_DATA_DIR / name
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        pytest.fail(f"{name} is not valid JSON: line {e.lineno} col {e.colno}: {e.msg}")


# ---------------------------------------------------------------------------
# Group 4: no CRLF (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TRACKED_JSON)
def test_json_no_crlf(name: str):
    if name in WJF_CRLF_EXEMPT:
        pytest.skip(f"{name} explicitly exempt")
    raw = (WIKI_DATA_DIR / name).read_bytes()
    assert b"\r\n" not in raw, f"{name} contains CRLF line endings"


# ---------------------------------------------------------------------------
# Group 5: exactly one final newline (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TRACKED_JSON)
def test_json_final_newline(name: str):
    if name in WJF_FINAL_NEWLINE_EXEMPT:
        pytest.skip(f"{name} explicitly exempt")
    raw = (WIKI_DATA_DIR / name).read_bytes()
    assert raw, f"{name} is empty"
    assert raw.endswith(b"\n"), f"{name} missing final newline"
    assert not raw.endswith(b"\n\n"), f"{name} has double final newline"


# ---------------------------------------------------------------------------
# Group 6: no trailing whitespace (parametrized)
# ---------------------------------------------------------------------------


_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


@pytest.mark.parametrize("name", TRACKED_JSON)
def test_json_no_trailing_whitespace(name: str):
    if name in WJF_TRAILING_WS_EXEMPT:
        pytest.skip(f"{name} explicitly exempt")
    text = (WIKI_DATA_DIR / name).read_text(encoding="utf-8")
    bad = []
    for i, line in enumerate(text.splitlines(), 1):
        if _TRAILING_WS_RE.search(line):
            bad.append(i)
    assert not bad, f"{name} has trailing whitespace on lines {bad[:10]}"


# ---------------------------------------------------------------------------
# Group 7: allow-list minimality (every exempt entry must violate its rule)
# ---------------------------------------------------------------------------


def test_valid_json_exempt_actually_violates():
    for name in WJF_VALID_JSON_EXEMPT:
        p = WIKI_DATA_DIR / name
        if not p.is_file():
            continue
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        pytest.fail(f"{name} is in WJF_VALID_JSON_EXEMPT but parses cleanly — remove exemption")


def test_crlf_exempt_actually_violates():
    for name in WJF_CRLF_EXEMPT:
        p = WIKI_DATA_DIR / name
        if not p.is_file():
            continue
        if b"\r\n" not in p.read_bytes():
            pytest.fail(f"{name} is in WJF_CRLF_EXEMPT but has no CRLF — remove exemption")


def test_final_newline_exempt_actually_violates():
    for name in WJF_FINAL_NEWLINE_EXEMPT:
        p = WIKI_DATA_DIR / name
        if not p.is_file():
            continue
        raw = p.read_bytes()
        if raw.endswith(b"\n") and not raw.endswith(b"\n\n"):
            pytest.fail(
                f"{name} is in WJF_FINAL_NEWLINE_EXEMPT but has exactly one final newline — remove exemption"
            )


def test_trailing_ws_exempt_actually_violates():
    for name in WJF_TRAILING_WS_EXEMPT:
        p = WIKI_DATA_DIR / name
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        if not any(_TRAILING_WS_RE.search(ln) for ln in text.splitlines()):
            pytest.fail(
                f"{name} is in WJF_TRAILING_WS_EXEMPT but has no trailing whitespace — remove exemption"
            )


def test_utf8_exempt_actually_violates():
    for name in WJF_UTF8_EXEMPT:
        p = WIKI_DATA_DIR / name
        if not p.is_file():
            continue
        try:
            p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        pytest.fail(f"{name} is in WJF_UTF8_EXEMPT but decodes as UTF-8 — remove exemption")


# ---------------------------------------------------------------------------
# Group 8: aggregate sanity (regex self-test + bounded test count)
# ---------------------------------------------------------------------------


def test_trailing_ws_regex_self_test():
    assert _TRAILING_WS_RE.search("ok   \n")
    assert _TRAILING_WS_RE.search("tab\t\n")
    assert _TRAILING_WS_RE.search("mixed \t\n")
    assert not _TRAILING_WS_RE.search("clean\n")
    assert not _TRAILING_WS_RE.search("nofinal")


def test_aggregate_bounded_test_count():
    # 3 fixture + 5 × N parametrized + 5 minimality + 2 self-test = 10 + 5N
    assert len(TRACKED_JSON) >= 50
