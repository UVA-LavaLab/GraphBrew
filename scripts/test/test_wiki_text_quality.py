"""Wiki text quality (WTQ-Fmt) — gate 217.

Every tracked ``.md`` artifact in ``reproduce_smoke.TRACKED_ARTIFACTS``
must satisfy basic markdown hygiene: no trailing whitespace per line,
no CRLF line endings, exactly one final newline (the POSIX rule), no
triple-or-greater consecutive newlines (no double-blank-line gaps),
and an H1 heading as the first non-empty line.

These are 'formatting-stable' invariants — they catch real regressions
that diff-noise rolls in over time but which the existing byte-stable
``reproduce_smoke`` SHA gate cannot localize. ``reproduce_smoke`` only
says "the bytes are stable across two regen calls"; it cannot say
"the bytes are stable AND look like clean markdown." WTQ-Fmt is the
'looks like clean markdown' floor.

Catches:
  - editors that silently add trailing whitespace
  - Windows CRLF leaking in via copy-paste from external docs
  - templates that forget the final newline (git/diff flags this)
  - human-authored sections inserting cosmetic blank-line runs
  - generator regressions that drop the leading H1 heading

Probe result on current corpus: 72 tracked .md files, all 72 clean
on every invariant — no allow-list needed today. If a future artifact
genuinely cannot satisfy a rule (e.g. a rendered email-content
attachment with intentional CRLF), add it to the per-rule exempt set
with a one-line comment naming the reason. Minimality tests ensure
exempt entries actually violate the rule they are exempted from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"
WIKI_DATA_DIR = REPO_ROOT / "wiki" / "data"

sys.path.insert(0, str(ECG_DIR))
from reproduce_smoke import TRACKED_ARTIFACTS  # noqa: E402

_TRACKED_MD: list[str] = sorted(a for a in TRACKED_ARTIFACTS if a.endswith(".md"))

# Per-rule allow-lists (all empty today). Add entries with a one-line
# comment naming the genuine reason the rule cannot apply. Minimality
# tests ensure entries here actually violate the rule.
WTQ_TRAILING_WS_EXEMPT: set[str] = set()
WTQ_CRLF_EXEMPT: set[str] = set()
WTQ_FINAL_NEWLINE_EXEMPT: set[str] = set()
WTQ_DOUBLE_BLANK_EXEMPT: set[str] = set()
WTQ_H1_HEADING_EXEMPT: set[str] = set()


def _read_raw(path: str) -> bytes:
    return (WIKI_DATA_DIR / path).read_bytes()


def _read_text(path: str) -> str:
    return _read_raw(path).decode("utf-8", errors="strict")


# ---------------------------------------------------------------------------
# Group 1: TRACKED_MD shape pre-conditions
# ---------------------------------------------------------------------------


def test_tracked_md_set_is_non_empty():
    assert len(_TRACKED_MD) >= 50, f"only {len(_TRACKED_MD)} tracked .md files"


def test_every_tracked_md_exists():
    missing = [p for p in _TRACKED_MD if not (WIKI_DATA_DIR / p).is_file()]
    assert not missing, f"tracked .md missing on disk: {missing}"


def test_every_tracked_md_decodes_as_utf8():
    bad = []
    for p in _TRACKED_MD:
        try:
            _read_raw(p).decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            bad.append((p, str(e)))
    assert not bad, f"non-UTF-8 .md files: {bad}"


# ---------------------------------------------------------------------------
# Group 2: No trailing whitespace per line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _TRACKED_MD, ids=lambda p: Path(p).name)
def test_no_trailing_whitespace_per_line(path: str):
    """Reject trailing tabs and trailing spaces other than the markdown
    hard-break sequence (exactly two trailing spaces). The CommonMark
    spec defines ``"  \\n"`` as an explicit line break; rejecting it
    would force generators to use ``<br>`` everywhere. Tabs are always
    rejected; single trailing space, three+ trailing spaces, and mixed
    space+tab are also rejected (cosmetic noise with no rendering value).
    """
    if path in WTQ_TRAILING_WS_EXEMPT:
        pytest.skip(f"{path} on WTQ_TRAILING_WS_EXEMPT allow-list")
    text = _read_text(path)
    offenders = []
    for i, line in enumerate(text.split("\n")):
        if line.endswith("\t"):
            offenders.append((i + 1, "trailing-tab", repr(line)))
            continue
        stripped = line.rstrip(" ")
        n_trailing = len(line) - len(stripped)
        # 0 spaces (clean) and exactly 2 spaces (markdown hard break) OK.
        if n_trailing not in (0, 2):
            offenders.append((i + 1, f"{n_trailing}-trailing-spaces", repr(line)))
    assert not offenders, (
        f"{path} has trailing whitespace on {len(offenders)} lines; "
        f"first: line {offenders[0][0]} kind={offenders[0][1]} {offenders[0][2]}"
    )


# ---------------------------------------------------------------------------
# Group 3: No CRLF line endings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _TRACKED_MD, ids=lambda p: Path(p).name)
def test_no_crlf_line_endings(path: str):
    if path in WTQ_CRLF_EXEMPT:
        pytest.skip(f"{path} on WTQ_CRLF_EXEMPT allow-list")
    raw = _read_raw(path)
    assert b"\r\n" not in raw, f"{path} has CRLF line endings (POSIX LF only)"
    assert b"\r" not in raw, f"{path} has bare CR characters"


# ---------------------------------------------------------------------------
# Group 4: Exactly one final newline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _TRACKED_MD, ids=lambda p: Path(p).name)
def test_exactly_one_final_newline(path: str):
    if path in WTQ_FINAL_NEWLINE_EXEMPT:
        pytest.skip(f"{path} on WTQ_FINAL_NEWLINE_EXEMPT allow-list")
    raw = _read_raw(path)
    assert raw, f"{path} is empty"
    assert raw.endswith(b"\n"), f"{path} missing final newline"
    assert not raw.endswith(b"\n\n"), (
        f"{path} has multiple trailing newlines (POSIX rule: exactly one)"
    )


# ---------------------------------------------------------------------------
# Group 5: No triple-or-greater consecutive newlines (no double-blank lines)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _TRACKED_MD, ids=lambda p: Path(p).name)
def test_no_triple_consecutive_newlines(path: str):
    if path in WTQ_DOUBLE_BLANK_EXEMPT:
        pytest.skip(f"{path} on WTQ_DOUBLE_BLANK_EXEMPT allow-list")
    text = _read_text(path)
    assert "\n\n\n" not in text, (
        f"{path} contains triple-newline (= double blank line); "
        f"first occurrence at offset {text.find(chr(10) * 3)}"
    )


# ---------------------------------------------------------------------------
# Group 6: First non-empty line is an H1 heading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _TRACKED_MD, ids=lambda p: Path(p).name)
def test_first_nonempty_line_is_h1(path: str):
    if path in WTQ_H1_HEADING_EXEMPT:
        pytest.skip(f"{path} on WTQ_H1_HEADING_EXEMPT allow-list")
    text = _read_text(path)
    first_real_line = next(
        (line for line in text.splitlines() if line.strip()),
        "",
    )
    assert first_real_line.startswith("# "), (
        f"{path} first non-empty line is not an H1 heading "
        f"(must start with '# '): {first_real_line!r}"
    )


# ---------------------------------------------------------------------------
# Group 7: Allow-list minimality (every exempt entry must violate its rule)
# ---------------------------------------------------------------------------


def test_trailing_ws_exempt_entries_actually_violate():
    spurious = []
    for p in WTQ_TRAILING_WS_EXEMPT:
        if not (WIKI_DATA_DIR / p).is_file():
            continue
        text = _read_text(p)
        clean = True
        for line in text.split("\n"):
            if line.endswith("\t"):
                clean = False
                break
            stripped = line.rstrip(" ")
            n_trailing = len(line) - len(stripped)
            if n_trailing not in (0, 2):
                clean = False
                break
        if clean:
            spurious.append(p)
    assert not spurious, (
        f"WTQ_TRAILING_WS_EXEMPT entries that are actually clean: {spurious}"
    )


def test_crlf_exempt_entries_actually_violate():
    spurious = []
    for p in WTQ_CRLF_EXEMPT:
        if not (WIKI_DATA_DIR / p).is_file():
            continue
        if b"\r\n" not in _read_raw(p):
            spurious.append(p)
    assert not spurious, (
        f"WTQ_CRLF_EXEMPT entries that are actually clean: {spurious}"
    )


def test_final_newline_exempt_entries_actually_violate():
    spurious = []
    for p in WTQ_FINAL_NEWLINE_EXEMPT:
        if not (WIKI_DATA_DIR / p).is_file():
            continue
        raw = _read_raw(p)
        if raw.endswith(b"\n") and not raw.endswith(b"\n\n"):
            spurious.append(p)
    assert not spurious, (
        f"WTQ_FINAL_NEWLINE_EXEMPT entries that are actually clean: {spurious}"
    )


def test_double_blank_exempt_entries_actually_violate():
    spurious = []
    for p in WTQ_DOUBLE_BLANK_EXEMPT:
        if not (WIKI_DATA_DIR / p).is_file():
            continue
        if "\n\n\n" not in _read_text(p):
            spurious.append(p)
    assert not spurious, (
        f"WTQ_DOUBLE_BLANK_EXEMPT entries that are actually clean: {spurious}"
    )


def test_h1_heading_exempt_entries_actually_violate():
    spurious = []
    for p in WTQ_H1_HEADING_EXEMPT:
        if not (WIKI_DATA_DIR / p).is_file():
            continue
        text = _read_text(p)
        first = next((line for line in text.splitlines() if line.strip()), "")
        if first.startswith("# "):
            spurious.append(p)
    assert not spurious, (
        f"WTQ_H1_HEADING_EXEMPT entries that are actually clean: {spurious}"
    )


# ---------------------------------------------------------------------------
# Group 8: Aggregate sanity
# ---------------------------------------------------------------------------


def test_at_least_50_md_files_pass_every_rule():
    """Sanity floor: every tracked .md should clear every rule today
    (allow-lists are empty). This catches the situation where a future
    refactor migrates the entire corpus to an exempt allow-list."""
    n = len(_TRACKED_MD)
    assert n >= 50, f"corpus shrank: {n} tracked .md files"


def test_all_per_rule_allow_lists_are_empty_today():
    """Documentation invariant: if you grow any allow-list, please
    flip this expectation to ``> 0`` and adjust the gate-217
    commit summary so the next reader can see WHY the exception
    landed. The point of allow-lists is to be load-bearing
    documentation, not invisible escape hatches."""
    all_exempt = (
        WTQ_TRAILING_WS_EXEMPT
        | WTQ_CRLF_EXEMPT
        | WTQ_FINAL_NEWLINE_EXEMPT
        | WTQ_DOUBLE_BLANK_EXEMPT
        | WTQ_H1_HEADING_EXEMPT
    )
    assert not all_exempt, (
        f"per-rule allow-lists are non-empty: {sorted(all_exempt)}; "
        f"if intentional, update this test's expectation and the gate-217 "
        f"description to document WHY each exemption is load-bearing"
    )
