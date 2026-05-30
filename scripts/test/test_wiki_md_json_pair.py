"""WMP-Pair — wiki .md ↔ .json companion-pair invariants gate.

Most tracked wiki/data/ artifacts come as `<stem>.md` + `<stem>.json`
pairs: the JSON carries the structured numerical content, the markdown
renders it for human review. Both are generated together by the same
upstream script and downstream gates depend on both being current and
consistent.

This gate locks five structural invariants on every md/json pair:

1. **Bijection (with allow-list)** — every .json stem in
   TRACKED_ARTIFACTS has a corresponding .md stem in TRACKED_ARTIFACTS
   (and vice versa). One md-only exception is allow-listed today
   (`literature_reproduction_summary` — a downstream synthesis with no
   independent JSON).
2. **Both files exist on disk.**
3. **Both files are non-empty** (> 0 bytes after read).
4. **Markdown has plausible JSON-sibling reference** — if the markdown
   names its companion json filename in inline text (a common pattern
   in the codebase), that file actually exists.
5. **Allow-list minimality** — every md-only exempt entry must
   actually have no JSON companion in TRACKED_ARTIFACTS.

Together with WTQ-Fmt (217, .md format), WJF-Fmt (218, .json format),
and PSL-Par (219, PYTEST_SUITES parity) this completes the wiki/data
integrity quad.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA_DIR = REPO_ROOT / "wiki" / "data"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "ecg"))
import reproduce_smoke  # type: ignore  # noqa: E402

ALL_TRACKED = list(reproduce_smoke.TRACKED_ARTIFACTS)
TRACKED_MD = sorted(n for n in ALL_TRACKED if n.endswith(".md"))
TRACKED_JSON = sorted(n for n in ALL_TRACKED if n.endswith(".json"))

MD_STEMS = {Path(n).stem for n in TRACKED_MD}
JSON_STEMS = {Path(n).stem for n in TRACKED_JSON}

# md-only stems are downstream syntheses with no independent JSON.
WMP_MD_ONLY_EXEMPT: set[str] = {
    "literature_reproduction_summary",  # synthesises lit_faith JSON; no own JSON
}

# json-only stems would be machine-only artifacts with no human view.
WMP_JSON_ONLY_EXEMPT: set[str] = set()


# ---------------------------------------------------------------------------
# Group 1 — Coverage sanity (3 tests)
# ---------------------------------------------------------------------------


def test_wiki_data_dir_exists():
    assert WIKI_DATA_DIR.is_dir()


def test_tracked_md_nonempty():
    assert len(TRACKED_MD) >= 50, f"too few tracked .md: {len(TRACKED_MD)}"


def test_tracked_json_nonempty():
    assert len(TRACKED_JSON) >= 50, f"too few tracked .json: {len(TRACKED_JSON)}"


# ---------------------------------------------------------------------------
# Group 2 — Bijection (.md ↔ .json stems)
# ---------------------------------------------------------------------------


def test_every_json_has_md_sibling():
    missing = JSON_STEMS - MD_STEMS - WMP_JSON_ONLY_EXEMPT
    assert not missing, f"json stems with no .md sibling in TRACKED_ARTIFACTS: {sorted(missing)}"


def test_every_md_has_json_sibling():
    missing = MD_STEMS - JSON_STEMS - WMP_MD_ONLY_EXEMPT
    assert not missing, f"md stems with no .json sibling in TRACKED_ARTIFACTS: {sorted(missing)}"


def test_md_only_exempt_minimality():
    for stem in WMP_MD_ONLY_EXEMPT:
        if stem in JSON_STEMS:
            pytest.fail(
                f"{stem!r} in WMP_MD_ONLY_EXEMPT but has a .json sibling — remove exemption"
            )


def test_json_only_exempt_minimality():
    for stem in WMP_JSON_ONLY_EXEMPT:
        if stem in MD_STEMS:
            pytest.fail(
                f"{stem!r} in WMP_JSON_ONLY_EXEMPT but has a .md sibling — remove exemption"
            )


# ---------------------------------------------------------------------------
# Group 3 — Both files exist on disk (parametrized × N pairs)
# ---------------------------------------------------------------------------


PAIRED_STEMS = sorted(MD_STEMS & JSON_STEMS)


@pytest.mark.parametrize("stem", PAIRED_STEMS)
def test_pair_md_exists(stem):
    assert (WIKI_DATA_DIR / f"{stem}.md").is_file(), f"missing wiki/data/{stem}.md"


@pytest.mark.parametrize("stem", PAIRED_STEMS)
def test_pair_json_exists(stem):
    assert (WIKI_DATA_DIR / f"{stem}.json").is_file(), f"missing wiki/data/{stem}.json"


# ---------------------------------------------------------------------------
# Group 4 — Both files are non-empty (parametrized × N pairs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem", PAIRED_STEMS)
def test_pair_md_nonempty(stem):
    raw = (WIKI_DATA_DIR / f"{stem}.md").read_bytes()
    assert len(raw) > 0, f"wiki/data/{stem}.md is empty"


@pytest.mark.parametrize("stem", PAIRED_STEMS)
def test_pair_json_nonempty(stem):
    raw = (WIKI_DATA_DIR / f"{stem}.json").read_bytes()
    assert len(raw) > 0, f"wiki/data/{stem}.json is empty"


# ---------------------------------------------------------------------------
# Group 5 — Markdown's referenced sibling files exist (parametrized × N)
# ---------------------------------------------------------------------------


_SIBLING_REF_RE = re.compile(
    r"`?wiki/data/(?P<name>[a-z0-9_]+\.(?:json|csv|md))`?", re.IGNORECASE
)


@pytest.mark.parametrize("stem", PAIRED_STEMS)
def test_md_referenced_siblings_exist(stem):
    text = (WIKI_DATA_DIR / f"{stem}.md").read_text(encoding="utf-8")
    referenced = set(_SIBLING_REF_RE.findall(text))
    missing = [r for r in referenced if not (WIKI_DATA_DIR / r).is_file()]
    assert not missing, f"{stem}.md references nonexistent siblings: {missing}"


# ---------------------------------------------------------------------------
# Group 6 — Aggregate sanity + regex self-test
# ---------------------------------------------------------------------------


def test_sibling_ref_regex_self_test():
    text = "see `wiki/data/foo.json` and `wiki/data/bar.csv`"
    matches = _SIBLING_REF_RE.findall(text)
    assert "foo.json" in matches
    assert "bar.csv" in matches


def test_aggregate_pair_count():
    # 71 pairs today; floor 50 ensures the gate isn't degenerate.
    assert len(PAIRED_STEMS) >= 50, f"too few md/json pairs: {len(PAIRED_STEMS)}"


def test_bijection_consistency():
    paired_count = len(PAIRED_STEMS)
    md_only = len(MD_STEMS - JSON_STEMS)
    json_only = len(JSON_STEMS - MD_STEMS)
    # paired + md_only = len(MD_STEMS); paired + json_only = len(JSON_STEMS)
    assert paired_count + md_only == len(MD_STEMS)
    assert paired_count + json_only == len(JSON_STEMS)
    assert md_only <= len(WMP_MD_ONLY_EXEMPT)
    assert json_only <= len(WMP_JSON_ONLY_EXEMPT)
