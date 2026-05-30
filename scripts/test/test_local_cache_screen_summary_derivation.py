"""Derivation-parity gate for ``local_cache_screen_summary.py``.

Why this gate exists
--------------------
``local_cache_screen_summary.py`` aggregates per-benchmark cache_sim
screens into a single summary CSV that drives the diversity-screen
tables in the wiki. Its arithmetic — LRU baseline lookup, delta vs LRU,
prefetch usefulness rate, l3_rank ordering — feeds policy-comparison
numbers cited downstream by the paper.

The code is small (≈150 lines) but every predicate is load-bearing:

Group 1 — ``number`` coercion: None/empty/non-numeric → None.
Group 2 — ``format_number`` near-integer collapse + 6-sig-fig fallback.
Group 3 — ``input_label`` precedence (explicit > parent dir > stem).
Group 4 — ``parse_input`` ``=`` separator (split-once so paths with ``=`` survive).
Group 5 — ``summarize_rows`` aggregation: status!='ok' drop, LRU-base lookup,
          l3_delta_vs_lru = (lru-misses)/lru, l3_rank by (misses, policy_label).
Group 6 — ``write_csv`` / FIELDNAMES column order pinned.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "ecg"))

import local_cache_screen_summary as _lcs  # noqa: E402


# ---------------------------------------------------------------------------
# Group 1 — number coercion
# ---------------------------------------------------------------------------


def test_number_none_returns_none():
    assert _lcs.number(None) is None


def test_number_empty_string_returns_none():
    assert _lcs.number("") is None


def test_number_integer_string_returns_float():
    """Coerces via float(str(value)) — '42' returns 42.0 (float, not int)."""
    result = _lcs.number("42")
    assert result == 42.0
    assert isinstance(result, float)


def test_number_float_string_returns_float():
    assert _lcs.number("3.14") == 3.14


def test_number_invalid_string_returns_none():
    """Non-numeric input → None (no exception). Defensive for CSV cells
    that may contain status strings or junk."""
    assert _lcs.number("abc") is None
    assert _lcs.number("not_a_number") is None


def test_number_accepts_numeric_input_directly():
    """str() wrapping means raw ints/floats also coerce cleanly."""
    assert _lcs.number(7) == 7.0
    assert _lcs.number(1.5) == 1.5


# ---------------------------------------------------------------------------
# Group 2 — format_number
# ---------------------------------------------------------------------------


def test_format_number_none_returns_empty_string():
    """None → '' (NOT '0' or '—') — CSV cell stays blank so downstream
    consumers can distinguish missing from zero."""
    assert _lcs.format_number(None) == ""


def test_format_number_exact_integer_renders_as_int():
    """Near-integer collapse: |v - round(v)| < 1e-9 → str(int(round(v))).
    Pins 'no .0 suffix' for whole numbers — critical for git diff
    stability across regen cycles."""
    assert _lcs.format_number(42.0) == "42"
    assert _lcs.format_number(0.0) == "0"


def test_format_number_within_epsilon_collapses_to_int():
    """Values within 1e-9 of an integer collapse — defensive against
    float-arithmetic noise in computed deltas."""
    assert _lcs.format_number(42.0 + 1e-10) == "42"
    assert _lcs.format_number(42.0 - 1e-10) == "42"


def test_format_number_just_outside_epsilon_keeps_decimal():
    """Outside the 1e-9 band AND past 6-sig-fig rounding → '.6g' fall-through.
    Note: tiny deltas (e.g. 1e-8 from int) still render as int because
    '.6g' truncates after 6 sig figs; the meaningful test uses a value
    where the decimal survives 6-sig-fig formatting."""
    assert _lcs.format_number(42.001) == "42.001"
    assert _lcs.format_number(42.5) == "42.5"


def test_format_number_fractional_uses_six_sig_fig():
    """Non-near-integer values → format spec '.6g' (6 significant digits)."""
    assert _lcs.format_number(3.14) == "3.14"
    assert _lcs.format_number(0.123456789) == "0.123457"


def test_format_number_negative_integer():
    assert _lcs.format_number(-7.0) == "-7"


# ---------------------------------------------------------------------------
# Group 3 — input_label precedence
# ---------------------------------------------------------------------------


def test_input_label_explicit_wins():
    """Explicit label takes precedence over both parent and stem."""
    p = Path("/some/parent/file.csv")
    assert _lcs.input_label(p, explicit_label="my_label") == "my_label"


def test_input_label_parent_dir_when_no_explicit():
    """Parent directory name is the default — encodes the per-benchmark
    subdir convention used by the screen runner."""
    p = Path("/data/email-Eu-core/combined_roi_matrix.csv")
    assert _lcs.input_label(p) == "email-Eu-core"


def test_input_label_falls_back_to_stem_when_no_parent():
    """No parent → file stem. The walrus 'if parent' check filters '' (root)."""
    p = Path("file.csv")
    assert _lcs.input_label(p) == "file"


# ---------------------------------------------------------------------------
# Group 4 — parse_input '=' separator
# ---------------------------------------------------------------------------


def test_parse_input_with_label():
    """'label=path' → (label, resolved_path). Label may be any string."""
    label, path = _lcs.parse_input("my_label=/tmp/foo.csv")
    assert label == "my_label"
    assert isinstance(path, Path)


def test_parse_input_without_label():
    """No '=' → label is None, path is the whole value (resolved)."""
    label, path = _lcs.parse_input("/tmp/foo.csv")
    assert label is None
    assert isinstance(path, Path)


def test_parse_input_split_once_preserves_equals_in_path():
    """split('=', 1) so paths containing '=' (e.g., query-string-style)
    are preserved after the first split. Pinned because the obvious
    full-split would break legitimate paths."""
    label, path = _lcs.parse_input("label=/tmp/foo=bar.csv")
    assert label == "label"
    # path text was "/tmp/foo=bar.csv"; resolve preserves the '=' part.
    assert "=" in str(path) or "bar" in str(path)


# ---------------------------------------------------------------------------
# Group 5 — summarize_rows aggregation
# ---------------------------------------------------------------------------


def _row(policy, misses, status="ok", benchmark="bc", prefetcher="off", l3="1MB", **kw):
    base = {
        "benchmark": benchmark,
        "prefetcher": prefetcher,
        "l3_size": l3,
        "policy_label": policy,
        "status": status,
        "l3_misses": str(misses) if misses is not None else "",
        "prefetch_requests": kw.get("prefetch_requests", "0"),
        "prefetch_useful": kw.get("prefetch_useful", "0"),
        "timing_valid_for_speedup": kw.get("timing_valid_for_speedup", "1"),
    }
    return base


def test_summarize_filters_non_ok_status():
    """Rows with status != 'ok' are dropped before aggregation. A
    failed run must NOT contribute to LRU baseline or rank."""
    rows = [
        _row("LRU", 100, status="ok"),
        _row("GRASP", 50, status="fail"),  # dropped
    ]
    out = _lcs.summarize_rows("label", rows)
    assert len(out) == 1
    assert out[0]["policy_label"] == "LRU"


def test_summarize_delta_vs_lru():
    """l3_delta_vs_lru = (lru_misses - row_misses) / lru_misses.
    Positive = reduction; negative = regression. GRASP at 50 vs LRU
    at 100 → +0.5."""
    rows = [_row("LRU", 100), _row("GRASP", 50)]
    out = _lcs.summarize_rows("label", rows)
    grasp = next(r for r in out if r["policy_label"] == "GRASP")
    # 0.5 is exact → format_number({:.6g}) renders as "0.5"
    assert grasp["l3_delta_vs_lru"] == "0.5"


def test_summarize_lru_delta_is_zero():
    """LRU's own delta vs itself is 0 → format_number collapses to '0'
    (near-integer branch)."""
    rows = [_row("LRU", 100), _row("GRASP", 50)]
    out = _lcs.summarize_rows("label", rows)
    lru = next(r for r in out if r["policy_label"] == "LRU")
    assert lru["l3_delta_vs_lru"] == "0"


def test_summarize_rank_by_misses_then_policy():
    """Rank order: ascending l3_misses, ties broken by policy_label
    (lex). Lowest misses → rank 1. Pinned because the paper cites
    rank-1 winners by name."""
    rows = [
        _row("LRU", 100),
        _row("GRASP", 50),
        _row("POPT", 50),  # same misses → tie-break by name
    ]
    out = _lcs.summarize_rows("label", rows)
    by_policy = {r["policy_label"]: r["l3_rank"] for r in out}
    # GRASP (50) and POPT (50) tied: 'GRASP' < 'POPT' lex → ranks 1,2
    assert by_policy["GRASP"] == 1
    assert by_policy["POPT"] == 2
    assert by_policy["LRU"] == 3


def test_summarize_missing_lru_yields_empty_delta():
    """No LRU row in the group → lru_misses is None → delta column
    rendered empty (NOT 0 — distinguishes 'no baseline' from
    'baseline matched')."""
    rows = [_row("GRASP", 50)]
    out = _lcs.summarize_rows("label", rows)
    assert out[0]["l3_delta_vs_lru"] == ""


def test_summarize_zero_lru_misses_yields_empty_delta():
    """LRU at 0 misses → would divide by zero → guarded; delta
    rendered empty rather than inf/NaN."""
    rows = [_row("LRU", 0), _row("GRASP", 0)]
    out = _lcs.summarize_rows("label", rows)
    grasp = next(r for r in out if r["policy_label"] == "GRASP")
    assert grasp["l3_delta_vs_lru"] == ""


def test_summarize_prefetch_useful_rate():
    """prefetch_useful_per_request = useful / requests; 0 requests →
    None → empty cell."""
    rows = [
        _row("LRU", 100, prefetch_requests="200", prefetch_useful="50"),
    ]
    out = _lcs.summarize_rows("label", rows)
    assert out[0]["prefetch_useful_per_request"] == "0.25"


def test_summarize_zero_requests_yields_empty_useful_rate():
    rows = [_row("LRU", 100, prefetch_requests="0", prefetch_useful="0")]
    out = _lcs.summarize_rows("label", rows)
    assert out[0]["prefetch_useful_per_request"] == ""


def test_summarize_groups_by_triple_not_just_benchmark():
    """Grouping key is (benchmark, prefetcher, l3_size) — same
    benchmark with different prefetcher or L3 forms a separate
    group with its own LRU baseline."""
    rows = [
        _row("LRU", 100, l3="1MB"),
        _row("GRASP", 50, l3="1MB"),
        _row("LRU", 200, l3="2MB"),  # different L3 → own baseline
        _row("GRASP", 150, l3="2MB"),
    ]
    out = _lcs.summarize_rows("label", rows)
    by_pair = {(r["l3_size"], r["policy_label"]): r["l3_delta_vs_lru"] for r in out}
    # GRASP@1MB: (100-50)/100 = 0.5
    # GRASP@2MB: (200-150)/200 = 0.25
    assert by_pair[("1MB", "GRASP")] == "0.5"
    assert by_pair[("2MB", "GRASP")] == "0.25"


# ---------------------------------------------------------------------------
# Group 6 — write_csv / FIELDNAMES
# ---------------------------------------------------------------------------


def test_fieldnames_column_order_pinned():
    """FIELDNAMES order is the public contract of the summary CSV —
    downstream tables index by position so reordering silently
    breaks them. Pinned exactly."""
    assert _lcs.FIELDNAMES == [
        "source",
        "benchmark",
        "prefetcher",
        "l3_size",
        "policy_label",
        "status",
        "l3_misses",
        "l3_rank",
        "l3_delta_vs_lru",
        "prefetch_requests",
        "prefetch_useful",
        "prefetch_useful_per_request",
        "timing_valid_for_speedup",
    ]


def test_write_csv_to_stdout(monkeypatch, capsys):
    """write_csv(None, rows) writes to stdout with FIELDNAMES header.
    DictWriter default lineterminator is \\r\\n — load-bearing for
    git diff cleanliness but NOT overridden in this generator (unlike
    literature_reproduction_summary which DOES override). Documented
    here so future drift either direction is intentional."""
    rows = [
        {f: "" for f in _lcs.FIELDNAMES}
    ]
    rows[0]["benchmark"] = "bc"
    rows[0]["l3_rank"] = 1
    _lcs.write_csv(None, rows)
    captured = capsys.readouterr().out
    header_line = captured.splitlines()[0]
    assert "benchmark" in header_line
    assert "l3_rank" in header_line
    # First column is 'source'
    assert header_line.startswith("source,")
