"""Derivation-parity gate for ``literature_reproduction_summary.py``.

Why this gate exists
--------------------
``literature_reproduction_summary.py`` is the paper-ready renderer that
groups lit-faithfulness cells by *citation* so reviewers can see one row
per published claim. Its outputs feed the wiki tables that show the
per-paper reproduction percentage and the per-citation roll-up — numbers
that appear verbatim in the paper introduction.

The byte layout of the markdown / CSV is load-bearing: the paper text
cites e.g. "Faldu HPCA20 reproduction% = 94.7%" and a silent rounding
or aggregation drift here breaks the paper claim. There is no on-disk
JSON artifact for this renderer (output is CSV+MD), so we gate the
predicate layer directly:

Group 1 — ``_verdict_glyph`` dispatch table.
Group 2 — ``_format_delta`` formatting (None → "—"; signed 3-decimal pp).
Group 3 — ``_expected_window`` assembly (sign + |Δ| window + ±tol).
Group 4 — ``_key`` citation ordering (CITATION_ORDER_PREFIX dispatch).
Group 5 — ``_paper_name`` n-citation rule (n≥2 → cross-paper).
Group 6 — ``_paper_rollup`` aggregation + ``render_csv`` / ``render_markdown``
          byte parity (reproduction% counts only ok+within_tolerance —
          known_deviation explicitly excluded).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "ecg"))

import literature_reproduction_summary as _lrs  # noqa: E402


# ---------------------------------------------------------------------------
# Group 1 — _verdict_glyph dispatch
# ---------------------------------------------------------------------------


def test_verdict_glyph_ok():
    assert _lrs._verdict_glyph("ok") == "✅"


def test_verdict_glyph_within_tolerance():
    assert _lrs._verdict_glyph("within_tolerance") == "🟡"


def test_verdict_glyph_known_deviation():
    assert _lrs._verdict_glyph("known_deviation") == "📘"


def test_verdict_glyph_disagree():
    assert _lrs._verdict_glyph("disagree") == "⛔"


def test_verdict_glyph_missing_and_insufficient():
    assert _lrs._verdict_glyph("missing") == "⏳"
    assert _lrs._verdict_glyph("insufficient_data") == "🔬"


def test_verdict_glyph_unknown_passes_through():
    """Unknown statuses pass through verbatim so wiring drift is visible
    in the rendered table rather than silently mapped to a wrong glyph."""
    assert _lrs._verdict_glyph("xyzzy") == "xyzzy"
    assert _lrs._verdict_glyph("") == ""


# ---------------------------------------------------------------------------
# Group 2 — _format_delta formatting
# ---------------------------------------------------------------------------


def test_format_delta_none_renders_em_dash():
    assert _lrs._format_delta({"delta_pct": None}) == "—"


def test_format_delta_missing_key_renders_em_dash():
    """`.get('delta_pct')` returns None for missing keys, same as explicit None."""
    assert _lrs._format_delta({}) == "—"


def test_format_delta_positive_is_signed():
    """Positive deltas use explicit '+' (so '+X' vs '-X' line up in tables)."""
    assert _lrs._format_delta({"delta_pct": 1.234}) == "+1.234pp"


def test_format_delta_negative_keeps_sign():
    assert _lrs._format_delta({"delta_pct": -14.766}) == "-14.766pp"


def test_format_delta_zero_is_signed_positive():
    """0.0 formats as '+0.000pp' under '+.3f' — load-bearing because a
    GRASP-LRU regression that flips zero-to-positive must not be hidden
    by an unsigned 0."""
    assert _lrs._format_delta({"delta_pct": 0.0}) == "+0.000pp"


def test_format_delta_rounds_to_three_decimals():
    """Renderer uses {:+.3f}; 4th decimal truncates with banker rounding."""
    assert _lrs._format_delta({"delta_pct": 1.23456}) == "+1.235pp"


# ---------------------------------------------------------------------------
# Group 3 — _expected_window assembly
# ---------------------------------------------------------------------------


def test_expected_window_sign_only():
    assert _lrs._expected_window({"expected_sign": "-"}) == "sign=-"


def test_expected_window_min_and_max():
    entry = {"expected_sign": "-", "min_abs_delta_pct": 1.0, "max_abs_delta_pct": 5.0}
    assert _lrs._expected_window(entry) == "sign=-, |Δ|∈[1.00,5.00]pp"


def test_expected_window_max_only():
    entry = {"expected_sign": "+", "max_abs_delta_pct": 0.5}
    assert _lrs._expected_window(entry) == "sign=+, |Δ|≤0.50pp"


def test_expected_window_min_only():
    """min-only branch — used by SMALL_CACHE-style 'must be at least' rules."""
    entry = {"expected_sign": "+", "min_abs_delta_pct": 2.0}
    assert _lrs._expected_window(entry) == "sign=+, |Δ|≥2.00pp"


def test_expected_window_with_tolerance():
    entry = {"expected_sign": "-", "max_abs_delta_pct": 5.0, "tolerance_pct": 0.5}
    assert _lrs._expected_window(entry) == "sign=-, |Δ|≤5.00pp, ±0.50pp"


def test_expected_window_empty_renders_em_dash():
    """All-empty entry must render '—' (not empty string) so renderer
    output stays grep-able and never produces an empty cell."""
    assert _lrs._expected_window({}) == "—"


def test_expected_window_sign_field_missing_string():
    """`.get('expected_sign', '')` returns '' which is falsy — sign part
    is skipped without raising."""
    entry = {"max_abs_delta_pct": 0.5}
    assert _lrs._expected_window(entry) == "|Δ|≤0.50pp"


# ---------------------------------------------------------------------------
# Group 4 — _key citation ordering
# ---------------------------------------------------------------------------


def test_key_orders_faldu_fig_before_faldu_section():
    """CITATION_ORDER_PREFIX puts 'Faldu et al. HPCA 2020 Fig' before
    'Faldu et al. HPCA 2020 §' so figure citations sort first."""
    fig = _lrs._key("Faldu et al. HPCA 2020 Fig 10")
    sec = _lrs._key("Faldu et al. HPCA 2020 §3.2")
    assert fig < sec
    assert fig[0] == 0
    assert sec[0] == 1


def test_key_orders_paper_citations_before_unknown():
    """Known-prefix citations get index < len(CITATION_ORDER_PREFIX);
    unknown citations get exactly len(CITATION_ORDER_PREFIX) so they
    always sort after every paper citation."""
    paper = _lrs._key("Faldu et al. HPCA 2020 Fig 10")
    unknown = _lrs._key("Random citation string")
    assert paper < unknown
    assert unknown[0] == len(_lrs.CITATION_ORDER_PREFIX)


def test_key_within_same_prefix_stable_by_name():
    """Two citations with the same prefix sort lexicographically by full
    citation string — deterministic per-row ordering inside a section."""
    a = _lrs._key("Faldu et al. HPCA 2020 Fig 10")
    b = _lrs._key("Faldu et al. HPCA 2020 Fig 12")
    assert a < b
    assert a[0] == b[0] == 0


def test_key_unknown_citations_sort_alphabetically():
    a = _lrs._key("ZZZ unknown")
    b = _lrs._key("AAA unknown")
    assert b < a


# ---------------------------------------------------------------------------
# Group 5 — _paper_name n-citation rule
# ---------------------------------------------------------------------------


def test_paper_name_faldu_single():
    assert _lrs._paper_name("Faldu et al. HPCA 2020 Fig 10") == "Faldu HPCA20"


def test_paper_name_balaji_single():
    assert _lrs._paper_name("Balaji & Lucia HPCA 2021 Fig 5") == "Balaji HPCA21"


def test_paper_name_jaleel_single():
    assert _lrs._paper_name("Jaleel et al. ISCA 2010 Table 3") == "Jaleel ISCA10"


def test_paper_name_cross_paper_two_citations():
    """n>=2 paper citations → 'cross-paper'. The Faldu+Balaji
    extrapolation rows are the canonical example: they don't count
    toward either paper's reproduction% on their own."""
    assert _lrs._paper_name("Faldu HPCA20 + Balaji HPCA21 extrapolation") == "cross-paper"


def test_paper_name_cross_paper_three_citations():
    assert _lrs._paper_name("Faldu + Balaji + Jaleel combined") == "cross-paper"


def test_paper_name_other_fallback():
    """Citations matching no known paper return 'other' — kept distinct
    from 'cross-paper' so the per-paper roll-up doesn't conflate
    unknown-source rows with multi-source extrapolations."""
    assert _lrs._paper_name("Anonymous workshop 2024") == "other"


def test_paper_name_substring_match_not_word_boundary():
    """Detection is substring-based: any occurrence of 'Faldu' counts.
    This is intentional — citations are free-form strings — but the
    test pins the behavior so future tightening to word-boundary is
    deliberate."""
    assert _lrs._paper_name("Falduism") == "Faldu HPCA20"


# ---------------------------------------------------------------------------
# Group 6 — _paper_rollup + render_csv / render_markdown byte parity
# ---------------------------------------------------------------------------


def _entry(citation, status, **kw):
    base = {
        "citation": citation,
        "status": status,
        "graph": kw.get("graph", "g"),
        "app": kw.get("app", "PR"),
        "l3_size": kw.get("l3_size", "1MB"),
        "policy": kw.get("policy", "GRASP"),
    }
    base.update(kw)
    return base


def test_paper_rollup_reproduction_pct_excludes_known_deviation():
    """Reproduction% = (ok + within_tolerance) / total. known_deviation
    is documented disagreement and is EXPLICITLY NOT counted as
    reproduction — this distinguishes 'we got it right' from 'we
    explicitly disagree on a documented basis'."""
    rows = [
        _entry("Faldu et al. HPCA 2020 Fig 10", "ok"),
        _entry("Faldu et al. HPCA 2020 Fig 10", "within_tolerance"),
        _entry("Faldu et al. HPCA 2020 Fig 10", "known_deviation"),
        _entry("Faldu et al. HPCA 2020 Fig 10", "disagree"),
    ]
    lines = _lrs._paper_rollup(rows)
    body = "\n".join(lines)
    assert "Faldu HPCA20" in body
    # 2 ok+wt / 4 total = 50.0%
    assert "50.0%" in body


def test_paper_rollup_paper_order_pinned():
    """paper_order is hard-coded: Faldu → Balaji → Jaleel → cross-paper
    → other. Any rearrangement here breaks the paper table layout."""
    rows = [
        _entry("Jaleel et al. ISCA 2010 Table 3", "ok"),
        _entry("Balaji & Lucia HPCA 2021 Fig 5", "ok"),
        _entry("Faldu et al. HPCA 2020 Fig 10", "ok"),
    ]
    lines = _lrs._paper_rollup(rows)
    body = "\n".join(lines)
    faldu_idx = body.index("Faldu HPCA20")
    balaji_idx = body.index("Balaji HPCA21")
    jaleel_idx = body.index("Jaleel ISCA10")
    assert faldu_idx < balaji_idx < jaleel_idx


def test_paper_rollup_skips_empty_papers():
    """Papers with zero rows are skipped — keeps the table compact."""
    rows = [_entry("Faldu et al. HPCA 2020 Fig 10", "ok")]
    lines = _lrs._paper_rollup(rows)
    body = "\n".join(lines)
    assert "Faldu HPCA20" in body
    assert "Balaji HPCA21" not in body
    assert "Jaleel ISCA10" not in body


def test_render_csv_uses_newline_line_terminator():
    """csv.DictWriter default lineterminator is '\\r\\n' which git diff
    flags as trailing whitespace on every line. The generator pins
    lineterminator='\\n' — this test catches accidental removal."""
    rows = [_entry("Faldu et al. HPCA 2020 Fig 10", "ok", delta_pct=-1.0)]
    csv_text = _lrs.render_csv(rows)
    assert "\r\n" not in csv_text
    assert csv_text.count("\n") == 2  # header + one data row


def test_render_csv_header_columns_pinned():
    """The CSV column order is part of the public contract — downstream
    notebooks index by position. Pinning here catches reorders."""
    csv_text = _lrs.render_csv([])
    header = csv_text.splitlines()[0]
    assert header == (
        "citation,graph,app,l3_size,policy,"
        "expected_sign,min_abs_delta_pct,max_abs_delta_pct,"
        "tolerance_pct,delta_pct,status"
    )


def test_render_markdown_trailing_newline():
    """render_markdown returns '\\n'.join(out) + '\\n' — exactly one
    trailing newline. Both 'no trailing newline' and 'two trailing
    newlines' break byte parity with the on-disk artifact."""
    rows = [_entry("Faldu et al. HPCA 2020 Fig 10", "ok", delta_pct=-1.0)]
    md = _lrs.render_markdown(rows)
    assert md.endswith("\n")
    assert not md.endswith("\n\n")
