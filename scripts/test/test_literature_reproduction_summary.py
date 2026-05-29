"""Tests for scripts/experiments/ecg/literature_reproduction_summary.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO_ROOT / "scripts" / "experiments" / "ecg" / "literature_reproduction_summary.py"
LIT_JSON = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"


def _load():
    spec = importlib.util.spec_from_file_location("lit_reproduction_summary", SUMMARY_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


def _entry(citation="X et al. 2020 Fig 1", status="ok", **kw):
    base = {
        "graph": "g", "app": "pr", "l3_size": "1MB", "policy": "GRASP",
        "expected_sign": "-", "min_abs_delta_pct": 1.0, "max_abs_delta_pct": 10.0,
        "tolerance_pct": 1.0, "delta_pct": -5.0, "status": status,
        "citation": citation,
    }
    base.update(kw)
    return base


def test_format_delta(mod):
    assert mod._format_delta({"delta_pct": -3.456}) == "-3.456pp"
    assert mod._format_delta({"delta_pct": 0.0}) == "+0.000pp"
    assert mod._format_delta({"delta_pct": None}) == "—"
    assert mod._format_delta({}) == "—"


def test_expected_window_full(mod):
    e = _entry()
    out = mod._expected_window(e)
    assert "sign=-" in out
    assert "Δ" in out and "1.00" in out and "10.00" in out
    assert "±1.00pp" in out


def test_expected_window_open_ended(mod):
    e = _entry(min_abs_delta_pct=None, max_abs_delta_pct=5.0)
    out = mod._expected_window(e)
    assert "|Δ|≤5.00pp" in out


def test_verdict_glyph(mod):
    assert mod._verdict_glyph("ok") == "✅"
    assert mod._verdict_glyph("known_deviation") == "📘"
    assert mod._verdict_glyph("disagree") == "⛔"
    assert mod._verdict_glyph("missing") == "⏳"


def test_render_markdown_groups_by_citation(mod):
    per_claim = [
        _entry(citation="Faldu et al. HPCA 2020 Fig 10", graph="web-Google"),
        _entry(citation="Faldu et al. HPCA 2020 Fig 10", graph="soc-pokec"),
        _entry(citation="Balaji & Lucia HPCA 2021 Fig 9", graph="cit-Patents", policy="POPT"),
    ]
    md = mod.render_markdown(per_claim)
    assert "## Faldu et al. HPCA 2020 Fig 10" in md
    assert "## Balaji & Lucia HPCA 2021 Fig 9" in md
    assert "Per-citation roll-up" in md
    # Citation ordering: Faldu paper-rooted before Balaji.
    assert md.index("Faldu et al. HPCA 2020 Fig 10") < md.index("Balaji & Lucia HPCA 2021 Fig 9")


def test_csv_has_fixed_header(mod):
    per_claim = [_entry()]
    csv_text = mod.render_csv(per_claim)
    header = csv_text.splitlines()[0]
    for col in ("citation", "graph", "app", "l3_size", "policy",
                "expected_sign", "delta_pct", "status"):
        assert col in header, f"missing {col}"


@pytest.mark.skipif(not LIT_JSON.exists(), reason="lit-faith JSON not produced yet")
def test_real_input_renders_without_error(mod):
    """Parity check: the real JSON should render cleanly."""
    raw = json.loads(LIT_JSON.read_text())
    per_claim = raw.get("per_claim", [])
    if not per_claim:
        pytest.skip("empty per_claim in lit-faith JSON")
    md = mod.render_markdown(per_claim)
    assert "# Literature reproduction summary" in md
    # Every citation in the JSON must appear in the output.
    cites = {e["citation"] for e in per_claim}
    for c in cites:
        assert c in md, f"citation missing: {c}"


def test_paper_name_classifies_single_paper_citations(mod):
    assert mod._paper_name("Faldu et al. HPCA 2020 Fig 10") == "Faldu HPCA20"
    assert mod._paper_name("Balaji & Lucia HPCA 2021 §6.3") == "Balaji HPCA21"
    assert mod._paper_name("Jaleel et al. ISCA 2010 §5.2") == "Jaleel ISCA10"


def test_paper_name_classifies_cross_paper_citations(mod):
    assert mod._paper_name(
        "Faldu HPCA20 §6.1 + Balaji HPCA21 Fig 9 cross-check"
    ) == "cross-paper"
    assert mod._paper_name(
        "Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1"
    ) == "cross-paper"


def test_paper_rollup_strict_reproduction_percentage(mod):
    rows = [
        _entry(citation="Faldu et al. HPCA 2020 Fig 10", status="ok"),
        _entry(citation="Faldu et al. HPCA 2020 Fig 10", status="ok"),
        _entry(citation="Faldu et al. HPCA 2020 Fig 11", status="within_tolerance"),
        _entry(citation="Faldu et al. HPCA 2020 Fig 11", status="known_deviation"),
    ]
    lines = mod._paper_rollup(rows)
    md = "\n".join(lines)
    assert "## Per-paper headline" in md
    # 4 cells: 2 ok + 1 within_tol = 3 reproduced, 1 known_dev not counted.
    assert "Faldu HPCA20 | 4 | 2 | 1 | 1" in md
    assert "75.0%" in md


def test_paper_rollup_known_deviation_excluded_from_reproduction(mod):
    """A paper whose every cell is known_deviation reproduces 0%, not 100%."""
    rows = [_entry(citation="Balaji & Lucia HPCA 2021 §6.3",
                   status="known_deviation") for _ in range(3)]
    md = "\n".join(mod._paper_rollup(rows))
    assert "Balaji HPCA21 | 3 | 0 | 0 | 3" in md
    assert "0.0%" in md


@pytest.mark.skipif(not LIT_JSON.exists(), reason="lit-faith JSON not produced yet")
def test_real_input_per_paper_headline_has_every_paper(mod):
    """All three cited papers must appear in the per-paper headline."""
    raw = json.loads(LIT_JSON.read_text())
    per_claim = raw.get("per_claim", [])
    if not per_claim:
        pytest.skip("empty per_claim in lit-faith JSON")
    md = mod.render_markdown(per_claim)
    # Per-paper headline must precede the per-citation roll-up.
    assert md.index("Per-paper headline") < md.index("Per-citation roll-up")
    for paper in ("Faldu HPCA20", "Balaji HPCA21", "Jaleel ISCA10"):
        assert paper in md, f"per-paper roll-up missing {paper}"
