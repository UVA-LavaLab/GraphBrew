"""Derivation-parity gate for the gem5-anchor summary generator
(GAS-Der) — locks the load-bearing predicates of
:func:`scripts.experiments.ecg.gem5_anchor_summary.evaluate_invariants`,
the three pinned tolerance constants (HEADLINE_MAX_GRASP_OVER_LRU_PP,
ASYMPTOTE_MAX_SPREAD_PCT, SMALL_CACHE_MIN_SPREAD_PP), the
`_pick_canonical_section` rule (same as literature_faithfulness), the
`_l3_sort_key` byte conversion, and the JSON write rule.

Why this exists
---------------
gem5_anchor.json and sniper_anchor.json are the two cycle-accurate
"reality checks" on the cache_sim sweep. The summary generator's
predicates encode load-bearing claims of the GRASP paper:

* HEADLINE_L3 = "256kB"  — GRASP ≤ LRU at L3=256kB (PR/BC regime).
  Tolerance HEADLINE_MAX_GRASP_OVER_LRU_PP = 0.45 pp was tightened
  from 0.5 once both gem5 + Sniper anchors landed; the worst
  observed |GRASP - LRU| margin was 0.328 pp (Sniper / pr / 256 kB).
  Leaves 0.122 pp slack. A regression past 0.45 must be SEEN.
* ASYMPTOTE_L3 = "2MB" — spread(GRASP,LRU,SRRIP) ≤ 1.0 pp at L3=2MB.
  All three converge into the asymptote regime once cache is big
  enough to hold the working set.
* SMALL_CACHE_L3 = "4kB" — spread ≥ 2.0 pp at L3=4kB (L-shape).
  Policies must NOT converge below capacity; a flat-at-4kB result
  would signal that policies stopped differentiating in the
  high-pressure regime — a behavioural regression.
* "no_error_rows" — zero error rows across all cells.

Drift in any of these constants or in the inequality sense of any
predicate would silently invalidate every gem5/Sniper anchor result
in the paper. The existing gem5-anchor + sniper-anchor pytest gates
test the *artifact* values, not the *predicates*; this new gate
locks the predicates themselves.

Test groups (28 tests, 6 groups):

* group 1 (5 tests) — module-level pinned constants
  (HEADLINE_L3, ASYMPTOTE_L3, SMALL_CACHE_L3, tolerance values,
  DEFAULT_SWEEP_ROOT/DEFAULT_SUBDIR).
* group 2 (4 tests) — top-level artifact shape (7 keys, counts
  field, sniper_anchor parity).
* group 3 (4 tests) — JSON write rule (indent=2, NO sort_keys,
  trailing newline) for both gem5 and sniper.
* group 4 (4 tests) — `_pick_canonical_section` (smallest non-zero
  section else first row; empty list → None; section coercion via
  `int(...)`).
* group 5 (4 tests) — `_l3_sort_key` byte conversion (kB / MB / B
  / GB unit dispatch; unknown unit → 0).
* group 6 (7 tests) — `evaluate_invariants` predicate parity:
  HEADLINE delta_pp inequality sense (≤ tolerance → ok);
  ASYMPTOTE spread inequality sense (≤ tolerance → ok);
  SMALL_CACHE spread inequality sense (≥ minimum → ok, STRICT
  below → disagree); missing policy triggers 'missing' (not
  'disagree'); no_error_rows totals error_rows across all cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from scripts.experiments.ecg import gem5_anchor_summary as gas  # noqa: E402


DATA = REPO_ROOT / "wiki" / "data"
GEM5_ARTIFACT = DATA / "gem5_anchor.json"
SNIPER_ARTIFACT = DATA / "sniper_anchor.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gem5_artifact() -> dict:
    return json.loads(GEM5_ARTIFACT.read_text())


@pytest.fixture(scope="module")
def sniper_artifact() -> dict:
    return json.loads(SNIPER_ARTIFACT.read_text())


def _cell(graph="email-Eu-core", app="pr", l3="256kB",
          miss_rate_by_policy=None, ok_rows=6, error_rows=0):
    c = gas.CellSummary(graph=graph, app=app, l3_size=l3)
    if miss_rate_by_policy:
        c.miss_rate_by_policy.update(miss_rate_by_policy)
    c.ok_rows = ok_rows
    c.error_rows = error_rows
    return c


# ---------------------------------------------------------------------------
# Group 1 — pinned constants (5 tests)
# ---------------------------------------------------------------------------


def test_headline_l3_pinned():
    """L3 = 256kB is the GRASP paper's PR/BC headline regime.
    Changing this would invalidate the paper's headline claim cell."""
    assert gas.HEADLINE_L3 == "256kB"


def test_asymptote_l3_pinned():
    """L3 = 2MB is the asymptote regime where all three policies
    converge. Changing this re-bases the asymptote invariant."""
    assert gas.ASYMPTOTE_L3 == "2MB"


def test_small_cache_l3_pinned():
    """L3 = 4kB is the L-shape companion: policies MUST diverge at
    this size. Loosening upward would dilute the L-shape signal."""
    assert gas.SMALL_CACHE_L3 == "4kB"


def test_tolerance_constants_pinned():
    """Three tolerance constants the paper depends on.
    HEADLINE_MAX_GRASP_OVER_LRU_PP was tightened 0.5→0.45 once both
    anchors landed (worst |Δ|=0.328 pp at Sniper/pr/256kB). Loosening
    must be intentional, never a typo."""
    assert gas.HEADLINE_MAX_GRASP_OVER_LRU_PP == 0.45
    assert gas.ASYMPTOTE_MAX_SPREAD_PCT == 1.0
    assert gas.SMALL_CACHE_MIN_SPREAD_PP == 2.0


def test_default_sweep_root_pinned():
    """Default gem5 sweep root + DBG subdir; CLI overrides these but
    the defaults document the canonical layout."""
    assert str(gas.DEFAULT_SWEEP_ROOT) == "/tmp/graphbrew-grasp-gem5-sweep"
    assert gas.DEFAULT_SUBDIR == "DBG"


# ---------------------------------------------------------------------------
# Group 2 — artifact top-level shape (4 tests)
# ---------------------------------------------------------------------------


def test_gem5_top_level_keys_pinned(gem5_artifact):
    """Seven top-level keys; no field drifts in. Shared shape with
    sniper_anchor (the two artifacts are produced by the SAME
    generator with different sweep roots)."""
    assert set(gem5_artifact.keys()) == {
        "apps_scope", "cells", "counts", "graphs_scope",
        "invariants", "sweep_root", "sweep_subdir",
    }


def test_sniper_top_level_keys_match_gem5(gem5_artifact, sniper_artifact):
    """Sniper anchor's top-level keys EXACTLY equal gem5 anchor's
    (single generator, both invocations). A drift indicates someone
    edited one anchor path's call site without the other."""
    assert set(sniper_artifact.keys()) == set(gem5_artifact.keys())


def test_counts_field_set_pinned(gem5_artifact):
    """counts has 4 fields: cells, invariants_ok, invariants_disagree,
    invariants_missing. These mirror the AnchorInvariant status
    vocabulary {ok / disagree / missing}."""
    assert set(gem5_artifact["counts"].keys()) == {
        "cells", "invariants_ok", "invariants_disagree", "invariants_missing",
    }


def test_invariant_status_vocabulary_pinned(gem5_artifact, sniper_artifact):
    """Every invariant.status is in the 3-element documented set.
    A new status leaking in here would unbalance the counts rollup."""
    allowed = {"ok", "disagree", "missing"}
    for art in (gem5_artifact, sniper_artifact):
        for inv in art["invariants"]:
            assert inv["status"] in allowed


# ---------------------------------------------------------------------------
# Group 3 — JSON write rule (4 tests)
# ---------------------------------------------------------------------------


def test_gem5_json_byte_parity_indent2_no_sort_keys_trailing_newline(gem5_artifact):
    """JSON write rule (line 343): json.dumps(payload, indent=2) +
    '\\n'. No sort_keys (insertion order matters for diff stability),
    trailing newline appended explicitly. Differs from gates 200/201
    which use sort_keys=True and no newline."""
    raw = GEM5_ARTIFACT.read_text()
    expected = json.dumps(gem5_artifact, indent=2) + "\n"
    assert raw == expected


def test_sniper_json_byte_parity_indent2_no_sort_keys_trailing_newline(sniper_artifact):
    """Same rule for sniper_anchor (single generator)."""
    raw = SNIPER_ARTIFACT.read_text()
    expected = json.dumps(sniper_artifact, indent=2) + "\n"
    assert raw == expected


def test_gem5_artifact_ends_with_newline():
    """Trailing-newline rule explicit (overrides Python json's default)."""
    raw = GEM5_ARTIFACT.read_text()
    assert raw.endswith("\n")


def test_gem5_artifact_does_not_use_sort_keys(gem5_artifact):
    """Negative parity check: sort_keys=True would alphabetise the
    top-level keys; the on-disk byte stream must NOT match that."""
    raw = GEM5_ARTIFACT.read_text()
    sorted_form = json.dumps(gem5_artifact, indent=2, sort_keys=True) + "\n"
    if set(gem5_artifact.keys()) != sorted(gem5_artifact.keys()):
        # Only meaningful when the natural insertion order differs
        # from sorted order; otherwise both representations coincide.
        assert raw != sorted_form


# ---------------------------------------------------------------------------
# Group 4 — _pick_canonical_section (4 tests)
# ---------------------------------------------------------------------------


def test_pick_canonical_section_prefers_smallest_nonzero():
    """Mirrors literature_faithfulness / sign_consistency: smallest
    non-zero section wins. Two comparators must agree on which ROI
    gem5 emitted when multiples exist."""
    rows = [
        {"section": "0", "l3_miss_rate": "0.10"},
        {"section": "3", "l3_miss_rate": "0.30"},
        {"section": "1", "l3_miss_rate": "0.20"},
        {"section": "2", "l3_miss_rate": "0.40"},
    ]
    picked = gas._pick_canonical_section(rows)
    assert picked["section"] == "1"
    assert picked["l3_miss_rate"] == "0.20"


def test_pick_canonical_section_falls_back_to_first_when_all_zero():
    """All rows section==0 → return first row (cache_sim path)."""
    rows = [
        {"section": "0", "l3_miss_rate": "0.10"},
        {"section": "0", "l3_miss_rate": "0.20"},
    ]
    picked = gas._pick_canonical_section(rows)
    assert picked["l3_miss_rate"] == "0.10"


def test_pick_canonical_section_empty_returns_none():
    """Empty input → None. The caller's loop SKIPS None to avoid
    blowing up on a fully-erroneous (graph, app, l3, policy) key."""
    assert gas._pick_canonical_section([]) is None


def test_pick_canonical_section_handles_none_or_missing_section_field():
    """A row with section=None or missing should coerce to 0 (the
    `int(r.get('section') or 0)` predicate). The row then participates
    in the zero/nonzero partition normally."""
    rows = [
        {"section": None, "l3_miss_rate": "0.10"},
        {"section": "2", "l3_miss_rate": "0.20"},
    ]
    picked = gas._pick_canonical_section(rows)
    # section=2 is the smallest non-zero
    assert picked["section"] == "2"


# ---------------------------------------------------------------------------
# Group 5 — _l3_sort_key byte conversion (4 tests)
# ---------------------------------------------------------------------------


def test_l3_sort_key_kb_conversion():
    """kB → ×1024."""
    assert gas._l3_sort_key("256kB") == 256 * 1024
    assert gas._l3_sort_key("4kB") == 4 * 1024
    assert gas._l3_sort_key("1kB") == 1024


def test_l3_sort_key_mb_conversion():
    """MB → ×1024×1024."""
    assert gas._l3_sort_key("2MB") == 2 * 1024 * 1024
    assert gas._l3_sort_key("1MB") == 1024 * 1024


def test_l3_sort_key_b_and_gb_units():
    """B → ×1. NOTE: GB falls back to 0 because the units dict
    iteration order tests 'B' before 'GB' and `endswith('B')` matches
    '1GB' first, then `int(float('1G'))` raises ValueError. This is a
    quirk of the unit table (no L3 size in any sweep uses 'B' or 'GB'
    standalone, only 'kB' / 'MB'), but documenting it locks the
    actual behavior so future fixes know what they're changing."""
    assert gas._l3_sort_key("4096B") == 4096
    assert gas._l3_sort_key("1GB") == 0  # known quirk: 'B' shadows 'GB'


def test_l3_sort_key_unknown_unit_returns_zero():
    """Unknown unit → 0 (defensive: never raise, just sort to the
    top of the table where the operator will notice)."""
    assert gas._l3_sort_key("invalid") == 0
    assert gas._l3_sort_key("") == 0
    assert gas._l3_sort_key("256kib") == 0  # case-sensitive


# ---------------------------------------------------------------------------
# Group 6 — evaluate_invariants predicate parity (7 tests)
# ---------------------------------------------------------------------------


def test_headline_invariant_ok_when_grasp_le_lru_within_tolerance():
    """grasp - lru ≤ 0.45 pp → ok. Predicate is
    `(grasp - lru) * 100 <= HEADLINE_MAX_GRASP_OVER_LRU_PP`
    (INCLUSIVE)."""
    cells = [_cell(l3="256kB",
                   miss_rate_by_policy={"GRASP": 0.1100, "LRU": 0.1080,
                                        "SRRIP": 0.1200})]
    # delta = (0.1100 - 0.1080) * 100 = 0.2 pp ≤ 0.45 → ok
    invs = gas.evaluate_invariants(cells, apps=("pr",), graphs=("email-Eu-core",))
    head = [i for i in invs if "GRASP_LE_LRU_headline" in i.name]
    assert head[0].status == "ok"


def test_headline_invariant_disagree_when_grasp_exceeds_tolerance():
    """grasp - lru > 0.45 pp → disagree. Predicate sense is STRICT >."""
    cells = [_cell(l3="256kB",
                   miss_rate_by_policy={"GRASP": 0.1100, "LRU": 0.1050,
                                        "SRRIP": 0.1200})]
    # delta = 0.5 pp > 0.45 → disagree
    invs = gas.evaluate_invariants(cells, apps=("pr",), graphs=("email-Eu-core",))
    head = [i for i in invs if "GRASP_LE_LRU_headline" in i.name]
    assert head[0].status == "disagree"


def test_asymptote_invariant_ok_when_spread_within_tolerance():
    """spread(GRASP,LRU,SRRIP) ≤ 1.0 pp at L3=2MB → ok. Predicate is
    `(max - min) * 100 <= 1.0` (INCLUSIVE)."""
    cells = [_cell(l3="2MB",
                   miss_rate_by_policy={"GRASP": 0.0500, "LRU": 0.0505,
                                        "SRRIP": 0.0510})]
    # spread = 0.10 pp ≤ 1.0 → ok
    invs = gas.evaluate_invariants(cells, apps=("pr",), graphs=("email-Eu-core",))
    asy = [i for i in invs if "asymptote_within" in i.name]
    assert asy[0].status == "ok"


def test_small_cache_invariant_ok_when_spread_meets_minimum():
    """spread ≥ 2.0 pp at L3=4kB → ok (L-shape holds). Predicate is
    `spread >= SMALL_CACHE_MIN_SPREAD_PP` (INCLUSIVE)."""
    cells = [_cell(l3="4kB",
                   miss_rate_by_policy={"GRASP": 0.40, "LRU": 0.45,
                                        "SRRIP": 0.50})]
    # spread = 10 pp ≥ 2 → ok
    invs = gas.evaluate_invariants(cells, apps=("pr",), graphs=("email-Eu-core",))
    sml = [i for i in invs if "small_cache_divergence" in i.name]
    assert sml[0].status == "ok"


def test_small_cache_invariant_disagree_when_policies_converge():
    """spread < 2.0 pp at L3=4kB → disagree (L-shape broken). Below the
    threshold, the predicate is STRICTLY <; tests the inverse polarity
    from the asymptote case to ensure no copy-paste flip happened."""
    cells = [_cell(l3="4kB",
                   miss_rate_by_policy={"GRASP": 0.500, "LRU": 0.505,
                                        "SRRIP": 0.510})]
    # spread = 1.0 pp < 2.0 → disagree (L-shape broken)
    invs = gas.evaluate_invariants(cells, apps=("pr",), graphs=("email-Eu-core",))
    sml = [i for i in invs if "small_cache_divergence" in i.name]
    assert sml[0].status == "disagree"


def test_missing_policy_triggers_missing_not_disagree():
    """When a required policy (GRASP/LRU/SRRIP) is absent at L3=256kB
    or 2MB, the invariant must be 'missing', NOT 'disagree'. This is
    the critical anchor for distinguishing 'sweep didn't produce
    data' from 'sweep produced data that disagrees with the paper'."""
    cells = [
        _cell(l3="256kB", miss_rate_by_policy={"LRU": 0.10, "SRRIP": 0.11}),
        _cell(l3="2MB", miss_rate_by_policy={"GRASP": 0.05, "LRU": 0.05}),
    ]
    invs = gas.evaluate_invariants(cells, apps=("pr",), graphs=("email-Eu-core",))
    head = [i for i in invs if "GRASP_LE_LRU_headline" in i.name]
    asy = [i for i in invs if "asymptote_within" in i.name]
    assert head[0].status == "missing"
    assert asy[0].status == "missing"


def test_no_error_rows_invariant_sums_across_all_cells():
    """no_error_rows invariant: total error_rows == 0 across ALL cells
    → ok; any error → disagree with count in detail. Predicate is
    SUM, not any-cell — a single error in any cell trips the gate."""
    cells_clean = [_cell(error_rows=0), _cell(app="bc", error_rows=0)]
    cells_dirty = [_cell(error_rows=0), _cell(app="bc", error_rows=1)]

    invs_clean = gas.evaluate_invariants(cells_clean, apps=("pr", "bc"),
                                         graphs=("email-Eu-core",))
    invs_dirty = gas.evaluate_invariants(cells_dirty, apps=("pr", "bc"),
                                         graphs=("email-Eu-core",))

    err_clean = [i for i in invs_clean if i.name == "no_error_rows"]
    err_dirty = [i for i in invs_dirty if i.name == "no_error_rows"]
    assert err_clean[0].status == "ok"
    assert err_dirty[0].status == "disagree"
    assert "1 error rows" in err_dirty[0].detail
