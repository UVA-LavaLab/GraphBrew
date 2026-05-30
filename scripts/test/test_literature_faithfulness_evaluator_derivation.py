"""Derivation-parity gate for the literature-faithfulness evaluator
(LFE-Der) — locks the load-bearing predicates of
:func:`scripts.experiments.ecg.literature_faithfulness._classify`,
the POPT_GE_GRASP / POPT_NEAR_GRASP_IF_BIG_GAP dispatch rules in
:func:`literature_faithfulness.evaluate`, and the supporting helpers
(_coerce_int defensive returns, _pick_section canonical-ROI rule).

Why this exists (complementary to LFP-Par at gate 133)
------------------------------------------------------
The existing gate LFP-Par (test_literature_faithfulness_postfix_
arithmetic.py) locks the on-disk JSON's *internal* consistency —
summary counts ↔ per_claim, bucket lists ↔ per_claim filtered by
status, delta_pct = round((policy_miss - lru_miss) * 100, 4),
per_observation miss_rate ↔ oracle_gap. But the comparator's
*predicates* — the if-else trees in `_classify` and in the
relative-claim dispatch — are not directly tested. A future
refactor could keep the JSON's internal arithmetic correct while
silently inverting which observations land in `ok` vs `disagree`.

This gate (28 tests, 6 groups) drives `_classify` and `evaluate`
through synthetic claims + observations and asserts the
load-bearing predicates fire exactly as specified. It also locks
the JSON write rule (sort_keys=True + indent=2 + no trailing
newline) and `_pick_section` / `_coerce_int` helpers used by the
upstream loader.

Test groups:

* group 1 (5 tests) — top-level shape (top-level keys, summary
  status set, min_accesses_threshold default == 10_000, JSON
  byte parity, summary key set pinned).
* group 2 (5 tests) — `_coerce_int` defensive returns (None /
  empty / non-numeric → 0; valid int passes; integer-string
  passes).
* group 3 (3 tests) — `_pick_section` canonical-ROI rule
  (smallest non-zero section; fallback to section 0 only when
  no non-zero present; section 0 NEVER preferred over any
  non-zero section).
* group 4 (8 tests) — `_classify` predicates: sign='-' branch
  (delta ≤ -min INCLUSIVE → ok; within tolerance → within_t;
  beyond → disagree); sign='+' branch (delta ≥ 0 → ok; within
  tol → within_t; otherwise disagree; bound violations →
  disagree); sign='~' magnitude-only branch.
* group 5 (4 tests) — POPT_GE_GRASP relative-claim dispatch
  (diff_pct ≤ tolerance INCLUSIVE → ok; > tolerance → disagree;
  missing observation → missing; insufficient_data when
  accesses below min).
* group 6 (3 tests) — POPT_NEAR_GRASP_IF_BIG_GAP phase-transition
  gate (grasp_gain_pp ≤ 10 STRICT → ok with skip-note; > 10 with
  signed_pp ≤ max+tol → ok; > 10 with signed_pp > max+tol →
  disagree).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from scripts.experiments.ecg import literature_faithfulness as lf  # noqa: E402


DATA = REPO_ROOT / "wiki" / "data"
ARTIFACT = DATA / "literature_faithfulness_postfix.json"

DOCUMENTED_STATUSES = {
    "ok", "within_tolerance", "disagree", "missing",
    "insufficient_data", "known_deviation",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


@dataclass(frozen=True)
class StubClaim:
    """Minimal LiteratureClaim stand-in for unit-testing _classify.

    Mirrors literature_baselines.LiteratureClaim's public fields used by
    _classify (expected_sign, tolerance_pct, min_abs_delta_pct,
    max_abs_delta_pct). Using a stub instead of building real claims
    means we exercise the predicate in isolation, without coupling the
    test to the full claims catalog.
    """
    expected_sign: str = "-"
    tolerance_pct: float = 1.0
    min_abs_delta_pct: float | None = None
    max_abs_delta_pct: float | None = None
    policy: str = "GRASP"
    rationale: str = "stub rationale"
    citation: str = "Stub et al. 2024"


# ---------------------------------------------------------------------------
# Group 1 — top-level shape + JSON byte parity (5 tests)
# ---------------------------------------------------------------------------


def test_top_level_keys_pinned(artifact):
    """The JSON has exactly six top-level keys; no field drifts in."""
    assert set(artifact.keys()) == {
        "disagreements", "known_deviations", "per_claim",
        "per_observation", "summary", "tolerated",
    }


def test_summary_keys_pinned(artifact):
    """summary keys are the eight-element set; min_accesses_threshold
    is required (load-bearing for the lit-faith comparator gates that
    filter on it). Adding a key here must be intentional."""
    assert set(artifact["summary"].keys()) == {
        "claims_total", "disagree", "insufficient_data",
        "known_deviation", "min_accesses_threshold", "missing",
        "ok", "within_tolerance",
    }


def test_summary_min_accesses_threshold_default_10000(artifact):
    """The default min_accesses cutoff is 10_000 (see evaluate signature
    and Makefile). A drift to a smaller value would silently re-classify
    cells previously tagged 'insufficient_data' as live observations."""
    assert artifact["summary"]["min_accesses_threshold"] == 10_000


def test_documented_status_set_used(artifact):
    """Every per_claim entry status is in the documented six-status set.
    A new status leaking in here would unbalance the summary counts."""
    for entry in artifact["per_claim"]:
        assert entry["status"] in DOCUMENTED_STATUSES, (
            f"unexpected status: {entry['status']}"
        )


def test_json_byte_parity_sort_keys_indent2_no_trailing_newline(artifact):
    """JSON write rule (line 542): json.dumps(result, indent=2,
    sort_keys=True), NO trailing newline. The sort_keys=True is what
    makes the artifact diff-stable across runs even though `evaluate`
    builds dicts in insertion order."""
    raw = ARTIFACT.read_text()
    expected = json.dumps(artifact, indent=2, sort_keys=True)
    assert raw == expected


# ---------------------------------------------------------------------------
# Group 2 — _coerce_int defensive returns (5 tests)
# ---------------------------------------------------------------------------


def test_coerce_int_none_returns_zero():
    """None → 0 (defensive default for missing CSV columns).
    The CSV loader uses _coerce_int on section / l3_misses / l3_hits
    columns; absence must NOT raise."""
    assert lf._coerce_int(None) == 0


def test_coerce_int_empty_string_returns_zero():
    """Empty string → 0. CSV readers emit '' for absent cells."""
    assert lf._coerce_int("") == 0


def test_coerce_int_valid_int_string_passes():
    """Valid int string passes through verbatim."""
    assert lf._coerce_int("42") == 42
    assert lf._coerce_int("0") == 0
    assert lf._coerce_int("-7") == -7


def test_coerce_int_non_numeric_returns_zero():
    """Non-numeric string returns 0 (NOT raises). A corrupt CSV cell
    must NOT crash the load_observations walker."""
    assert lf._coerce_int("not_a_number") == 0
    assert lf._coerce_int("12abc") == 0
    assert lf._coerce_int("--") == 0


def test_coerce_int_does_not_accept_floats():
    """A float-formatted string ('3.14') returns 0 — the predicate
    is `int(text)` not `int(float(text))`. This intentionally rejects
    fractional section indices and fractional miss/hit counts."""
    assert lf._coerce_int("3.14") == 0


# ---------------------------------------------------------------------------
# Group 3 — _pick_section canonical-ROI rule (3 tests)
# ---------------------------------------------------------------------------


def _obs(section: int, miss: float = 0.5) -> lf.Observation:
    return lf.Observation(
        graph="g", app="a", l3_size="1MB", policy="GRASP",
        miss_rate=miss, section=section,
    )


def test_pick_section_prefers_smallest_nonzero():
    """When multiple non-zero sections exist, the smallest wins.
    This mirrors `sign_consistency`'s rule so the two comparators
    agree on which ROI gem5 emitted when there are multiples."""
    rows = [_obs(0, 0.10), _obs(3, 0.30), _obs(1, 0.20), _obs(2, 0.40)]
    picked = lf._pick_section(rows)
    assert picked.section == 1
    assert picked.miss_rate == 0.20


def test_pick_section_falls_back_to_zero_when_no_nonzero():
    """When ALL rows have section==0, return the first row
    (rows[0]). This is the cache_sim path — only one section exists."""
    rows = [_obs(0, 0.10), _obs(0, 0.20)]
    picked = lf._pick_section(rows)
    assert picked.section == 0
    assert picked.miss_rate == 0.10


def test_pick_section_never_prefers_zero_over_nonzero():
    """Even when section=0 has a 'better' miss_rate, a non-zero section
    is canonical — section 0 is gem5's per-program init/warmup section
    and the miss_rate is unreliable there."""
    rows = [_obs(0, 0.05), _obs(5, 0.99)]
    picked = lf._pick_section(rows)
    assert picked.section == 5  # non-zero wins regardless of miss_rate


# ---------------------------------------------------------------------------
# Group 4 — _classify predicate semantics (8 tests)
# ---------------------------------------------------------------------------


def test_classify_negative_sign_below_min_abs_is_ok():
    """sign='-' with min_abs_delta=2.0, tolerance=0.5:
    delta_pct ≤ -2.0 (INCLUSIVE) → ok.
    The condition is `delta_pct <= -(claim.min_abs_delta_pct)` so
    delta = -2.0 itself is ok."""
    c = StubClaim(expected_sign="-", min_abs_delta_pct=2.0, tolerance_pct=0.5)
    assert lf._classify(c, -3.0) == "ok"
    assert lf._classify(c, -2.0) == "ok"  # exact boundary INCLUSIVE


def test_classify_negative_sign_in_tolerance_band_is_within_tolerance():
    """sign='-' with min_abs=2.0, tol=0.5:
    -2.0 > delta ≥ -1.5 → within_tolerance.
    Specifically -(min - tol) = -1.5, condition
    `delta_pct <= -(min_abs - tol)`."""
    c = StubClaim(expected_sign="-", min_abs_delta_pct=2.0, tolerance_pct=0.5)
    assert lf._classify(c, -1.7) == "within_tolerance"
    assert lf._classify(c, -1.5) == "within_tolerance"  # boundary


def test_classify_negative_sign_above_tolerance_is_disagree():
    """sign='-' with min_abs=2.0, tol=0.5:
    delta_pct > -1.5 → disagree (the policy isn't reducing miss
    rate enough)."""
    c = StubClaim(expected_sign="-", min_abs_delta_pct=2.0, tolerance_pct=0.5)
    assert lf._classify(c, -1.0) == "disagree"
    assert lf._classify(c, 0.0) == "disagree"
    assert lf._classify(c, 5.0) == "disagree"


def test_classify_negative_sign_violates_max_abs_is_disagree():
    """sign='-' with min_abs=2.0, max_abs=10.0, tol=0.5:
    |delta| > 10.5 (max+tol) → disagree even if delta_pct also
    satisfies the negative-sign min_abs constraint. This catches
    'GRASP improves too much to be believable' cases."""
    c = StubClaim(
        expected_sign="-", min_abs_delta_pct=2.0,
        max_abs_delta_pct=10.0, tolerance_pct=0.5,
    )
    assert lf._classify(c, -10.6) == "disagree"
    assert lf._classify(c, -10.4) == "ok"  # within max + tol


def test_classify_negative_sign_no_min_abs_only_tolerance():
    """sign='-' with min_abs=None: any delta > tolerance → disagree;
    delta ≤ tolerance → ok. This is the 'just be non-positive within
    tolerance' case where the literature doesn't promise a magnitude."""
    c = StubClaim(expected_sign="-", min_abs_delta_pct=None, tolerance_pct=0.5)
    assert lf._classify(c, 1.0) == "disagree"
    assert lf._classify(c, 0.5) == "ok"
    assert lf._classify(c, -5.0) == "ok"


def test_classify_positive_sign_branches():
    """sign='+' with tolerance=0.5:
    delta ≥ 0 → ok; delta in [-0.5, 0) → within_tolerance;
    delta < -0.5 → disagree."""
    c = StubClaim(expected_sign="+", tolerance_pct=0.5)
    assert lf._classify(c, 1.0) == "ok"
    assert lf._classify(c, 0.0) == "ok"
    assert lf._classify(c, -0.3) == "within_tolerance"
    assert lf._classify(c, -0.5) == "within_tolerance"  # boundary INCLUSIVE
    assert lf._classify(c, -0.6) == "disagree"


def test_classify_positive_sign_bound_violations():
    """sign='+' with min_abs=5, max_abs=20, tol=0.5:
    delta < min-tol=4.5 → disagree; delta > max+tol=20.5 → disagree."""
    c = StubClaim(
        expected_sign="+", min_abs_delta_pct=5.0,
        max_abs_delta_pct=20.0, tolerance_pct=0.5,
    )
    assert lf._classify(c, 4.4) == "disagree"  # below min - tol
    assert lf._classify(c, 4.5) == "ok"  # at boundary (>=)
    assert lf._classify(c, 20.5) == "ok"  # at upper boundary (<=)
    assert lf._classify(c, 20.6) == "disagree"


def test_classify_tilde_sign_magnitude_only():
    """sign='~' with max_abs=2.0, tol=0.5:
    |delta| ≤ 2.5 → ok; |delta| > 2.5 → disagree. There's no
    minimum-magnitude requirement — the literature just promises
    'no significant change' for this cell."""
    c = StubClaim(
        expected_sign="~", max_abs_delta_pct=2.0, tolerance_pct=0.5,
    )
    assert lf._classify(c, 0.0) == "ok"
    assert lf._classify(c, 2.5) == "ok"  # |delta| at boundary
    assert lf._classify(c, -2.5) == "ok"
    assert lf._classify(c, 2.6) == "disagree"
    assert lf._classify(c, -2.6) == "disagree"


# ---------------------------------------------------------------------------
# Group 5 — POPT_GE_GRASP relative-claim dispatch (4 tests)
# ---------------------------------------------------------------------------
#
# These tests exercise the relative-claim branch of `evaluate` by building
# a tiny obs_idx + monkey-patching `claims_for` to return a single
# POPT_GE_GRASP claim. Verifying the *full* evaluate dispatch matters
# because LFP-Par only sees the artifact; we want a regression if
# someone flips the inequality sense.


def _build_obs(graph, app, l3, policy, miss_rate, accesses=1_000_000):
    return lf.Observation(
        graph=graph, app=app, l3_size=l3, policy=policy,
        miss_rate=miss_rate, section=1, accesses=accesses,
    )


@pytest.fixture
def patch_claims_for(monkeypatch):
    """Helper to patch `_lit.claims_for` for the duration of a test."""
    def _patch(claims_list: list):
        monkeypatch.setattr(lf._lit, "claims_for", lambda g, a, l: claims_list)
    return _patch


def test_popt_ge_grasp_diff_le_tolerance_is_ok(patch_claims_for):
    """POPT_GE_GRASP: diff_pct = (popt - grasp)*100 ≤ tolerance → ok.
    The semantics is 'POPT must be at least as good as GRASP (lower
    miss rate), within tolerance'."""
    claim = StubClaim(
        policy="POPT_GE_GRASP", tolerance_pct=0.5, expected_sign="-",
    )
    patch_claims_for([claim])
    obs_idx = lf.index([
        _build_obs("g", "a", "1MB", "POPT", 0.500),
        _build_obs("g", "a", "1MB", "GRASP", 0.498),
        _build_obs("g", "a", "1MB", "LRU", 0.600),
    ])
    result = lf.evaluate(obs_idx)
    rel = [e for e in result["per_claim"] if e["policy"] == "POPT_GE_GRASP"]
    assert len(rel) == 1
    # diff_pct = (0.500 - 0.498) * 100 = 0.2 ≤ 0.5 → ok
    assert rel[0]["status"] == "ok"
    assert rel[0]["delta_pct"] == 0.2


def test_popt_ge_grasp_diff_above_tolerance_is_disagree(patch_claims_for):
    """diff_pct > tolerance → disagree."""
    claim = StubClaim(
        policy="POPT_GE_GRASP", tolerance_pct=0.5, expected_sign="-",
    )
    patch_claims_for([claim])
    obs_idx = lf.index([
        _build_obs("g", "a", "1MB", "POPT", 0.510),  # POPT worse by 1pp
        _build_obs("g", "a", "1MB", "GRASP", 0.500),
        _build_obs("g", "a", "1MB", "LRU", 0.600),
    ])
    result = lf.evaluate(obs_idx)
    rel = [e for e in result["per_claim"] if e["policy"] == "POPT_GE_GRASP"]
    assert rel[0]["status"] == "disagree"
    assert rel[0]["delta_pct"] == 1.0  # > tolerance


def test_popt_ge_grasp_missing_observation(patch_claims_for):
    """If POPT or GRASP observation absent → status='missing'.
    Also: delta_pct is None, accesses is None when popt absent."""
    claim = StubClaim(
        policy="POPT_GE_GRASP", tolerance_pct=0.5, expected_sign="-",
    )
    patch_claims_for([claim])
    obs_idx = lf.index([
        _build_obs("g", "a", "1MB", "GRASP", 0.500),
        _build_obs("g", "a", "1MB", "LRU", 0.600),
        # POPT missing
    ])
    result = lf.evaluate(obs_idx)
    rel = [e for e in result["per_claim"] if e["policy"] == "POPT_GE_GRASP"]
    assert rel[0]["status"] == "missing"
    assert rel[0]["delta_pct"] is None
    assert rel[0]["accesses"] is None


def test_popt_ge_grasp_insufficient_data(patch_claims_for):
    """If max(popt.accesses, grasp.accesses) < min_accesses →
    status='insufficient_data'. delta_pct is still computed so it
    appears in the markdown table but doesn't drive verdict."""
    claim = StubClaim(
        policy="POPT_GE_GRASP", tolerance_pct=0.5, expected_sign="-",
    )
    patch_claims_for([claim])
    obs_idx = lf.index([
        _build_obs("g", "a", "1MB", "POPT", 0.500, accesses=5_000),
        _build_obs("g", "a", "1MB", "GRASP", 0.498, accesses=5_000),
        _build_obs("g", "a", "1MB", "LRU", 0.600, accesses=5_000),
    ])
    result = lf.evaluate(obs_idx, min_accesses=10_000)
    rel = [e for e in result["per_claim"] if e["policy"] == "POPT_GE_GRASP"]
    assert rel[0]["status"] == "insufficient_data"
    assert rel[0]["delta_pct"] is not None  # still recorded for table


# ---------------------------------------------------------------------------
# Group 6 — POPT_NEAR_GRASP_IF_BIG_GAP phase-transition gate (3 tests)
# ---------------------------------------------------------------------------


def test_popt_near_grasp_below_phase_threshold_is_ok_with_note(patch_claims_for):
    """grasp_gain_pp = (lru - grasp) × 100 ≤ 10 → status='ok' with
    note='not in phase-transition regime; assertion not triggered'.
    Below the 10pp threshold the comparator does NOT assert POPT≈GRASP."""
    claim = StubClaim(
        policy="POPT_NEAR_GRASP_IF_BIG_GAP",
        max_abs_delta_pct=2.0,
        tolerance_pct=0.5,
        expected_sign="~",
    )
    patch_claims_for([claim])
    obs_idx = lf.index([
        _build_obs("g", "a", "1MB", "POPT", 0.700),  # POPT much worse
        _build_obs("g", "a", "1MB", "GRASP", 0.500),
        _build_obs("g", "a", "1MB", "LRU", 0.550),  # gap only 5pp
    ])
    result = lf.evaluate(obs_idx)
    rel = [e for e in result["per_claim"]
           if e["policy"] == "POPT_NEAR_GRASP_IF_BIG_GAP"]
    assert rel[0]["status"] == "ok"
    assert "not in phase-transition regime" in rel[0].get("note", "")


def test_popt_near_grasp_above_threshold_within_tolerance_is_ok(patch_claims_for):
    """grasp_gain_pp > 10 AND signed_pp ≤ max_abs + tol → ok.
    Phase-transition regime triggered AND POPT close enough to GRASP."""
    claim = StubClaim(
        policy="POPT_NEAR_GRASP_IF_BIG_GAP",
        max_abs_delta_pct=2.0,
        tolerance_pct=0.5,
        expected_sign="~",
    )
    patch_claims_for([claim])
    obs_idx = lf.index([
        _build_obs("g", "a", "1MB", "POPT", 0.520),  # POPT 2pp worse than GRASP
        _build_obs("g", "a", "1MB", "GRASP", 0.500),
        _build_obs("g", "a", "1MB", "LRU", 0.650),  # gap = 15pp (> 10)
    ])
    result = lf.evaluate(obs_idx)
    rel = [e for e in result["per_claim"]
           if e["policy"] == "POPT_NEAR_GRASP_IF_BIG_GAP"]
    # signed_pp = (0.52 - 0.50) * 100 = 2.0 ≤ max+tol=2.5 → ok
    assert rel[0]["status"] == "ok"


def test_popt_near_grasp_above_threshold_beyond_tolerance_is_disagree(
    patch_claims_for,
):
    """grasp_gain_pp > 10 AND signed_pp > max_abs + tol → disagree.
    POPT diverges from GRASP in the phase-transition regime — one
    of the two oracle-aware policies is misbehaving."""
    claim = StubClaim(
        policy="POPT_NEAR_GRASP_IF_BIG_GAP",
        max_abs_delta_pct=2.0,
        tolerance_pct=0.5,
        expected_sign="~",
    )
    patch_claims_for([claim])
    obs_idx = lf.index([
        _build_obs("g", "a", "1MB", "POPT", 0.530),  # POPT 3pp worse
        _build_obs("g", "a", "1MB", "GRASP", 0.500),
        _build_obs("g", "a", "1MB", "LRU", 0.650),  # gap = 15pp
    ])
    result = lf.evaluate(obs_idx)
    rel = [e for e in result["per_claim"]
           if e["policy"] == "POPT_NEAR_GRASP_IF_BIG_GAP"]
    # signed_pp = 3.0 > max+tol=2.5 → disagree
    assert rel[0]["status"] == "disagree"
