"""Derivation-parity gate for the GRASP-vs-LRU sign comparator
(SCD-Der) — locks the load-bearing predicates of
:mod:`scripts.experiments.ecg.sign_consistency`.

Why this exists
---------------
sign_consistency is the original GRASP-paper reality-check: for every
(graph, app, L3) cell in the cache_sim reference, GRASP - LRU must
have the SAME SIGN in the gem5 and Sniper timing simulators
(otherwise one of the three simulators is reporting a sign-inverted
hit/miss curve). The MANDATORY_L3_SIZES tuple (4kB, 32kB) names the
two L3 sizes where this MUST hold — larger L3s converge into the
asymptote regime and are reported as warnings only.

The script is also the *canonical implementation* of two helpers
that downstream comparators (literature_faithfulness, gem5_anchor_
summary) deliberately mirror:

* `_pick`: smallest non-zero section if any, else section 0. The
  literature_faithfulness `_pick_section` and gem5_anchor_summary
  `_pick_canonical_section` are docstring-tagged as 'mirror' of
  this rule. Drift here silently breaks all three.
* `_coerce_int` / `_coerce_float`: defensive ('', 'None', None →
  None; otherwise int(float(value)) / float(value), trap ValueError).

The existing integration gate (test_grasp_sign_consistency.py) runs
`evaluate` against real sweep data and asserts no disagreements,
but does NOT test the predicates in isolation. A regression that
inverted the `_sign` epsilon test (1e-9 → -1e-9) would still pass
the integration test on most cells, but secretly mis-classify any
GRASP-LRU delta in the 0..1e-9 numerical-noise band.

Test groups (28 tests, 6 groups):

* group 1 (4 tests) — pinned constants (MANDATORY_L3_SIZES,
  PolicyRow frozen dataclass with 7 fields).
* group 2 (6 tests) — `_coerce_float` / `_coerce_int` defensive
  returns (None / '' / 'None' → None; int via int(float()); reject
  on TypeError/ValueError).
* group 3 (4 tests) — `_pick` canonical-ROI rule (smallest non-zero
  section; fallback to first row; policy + l3_size double filter;
  no match → None).
* group 4 (5 tests) — `_sign` classifier (None → 'n/a'; +ε threshold
  ±1e-9 INCLUSIVE-zero band; positive / negative / zero outputs).
* group 5 (4 tests) — `compute_deltas` (per-L3 dict; GRASP - LRU
  arithmetic; missing GRASP or LRU → None; sizes derived from rows).
* group 6 (5 tests) — `evaluate` predicate dispatch (status='ok' when
  signs agree or zero-band; status='disagree' when signs differ;
  mandatory_violations vs warnings bucket dispatch; missing data →
  warnings bucket; pair block structure).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from scripts.experiments.ecg import sign_consistency as sc  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(simulator="cache_sim", graph="g", app="pr", l3="4kB",
         policy="GRASP", miss_rate=0.1, section=0):
    return sc.PolicyRow(
        simulator=simulator, graph=graph, app=app, l3_size=l3,
        policy=policy, miss_rate=miss_rate, section=section,
    )


# ---------------------------------------------------------------------------
# Group 1 — pinned constants and dataclass shape (4 tests)
# ---------------------------------------------------------------------------


def test_mandatory_l3_sizes_pinned():
    """MANDATORY_L3_SIZES = ('4kB', '32kB'). These two L3 sizes name
    the cells where GRASP-LRU MUST agree in sign across cache_sim /
    gem5 / Sniper — drift here changes which cells trigger the
    'mandatory_violations' bucket (paper-blocking failures)."""
    assert sc.MANDATORY_L3_SIZES == ("4kB", "32kB")


def test_mandatory_l3_sizes_is_tuple_not_list():
    """Tuple, not list — immutable so other modules importing the
    constant can't accidentally mutate the canonical set."""
    assert isinstance(sc.MANDATORY_L3_SIZES, tuple)


def test_policyrow_is_frozen_dataclass():
    """PolicyRow is frozen — predicate functions must not mutate
    rows they receive. A regression that unfroze the dataclass would
    let `_pick` accidentally rewrite section indices."""
    r = _row()
    with pytest.raises((AttributeError, Exception)):
        r.section = 99


def test_policyrow_has_seven_fields_with_section_default_zero():
    """PolicyRow fields exactly = {simulator, graph, app, l3_size,
    policy, miss_rate, section}; section defaults to 0 (the cache_sim
    single-section convention)."""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(sc.PolicyRow)}
    assert fields == {
        "simulator", "graph", "app", "l3_size",
        "policy", "miss_rate", "section",
    }
    r = sc.PolicyRow(
        simulator="cache_sim", graph="g", app="pr",
        l3_size="4kB", policy="GRASP", miss_rate=0.1,
    )
    assert r.section == 0


# ---------------------------------------------------------------------------
# Group 2 — _coerce_float / _coerce_int defensive returns (6 tests)
# ---------------------------------------------------------------------------


def test_coerce_float_treats_none_empty_string_literal_as_none():
    """None / '' / 'None' → None. The 'None' literal sentinel comes
    from CSV cells where a prior pipeline wrote str(None)."""
    assert sc._coerce_float(None) is None
    assert sc._coerce_float("") is None
    assert sc._coerce_float("None") is None


def test_coerce_float_valid_inputs_pass():
    """Numeric strings parse; ints / floats pass through."""
    assert sc._coerce_float("3.14") == pytest.approx(3.14)
    assert sc._coerce_float("0") == 0.0
    assert sc._coerce_float(42) == 42.0


def test_coerce_float_traps_typeerror_and_valueerror():
    """Non-coercible inputs return None — load_roi_matrix MUST NOT
    crash on a single corrupt cell."""
    assert sc._coerce_float("not_a_number") is None
    assert sc._coerce_float([1, 2]) is None
    assert sc._coerce_float({}) is None


def test_coerce_int_treats_none_empty_string_literal_as_none():
    """Same sentinel set as _coerce_float."""
    assert sc._coerce_int(None) is None
    assert sc._coerce_int("") is None
    assert sc._coerce_int("None") is None


def test_coerce_int_uses_int_of_float_for_string_inputs():
    """Predicate is int(float(value)) — so '3.7' parses to 3 (truncated
    toward zero, not rounded). This differs from gem5_anchor_summary's
    `_coerce_int` which is just int(text). Documenting the difference
    so a future unification is intentional, not accidental."""
    assert sc._coerce_int("3.7") == 3
    assert sc._coerce_int("3.14") == 3
    assert sc._coerce_int("0") == 0
    assert sc._coerce_int("-2.9") == -2  # truncation toward zero


def test_coerce_int_traps_errors():
    """Returns None on non-numeric (load_roi_matrix `_coerce_int(...) or 0`
    then turns this into the section=0 fallback)."""
    assert sc._coerce_int("not_a_number") is None
    assert sc._coerce_int([1]) is None
    assert sc._coerce_int({}) is None


# ---------------------------------------------------------------------------
# Group 3 — _pick canonical-ROI rule (4 tests)
# ---------------------------------------------------------------------------


def test_pick_prefers_smallest_nonzero_section():
    """Canonical rule (mirrored by lit-faith._pick_section + gem5_anchor_
    summary._pick_canonical_section): smallest non-zero section wins.
    Section 2 = m5_dump_stats (post-ROI cumulative, includes teardown
    noise on tiny graphs); section 1 = m5_work_end (canonical ROI)."""
    rows = [
        _row(section=0, miss_rate=0.10),
        _row(section=3, miss_rate=0.30),
        _row(section=1, miss_rate=0.20),
        _row(section=2, miss_rate=0.25),
    ]
    picked = sc._pick(rows, "GRASP", "4kB")
    assert picked.section == 1
    assert picked.miss_rate == 0.20


def test_pick_falls_back_to_first_when_all_section_zero():
    """All section==0 → return first matching row (cache_sim /
    Sniper paths — they emit only one section per cell)."""
    rows = [
        _row(section=0, miss_rate=0.10),
        _row(section=0, miss_rate=0.20),
    ]
    picked = sc._pick(rows, "GRASP", "4kB")
    assert picked.miss_rate == 0.10


def test_pick_filters_by_both_policy_and_l3_size():
    """Both filters AND'd — wrong policy or wrong L3 → excluded."""
    rows = [
        _row(policy="GRASP", l3="4kB", miss_rate=0.10),
        _row(policy="LRU", l3="4kB", miss_rate=0.20),
        _row(policy="GRASP", l3="32kB", miss_rate=0.30),
    ]
    picked = sc._pick(rows, "GRASP", "4kB")
    assert picked.miss_rate == 0.10
    assert sc._pick(rows, "SRRIP", "4kB") is None  # wrong policy
    assert sc._pick(rows, "GRASP", "1MB") is None  # wrong L3


def test_pick_no_match_returns_none():
    """No matching rows → None (caller responsible for handling)."""
    assert sc._pick([], "GRASP", "4kB") is None
    assert sc._pick([_row(policy="LRU")], "GRASP", "4kB") is None


# ---------------------------------------------------------------------------
# Group 4 — _sign classifier (5 tests)
# ---------------------------------------------------------------------------


def test_sign_none_returns_n_a():
    """None → 'n/a' (sentinel for missing observations; downstream
    `evaluate` triggers status='missing' on this)."""
    assert sc._sign(None) == "n/a"


def test_sign_strictly_negative_returns_minus():
    """value < -1e-9 → '-'. STRICT inequality with epsilon = 1e-9."""
    assert sc._sign(-1.0) == "-"
    assert sc._sign(-0.001) == "-"
    assert sc._sign(-1e-8) == "-"  # |value| > 1e-9 → '-'


def test_sign_strictly_positive_returns_plus():
    """value > 1e-9 → '+'. STRICT inequality with epsilon = 1e-9."""
    assert sc._sign(1.0) == "+"
    assert sc._sign(0.001) == "+"
    assert sc._sign(1e-8) == "+"


def test_sign_zero_band_inclusive_returns_zero():
    """|value| <= 1e-9 → '0'. The zero band is INCLUSIVE — exactly
    ±1e-9 maps to '0'. This is the numerical-noise tolerance: any
    GRASP-LRU delta smaller than 1 part in 1e9 is treated as
    sign-agnostic. Downstream `evaluate` collapses '0' against any
    sign as agreement (`or '0' in (ref_sign, sim_sign)`)."""
    assert sc._sign(0.0) == "0"
    assert sc._sign(1e-9) == "0"  # boundary: NOT strict +
    assert sc._sign(-1e-9) == "0"  # boundary: NOT strict -
    assert sc._sign(1e-10) == "0"


def test_sign_returns_string_not_enum():
    """`_sign` returns a string literal — downstream pickles and
    JSON-serialises this verbatim. An enum / dataclass return would
    break json.dumps(..., default=str) round-trips."""
    assert isinstance(sc._sign(1.0), str)
    assert isinstance(sc._sign(None), str)


# ---------------------------------------------------------------------------
# Group 5 — compute_deltas (4 tests)
# ---------------------------------------------------------------------------


def test_compute_deltas_arithmetic_is_grasp_minus_lru():
    """delta = miss_rate(GRASP) - miss_rate(LRU). Sign convention:
    negative = GRASP improvement (lower miss rate). This is the
    sense the paper's headline cells depend on."""
    rows = [
        _row(policy="GRASP", l3="4kB", miss_rate=0.30),
        _row(policy="LRU", l3="4kB", miss_rate=0.40),
    ]
    deltas = sc.compute_deltas(rows)
    assert deltas["4kB"] == pytest.approx(-0.10)  # GRASP improves by 10pp


def test_compute_deltas_emits_one_entry_per_l3_size():
    """Sizes derived as `sorted({r.l3_size for r in rows})`. Output
    dict has one entry per distinct L3 size in the input."""
    rows = [
        _row(policy="GRASP", l3="4kB", miss_rate=0.30),
        _row(policy="LRU", l3="4kB", miss_rate=0.40),
        _row(policy="GRASP", l3="32kB", miss_rate=0.10),
        _row(policy="LRU", l3="32kB", miss_rate=0.12),
    ]
    deltas = sc.compute_deltas(rows)
    assert set(deltas.keys()) == {"4kB", "32kB"}


def test_compute_deltas_returns_none_when_grasp_or_lru_missing():
    """Either side missing → None for that L3 (downstream `_sign`
    maps None → 'n/a' → status='missing' in evaluate)."""
    rows = [
        _row(policy="GRASP", l3="4kB", miss_rate=0.30),
        # LRU missing at 4kB
    ]
    deltas = sc.compute_deltas(rows)
    assert deltas["4kB"] is None


def test_compute_deltas_uses_smallest_nonzero_section_per_policy():
    """The per-policy _pick rule is applied inside compute_deltas
    so multi-section sweeps (gem5) get the canonical ROI section,
    not the post-ROI cumulative section."""
    rows = [
        # GRASP: section 1 (ROI) = 0.20, section 2 (post-ROI noise) = 0.50
        _row(policy="GRASP", l3="4kB", miss_rate=0.50, section=2),
        _row(policy="GRASP", l3="4kB", miss_rate=0.20, section=1),
        # LRU: section 1 = 0.30
        _row(policy="LRU", l3="4kB", miss_rate=0.30, section=1),
    ]
    deltas = sc.compute_deltas(rows)
    # Should use GRASP@section1=0.20, NOT GRASP@section2=0.50
    assert deltas["4kB"] == pytest.approx(-0.10)  # 0.20 - 0.30


# ---------------------------------------------------------------------------
# Group 6 — evaluate predicate dispatch (5 tests)
# ---------------------------------------------------------------------------


def test_evaluate_pair_block_has_required_fields(tmp_path):
    """Each pair block has {graph, app, cache_sim_csv, deltas},
    deltas is a dict keyed by simulator name. Missing CSVs are
    handled by `load_roi_matrix` returning [] — block still emitted."""
    summary = sc.evaluate(
        cache_root=tmp_path / "cache",
        gem5_root=None,
        sniper_root=None,
        pairs=[("g", "pr")],
    )
    assert len(summary["pairs"]) == 1
    block = summary["pairs"][0]
    assert {"graph", "app", "cache_sim_csv", "deltas"}.issubset(block.keys())
    assert "cache_sim" in block["deltas"]


def test_evaluate_summary_has_three_top_level_keys(tmp_path):
    """summary = {pairs, mandatory_violations, warnings}. Three
    output buckets pinned."""
    summary = sc.evaluate(
        cache_root=tmp_path / "cache",
        gem5_root=None, sniper_root=None,
        pairs=[("g", "pr")],
    )
    assert set(summary.keys()) == {"pairs", "mandatory_violations", "warnings"}
    assert isinstance(summary["mandatory_violations"], list)
    assert isinstance(summary["warnings"], list)


def test_evaluate_disagree_at_mandatory_size_routes_to_mandatory_violations(
    tmp_path, monkeypatch,
):
    """When ref_sign ≠ sim_sign AND L3 ∈ MANDATORY_L3_SIZES,
    record goes to mandatory_violations bucket (paper-blocking).
    We monkey-patch load_roi_matrix to inject a disagreement at 4kB."""

    def fake_load(path, simulator, graph, app):
        if simulator == "cache_sim":
            return [
                _row(simulator="cache_sim", policy="GRASP",
                     l3="4kB", miss_rate=0.30),
                _row(simulator="cache_sim", policy="LRU",
                     l3="4kB", miss_rate=0.40),
            ]
        # gem5 has GRASP > LRU at 4kB → opposite sign
        return [
            _row(simulator=simulator, policy="GRASP",
                 l3="4kB", miss_rate=0.50),
            _row(simulator=simulator, policy="LRU",
                 l3="4kB", miss_rate=0.40),
        ]

    monkeypatch.setattr(sc, "load_roi_matrix", fake_load)
    summary = sc.evaluate(
        cache_root=tmp_path / "cache",
        gem5_root=tmp_path / "gem5",
        sniper_root=None,
        pairs=[("g", "pr")],
    )
    # cache_sim delta = -0.10 (-), gem5 delta = +0.10 (+) → disagree
    # 4kB is mandatory → mandatory_violations bucket
    assert len(summary["mandatory_violations"]) == 1
    assert summary["mandatory_violations"][0]["l3_size"] == "4kB"
    assert summary["mandatory_violations"][0]["status"] == "disagree"


def test_evaluate_disagree_at_non_mandatory_size_routes_to_warnings(
    tmp_path, monkeypatch,
):
    """When L3 NOT in MANDATORY_L3_SIZES, disagreement → warnings
    (large L3 converges into asymptote, sign-flips are expected
    numerical noise)."""

    def fake_load(path, simulator, graph, app):
        if simulator == "cache_sim":
            return [
                _row(simulator="cache_sim", policy="GRASP",
                     l3="1MB", miss_rate=0.05),
                _row(simulator="cache_sim", policy="LRU",
                     l3="1MB", miss_rate=0.06),
            ]
        return [
            _row(simulator=simulator, policy="GRASP",
                 l3="1MB", miss_rate=0.07),
            _row(simulator=simulator, policy="LRU",
                 l3="1MB", miss_rate=0.06),
        ]

    monkeypatch.setattr(sc, "load_roi_matrix", fake_load)
    summary = sc.evaluate(
        cache_root=tmp_path / "cache",
        gem5_root=tmp_path / "gem5",
        sniper_root=None,
        pairs=[("g", "pr")],
    )
    # 1MB NOT in MANDATORY_L3_SIZES → warnings bucket
    assert len(summary["mandatory_violations"]) == 0
    assert len(summary["warnings"]) == 1
    assert summary["warnings"][0]["status"] == "disagree"
    assert summary["warnings"][0]["l3_size"] == "1MB"


def test_evaluate_zero_sign_collapses_to_ok(tmp_path, monkeypatch):
    """When EITHER ref_sign OR sim_sign is '0' (within ±1e-9), status
    is 'ok' regardless of the other sign. This is the documented
    `or '0' in (ref_sign, sim_sign)` predicate that prevents
    numerical-noise sign-flips from being treated as disagreement
    when one side is essentially zero."""

    def fake_load(path, simulator, graph, app):
        if simulator == "cache_sim":
            # cache_sim: identical miss_rates → delta=0 → sign='0'
            return [
                _row(simulator="cache_sim", policy="GRASP",
                     l3="4kB", miss_rate=0.40),
                _row(simulator="cache_sim", policy="LRU",
                     l3="4kB", miss_rate=0.40),
            ]
        # gem5: clearly negative delta → sign='-'
        return [
            _row(simulator=simulator, policy="GRASP",
                 l3="4kB", miss_rate=0.30),
            _row(simulator=simulator, policy="LRU",
                 l3="4kB", miss_rate=0.40),
        ]

    monkeypatch.setattr(sc, "load_roi_matrix", fake_load)
    summary = sc.evaluate(
        cache_root=tmp_path / "cache",
        gem5_root=tmp_path / "gem5",
        sniper_root=None,
        pairs=[("g", "pr")],
    )
    # '0' on cache_sim collapses with '-' on gem5 → status='ok',
    # NOT routed to mandatory_violations or warnings
    assert len(summary["mandatory_violations"]) == 0
    assert len(summary["warnings"]) == 0
