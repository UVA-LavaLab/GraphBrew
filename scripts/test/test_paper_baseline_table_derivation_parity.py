"""Derivation parity gate for ``wiki/data/paper_baseline_table.json``.

Locks the paper baseline table against its two upstream sources so any
silent drift in the reducer (section pick, dedup, delta math, verdict
classification) trips a test before the dashboard re-publishes:

    sweep roi_matrix.csv  ──► literature_faithfulness.load_observations()
                              │
                              ▼
    wiki/data/literature_faithfulness_postfix.json#per_observation
                              │
            paper_baseline_table.py:build_rows() (this artifact)
                              │
                              ▼
                wiki/data/paper_baseline_table.json   ← gate target

Cross-source ground truth:

* ``literature_faithfulness_postfix.json#per_observation`` provides
  ``miss_rate`` per ``(graph, app, l3, policy)``.  The paper-baseline
  table consumes the exact same loader (``_lf.load_observations``) and
  must agree exactly on miss-rate values.
* ``scripts/experiments/ecg/literature_baselines.PER_GRAPH_CLAIMS`` is
  the only source of verdicts: filtered through ``_index_claims`` (drops
  ``POPT_GE_GRASP`` / ``POPT_NEAR_GRASP_IF_BIG_GAP`` pseudo-policies)
  this leaves only ``GRASP`` and ``POPT`` claims, so verdict policies
  must be a subset of ``{GRASP, POPT}``.

If the loader, the delta math, the verdict classifier, or the claim
table is silently changed, this test breaks and the dashboard refuses
to re-publish until the invariant is restored.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"
PBT_PATH = WIKI_DATA / "paper_baseline_table.json"
LF_POST_PATH = WIKI_DATA / "literature_faithfulness_postfix.json"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")
VERDICT_LABELS = {"ok", "within_tol", "DISAGREE", "insufficient", "no_lru"}

# Mirror of ``_index_claims`` filter set in paper_baseline_table.py.
PSEUDO_POLICIES = {"POPT_GE_GRASP", "POPT_NEAR_GRASP_IF_BIG_GAP"}

# Mirror of the Makefile invocation default (no override at lit-table).
MIN_ACCESSES = 10_000


@pytest.fixture(scope="module")
def pbt() -> list[dict]:
    if not PBT_PATH.exists():
        pytest.skip(f"missing {PBT_PATH}")
    return json.loads(PBT_PATH.read_text())


@pytest.fixture(scope="module")
def per_observation() -> dict[tuple[str, str, str, str], dict]:
    if not LF_POST_PATH.exists():
        pytest.skip(f"missing {LF_POST_PATH}")
    payload = json.loads(LF_POST_PATH.read_text())
    out: dict[tuple[str, str, str, str], dict] = {}
    for row in payload["per_observation"]:
        out[(row["graph"], row["app"], row["l3_size"], row["policy"])] = row
    return out


@pytest.fixture(scope="module")
def per_claim() -> list[dict]:
    if not LF_POST_PATH.exists():
        pytest.skip(f"missing {LF_POST_PATH}")
    payload = json.loads(LF_POST_PATH.read_text())
    return payload["per_claim"]


@pytest.fixture(scope="module")
def per_graph_claims():
    """Load ``literature_baselines.PER_GRAPH_CLAIMS`` via the same dynamic
    import shim the generator uses, so we exercise the same claim table.
    """
    here = REPO_ROOT / "scripts" / "experiments" / "ecg"
    spec = importlib.util.spec_from_file_location(
        "literature_baselines", here / "literature_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["literature_baselines"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return [c for c in module.PER_GRAPH_CLAIMS if c.policy not in PSEUDO_POLICIES]


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_is_nonempty_list(pbt):
    assert isinstance(pbt, list)
    assert len(pbt) > 0, "paper_baseline_table.json must not be empty"


def test_every_row_has_canonical_keys(pbt):
    expected = {
        "graph", "app", "l3_size",
        "miss_rate", "delta_pp_vs_lru", "verdict", "accesses",
    }
    for r in pbt:
        assert set(r.keys()) == expected, (
            f"unexpected key set on row {(r.get('graph'), r.get('app'), r.get('l3_size'))}: "
            f"{set(r.keys()) ^ expected}"
        )


def test_rows_are_sorted_by_graph_app_l3(pbt):
    keys = [(r["graph"], r["app"], r["l3_size"]) for r in pbt]
    assert keys == sorted(keys), (
        "paper_baseline_table.json rows must be sorted by (graph, app, l3_size); "
        "the generator emits rows in sorted(grouped.items()) order."
    )


def test_graph_app_l3_tuples_are_unique(pbt):
    keys = [(r["graph"], r["app"], r["l3_size"]) for r in pbt]
    dupes = [k for k, n in Counter(keys).items() if n > 1]
    assert not dupes, f"duplicate (graph, app, l3) rows: {dupes}"


def test_policies_are_canonical(pbt):
    for r in pbt:
        for p in r["miss_rate"]:
            assert p in POLICIES, f"unexpected policy {p!r} in row {r}"
        for p in r["delta_pp_vs_lru"]:
            assert p in POLICIES, f"unexpected policy {p!r} in delta row {r}"
        for p in r["verdict"]:
            assert p in POLICIES, f"unexpected verdict policy {p!r} in row {r}"


# ----------------------------------------------------------------------
# Group B: miss_rate cross-source parity with per_observation
# ----------------------------------------------------------------------

def test_miss_rate_matches_per_observation_exactly(pbt, per_observation):
    for r in pbt:
        for pol, mr in r["miss_rate"].items():
            obs = per_observation.get((r["graph"], r["app"], r["l3_size"], pol))
            assert obs is not None, (
                f"paper_baseline row {(r['graph'], r['app'], r['l3_size'], pol)} "
                f"has no matching per_observation entry"
            )
            assert obs["miss_rate"] == mr, (
                f"miss_rate mismatch at {(r['graph'], r['app'], r['l3_size'], pol)}: "
                f"per_observation={obs['miss_rate']!r} vs paper_baseline={mr!r}"
            )


def test_miss_rates_are_proper_probabilities(pbt):
    for r in pbt:
        for pol, mr in r["miss_rate"].items():
            assert 0.0 <= mr <= 1.0, (
                f"miss_rate out of [0,1] at {(r['graph'], r['app'], r['l3_size'], pol)}: {mr!r}"
            )


def test_every_pbt_row_has_at_least_one_observed_policy(pbt):
    for r in pbt:
        assert r["miss_rate"], (
            f"empty miss_rate dict at {(r['graph'], r['app'], r['l3_size'])} — "
            "grouped rows with no policies should never be emitted"
        )


def test_every_lru_in_per_observation_propagates_to_pbt(pbt, per_observation):
    """If per_observation has an LRU entry for a (graph, app, l3) and the
    paper-baseline table has any row for that triple, then LRU must be
    present in that row's miss_rate (paper_baseline never drops LRU).
    """
    by_triple: dict[tuple[str, str, str], dict] = {
        (r["graph"], r["app"], r["l3_size"]): r for r in pbt
    }
    for (g, a, l, p), obs in per_observation.items():
        if p != "LRU":
            continue
        row = by_triple.get((g, a, l))
        if row is None:
            continue
        assert "LRU" in row["miss_rate"], (
            f"LRU observation at {(g, a, l)} not propagated into paper_baseline row"
        )
        assert row["miss_rate"]["LRU"] == obs["miss_rate"], (
            f"LRU miss_rate mismatch at {(g, a, l)}: "
            f"per_observation={obs['miss_rate']!r} vs paper_baseline={row['miss_rate']['LRU']!r}"
        )


# ----------------------------------------------------------------------
# Group C: delta_pp_vs_lru math
# ----------------------------------------------------------------------

def test_delta_for_lru_is_exactly_zero(pbt):
    for r in pbt:
        if "LRU" not in r["miss_rate"]:
            continue
        assert "LRU" in r["delta_pp_vs_lru"]
        assert r["delta_pp_vs_lru"]["LRU"] == 0.0, (
            f"LRU delta must be 0.0 at {(r['graph'], r['app'], r['l3_size'])}, "
            f"got {r['delta_pp_vs_lru']['LRU']!r}"
        )


def test_delta_for_non_lru_matches_raw_subtraction(pbt):
    """delta_pp[p] = (miss[p] - miss[LRU]) * 100.0 reproduced exactly."""
    for r in pbt:
        lru_mr = r["miss_rate"].get("LRU")
        if lru_mr is None:
            continue
        for pol in ("SRRIP", "GRASP", "POPT"):
            mr = r["miss_rate"].get(pol)
            if mr is None:
                continue
            expected = (mr - lru_mr) * 100.0
            actual = r["delta_pp_vs_lru"].get(pol)
            assert actual is not None, (
                f"missing delta for {pol} at {(r['graph'], r['app'], r['l3_size'])}"
            )
            assert actual == expected, (
                f"delta_pp mismatch at {(r['graph'], r['app'], r['l3_size'], pol)}: "
                f"expected (raw){expected!r}, got {actual!r}"
            )


def test_delta_keys_are_subset_of_miss_rate_keys(pbt):
    for r in pbt:
        miss_keys = set(r["miss_rate"])
        delta_keys = set(r["delta_pp_vs_lru"])
        assert delta_keys <= miss_keys, (
            f"delta has keys not in miss_rate at {(r['graph'], r['app'], r['l3_size'])}: "
            f"{delta_keys - miss_keys}"
        )


def test_delta_absent_when_lru_absent(pbt):
    """When LRU has no observation in the row, no non-LRU delta should be
    populated — generator only writes ``delta[pol]`` when ``lru is not
    None``.
    """
    for r in pbt:
        if "LRU" in r["miss_rate"]:
            continue
        for pol in ("SRRIP", "GRASP", "POPT"):
            assert pol not in r["delta_pp_vs_lru"], (
                f"unexpected {pol} delta at {(r['graph'], r['app'], r['l3_size'])} "
                "with no LRU baseline"
            )


# ----------------------------------------------------------------------
# Group D: verdict cross-source parity with PER_GRAPH_CLAIMS
# ----------------------------------------------------------------------

def test_verdict_policies_are_only_grasp_or_popt(pbt):
    """``_index_claims`` filters pseudo-policies; the surviving
    ``PER_GRAPH_CLAIMS`` only mention GRASP and POPT — so paper-baseline
    can never produce a verdict for SRRIP or LRU.
    """
    seen = {p for r in pbt for p in r["verdict"]}
    assert seen <= {"GRASP", "POPT"}, (
        f"verdict policies leaked outside {{GRASP, POPT}}: {seen}"
    )


def test_verdict_labels_are_in_allowed_set(pbt):
    for r in pbt:
        for pol, v in r["verdict"].items():
            assert v in VERDICT_LABELS, (
                f"unknown verdict label {v!r} at {(r['graph'], r['app'], r['l3_size'], pol)}"
            )


def test_verdict_count_matches_per_graph_claims(pbt, per_graph_claims):
    """Every PER_GRAPH_CLAIMS row whose ``(graph, app, l3, policy)`` also
    has an observation must produce exactly one verdict; rows with no
    matching observation are silently skipped (build_rows iterates the
    observed policy_map first).
    """
    have_obs: set[tuple[str, str, str, str]] = set()
    for r in pbt:
        for pol in r["miss_rate"]:
            have_obs.add((r["graph"], r["app"], r["l3_size"], pol))

    expected_verdicts = sum(
        1 for c in per_graph_claims
        if (c.graph, c.app, c.l3_size, c.policy) in have_obs
    )
    actual_verdicts = sum(len(r["verdict"]) for r in pbt)
    assert expected_verdicts == actual_verdicts, (
        f"verdict cardinality drift: PER_GRAPH_CLAIMS claims matching observations "
        f"= {expected_verdicts}, paper_baseline verdicts = {actual_verdicts}"
    )


def test_every_per_graph_claim_with_observation_has_a_verdict(pbt, per_graph_claims):
    by_triple: dict[tuple[str, str, str], dict] = {
        (r["graph"], r["app"], r["l3_size"]): r for r in pbt
    }
    for c in per_graph_claims:
        row = by_triple.get((c.graph, c.app, c.l3_size))
        if row is None:
            continue
        if c.policy not in row["miss_rate"]:
            continue  # claim exists but no observation for that policy
        assert c.policy in row["verdict"], (
            f"missing verdict for claim {(c.graph, c.app, c.l3_size, c.policy)}"
        )


def test_verdict_matches_recomputed_classifier(pbt, per_graph_claims):
    """Re-derive the verdict label from scratch using the same math the
    generator uses (``_verdict_for``) and assert exact agreement.  This
    locks the four classifier branches against silent drift.
    """
    by_triple: dict[tuple[str, str, str], dict] = {
        (r["graph"], r["app"], r["l3_size"]): r for r in pbt
    }
    for c in per_graph_claims:
        row = by_triple.get((c.graph, c.app, c.l3_size))
        if row is None:
            continue
        if c.policy not in row["miss_rate"]:
            continue
        lru_mr = row["miss_rate"].get("LRU")
        if lru_mr is None:
            assert row["verdict"][c.policy] == "no_lru"
            continue
        # We can't recover policy.accesses from per_observation here, but the
        # paper-baseline row exposes ``accesses = max over policies``.  All
        # current rows are far above MIN_ACCESSES; assert that, then proceed
        # with the same five-branch classifier in _verdict_for.
        assert row["accesses"] >= MIN_ACCESSES, (
            f"row {(c.graph, c.app, c.l3_size)} below MIN_ACCESSES; classifier "
            f"would short-circuit to 'insufficient' — gate must be updated."
        )
        delta_pp = (row["miss_rate"][c.policy] - lru_mr) * 100.0
        lo = c.min_abs_delta_pct if c.min_abs_delta_pct is not None else 0.0
        hi = c.max_abs_delta_pct if c.max_abs_delta_pct is not None else math.inf
        tol = c.tolerance_pct
        sign = c.expected_sign
        sign_ok = (sign == "~") or (
            (sign == "-" and delta_pp <= 0) or (sign == "+" and delta_pp >= 0)
        )
        mag = abs(delta_pp)
        if not sign_ok and mag > tol:
            expected = "DISAGREE"
        elif mag < lo - tol or mag > hi + tol:
            expected = "DISAGREE"
        elif mag < lo or mag > hi:
            expected = "within_tol"
        else:
            expected = "ok"
        actual = row["verdict"][c.policy]
        assert actual == expected, (
            f"verdict mismatch at {(c.graph, c.app, c.l3_size, c.policy)}: "
            f"expected {expected!r}, got {actual!r} (Δ={delta_pp:+.4f} pp, "
            f"tol={tol}, lo={lo}, hi={hi}, sign={sign})"
        )


# ----------------------------------------------------------------------
# Group E: accesses & cross-source bounds
# ----------------------------------------------------------------------

def test_accesses_is_nonnegative_int(pbt):
    for r in pbt:
        assert isinstance(r["accesses"], int), (
            f"accesses must be int at {(r['graph'], r['app'], r['l3_size'])}, "
            f"got {type(r['accesses']).__name__}"
        )
        assert r["accesses"] >= 0


def test_accesses_bound_against_per_claim(pbt, per_claim):
    """``paper_baseline.accesses`` is the max across policies for a triple;
    every per_claim row carries the policy's own ``accesses`` — so for any
    overlap the table value must be ``>=`` the claim value.
    """
    by_triple: dict[tuple[str, str, str], int] = {
        (r["graph"], r["app"], r["l3_size"]): r["accesses"] for r in pbt
    }
    checked = 0
    for c in per_claim:
        n = by_triple.get((c["graph"], c["app"], c["l3_size"]))
        if n is None:
            continue
        if c.get("accesses") is None:
            continue
        assert n >= c["accesses"], (
            f"paper_baseline.accesses={n} is below per_claim.accesses={c['accesses']} "
            f"at {(c['graph'], c['app'], c['l3_size'])}"
        )
        checked += 1
    assert checked > 0, "no per_claim/paper_baseline overlap found — gate inert"


def test_all_rows_meet_min_accesses_threshold(pbt):
    """If anyone lowers MIN_ACCESSES below the floor used by ``lit-table``,
    or if a degenerate sweep produces a near-empty row, the recomputed
    verdict branch in ``test_verdict_matches_recomputed_classifier``
    silently flips to 'insufficient'.  Pin the precondition here.
    """
    for r in pbt:
        if not r["verdict"]:
            continue  # rows without a matching claim are irrelevant
        assert r["accesses"] >= MIN_ACCESSES, (
            f"row {(r['graph'], r['app'], r['l3_size'])} has accesses="
            f"{r['accesses']} below MIN_ACCESSES={MIN_ACCESSES}; verdict "
            "classifier would short-circuit to 'insufficient'"
        )


def test_round_trip_sort_keys_serialization(pbt):
    """The generator emits the JSON with ``indent=2, sort_keys=True``.
    Re-serialise and compare against the on-disk file to catch any
    accidental key reordering or whitespace drift introduced by hand
    edits to ``paper_baseline_table.json``.
    """
    on_disk = PBT_PATH.read_text()
    canonical = json.dumps(pbt, indent=2, sort_keys=True)
    assert on_disk.strip() == canonical.strip(), (
        "paper_baseline_table.json is not in canonical sort_keys=True / "
        "indent=2 form — re-run `make lit-table` to regenerate."
    )
