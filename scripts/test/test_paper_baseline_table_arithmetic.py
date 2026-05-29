"""Gate 132 — arithmetic + cross-source audit of `paper_baseline_table.json`.

Independently audits the paper-ready baseline table (paste-ready for the
paper's baselines section) for internal arithmetic consistency and
cross-source parity with `oracle_gap.json`:

* miss_rate[policy] for every (graph, app, l3, policy) tuple must match
  the value in oracle_gap.json to within 1e-9 (no silent drift between
  the comparator pipeline and the oracle_gap aggregator)
* delta_pp_vs_lru[policy] must equal (miss_rate[policy] - miss_rate.LRU)
  * 100.0 for every non-LRU policy, and exactly 0.0 for LRU
* verdict keys are restricted to {GRASP, POPT} per generator
  (literature claims only apply to oracle-aware policies)
* verdict values are drawn from the documented label set
* per-row policy keys are subset of (LRU, SRRIP, GRASP, POPT)
* rows are sorted lexicographically by (graph, app, l3_size)

This artifact is paste-ready into the paper, so any silent corruption
would land directly in the submission.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
TABLE_PATH = REPO_ROOT / "wiki" / "data" / "paper_baseline_table.json"

POLICY_COLS = ("LRU", "SRRIP", "GRASP", "POPT")
VERDICT_LABELS = frozenset(
    {"ok", "within_tol", "DISAGREE", "insufficient", "no_lru", ""}
)
ORACLE_AWARE_POLICIES = frozenset({"GRASP", "POPT"})
EPS = 1e-9


@pytest.fixture(scope="module")
def table() -> list:
    return json.loads(TABLE_PATH.read_text())


@pytest.fixture(scope="module")
def oracle_miss_idx() -> dict:
    rows = json.loads(ORACLE_PATH.read_text())["rows"]
    return {(r["graph"], r["app"], r["l3_size"], r["policy"]): float(r["miss_rate"])
            for r in rows}


# ---------- Group 1: structural sanity ----------

def test_table_is_nonempty_list(table):
    assert isinstance(table, list)
    assert len(table) > 0


def test_table_rows_have_required_fields(table):
    required = {"graph", "app", "l3_size", "miss_rate",
                "delta_pp_vs_lru", "verdict", "accesses"}
    for r in table:
        assert required.issubset(set(r.keys())), set(r.keys())


def test_table_sorted_by_graph_app_l3(table):
    keys = [(r["graph"], r["app"], r["l3_size"]) for r in table]
    assert keys == sorted(keys)


def test_table_rows_unique_by_graph_app_l3(table):
    keys = [(r["graph"], r["app"], r["l3_size"]) for r in table]
    assert len(keys) == len(set(keys))


# ---------- Group 2: per-row policy/verdict key validation ----------

def test_miss_rate_keys_are_valid_policies(table):
    for r in table:
        for p in r["miss_rate"]:
            assert p in POLICY_COLS, (r["graph"], r["app"], r["l3_size"], p)


def test_delta_keys_subset_of_miss_rate_keys(table):
    for r in table:
        assert set(r["delta_pp_vs_lru"]).issubset(set(r["miss_rate"])), \
            (r["graph"], r["app"], r["l3_size"])


def test_verdict_keys_restricted_to_oracle_aware(table):
    for r in table:
        for p in r["verdict"]:
            assert p in ORACLE_AWARE_POLICIES, \
                (r["graph"], r["app"], r["l3_size"], p)


def test_verdict_values_in_documented_set(table):
    for r in table:
        for p, v in r["verdict"].items():
            assert v in VERDICT_LABELS, \
                (r["graph"], r["app"], r["l3_size"], p, v)


def test_accesses_nonnegative_integers(table):
    for r in table:
        assert isinstance(r["accesses"], int) and r["accesses"] >= 0, r


# ---------- Group 3: delta arithmetic ----------

def test_delta_lru_is_zero_when_present(table):
    for r in table:
        if "LRU" in r["delta_pp_vs_lru"]:
            assert r["delta_pp_vs_lru"]["LRU"] == 0.0, \
                (r["graph"], r["app"], r["l3_size"])


def test_delta_nonlru_matches_miss_diff_x100(table):
    for r in table:
        if "LRU" not in r["miss_rate"]:
            continue
        lru = r["miss_rate"]["LRU"]
        for pol, mr in r["miss_rate"].items():
            if pol == "LRU":
                continue
            if pol not in r["delta_pp_vs_lru"]:
                continue
            expected = (mr - lru) * 100.0
            got = r["delta_pp_vs_lru"][pol]
            assert abs(got - expected) < EPS, \
                (r["graph"], r["app"], r["l3_size"], pol, got, expected)


def test_delta_only_for_policies_with_lru_or_self(table):
    """delta is set only when LRU is present (and 0.0 for LRU itself)."""
    for r in table:
        has_lru = "LRU" in r["miss_rate"]
        if not has_lru:
            assert r["delta_pp_vs_lru"] == {}, \
                (r["graph"], r["app"], r["l3_size"])


# ---------- Group 4: cross-source parity with oracle_gap ----------

def test_every_miss_rate_matches_oracle_gap(table, oracle_miss_idx):
    """Every (graph, app, l3, policy) in the paper table must match the
    same key's miss_rate in oracle_gap.json — both pipelines aggregate the
    same observation; any drift = silent corruption."""
    mismatches = []
    for r in table:
        for pol, mr in r["miss_rate"].items():
            k = (r["graph"], r["app"], r["l3_size"], pol)
            if k not in oracle_miss_idx:
                continue
            if abs(oracle_miss_idx[k] - mr) > EPS:
                mismatches.append((k, oracle_miss_idx[k], mr))
    assert not mismatches, mismatches[:5]


def test_every_keyed_observation_is_in_oracle_gap(table, oracle_miss_idx):
    """For paper-scope L3 (1MB/4MB/8MB), every cell present here must
    also exist in oracle_gap; non-paper-scope cells may legitimately
    differ in coverage."""
    PAPER_L3 = ("1MB", "4MB", "8MB")
    missing = []
    for r in table:
        if r["l3_size"] not in PAPER_L3:
            continue
        for pol in r["miss_rate"]:
            k = (r["graph"], r["app"], r["l3_size"], pol)
            if k not in oracle_miss_idx:
                missing.append(k)
    assert not missing, missing[:5]


def test_at_least_one_verdict_per_oracle_aware_when_lru_present(table):
    """When LRU is present, at least one of GRASP/POPT rows should have
    a verdict somewhere in the table (sanity check that lit claims are
    actually being indexed at all)."""
    any_verdict = any(r["verdict"] for r in table)
    assert any_verdict, "no verdicts at all in entire baseline table"
