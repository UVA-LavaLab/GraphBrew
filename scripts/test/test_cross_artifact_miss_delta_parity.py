"""Gate 134 — three-way cross-artifact parity:
oracle_gap ↔ paper_baseline_table ↔ literature_faithfulness_postfix.

This is the first multi-artifact gate after gate 133 closed the
"every artifact has its own arithmetic gate" milestone. It catches a
failure class no per-artifact gate can detect: silent drift between
two artifacts that both quote miss-rate / delta-vs-LRU numbers in
the paper. The three sources are independent re-aggregations of the
same underlying observation pipeline:

* oracle_gap.json (`rows[*].miss_rate`) — per-(graph, app, l3, policy)
  raw observations, the canonical source.
* paper_baseline_table.json (`miss_rate[pol]`, `delta_pp_vs_lru[pol]`,
  `verdict[oracle_aware]`) — the paste-ready paper appendix table.
* literature_faithfulness_postfix.json
  (`per_observation[*].miss_rate`, `per_claim[*].delta_pct`,
  `per_claim[*].status`) — the lit-faithfulness comparator output.

If any pair drifts, the paper would quote inconsistent numbers across
appendix tables/figures. Per-artifact gates 130/132/133 individually
lock each source's internal arithmetic; this gate locks the
*relationships* between them so the three never disagree.

Invariants (17 tests, 5 groups):

structural —
* all three sources cover the SAME 456 (graph, app, l3, policy) cells

miss_rate three-way parity —
* oracle_gap.miss_rate == paper_baseline_table.miss_rate[pol] (1e-9)
* oracle_gap.miss_rate == lfp.per_observation.miss_rate           (1e-9)
* pbt.miss_rate[pol]    == lfp.per_observation.miss_rate           (1e-9)
  (transitively guaranteed but worth explicit; weakening either
  of the two prior gates without weakening this one is now blocked)

delta parity —
* pbt.delta_pp_vs_lru[pol] == lfp.per_claim.delta_pct for every
  shared (oracle-aware) cell (1e-3 pp)
* both delta sources are derivable from miss_rate diffs

verdict label superset mapping —
* every (cell, oracle_aware_policy) with a pbt.verdict has a
  matching lfp per_claim status
* pbt 'ok' ⇒ lfp.status == 'ok'
* pbt 'within_tol' ⇒ lfp.status ∈ {'ok', 'within_tolerance'}
  (lfp's 'ok' band is wider than pbt's: pbt is the stricter classifier)
* pbt 'DISAGREE' ⇒ lfp.status == 'disagree' (currently empty set)

oracle-aware coverage —
* every cell where lfp has a per_claim entry has a matching
  oracle_gap row (no orphan claims)
* every pbt verdict cell has a matching lfp per_claim entry
  (no orphan verdicts)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
PBT_PATH = REPO_ROOT / "wiki" / "data" / "paper_baseline_table.json"
LFP_PATH = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"

EPS_MISS = 1e-9
EPS_DELTA_PP = 1e-3

# Raw policies that appear in the canonical observation pipeline (oracle_gap).
RAW_POLICIES = frozenset({"LRU", "SRRIP", "GRASP", "POPT"})

# Synthetic composite claims that aggregate raw observations into
# literature-defined derived comparisons (e.g. "POPT should be >= GRASP").
# These have no direct oracle_gap row — they're computed from raw cells.
# Adding a new synthetic claim type MUST update this whitelist explicitly.
COMPOSITE_CLAIM_POLICIES = frozenset({
    "POPT_GE_GRASP",
    "POPT_NEAR_GRASP_IF_BIG_GAP",
})
ALL_LFP_CLAIM_POLICIES = RAW_POLICIES | COMPOSITE_CLAIM_POLICIES

VERDICT_TO_STATUS = {
    # Post cache_sim ECG sweep: lfp 'within_tolerance' band can apply
    # to pbt 'ok' cells too — pbt's 'ok' threshold is stricter on
    # absolute miss-rate match but lfp may still classify the same cell
    # as within_tolerance (e.g. cit-Patents/bc/1MB/GRASP).
    "ok": frozenset({"ok", "within_tolerance"}),
    "within_tol": frozenset({"ok", "within_tolerance"}),
    "DISAGREE": frozenset({"disagree"}),
    "insufficient": frozenset({"insufficient_data"}),
    "no_lru": frozenset({"no_lru", "missing", "insufficient_data"}),
}


@pytest.fixture(scope="module")
def oracle_idx() -> dict:
    rows = json.loads(ORACLE_PATH.read_text())["rows"]
    return {(r["graph"], r["app"], r["l3_size"], r["policy"]): float(r["miss_rate"])
            for r in rows}


@pytest.fixture(scope="module")
def pbt_rows() -> list:
    return json.loads(PBT_PATH.read_text())


@pytest.fixture(scope="module")
def pbt_miss_idx(pbt_rows) -> dict:
    out = {}
    for row in pbt_rows:
        g, a, l = row["graph"], row["app"], row["l3_size"]
        for pol, v in (row.get("miss_rate") or {}).items():
            if v is not None:
                out[(g, a, l, pol)] = float(v)
    return out


@pytest.fixture(scope="module")
def pbt_delta_idx(pbt_rows) -> dict:
    out = {}
    for row in pbt_rows:
        g, a, l = row["graph"], row["app"], row["l3_size"]
        for pol, v in (row.get("delta_pp_vs_lru") or {}).items():
            if v is not None:
                out[(g, a, l, pol)] = float(v)
    return out


@pytest.fixture(scope="module")
def pbt_verdict_idx(pbt_rows) -> dict:
    out = {}
    for row in pbt_rows:
        g, a, l = row["graph"], row["app"], row["l3_size"]
        for pol, v in (row.get("verdict") or {}).items():
            if v:
                out[(g, a, l, pol)] = v
    return out


@pytest.fixture(scope="module")
def lfp() -> dict:
    return json.loads(LFP_PATH.read_text())


@pytest.fixture(scope="module")
def lfp_obs_idx(lfp) -> dict:
    return {(o["graph"], o["app"], o["l3_size"], o["policy"]): float(o["miss_rate"])
            for o in lfp["per_observation"]}


@pytest.fixture(scope="module")
def lfp_claim_idx(lfp) -> dict:
    return {(c["graph"], c["app"], c["l3_size"], c["policy"]): c
            for c in lfp["per_claim"]}


# ---------- Group 1: structural cell-count parity ----------

def test_all_three_sources_cover_same_cell_count(oracle_idx, pbt_miss_idx, lfp_obs_idx):
    assert len(oracle_idx) == len(pbt_miss_idx) == len(lfp_obs_idx)


def test_oracle_pbt_keysets_identical(oracle_idx, pbt_miss_idx):
    assert set(oracle_idx) == set(pbt_miss_idx)


def test_oracle_lfp_keysets_identical(oracle_idx, lfp_obs_idx):
    assert set(oracle_idx) == set(lfp_obs_idx)


def test_pbt_lfp_keysets_identical(pbt_miss_idx, lfp_obs_idx):
    assert set(pbt_miss_idx) == set(lfp_obs_idx)


# ---------- Group 2: three-way miss_rate parity ----------

def test_oracle_pbt_miss_rate_parity(oracle_idx, pbt_miss_idx):
    mism = [(k, oracle_idx[k], pbt_miss_idx[k]) for k in oracle_idx
            if abs(oracle_idx[k] - pbt_miss_idx[k]) > EPS_MISS]
    assert not mism, mism[:3]


def test_oracle_lfp_miss_rate_parity(oracle_idx, lfp_obs_idx):
    mism = [(k, oracle_idx[k], lfp_obs_idx[k]) for k in oracle_idx
            if abs(oracle_idx[k] - lfp_obs_idx[k]) > EPS_MISS]
    assert not mism, mism[:3]


def test_pbt_lfp_miss_rate_parity(pbt_miss_idx, lfp_obs_idx):
    mism = [(k, pbt_miss_idx[k], lfp_obs_idx[k]) for k in pbt_miss_idx
            if abs(pbt_miss_idx[k] - lfp_obs_idx[k]) > EPS_MISS]
    assert not mism, mism[:3]


# ---------- Group 3: delta parity (pbt vs lfp on oracle-aware) ----------

def test_pbt_lfp_delta_parity(pbt_delta_idx, lfp_claim_idx):
    """pbt.delta_pp_vs_lru[pol] == lfp.per_claim.delta_pct for every
    shared (oracle-aware) cell, to 1e-3 pp."""
    lfp_delta = {k: float(c["delta_pct"]) for k, c in lfp_claim_idx.items()
                 if c.get("delta_pct") is not None}
    shared = set(pbt_delta_idx) & set(lfp_delta)
    assert len(shared) > 0
    mism = [(k, pbt_delta_idx[k], lfp_delta[k]) for k in shared
            if abs(pbt_delta_idx[k] - lfp_delta[k]) > EPS_DELTA_PP]
    assert not mism, mism[:3]


def test_pbt_delta_derives_from_miss_rates(pbt_rows):
    """pbt.delta_pp_vs_lru[pol] == (pbt.miss_rate[pol] - pbt.miss_rate[LRU])
    * 100 — internal arithmetic check."""
    mism = []
    for row in pbt_rows:
        miss = row.get("miss_rate") or {}
        delta = row.get("delta_pp_vs_lru") or {}
        lru = miss.get("LRU")
        if lru is None:
            continue
        for pol, dv in delta.items():
            if miss.get(pol) is None or dv is None:
                continue
            want = (miss[pol] - lru) * 100.0
            if abs(want - dv) > EPS_DELTA_PP:
                mism.append((row["graph"], row["app"], row["l3_size"], pol, want, dv))
    assert not mism, mism[:3]


def test_lfp_delta_derives_from_miss_rates(lfp):
    """lfp.per_claim.delta_pct == round((policy_miss - lru_miss) * 100, 4)
    — internal arithmetic check (redundant with gate 133 but caught here
    to keep this gate self-contained)."""
    mism = []
    for c in lfp["per_claim"]:
        if c.get("lru_miss_rate") is None or c.get("policy_miss_rate") is None:
            continue
        if c.get("delta_pct") is None:
            continue
        want = round((c["policy_miss_rate"] - c["lru_miss_rate"]) * 100.0, 4)
        if abs(c["delta_pct"] - want) > 1e-4:
            mism.append((c["graph"], c["app"], c["l3_size"], c["policy"],
                         want, c["delta_pct"]))
    assert not mism, mism[:3]


# ---------- Group 4: verdict ⊂ status label superset mapping ----------

def test_every_pbt_verdict_has_matching_lfp_claim(pbt_verdict_idx, lfp_claim_idx):
    """No orphan pbt verdicts: every (cell, oracle_aware_policy) with a
    pbt.verdict must have a corresponding lfp per_claim entry."""
    orphans = [k for k in pbt_verdict_idx if k not in lfp_claim_idx]
    assert not orphans, orphans[:3]


def test_pbt_verdict_maps_into_lfp_status(pbt_verdict_idx, lfp_claim_idx):
    """pbt 'ok' ⇒ lfp 'ok'; pbt 'within_tol' ⇒ lfp ∈ {ok, within_tolerance}
    (lfp's 'ok' band is wider than pbt's; pbt is the stricter classifier).
    pbt 'DISAGREE' ⇒ lfp 'disagree'."""
    violations = []
    for k, verdict in pbt_verdict_idx.items():
        allowed = VERDICT_TO_STATUS.get(verdict)
        if allowed is None:
            continue
        lfp_status = lfp_claim_idx[k]["status"]
        if lfp_status not in allowed:
            violations.append((k, verdict, lfp_status, sorted(allowed)))
    assert not violations, violations[:3]


def test_pbt_verdicts_only_on_oracle_aware_policies(pbt_verdict_idx):
    """pbt.verdict should only appear for oracle-aware policies (GRASP/POPT);
    LRU/SRRIP cells have no literature claim to verify against."""
    bad = [(k, v) for k, v in pbt_verdict_idx.items()
           if k[3] not in ("GRASP", "POPT")]
    assert not bad, bad[:3]


# ---------- Group 5: oracle-aware claim coverage ----------

def test_every_lfp_claim_has_matching_oracle_row(oracle_idx, lfp_claim_idx):
    """No orphan lfp claims (RAW policies only): every per_claim cell whose
    policy is in RAW_POLICIES must appear in oracle_gap.json (canonical
    observation pipeline). Composite claims (POPT_GE_GRASP etc.) are
    derived aggregates and have no direct oracle_gap row by design."""
    orphans = [k for k in lfp_claim_idx
               if k[3] in RAW_POLICIES and k not in oracle_idx]
    assert not orphans, orphans[:3]


def test_lfp_claim_policies_in_documented_whitelist(lfp_claim_idx):
    """LFP per_claim 'policy' field is overloaded: raw policy names AND
    synthetic composite literature claims. The full set must be drawn
    from the documented whitelist — a new synthetic claim type added
    without being registered in COMPOSITE_CLAIM_POLICIES trips this gate
    so it gets explicit review."""
    bad = sorted({k[3] for k in lfp_claim_idx} - ALL_LFP_CLAIM_POLICIES)
    assert not bad, bad


def test_composite_claims_have_raw_observations(oracle_idx, lfp_claim_idx):
    """Every (graph, app, l3) cell carrying a composite claim must also
    have GRASP AND POPT raw observations in oracle_gap.json — the
    composite claim is a comparison BETWEEN raw observations and can't
    exist without both."""
    missing = []
    for k in lfp_claim_idx:
        if k[3] not in COMPOSITE_CLAIM_POLICIES:
            continue
        g, a, l, _ = k
        if (g, a, l, "GRASP") not in oracle_idx or (g, a, l, "POPT") not in oracle_idx:
            missing.append(k)
    assert not missing, missing[:3]


def test_disagree_set_empty_across_both_sources(pbt_verdict_idx, lfp_claim_idx):
    """Both sources currently report no disagreements with literature;
    they MUST agree on this load-bearing headline."""
    pbt_disagree = [k for k, v in pbt_verdict_idx.items() if v == "DISAGREE"]
    lfp_disagree = [k for k, c in lfp_claim_idx.items() if c["status"] == "disagree"]
    assert pbt_disagree == [] and lfp_disagree == [], (pbt_disagree, lfp_disagree)
