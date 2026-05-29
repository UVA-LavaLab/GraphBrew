"""Gate 133 — arithmetic + bucket-consistency audit of
`literature_faithfulness_postfix.json`.

The literature-faithfulness comparator output is the largest artifact
(~360 KB) in `wiki/data/` and the headline source of the paper's
"how faithful are we to published baselines?" claim. It must be
internally self-consistent (per_claim status counts ↔ summary,
bucket lists ↔ per_claim filtered by status) and cross-consistent
with oracle_gap.json (per_observation miss_rate values must match the
canonical observation pipeline).

Invariants:

* summary counts (ok / within_tolerance / known_deviation / disagree
  / insufficient_data / missing) reproduce exactly from per_claim.
* summary.claims_total == len(per_claim).
* tolerated / known_deviations / disagreements lists equal the
  per_claim entries filtered to status ∈ {within_tolerance,
  known_deviation, disagree} (by (graph, app, l3_size, policy)).
* per_claim.delta_pct == round((policy_miss_rate - lru_miss_rate) *
  100.0, 4) (to 1e-4 tolerance) when both miss rates present.
* per_claim.status drawn from documented label set.
* per_observation.miss_rate matches oracle_gap.json for every shared
  (graph, app, l3, policy) key to 1e-9 (cross-source parity).
* per_claim entries reference a citation literal (paper-traceability).
* No DISAGREE entries present (paper is currently faithful).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POSTFIX_PATH = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"
ORACLE_PATH = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"

STATUS_LABELS = frozenset(
    {"ok", "within_tolerance", "known_deviation", "disagree",
     "insufficient_data", "missing", "no_lru"}
)
EPS_MISS = 1e-9
EPS_PCT = 1e-4


def _key(r):
    return (r["graph"], r["app"], r["l3_size"], r["policy"])


@pytest.fixture(scope="module")
def postfix() -> dict:
    return json.loads(POSTFIX_PATH.read_text())


@pytest.fixture(scope="module")
def oracle_miss_idx() -> dict:
    rows = json.loads(ORACLE_PATH.read_text())["rows"]
    return {(r["graph"], r["app"], r["l3_size"], r["policy"]): float(r["miss_rate"])
            for r in rows}


# ---------- Group 1: structural ----------

def test_top_level_keys_present(postfix):
    expected = {"summary", "per_claim", "per_observation",
                "tolerated", "known_deviations", "disagreements"}
    assert expected.issubset(set(postfix.keys()))


def test_summary_has_required_counts(postfix):
    required = {"claims_total", "ok", "within_tolerance", "known_deviation",
                "disagree", "insufficient_data", "missing"}
    assert required.issubset(set(postfix["summary"].keys()))


def test_per_claim_is_nonempty_list(postfix):
    assert isinstance(postfix["per_claim"], list)
    assert len(postfix["per_claim"]) > 0


def test_per_observation_is_nonempty_list(postfix):
    assert isinstance(postfix["per_observation"], list)
    assert len(postfix["per_observation"]) > 0


# ---------- Group 2: summary counts reproduce from per_claim ----------

def test_summary_claims_total_matches_per_claim_length(postfix):
    assert postfix["summary"]["claims_total"] == len(postfix["per_claim"])


def test_summary_status_counts_reproduce(postfix):
    counts = Counter(c["status"] for c in postfix["per_claim"])
    s = postfix["summary"]
    for label in ("ok", "within_tolerance", "known_deviation", "disagree",
                  "insufficient_data", "missing"):
        assert s[label] == counts.get(label, 0), label


def test_summary_counts_sum_to_total(postfix):
    s = postfix["summary"]
    bucket_sum = (s["ok"] + s["within_tolerance"] + s["known_deviation"]
                  + s["disagree"] + s["insufficient_data"] + s["missing"])
    assert bucket_sum == s["claims_total"]


def test_all_per_claim_statuses_in_documented_set(postfix):
    for c in postfix["per_claim"]:
        assert c["status"] in STATUS_LABELS, c["status"]


# ---------- Group 3: bucket lists match per_claim filters ----------

def test_tolerated_bucket_matches_within_tolerance(postfix):
    got = sorted(_key(c) for c in postfix["per_claim"]
                 if c["status"] == "within_tolerance")
    want = sorted(_key(c) for c in postfix["tolerated"])
    assert got == want


def test_known_deviations_bucket_matches(postfix):
    got = sorted(_key(c) for c in postfix["per_claim"]
                 if c["status"] == "known_deviation")
    want = sorted(_key(c) for c in postfix["known_deviations"])
    assert got == want


def test_disagreements_bucket_matches(postfix):
    got = sorted(_key(c) for c in postfix["per_claim"]
                 if c["status"] == "disagree")
    want = sorted(_key(c) for c in postfix["disagreements"])
    assert got == want


def test_no_disagreements_today(postfix):
    """The paper is currently faithful — no DISAGREE entries should exist.
    This is the load-bearing headline of the lit-faith comparator."""
    assert postfix["summary"]["disagree"] == 0
    assert len(postfix["disagreements"]) == 0


# ---------- Group 4: arithmetic + cross-source parity ----------

def test_per_claim_delta_pct_reproduces(postfix):
    """delta_pct == round((policy_miss - lru_miss) * 100, 4) when both present."""
    mismatches = []
    checked = 0
    for c in postfix["per_claim"]:
        if c.get("lru_miss_rate") is None or c.get("policy_miss_rate") is None:
            continue
        if c.get("delta_pct") is None:
            continue
        want = round((c["policy_miss_rate"] - c["lru_miss_rate"]) * 100.0, 4)
        if abs(c["delta_pct"] - want) > EPS_PCT:
            mismatches.append((_key(c), c["delta_pct"], want))
        checked += 1
    assert checked > 0, "no per_claim entries with both miss rates present"
    assert not mismatches, mismatches[:3]


def test_per_observation_miss_rate_matches_oracle_gap(postfix, oracle_miss_idx):
    """Every per_observation row's miss_rate must match oracle_gap.json
    for the same (graph, app, l3, policy) key."""
    mismatches = []
    checked = 0
    for o in postfix["per_observation"]:
        k = (o["graph"], o["app"], o["l3_size"], o["policy"])
        if k not in oracle_miss_idx:
            continue
        if abs(oracle_miss_idx[k] - o["miss_rate"]) > EPS_MISS:
            mismatches.append((k, oracle_miss_idx[k], o["miss_rate"]))
        checked += 1
    assert checked > 0
    assert not mismatches, mismatches[:3]


def test_per_claim_has_citation_literal(postfix):
    """Every per_claim entry must carry a non-empty citation literal
    pointing back to a paper figure or section (paper-traceability)."""
    no_cite = [c for c in postfix["per_claim"]
               if not c.get("citation") or not isinstance(c.get("citation"), str)]
    assert not no_cite, no_cite[:3]


def test_known_deviations_have_root_cause(postfix):
    """Every known_deviation entry must carry a known_deviation_reason
    (otherwise it's a silent waiver, not a documented exception)."""
    no_reason = [c for c in postfix["known_deviations"]
                 if not c.get("known_deviation_reason")
                 or not isinstance(c.get("known_deviation_reason"), str)]
    assert not no_reason, no_reason[:3]


def test_min_accesses_threshold_positive(postfix):
    assert postfix["summary"]["min_accesses_threshold"] > 0
