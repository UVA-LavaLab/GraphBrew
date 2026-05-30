"""LIT-Mono (gate 229): pytest invariants for the cache-size monotonicity audit.

Locks the physical invariant that miss rate is a non-increasing function
of LLC size for every (graph, app, policy) triple in the lit-faith
corpus, plus floors on the median slope-per-doubling so the corpus is
exercising real cache pressure (not saturated noise).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "wiki" / "data" / "lit_faith_monotonicity.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    assert AUDIT_PATH.exists(), (
        f"{AUDIT_PATH} missing — run `make lit-monotonicity` (gate 229)"
    )
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_schema_keys(audit: dict) -> None:
    for key in ("schema_version", "tolerance", "summary",
                "violations_lru", "violations_policy",
                "saturated_triples", "triples"):
        assert key in audit, f"missing top-level key {key!r}"
    assert audit["schema_version"] == 1


def test_tolerance_pinning(audit: dict) -> None:
    assert audit["tolerance"] == 0.005, (
        "monotonicity tolerance must remain 0.5 pp — changing it without "
        "this gate's review hides real regressions"
    )


def test_audit_coverage_floor(audit: dict) -> None:
    s = audit["summary"]
    assert s["triples_audited"] >= 25, (
        f"only {s['triples_audited']} triples audited; lit-faith corpus "
        "lost L3-sweep coverage"
    )
    assert s["rows_with_rates"] >= 100


def test_no_lru_monotonicity_violations(audit: dict) -> None:
    if audit["violations_lru"]:
        details = "; ".join(
            f"{v['graph']}/{v['app']}/{v['policy']}: "
            f"{v['l3_smaller']}→{v['l3_larger']} Δ {v['delta']:+.4f}"
            for v in audit["violations_lru"][:5]
        )
        pytest.fail(f"{len(audit['violations_lru'])} LRU monotonicity "
                    f"violations: {details}")


def test_no_policy_monotonicity_violations(audit: dict) -> None:
    if audit["violations_policy"]:
        details = "; ".join(
            f"{v['graph']}/{v['app']}/{v['policy']}: "
            f"{v['l3_smaller']}→{v['l3_larger']} Δ {v['delta']:+.4f}"
            for v in audit["violations_policy"][:5]
        )
        pytest.fail(f"{len(audit['violations_policy'])} policy monotonicity "
                    f"violations: {details}")


def test_saturated_count_ceiling(audit: dict) -> None:
    """At most 5 triples may be saturated (< 1 pp total drop across the
    full L3 sweep). Today: 1 (com-orkut/bfs/SRRIP, expected — bfs on
    orkut is capacity-bound at any LLC ≤ 8MB)."""
    s = audit["summary"]
    assert s["saturated_count"] <= 5, (
        f"{s['saturated_count']} saturated triples — too many workloads "
        "are below the cache-pressure noise floor"
    )


def test_median_slope_lru_floor(audit: dict) -> None:
    s = audit["summary"]
    assert s["median_slope_lru"] >= 0.05, (
        f"median LRU slope per log2(L3) is {s['median_slope_lru']:.4f}, "
        "below 0.05 — the corpus has lost cache-pressure sensitivity"
    )


def test_median_slope_policy_floor(audit: dict) -> None:
    s = audit["summary"]
    assert s["median_slope_policy"] >= 0.05, (
        f"median policy slope per log2(L3) is {s['median_slope_policy']:.4f}, "
        "below 0.05"
    )


def test_min_slope_strictly_positive(audit: dict) -> None:
    """Every triple's average slope must be ≥ 0 — otherwise miss rate
    increased on average as cache size grew."""
    s = audit["summary"]
    assert s["min_slope_lru"] >= 0.0, (
        f"a triple has negative LRU slope per log2(L3): {s['min_slope_lru']}"
    )
    assert s["min_slope_policy"] >= 0.0, (
        f"a triple has negative policy slope: {s['min_slope_policy']}"
    )


def test_per_triple_l3_count_floor(audit: dict) -> None:
    """Every audited triple must have ≥ 2 L3 samples (precondition).
    Catches an audit-loop bug if it ever regresses."""
    for t in audit["triples"]:
        assert t["l3_count"] >= 2, (
            f"triple {t['graph']}/{t['app']}/{t['policy']} entered audit "
            f"with only {t['l3_count']} L3 samples"
        )


def test_per_triple_miss_rates_within_unit(audit: dict) -> None:
    for t in audit["triples"]:
        assert 0.0 <= t["lru_miss_min"] <= t["lru_miss_max"] <= 1.0, (
            f"triple {t['graph']}/{t['app']}/{t['policy']} LRU miss-rate "
            f"out of [0,1]"
        )
        assert 0.0 <= t["policy_miss_min"] <= t["policy_miss_max"] <= 1.0, (
            f"triple {t['graph']}/{t['app']}/{t['policy']} policy miss-rate "
            f"out of [0,1]"
        )


def test_per_triple_samples_sorted_by_l3(audit: dict) -> None:
    L3_BYTES = {"4kB": 4*1024, "16kB": 16*1024, "64kB": 64*1024,
                "256kB": 256*1024, "1MB": 1024*1024, "4MB": 4*1024*1024,
                "8MB": 8*1024*1024, "16MB": 16*1024*1024}
    for t in audit["triples"]:
        prev = -1
        for s in t["samples"]:
            assert s["l3_size"] in L3_BYTES, (
                f"unknown l3 label {s['l3_size']!r} in triple"
            )
            b = L3_BYTES[s["l3_size"]]
            assert b > prev, (
                f"triple {t['graph']}/{t['app']}/{t['policy']} samples "
                "not sorted ascending by L3 size"
            )
            prev = b


def test_per_triple_policy_le_lru(audit: dict) -> None:
    """For every triple, the policy's miss-rate at each L3 point should
    be ≤ LRU's at the same point (modulo 1 pp). This is the lit-faith
    "policy is at least as good as LRU on average" stance; rare cells
    can drift slightly but a systematic LRU-beats-policy pattern means
    the comparator swapped columns."""
    tolerance = 0.01
    flipped = []
    for t in audit["triples"]:
        for s in t["samples"]:
            if s["policy_miss_rate"] > s["lru_miss_rate"] + tolerance:
                flipped.append({
                    "graph":  t["graph"],
                    "app":    t["app"],
                    "policy": t["policy"],
                    "l3":     s["l3_size"],
                    "lru":    s["lru_miss_rate"],
                    "policy_mr": s["policy_miss_rate"],
                })
    # Some SRRIP cells genuinely exceed LRU by < 1 pp — allowed.
    # Hard ceiling: at most 5 per-cell flips across the audit.
    assert len(flipped) <= 5, (
        f"{len(flipped)} cells have policy_miss_rate > lru_miss_rate "
        f"by > 1 pp (first 5: " +
        "; ".join(f"{f['graph']}/{f['app']}/{f['policy']}@{f['l3']} "
                  f"{f['policy_mr']:.4f} vs {f['lru']:.4f}" for f in flipped[:5])
        + ")"
    )


def test_slope_max_below_unity(audit: dict) -> None:
    """No slope should exceed 1.0 per log2(L3) doubling — that would
    mean miss rate dropped by > 100 % per cache doubling, which is
    arithmetically impossible (miss rate is bounded by [0, 1])."""
    s = audit["summary"]
    assert s["max_slope_lru"] <= 1.0, (
        f"impossible LRU slope {s['max_slope_lru']}"
    )
    assert s["max_slope_policy"] <= 1.0, (
        f"impossible policy slope {s['max_slope_policy']}"
    )


def test_summary_counts_consistent(audit: dict) -> None:
    s = audit["summary"]
    assert s["triples_audited"] + s["triples_singleton"] == s["triple_count"]
    assert len(audit["triples"]) == s["triples_audited"]
    assert len(audit["violations_lru"]) == s["violations_lru"]
    assert len(audit["violations_policy"]) == s["violations_policy"]
    assert len(audit["saturated_triples"]) == s["saturated_count"]


def test_dedup_triples(audit: dict) -> None:
    seen = set()
    for t in audit["triples"]:
        k = (t["graph"], t["app"], t["policy"])
        assert k not in seen, f"duplicate triple {k}"
        seen.add(k)


def test_axis_coverage_floor(audit: dict) -> None:
    """Audited triples must span ≥ 4 graphs, ≥ 3 apps, ≥ 2 policies."""
    graphs   = {t["graph"]  for t in audit["triples"]}
    apps     = {t["app"]    for t in audit["triples"]}
    policies = {t["policy"] for t in audit["triples"]}
    assert len(graphs)   >= 4, f"only {len(graphs)} graphs"
    assert len(apps)     >= 3, f"only {len(apps)} apps"
    assert len(policies) >= 2, f"only {len(policies)} policies — "\
        "LRU rows for POPT_GE_GRASP/POPT_NEAR_GRASP_IF_BIG_GAP aren't "\
        "populated (those compare against GRASP); GRASP+SRRIP coverage "\
        "is the minimum"


def test_at_least_one_lru_drop_above_30pp(audit: dict) -> None:
    """The corpus must contain at least one triple where LRU miss rate
    drops by > 30 pp across the L3 sweep — confirms we're sampling a
    real cache-pressure regime, not just saturated workloads."""
    big = [t for t in audit["triples"] if t["lru_total_drop"] >= 0.30]
    assert len(big) >= 1, (
        "no triple shows ≥ 30 pp LRU drop across the L3 sweep — "
        "corpus is too saturated"
    )
