"""LIT-Acc (gate 227): pytest invariants for the accesses-floor audit.

Pins floors, bucket distribution, per-axis presence, and the violation
count of `wiki/data/lit_faith_accesses.json`. Floors are calibrated
about an order of magnitude below today's empirical minimums so a real
warmup-truncation regression trips immediately without churn from
ordinary corpus variation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "wiki" / "data" / "lit_faith_accesses.json"


@pytest.fixture(scope="module")
def audit() -> dict:
    assert AUDIT_PATH.exists(), (
        f"{AUDIT_PATH} missing — run `make lit-accesses` (gate 227)"
    )
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_schema_keys(audit: dict) -> None:
    for key in ("schema_version", "summary", "floors", "buckets",
                "per_graph", "per_app", "per_l3", "per_policy",
                "per_graph_app", "violations", "per_row"):
        assert key in audit, f"missing top-level key {key!r}"
    assert audit["schema_version"] == 1


def test_summary_row_count_floor(audit: dict) -> None:
    s = audit["summary"]
    assert s["total_rows"] >= 250, (
        f"lit-faith corpus shrank below 250 rows ({s['total_rows']}); "
        "investigate whether comparator silently dropped graphs/apps."
    )
    assert s["production_rows"] >= 220
    assert s["smoke_rows"] >= 10


def test_no_floor_violations(audit: dict) -> None:
    viol = audit["violations"]
    assert viol == [], (
        f"{len(viol)} accesses-floor violations: " +
        "; ".join(f"{v['graph']}/{v['app']}/{v['policy']}/{v['l3_size']}"
                  f" acc={v['accesses']:,} < floor={v['floor']:,}"
                  for v in viol[:5])
    )


def test_production_min_accesses_floor(audit: dict) -> None:
    s = audit["summary"]
    # Today's actual minimum (web-Google) is ~735k. Floor at 500k so a
    # 30% drop trips; ordinary regen does not.
    assert s["production_min_accesses"] >= 500_000, (
        f"production min accesses regressed to {s['production_min_accesses']:,}"
    )


def test_production_median_accesses_floor(audit: dict) -> None:
    s = audit["summary"]
    # Today's median ~15.9M. Floor at 5M.
    assert s["production_median_accesses"] >= 5_000_000


def test_smoke_min_accesses_floor(audit: dict) -> None:
    s = audit["summary"]
    # email-Eu-core bfs is intentionally tiny but should never collapse
    # to zero accesses.
    assert s["smoke_min_accesses"] >= 20_000


def test_smoke_min_below_production_min(audit: dict) -> None:
    s = audit["summary"]
    assert s["smoke_min_accesses"] < s["production_min_accesses"], (
        "smoke graph (email-Eu-core) min accesses should remain below "
        "production min — otherwise the dev-smoke graph isn't actually a smoke."
    )


def test_corpus_axis_floors(audit: dict) -> None:
    s = audit["summary"]
    assert s["graph_count"] >= 6
    assert s["app_count"]   >= 4
    assert s["l3_count"]    >= 3
    assert s["policy_count"] >= 3


def test_floors_table_pinning(audit: dict) -> None:
    floors = audit["floors"]
    assert floors["production_per_app"] == {
        "bc":   1_000_000,
        "bfs":    500_000,
        "cc":   1_000_000,
        "pr":   1_000_000,
        "sssp": 500_000,
    }
    assert floors["smoke_per_app"]["bfs"] == 20_000
    assert floors["smoke_graphs"] == ["email-Eu-core"]


def test_bucket_definitions(audit: dict) -> None:
    names = [b["name"] for b in audit["buckets"]]
    assert names == ["tiny", "small", "medium", "large", "huge"]
    edges = [b["upper_exclusive"] for b in audit["buckets"]]
    assert edges == [100_000, 1_000_000, 10_000_000, 100_000_000, None]


def test_production_no_tiny_bucket(audit: dict) -> None:
    """Production graphs (excluding email-Eu-core) must have zero rows in
    the `tiny` bucket (< 100k accesses). If this fires, a production
    workload was silently truncated to a warmup-only trace."""
    tiny_prod = audit["summary"]["buckets_production"].get("tiny", 0)
    assert tiny_prod == 0, (
        f"{tiny_prod} production rows have <100k accesses — warmup leak"
    )


def test_production_medium_or_larger_majority(audit: dict) -> None:
    """At least 60% of production rows must reach the medium bucket
    (≥ 1M accesses) so the corpus is dominated by post-warmup traces."""
    buckets = audit["summary"]["buckets_production"]
    total = sum(buckets.values()) or 1
    heavy = sum(buckets.get(b, 0) for b in ("small", "medium", "large", "huge"))
    frac = heavy / total
    assert frac >= 0.6, (
        f"production rows in small+ buckets fraction {frac:.2%} < 60%"
    )


def test_production_large_bucket_floor(audit: dict) -> None:
    """At least 25% of production rows must be in the `large` or `huge`
    bucket (≥ 10M accesses) — guarantees the corpus exercises real
    cache-pressure regimes, not just warmup."""
    buckets = audit["summary"]["buckets_production"]
    total = sum(buckets.values()) or 1
    big = buckets.get("large", 0) + buckets.get("huge", 0)
    frac = big / total
    assert frac >= 0.25, (
        f"production large+huge bucket fraction {frac:.2%} < 25%"
    )


def test_every_graph_has_summary(audit: dict) -> None:
    for graph in audit["floors"]["smoke_graphs"]:
        assert graph in audit["per_graph"], (
            f"smoke graph {graph} missing from per_graph aggregates"
        )
    # Production graphs we always expect today.
    for graph in ("cit-Patents", "soc-LiveJournal1", "soc-pokec",
                   "web-Google", "roadNet-CA"):
        assert graph in audit["per_graph"], (
            f"production graph {graph} missing from per_graph aggregates"
        )


def test_per_graph_production_min_floor(audit: dict) -> None:
    """Every production graph's min accesses (across all apps/L3/policies)
    must clear 500k. Catches per-graph warmup regressions even when the
    corpus-wide minimum is healthy."""
    smoke = set(audit["floors"]["smoke_graphs"])
    bad = [(g, stat["min"]) for g, stat in audit["per_graph"].items()
           if g not in smoke and stat["min"] < 500_000]
    assert not bad, f"production graphs below 500k min accesses: {bad}"


def test_per_app_production_min_floor(audit: dict) -> None:
    """Every app aggregated across production graphs must clear 200k.
    The per-row floor table is tighter; this is the corpus-wide backstop."""
    smoke_graphs = set(audit["floors"]["smoke_graphs"])
    by_app: dict[str, list[int]] = {}
    for r in audit["per_row"]:
        if r["graph"] in smoke_graphs:
            continue
        by_app.setdefault(r["app"], []).append(r["accesses"])
    for app, vals in by_app.items():
        assert min(vals) >= 200_000, (
            f"app {app} production min accesses {min(vals):,} < 200,000"
        )


def test_per_row_count_matches_summary(audit: dict) -> None:
    assert len(audit["per_row"]) == audit["summary"]["total_rows"]


def test_per_row_buckets_sum_to_total(audit: dict) -> None:
    total = sum(audit["summary"]["buckets_all"].values())
    assert total == audit["summary"]["total_rows"]


def test_per_row_log10_finite(audit: dict) -> None:
    for r in audit["per_row"]:
        if r["log10"] is not None:
            assert 4.0 <= r["log10"] <= 10.0, (
                f"log10(accesses) out of range for {r['graph']}/{r['app']}: {r['log10']}"
            )


def test_per_row_smoke_flag_consistency(audit: dict) -> None:
    smoke_graphs = set(audit["floors"]["smoke_graphs"])
    for r in audit["per_row"]:
        assert r["smoke"] == (r["graph"] in smoke_graphs), (
            f"smoke flag mismatch on row {r}"
        )


def test_violation_shortfall_pct_well_formed(audit: dict) -> None:
    for v in audit["violations"]:
        assert 0.0 < v["shortfall_pct"] <= 100.0
        assert v["shortfall"] == v["floor"] - v["accesses"]


def test_email_eu_core_bfs_is_smallest(audit: dict) -> None:
    """email-Eu-core/bfs is the canonical tiny dev-smoke trace; if any
    other (graph, app) drops below it, something is wrong."""
    key = "email-Eu-core::bfs"
    assert key in audit["per_graph_app"]
    eec_bfs_min = audit["per_graph_app"][key]["min"]
    smaller = [(k, stat["min"]) for k, stat in audit["per_graph_app"].items()
               if k != key and stat["min"] < eec_bfs_min]
    assert not smaller, (
        f"non-smoke (graph, app) trace smaller than email-Eu-core/bfs: {smaller}"
    )


def test_dedup_per_row(audit: dict) -> None:
    seen = set()
    for r in audit["per_row"]:
        k = (r["graph"], r["app"], r["policy"], r["l3_size"])
        assert k not in seen, f"duplicate per_row key {k}"
        seen.add(k)


def test_violation_count_matches_summary(audit: dict) -> None:
    assert audit["summary"]["violation_count"] == len(audit["violations"])
