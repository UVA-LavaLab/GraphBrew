"""Gate 54: cross-generator gap_pp parity invariants.

Three load-bearing aggregators surface the same per-(app, policy, L3)
gap_pp data through different schemas. If any of them silently drifts
(stale cache, rounding change, aggregation bug) the paper's narrative
breaks invisibly. This gate verifies byte-equivalent agreement (within
a 1e-3 pp tolerance) across all three.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = REPO_ROOT / "wiki" / "data" / "cross_generator_gap_parity.json"

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}
EXPECTED_L3 = {"1MB", "4MB", "8MB"}


@pytest.fixture(scope="module")
def payload():
    if not PAYLOAD.exists():
        pytest.skip(f"missing {PAYLOAD}; run `make lit-cross-generator-gap-parity`")
    return json.loads(PAYLOAD.read_text())


def test_meta_tolerance_tight(payload):
    assert payload["meta"]["tolerance_pp"] <= 1e-3


def test_meta_scope_is_paper_l3(payload):
    assert set(payload["meta"]["scope_l3_sizes"]) == EXPECTED_L3


def test_zero_mismatches(payload):
    """The hard invariant: every shared triple agrees within tolerance."""
    assert payload["meta"]["n_mismatches"] == 0, (
        f"{payload['meta']['n_mismatches']} cross-generator gap_pp "
        f"mismatches; see wiki/data/cross_generator_gap_parity.md"
    )


def test_full_coverage_of_paper_grid(payload):
    """All 5 apps x 4 policies x 3 L3 = 60 paper-grid triples present."""
    assert payload["meta"]["n_cells_checked"] >= 60
    triples = {
        (c["app"], c["policy"], c["l3_size"]) for c in payload["cells"]
    }
    for a in EXPECTED_APPS:
        for p in EXPECTED_POLICIES:
            for l3 in EXPECTED_L3:
                assert (a, p, l3) in triples, f"missing triple ({a}, {p}, {l3})"


def test_every_paper_triple_has_all_three_sources(payload):
    """No triple is silently missing from one of the three generators."""
    paper_triples = {
        (a, p, l3)
        for a in EXPECTED_APPS
        for p in EXPECTED_POLICIES
        for l3 in EXPECTED_L3
    }
    by_triple = {(c["app"], c["policy"], c["l3_size"]): c for c in payload["cells"]}
    for t in paper_triples:
        c = by_triple[t]
        assert c["all_three_present"], (
            f"triple {t} missing from at least one generator: "
            f"raw={c['raw_mean_gap_pp']}, auc={c['auc_trajectory_gap_pp']}, "
            f"slope={c['slope_gap_pp']}"
        )


def test_spread_uniformly_tiny(payload):
    """Every spread should be effectively zero, not just under tolerance."""
    threshold = payload["meta"]["tolerance_pp"]
    over = [c for c in payload["cells"] if c["spread_pp"] > threshold]
    assert not over, f"{len(over)} cells exceed tolerance: {over[:3]}"


def test_raw_means_have_at_least_5_graphs(payload):
    """Each (app, policy, L3) mean should aggregate >=5 per-graph cells."""
    sparse = [
        c for c in payload["cells"]
        if c["all_three_present"] and (c["n_graphs_in_raw"] or 0) < 5
    ]
    assert not sparse, (
        f"{len(sparse)} cells have <5 raw graphs in their mean: "
        f"{[(c['app'], c['policy'], c['l3_size'], c['n_graphs_in_raw']) for c in sparse[:5]]}"
    )


def test_known_anchor_pr_popt_1mb(payload):
    """Pin pr/POPT/1MB to ~0.0 pp across all three generators."""
    by_triple = {(c["app"], c["policy"], c["l3_size"]): c for c in payload["cells"]}
    c = by_triple[("pr", "POPT", "1MB")]
    assert abs(c["raw_mean_gap_pp"] - 0.0) < 0.02
    assert abs(c["auc_trajectory_gap_pp"] - 0.0) < 0.02
    assert abs(c["slope_gap_pp"] - 0.0) < 0.02


def test_known_anchor_pr_lru_8mb(payload):
    """Pin pr/LRU/8MB to ~5.88 pp across all three generators."""
    by_triple = {(c["app"], c["policy"], c["l3_size"]): c for c in payload["cells"]}
    c = by_triple[("pr", "LRU", "8MB")]
    assert abs(c["raw_mean_gap_pp"] - 5.8758) < 0.10
    assert abs(c["auc_trajectory_gap_pp"] - 5.8758) < 0.10
    assert abs(c["slope_gap_pp"] - 5.8758) < 0.10


def test_popt_beats_lru_at_largest_l3(payload):
    """Cross-gen sanity: at 8MB, POPT mean <= LRU mean for every app.

    The reverse can happen at 1MB (cache too small for any policy to
    help — see bfs/1MB and bc/1MB), so only assert at the largest
    paper L3.
    """
    by_triple = {(c["app"], c["policy"], c["l3_size"]): c for c in payload["cells"]}
    violations = []
    for a in EXPECTED_APPS:
        popt = by_triple[(a, "POPT", "8MB")]["raw_mean_gap_pp"]
        lru = by_triple[(a, "LRU", "8MB")]["raw_mean_gap_pp"]
        if popt > lru:
            violations.append((a, popt, lru))
    assert not violations, f"POPT > LRU @ 8MB detected: {violations}"


def test_anchor_against_oracle_gap_summary(payload):
    """Cross-check raw means match aggregate fields in oracle_gap.json summary."""
    raw_path = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
    if not raw_path.exists():
        pytest.skip("oracle_gap.json missing")
    raw = json.loads(raw_path.read_text())
    # Use the raw mean for a single anchor and re-derive from rows to confirm
    # parity logic doesn't drift from the source-of-truth aggregator.
    by_triple = {(c["app"], c["policy"], c["l3_size"]): c for c in payload["cells"]}
    by_l3_pol = {}
    for r in raw["rows"]:
        if r["l3_size"] not in EXPECTED_L3 or r["app"] != "pr":
            continue
        by_l3_pol.setdefault(("pr", r["policy"], r["l3_size"]), []).append(
            float(r["gap_pp"])
        )
    for k, vs in by_l3_pol.items():
        expected = sum(vs) / len(vs)
        actual = by_triple[k]["raw_mean_gap_pp"]
        assert abs(actual - expected) < 1e-3, (
            f"parity raw-mean for {k} drifted from oracle_gap.json: "
            f"actual={actual}, expected={expected}"
        )


def test_no_negative_gap_in_means(payload):
    """A negative mean gap_pp at the corpus level would be physically wrong."""
    neg = [c for c in payload["cells"] if (c["raw_mean_gap_pp"] or 0) < 0]
    assert not neg, f"negative mean gap_pp found: {neg[:3]}"
