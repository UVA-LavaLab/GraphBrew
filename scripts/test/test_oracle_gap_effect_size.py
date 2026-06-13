"""Gate: Cliff's delta + Mann-Whitney U on raw oracle-gap distributions.

Wilson (gate 36) and Cohen's h (gate 37) test win-count claims.
This gate tests the *gap-magnitude* claims by running standard
nonparametric tools on the raw ``gap_pp`` distributions per
(app, policy).

Key advantages:
* Nonparametric — outlier-robust, no normality assumption.
* Cliff's delta is rank-based, so it survives the wide gap_pp
  ranges in the data (LRU can have 20pp gaps; POPT 0pp gaps).
* MW-U gives an honest p-value on stochastic dominance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = REPO_ROOT / "wiki" / "data" / "oracle_gap_effect_size.json"

LARGE = 0.474

REQUIRED_APPS = ("pr", "bc", "cc", "bfs", "sssp")
PVAL_STRONG = 0.001
PVAL_WEAK = 0.01


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip(
            f"missing {PAYLOAD}; run `make lit-gap-effect-size` "
            f"(or `make lit-oracle-gap lit-gap-effect-size`)."
        )
    return json.loads(PAYLOAD.read_text())


def _cmp(payload: dict, app: str, a: str, b: str) -> dict:
    for c in payload["per_app"][app]["comparisons"]:
        if c["a"] == a and c["b"] == b:
            return c
    raise KeyError(f"no comparison {app}: {a} vs {b}")


def test_schema_has_per_app_and_large_pairs(payload):
    assert "per_app" in payload
    assert "large_negative_deltas" in payload
    for app in REQUIRED_APPS:
        assert app in payload["per_app"], f"missing {app}"


def test_each_app_has_complete_pairwise_table(payload):
    """4 policies → 12 ordered pairs per app."""
    for app in REQUIRED_APPS:
        comps = payload["per_app"][app]["comparisons"]
        assert len(comps) == 12, f"{app} has {len(comps)} comparisons, expected 12"


def test_cliffs_delta_symmetric_within_sign(payload):
    """For any (a, b), d(a,b) = -d(b,a) by definition."""
    for app in REQUIRED_APPS:
        comps = payload["per_app"][app]["comparisons"]
        lookup = {(c["a"], c["b"]): c["cliffs_delta_a_minus_b"] for c in comps}
        for (a, b), d_ab in lookup.items():
            d_ba = lookup.get((b, a))
            if d_ba is None:
                continue
            assert d_ab == pytest.approx(-d_ba, abs=1e-9), (
                f"{app}: d({a},{b})={d_ab} not negated as d({b},{a})={d_ba}"
            )


def test_pr_popt_dominates_lru_srrip_large_pval_zero(payload):
    """pr/POPT vs LRU and SRRIP: d ≤ -0.71, MW p essentially 0.

    Re-pinned 2026-06-13 for charged-POPT corpus.
    """
    for loser in ("LRU", "SRRIP"):
        c = _cmp(payload, "pr", "POPT", loser)
        assert c["magnitude"] == "large"
        assert c["cliffs_delta_a_minus_b"] <= -0.71, (
            f"pr/POPT vs {loser}: d collapsed: {c}"
        )
        assert c["mannwhitney_p"] < PVAL_STRONG, c
        assert c["stochastically_smaller"] == "POPT"


def test_pr_popt_beats_grasp_at_large_effect(payload):
    """pr POPT vs GRASP: large effect AND MW p < 0.01."""
    c = _cmp(payload, "pr", "POPT", "GRASP")
    assert c["magnitude"] == "large", c
    assert c["cliffs_delta_a_minus_b"] < -0.4, c
    assert c["mannwhitney_p"] < PVAL_WEAK, c
    assert c["stochastically_smaller"] == "POPT"


def test_cc_popt_dominates_lru_srrip_at_large_effect(payload):
    """Charged corpus: cc is not uniformly GRASP-favored; POPT beats blind baselines."""
    for loser in ("LRU", "SRRIP"):
        c = _cmp(payload, "cc", "POPT", loser)
        assert c["magnitude"] == "large", f"cc/POPT vs {loser}: {c}"
        assert c["mannwhitney_p"] < PVAL_WEAK, c
        assert c["stochastically_smaller"] == "POPT"


def test_bc_grasp_beats_lru_large(payload):
    """bc/GRASP vs LRU: large effect, MW p < 0.01."""
    c = _cmp(payload, "bc", "GRASP", "LRU")
    assert c["magnitude"] == "large", c
    assert c["mannwhitney_p"] < PVAL_WEAK, c


def test_bfs_popt_beats_lru_srrip_medium_or_better(payload):
    """Charged corpus: bfs/POPT still beats LRU/SRRIP, but effect is medium."""
    for loser in ("LRU", "SRRIP"):
        c = _cmp(payload, "bfs", "POPT", loser)
        assert c["magnitude"] in {"medium", "large"}, f"bfs/POPT vs {loser}: {c}"
        assert c["mannwhitney_p"] < PVAL_WEAK, c


def test_sssp_grasp_large_effect_vs_lru_popt(payload):
    """Charged corpus: sssp now has large raw-gap separation favoring GRASP."""
    for loser in ("LRU", "POPT"):
        c = _cmp(payload, "sssp", "GRASP", loser)
        assert c["magnitude"] == "large", c
        assert c["stochastically_smaller"] == "GRASP"


def test_cc_grasp_vs_popt_is_diagnostic_not_claim(payload):
    """Charged corpus: cc POPT-vs-GRASP is weak and graph-dependent."""
    c = _cmp(payload, "cc", "GRASP", "POPT")
    assert c["magnitude"] == "negligible", c
    assert c["mannwhitney_p"] > 0.05, c


def test_large_negative_deltas_list_sorted_ascending(payload):
    """large_negative_deltas sorted by d ascending (most-negative first)."""
    ds = [r["cliffs_delta_a_minus_b"] for r in payload["large_negative_deltas"]]
    assert ds == sorted(ds), f"large_negative_deltas not sorted: {ds}"
    if ds:
        assert ds[0] < 0, f"first entry not negative: {ds[0]}"
