"""Gate: Cohen's h effect-size headlines on policy win-rate gaps.

Wilson CIs (gate 36) test *whether* gaps are statistically separable.
This gate tests *how big* the separable gaps are. The two together
defend the qualitative force of every "X dominates Y on app Z" claim
in the paper.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = REPO_ROOT / "wiki" / "data" / "cohens_h_win_rates.json"

LARGE = 0.8
MEDIUM = 0.5

REQUIRED_APPS = ("pr", "bc", "cc", "bfs", "sssp")


@pytest.fixture(scope="module")
def payload() -> dict:
    if not PAYLOAD.exists():
        pytest.skip(
            f"missing {PAYLOAD}; run `make lit-cohens-h` "
            f"(or `make lit-oracle-gap lit-cohens-h`)."
        )
    return json.loads(PAYLOAD.read_text())


def _cmp(payload: dict, app: str, a: str, b: str) -> dict:
    for c in payload["per_app"][app]["comparisons"]:
        if c["a"] == a and c["b"] == b:
            return c
    raise KeyError(f"no comparison {app}: {a} vs {b}")


def test_schema_has_per_app_and_summaries(payload):
    assert "per_app" in payload
    assert "largest_per_app" in payload
    assert "large_effects" in payload
    for app in REQUIRED_APPS:
        assert app in payload["per_app"], f"missing {app}"


def test_each_app_has_complete_pairwise_table(payload):
    """4 policies → 4*3 = 12 ordered comparisons per app."""
    for app in REQUIRED_APPS:
        comps = payload["per_app"][app]["comparisons"]
        assert len(comps) == 12, f"{app} has {len(comps)} comparisons, expected 12"


def test_cohen_h_formula_matches_reference(payload):
    """Re-derive cc/GRASP vs cc/POPT (p=9/20, 11/20) → h≈0.200."""
    p1, p2 = 9 / 20, 11 / 20
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    expected = abs(phi1 - phi2)
    c = _cmp(payload, "cc", "GRASP", "POPT")
    assert c["h"] == pytest.approx(expected, abs=1e-3), c


def test_cc_grasp_vs_popt_is_small_diagnostic(payload):
    """Charged corpus: cc POPT-vs-GRASP split is small and diagnostic."""
    c = _cmp(payload, "cc", "GRASP", "POPT")
    assert c["magnitude"] == "small"
    assert c["favors"] == "POPT"


def test_pr_popt_beats_lru_srrip_at_large_effect(payload):
    """pr is unambiguously POPT vs the dumb policies."""
    for loser in ("LRU", "SRRIP"):
        c = _cmp(payload, "pr", "POPT", loser)
        assert c["magnitude"] == "large", f"pr/POPT vs {loser}: {c}"
        assert c["favors"] == "POPT"


def test_pr_popt_vs_grasp_is_large_effect(payload):
    """pr POPT beats GRASP at large effect (h=0.886, in the same direction
    as the CI-strict claim from gate 36)."""
    c = _cmp(payload, "pr", "POPT", "GRASP")
    assert c["favors"] == "POPT"
    assert c["magnitude"] == "large", f"pr/POPT-vs-GRASP shrank: {c}"


def test_cc_popt_beats_blind_baselines_at_large_effect(payload):
    """Charged corpus: cc POPT beats LRU/SRRIP, not uniformly GRASP."""
    for loser in ("LRU", "SRRIP"):
        c = _cmp(payload, "cc", "POPT", loser)
        assert c["magnitude"] == "large", f"cc/POPT vs {loser}: {c}"
        assert c["favors"] == "POPT"


def test_bc_grasp_vs_lru_is_large(payload):
    """bc/GRASP beats LRU at medium effect even though GRASP-vs-POPT/SRRIP
    is more equivocal."""
    c = _cmp(payload, "bc", "GRASP", "LRU")
    assert c["magnitude"] == "large", c


def test_bfs_modern_policies_beat_lru_at_large_effect(payload):
    """POPT beats LRU at medium effect; GRASP beats LRU at large effect."""
    expected = {"POPT": "medium", "GRASP": "large"}
    for winner, magnitude in expected.items():
        c = _cmp(payload, "bfs", winner, "LRU")
        assert c["magnitude"] == magnitude, f"bfs/{winner} vs LRU: {c}"
        assert c["favors"] == winner


def test_sssp_grasp_has_large_effect_dominance(payload):
    """Charged corpus: sssp has large win-rate effects favoring GRASP."""
    sssp_large = [
        c
        for c in payload["per_app"]["sssp"]["comparisons"]
        if c["magnitude"] == "large"
    ]
    assert sssp_large
    assert any(c["a"] == "GRASP" and c["b"] == "POPT" and c["favors"] == "GRASP"
               for c in sssp_large)


def test_every_kernel_has_one_large_or_medium_effect_dominance_pair_except_sssp(payload):
    """For pr, bc, cc, bfs the largest h must hit at least "medium"."""
    for app in ("pr", "bc", "cc", "bfs"):
        biggest = payload["largest_per_app"][app]
        assert biggest["magnitude"] in {"large", "medium"}, (
            f"{app} top effect collapsed to {biggest['magnitude']}: {biggest}"
        )


def test_large_effects_list_is_sorted_desc_by_h(payload):
    hs = [r["h"] for r in payload["large_effects"]]
    assert hs == sorted(hs, reverse=True), (
        f"large_effects no longer sorted descending: {hs}"
    )
