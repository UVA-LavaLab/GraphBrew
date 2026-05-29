"""Gate 95 — family clustering vs sensitivity vs winner-table 3-way agreement.

Three independent artifacts answer the question "which policy wins where"
from different statistical angles:

- ``family_policy_auc_clustering``: argmax of oracle-gap AUC per
  (family, app), clustered globally; emits a deviation_set that pins
  exactly which (family, app) pairs are allowed to differ from the
  global winner.
- ``family_sensitivity``: bootstrap-style claim stability over family
  relabelings; publishes 7 canonical claims with their stable-fraction.
- ``policy_winner_table``: cell-by-cell winner (lowest miss-rate)
  aggregated to (app, family, policy, regime) wins.

If the three drift apart silently — say a generator change that flips
a per-family winner without updating the clustering pin — every family
narrative in the paper becomes load-bearing-but-untestable.

This gate locks the global-winner partition, the AUC-clustering
deviation_set, the 3 stable family-sensitivity claims, the per-app
and per-family cell-count totals, and the cross-artifact requirement
that the cell-wins argmax agrees with the AUC-derived global winner.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FPAC_PATH = REPO / "wiki/data/family_policy_auc_clustering.json"
FS_PATH = REPO / "wiki/data/family_sensitivity.json"
PWT_PATH = REPO / "wiki/data/policy_winner_table.json"

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_POLICIES = {"GRASP", "LRU", "POPT", "SRRIP"}
EXPECTED_FAMILIES = {"citation", "mesh", "road", "social", "web"}
EXPECTED_QUALIFYING_FAMILIES = {"citation", "social", "web"}
EXPECTED_NON_QUALIFYING_FAMILIES = {"mesh", "road"}

EXPECTED_GLOBAL_WINNER = {
    "bc": "GRASP",
    "bfs": "POPT",
    "cc": "GRASP",
    "pr": "POPT",
    "sssp": "POPT",
}
EXPECTED_GLOBAL_CLUSTERS = {
    "GRASP": {"bc", "cc"},
    "POPT": {"bfs", "pr", "sssp"},
}
EXPECTED_DEVIATIONS = frozenset(
    {
        ("citation", "bfs"),
        ("citation", "sssp"),
    }
)

EXPECTED_APP_CELL_COUNT = {"bc": 23, "bfs": 23, "cc": 20, "pr": 28, "sssp": 20}
EXPECTED_FAMILY_CELL_COUNT = {
    "citation": 15,
    "mesh": 5,
    "road": 25,
    "social": 54,
    "web": 15,
}
EXPECTED_TOTAL_CELLS = 114

EXPECTED_CLAIM_COUNT = 7
EXPECTED_STABILITY_FLOOR = 0.95
EXPECTED_STABLE_CLAIMS = frozenset(
    {
        "GRASP < LRU on social",
        "POPT < GRASP on road",
        "POPT < LRU on social",
    }
)


def _load(path: Path) -> dict:
    assert path.exists(), f"missing artifact: {path}"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# family_policy_auc_clustering — single-artifact invariants
# ---------------------------------------------------------------------------


def test_fpac_meta_scope_is_locked():
    meta = _load(FPAC_PATH)["meta"]
    assert set(meta["apps"]) == EXPECTED_APPS
    assert set(meta["policies"]) == EXPECTED_POLICIES
    assert set(meta["families"]) == EXPECTED_FAMILIES
    assert set(meta["qualifying_families"]) == EXPECTED_QUALIFYING_FAMILIES
    assert meta["n_families_with_intra_dominates"] == len(EXPECTED_QUALIFYING_FAMILIES)


def test_fpac_global_winner_by_app_is_locked():
    meta = _load(FPAC_PATH)["meta"]
    assert meta["global_winner_by_app"] == EXPECTED_GLOBAL_WINNER


def test_fpac_global_clusters_cover_all_apps_disjointly():
    meta = _load(FPAC_PATH)["meta"]
    clusters = {pol: set(apps) for pol, apps in meta["global_clusters"].items()}
    assert clusters == EXPECTED_GLOBAL_CLUSTERS
    seen = set()
    for apps in clusters.values():
        assert seen.isdisjoint(apps), f"app appears in multiple clusters: {apps & seen}"
        seen |= apps
    assert seen == EXPECTED_APPS


def test_fpac_deviation_set_observed_equals_pinned_and_verdict_pass():
    meta = _load(FPAC_PATH)["meta"]
    ds = meta["deviation_set"]
    observed = frozenset((d["family"], d["app"]) for d in ds["observed"])
    pinned = frozenset((d["family"], d["app"]) for d in ds["pinned"])
    assert observed == pinned, (
        f"observed != pinned: only-in-observed={observed - pinned} only-in-pinned={pinned - observed}"
    )
    assert observed == EXPECTED_DEVIATIONS
    assert ds["gone_vs_pin"] == []
    assert ds["new_vs_pin"] == []
    assert meta["cluster_invariance_verdict"] == "PASS"


def test_fpac_qualified_families_intra_dominates_and_non_qualified_have_no_winner_by_app():
    per_family = _load(FPAC_PATH)["per_family"]
    for fam in EXPECTED_QUALIFYING_FAMILIES:
        payload = per_family[fam]
        assert payload["qualified"] is True
        assert payload["intra_dominates"] is True
        assert "winner_by_app" in payload, f"qualified family {fam} missing winner_by_app"
        assert payload["intra_cluster_mean_r"] > payload["inter_cluster_mean_r"], (
            f"family {fam}: intra_cluster_mean_r={payload['intra_cluster_mean_r']} "
            f"!> inter_cluster_mean_r={payload['inter_cluster_mean_r']}"
        )
    for fam in EXPECTED_NON_QUALIFYING_FAMILIES:
        payload = per_family[fam]
        assert payload["qualified"] is False
        assert "winner_by_app" not in payload, f"non-qualified family {fam} has winner_by_app"


# ---------------------------------------------------------------------------
# family_sensitivity — single-artifact invariants
# ---------------------------------------------------------------------------


def test_fs_canonical_claims_and_state_consistent():
    fs = _load(FS_PATH)
    assert len(fs["canonical_claims"]) == EXPECTED_CLAIM_COUNT
    state_keys = set(fs["canonical_state"].keys())
    claim_labels = {c["claim"] for c in fs["canonical_claims"]}
    assert state_keys == claim_labels, (
        f"canonical_state keys disagree with canonical_claims labels: "
        f"only-in-state={state_keys - claim_labels} only-in-claims={claim_labels - state_keys}"
    )


def test_fs_stable_claim_set_matches_expected():
    fs = _load(FS_PATH)
    assert math.isclose(fs["meta"]["stability_floor"], EXPECTED_STABILITY_FLOOR)
    floor = fs["meta"]["stability_floor"]
    stable = frozenset(label for label, frac in fs["canonical_state"].items() if frac >= floor)
    assert stable == EXPECTED_STABLE_CLAIMS, (
        f"stable claim set drifted: only-in-actual={stable - EXPECTED_STABLE_CLAIMS} "
        f"only-in-expected={EXPECTED_STABLE_CLAIMS - stable}"
    )


def test_fs_canonical_claims_n_match_family_cell_counts():
    fs = _load(FS_PATH)
    for claim in fs["canonical_claims"]:
        fam = claim["family"]
        expected = EXPECTED_FAMILY_CELL_COUNT[fam]
        assert claim["n_a"] == expected, (
            f"claim '{claim['claim']}': n_a={claim['n_a']} != expected family count {expected}"
        )
        assert claim["n_b"] == expected, (
            f"claim '{claim['claim']}': n_b={claim['n_b']} != expected family count {expected}"
        )


# ---------------------------------------------------------------------------
# policy_winner_table — single-artifact invariants
# ---------------------------------------------------------------------------


def test_pwt_total_cells_and_wins_by_policy_sum_to_total():
    pwt = _load(PWT_PATH)
    summary = pwt["summary"]
    assert summary["n_cells"] == EXPECTED_TOTAL_CELLS
    assert len(pwt["cells"]) == EXPECTED_TOTAL_CELLS
    assert sum(summary["wins_by_policy"].values()) == EXPECTED_TOTAL_CELLS


def test_pwt_wins_by_app_matches_expected_cell_counts():
    summary = _load(PWT_PATH)["summary"]
    for app, expected in EXPECTED_APP_CELL_COUNT.items():
        assert sum(summary["wins_by_app"][app].values()) == expected, (
            f"app {app}: wins_by_app sum != expected {expected}"
        )


def test_pwt_wins_by_family_matches_expected_cell_counts():
    summary = _load(PWT_PATH)["summary"]
    for fam, expected in EXPECTED_FAMILY_CELL_COUNT.items():
        assert sum(summary["wins_by_family"][fam].values()) == expected, (
            f"family {fam}: wins_by_family sum != expected {expected}"
        )


# ---------------------------------------------------------------------------
# Cross-artifact agreement
# ---------------------------------------------------------------------------


def test_xartifact_pwt_argmax_per_app_equals_fpac_global_winner():
    pwt_wins = _load(PWT_PATH)["summary"]["wins_by_app"]
    fpac_winner = _load(FPAC_PATH)["meta"]["global_winner_by_app"]
    for app in EXPECTED_APPS:
        wins = pwt_wins[app]
        # Tie-break by alphabetical (deterministic) — but if there's a tie at the top,
        # the gate exposes the ambiguity. The current data has no top-tie on any app.
        ordered = sorted(wins.items(), key=lambda kv: (-kv[1], kv[0]))
        top_pol, top_count = ordered[0]
        if len(ordered) > 1 and ordered[1][1] == top_count:
            raise AssertionError(
                f"app {app}: top-policy tie between {top_pol} and {ordered[1][0]} at {top_count} wins"
            )
        assert top_pol == fpac_winner[app], (
            f"app {app}: pwt argmax={top_pol} disagrees with fpac global winner={fpac_winner[app]}"
        )


def test_xartifact_deviation_set_recomputable_from_per_family_qualified_winners():
    fpac = _load(FPAC_PATH)
    global_winner = fpac["meta"]["global_winner_by_app"]
    recomputed = set()
    for fam, payload in fpac["per_family"].items():
        if not payload.get("qualified"):
            continue
        for app, w in payload["winner_by_app"].items():
            if w != global_winner[app]:
                recomputed.add((fam, app))
    pinned = frozenset(
        (d["family"], d["app"]) for d in fpac["meta"]["deviation_set"]["pinned"]
    )
    assert frozenset(recomputed) == pinned == EXPECTED_DEVIATIONS, (
        f"recomputed={recomputed}, pinned={pinned}, expected={EXPECTED_DEVIATIONS}"
    )
