"""
Confidence gate 116 — lofo_robustness ↔ leave_one_graph_out parity + arithmetic.

Two complementary perturbation analyses of the per-app winner stability:
- lofo_robustness drops one *family* at a time (citation, mesh, road,
  social, web) from the (graph × l3) data and recomputes the winner;
- leave_one_graph_out drops one *graph* at a time from the same source
  and recomputes the winner.

Both are computed from oracle_gap.json. They disagree on the underlying
scope (lofo restricts to scope_l3_sizes=[1MB, 4MB, 8MB]; logo uses all
l3 sizes), so per-app full-corpus win_counts will differ between the
two artifacts and we deliberately do NOT compare them cell-for-cell.

What MUST match between them is the app-level fragility classification:
if dropping any family flips the winner, the app is fragile by lofo;
if dropping any graph flips the winner, the app is fragile by logo.
Empirically the two methods agree (bfs and sssp are fragile by both,
bc / cc / pr are robust by both). This gate locks that agreement and
also recomputes every internal arithmetic field of each artifact.

Cohen 1988 / Wilson are about effect-size views of the same wins;
this gate is about leave-one-out sensitivity — together with gate 115
they triangulate that the win-rate story is both effect-sized and
perturbation-robust.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[2] / "wiki" / "data"
LOFO_PATH = DATA / "lofo_robustness.json"
LOGO_PATH = DATA / "leave_one_graph_out.json"

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}
EXPECTED_FAMILIES = {"citation", "mesh", "road", "social", "web"}


@pytest.fixture(scope="module")
def lofo() -> dict:
    assert LOFO_PATH.exists(), f"missing artifact: {LOFO_PATH}"
    return json.loads(LOFO_PATH.read_text())


@pytest.fixture(scope="module")
def logo() -> dict:
    assert LOGO_PATH.exists(), f"missing artifact: {LOGO_PATH}"
    return json.loads(LOGO_PATH.read_text())


def _full_corpus_math(full: dict) -> tuple[int, int, bool, int]:
    """Returns (top_wins, runner_up_wins, unique_top, margin) recomputed."""
    wc = full["win_counts"]
    values = sorted(wc.values(), reverse=True)
    top = values[0]
    runner_up = values[1] if len(values) > 1 else 0
    unique = sum(1 for v in wc.values() if v == top) == 1
    return top, runner_up, unique, top - runner_up


# ---------------------------------------------------------------------------
# Group A — meta structure + parity
# ---------------------------------------------------------------------------

def test_both_artifacts_share_apps(lofo, logo):
    assert set(lofo["meta"]["apps"]) == EXPECTED_APPS
    assert set(logo["meta"]["apps"]) == EXPECTED_APPS


def test_lofo_meta_universe_is_expected(lofo):
    m = lofo["meta"]
    assert m["source"] == "wiki/data/oracle_gap.json"
    assert set(m["families"]) == EXPECTED_FAMILIES
    assert m["n_apps"] == 5
    assert m["n_families"] == 5
    assert m["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]
    assert m["n_rows_in_scope"] > 0
    assert m["n_fragile_apps"] + m["n_robust_apps"] == m["n_apps"]
    assert 0.0 <= m["robustness_fraction"] <= 1.0
    assert abs(m["robustness_fraction"] - m["n_robust_apps"] / m["n_apps"]) < 1e-6


def test_logo_meta_universe_is_expected(logo):
    m = logo["meta"]
    assert m["n_fragile_apps"] + m["n_robust_apps"] == 5
    assert len(m["graphs"]) == m["n_graphs"]
    assert m["n_rows"] > 0


def test_fragile_apps_match_between_perturbation_methods(lofo, logo):
    assert set(lofo["meta"]["fragile_apps"]) == set(logo["meta"]["fragile_apps"]), (
        f"lofo fragile {lofo['meta']['fragile_apps']} != logo fragile {logo['meta']['fragile_apps']}"
    )


def test_robust_apps_match_between_perturbation_methods(lofo, logo):
    assert set(lofo["meta"]["robust_apps"]) == set(logo["meta"]["robust_apps"]), (
        f"lofo robust {lofo['meta']['robust_apps']} != logo robust {logo['meta']['robust_apps']}"
    )


# ---------------------------------------------------------------------------
# Group B — meta ↔ per_app self-consistency (within each artifact)
# ---------------------------------------------------------------------------

def test_lofo_per_app_robust_classification_matches_meta(lofo):
    derived_robust = sorted(a for a, pd in lofo["per_app"].items() if pd["is_lofo_robust"])
    derived_fragile = sorted(a for a, pd in lofo["per_app"].items() if not pd["is_lofo_robust"])
    assert derived_robust == sorted(lofo["meta"]["robust_apps"])
    assert derived_fragile == sorted(lofo["meta"]["fragile_apps"])


def test_logo_per_app_robust_classification_matches_meta(logo):
    derived_robust = sorted(a for a, pd in logo["per_app"].items() if pd["is_logo_robust"])
    derived_fragile = sorted(a for a, pd in logo["per_app"].items() if not pd["is_logo_robust"])
    assert derived_robust == sorted(logo["meta"]["robust_apps"])
    assert derived_fragile == sorted(logo["meta"]["fragile_apps"])


# ---------------------------------------------------------------------------
# Group C — full_corpus arithmetic (both artifacts share the same shape)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["lofo", "logo"])
def test_full_corpus_arithmetic(which, lofo, logo):
    art = lofo if which == "lofo" else logo
    for app, pd in art["per_app"].items():
        fc = pd["full_corpus"]
        # win_counts policies are a subset of expected
        assert set(fc["win_counts"]) <= EXPECTED_POLICIES, f"{which}/{app}: unknown policies in win_counts"
        # top_policy must be a winner with maximal wins
        max_wins = max(fc["win_counts"].values())
        winners_at_top = {p for p, v in fc["win_counts"].items() if v == max_wins}
        assert fc["top_policy"] in winners_at_top, f"{which}/{app}: top_policy not at the max"
        top, runner_up, unique, margin = _full_corpus_math(fc)
        assert fc["top_wins"] == top, f"{which}/{app}: top_wins mismatch"
        assert fc["runner_up_wins"] == runner_up, f"{which}/{app}: runner_up_wins mismatch"
        assert fc["unique_top"] == unique, f"{which}/{app}: unique_top mismatch"
        assert fc["margin"] == margin, f"{which}/{app}: margin mismatch"


# ---------------------------------------------------------------------------
# Group D — drops + fragile list arithmetic (artifact-specific keys)
# ---------------------------------------------------------------------------

def test_lofo_drops_universe_is_families(lofo):
    for app, pd in lofo["per_app"].items():
        assert set(pd["drops"]) == EXPECTED_FAMILIES, f"{app}: lofo drop set != families"
        assert pd["n_drops"] == len(pd["drops"]) == 5


def test_logo_drops_universe_is_graphs(logo):
    expected_graphs = set(logo["meta"]["graphs"])
    for app, pd in logo["per_app"].items():
        assert set(pd["drops"]) == expected_graphs, f"{app}: logo drop set != meta.graphs"
        assert pd["n_drops"] == len(pd["drops"]) == len(expected_graphs)


def test_drops_same_winner_flag_matches_top_policy_and_unique_top(lofo, logo):
    for which, art in (("lofo", lofo), ("logo", logo)):
        for app, pd in art["per_app"].items():
            fc_top = pd["full_corpus"]["top_policy"]
            for key, drop in pd["drops"].items():
                expected_same = (drop["top_policy"] == fc_top) and drop["unique_top"]
                assert drop["same_winner_as_full"] == expected_same, (
                    f"{which}/{app}/{key}: same_winner_as_full={drop['same_winner_as_full']} "
                    f"but derived={expected_same} (top_policy={drop['top_policy']} vs fc={fc_top}, "
                    f"unique_top={drop['unique_top']})"
                )


def test_n_robust_drops_matches_count_of_same_winner_drops(lofo, logo):
    for which, art in (("lofo", lofo), ("logo", logo)):
        for app, pd in art["per_app"].items():
            expected = sum(1 for d in pd["drops"].values() if d["same_winner_as_full"])
            assert pd["n_robust_drops"] == expected, f"{which}/{app}: n_robust_drops mismatch"


def test_fragile_drops_list_matches_drops_with_winner_flip(lofo, logo):
    # lofo uses 'fragile_family_drops', logo uses 'fragile_drops'.
    for which, art, key in (("lofo", lofo, "fragile_family_drops"),
                            ("logo", logo, "fragile_drops")):
        for app, pd in art["per_app"].items():
            expected = sorted(k for k, d in pd["drops"].items() if not d["same_winner_as_full"])
            assert pd[key] == expected, (
                f"{which}/{app}: {key}={pd[key]} but derived={expected}"
            )


def test_is_robust_iff_no_fragile_drops(lofo, logo):
    for which, art, drop_key, flag_key in (
        ("lofo", lofo, "fragile_family_drops", "is_lofo_robust"),
        ("logo", logo, "fragile_drops", "is_logo_robust"),
    ):
        for app, pd in art["per_app"].items():
            expected = (len(pd[drop_key]) == 0)
            assert pd[flag_key] == expected, (
                f"{which}/{app}: {flag_key}={pd[flag_key]} but len({drop_key})={len(pd[drop_key])}"
            )
