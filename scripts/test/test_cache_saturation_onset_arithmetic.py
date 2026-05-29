"""Gate 118 — cache_saturation_onset.json arithmetic + policy ranking.

Locks the L3 saturation-onset table built from oracle_gap_auc.json:

    - per-app/per-policy octave arithmetic (delta_gap_pp,
      slope_pp_per_octave) recomputed from the source trajectories;
    - saturation_onset reproduced from the exact step-down rule used by
      cache_saturation_onset._onset(): the smallest L3 from which all
      remaining octaves shrink within (-threshold, 0] (anti-scaling
      octaves never count as saturated);
    - saturated_within_paper_l3 == (onset != 'never');
    - per_policy aggregation (onset_counts, n_saturated,
      n_never_saturated, sorted apps) matches per_app;
    - saturation_rank_by_policy reproduced from the documented sort key
      (more 1MB-saturated wins, then more 4MB-saturated, then fewer
      never-saturated);
    - meta consistency (apps, policies, n_apps, n_policies, source).

These tests pin the "POPT saturates earliest, SRRIP last" narrative the
paper uses to justify the 4MB ceiling as the saturation regime.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/cache_saturation_onset.json")
SOURCE = Path("wiki/data/oracle_gap_auc.json")

THRESH = 0.5
L3_MB = {"1MB": 1, "4MB": 4, "8MB": 8, "16MB": 16, "32MB": 32}
SLOPE_TOL = 1e-3
GAP_TOL = 1e-3


@pytest.fixture(scope="module")
def data():
    assert ARTIFACT.exists(), f"missing artifact: {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def source():
    assert SOURCE.exists(), f"missing source artifact: {SOURCE}"
    return json.loads(SOURCE.read_text())


def _onset(octaves: list[dict]) -> str:
    for i, oct_ in enumerate(octaves):
        if all(
            o["delta_gap_pp"] > -THRESH and o["delta_gap_pp"] <= 0
            for o in octaves[i:]
        ):
            return oct_["from"]
    last = octaves[-1]
    if last["delta_gap_pp"] > -THRESH and last["delta_gap_pp"] <= 0:
        return last["from"]
    return "never"


# ── group 1: meta + structure ────────────────────────────────────────────


def test_meta_threshold_and_scope(data):
    assert data["meta"]["saturation_threshold_pp_per_octave"] == THRESH
    assert data["meta"]["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]
    assert data["meta"]["source"].endswith("oracle_gap_auc.json")


def test_meta_apps_and_policies_match_per_app(data):
    apps = sorted(data["per_app"].keys())
    assert data["meta"]["apps"] == apps
    assert data["meta"]["n_apps"] == len(apps)
    policies = sorted(data["per_policy"].keys())
    assert data["meta"]["policies"] == policies
    expected_npol = max((len(b) for b in data["per_app"].values()), default=0)
    assert data["meta"]["n_policies"] == expected_npol


# ── group 2: octave arithmetic from source ───────────────────────────────


def test_octave_gap_endpoints_match_source(data, source):
    for app, app_blob in data["per_app"].items():
        traj_by_pol = source["per_app"][app]["trajectory_by_policy"]
        for pol, blob in app_blob.items():
            traj = traj_by_pol[pol]
            for oct_ in blob["octaves"]:
                assert math.isclose(oct_["gap_from"], traj[oct_["from"]], abs_tol=GAP_TOL), (
                    f"{app}/{pol} {oct_['from']}→{oct_['to']}: gap_from"
                )
                assert math.isclose(oct_["gap_to"], traj[oct_["to"]], abs_tol=GAP_TOL), (
                    f"{app}/{pol} {oct_['from']}→{oct_['to']}: gap_to"
                )


def test_octave_delta_and_slope_correct(data):
    for app, app_blob in data["per_app"].items():
        for pol, blob in app_blob.items():
            for oct_ in blob["octaves"]:
                expected_delta = oct_["gap_to"] - oct_["gap_from"]
                d_log = math.log2(L3_MB[oct_["to"]]) - math.log2(L3_MB[oct_["from"]])
                expected_slope = -expected_delta / d_log if d_log > 0 else 0.0
                assert math.isclose(oct_["delta_gap_pp"], expected_delta, abs_tol=GAP_TOL), (
                    f"{app}/{pol}: delta_gap_pp mismatch"
                )
                assert math.isclose(oct_["slope_pp_per_octave"], expected_slope, abs_tol=SLOPE_TOL), (
                    f"{app}/{pol}: slope_pp_per_octave mismatch"
                )


def test_final_octave_fields_match_last_octave(data):
    for app, app_blob in data["per_app"].items():
        for pol, blob in app_blob.items():
            last = blob["octaves"][-1]
            assert math.isclose(
                blob["final_octave_slope_pp"], last["slope_pp_per_octave"], abs_tol=SLOPE_TOL
            ), f"{app}/{pol}: final_octave_slope_pp"
            assert math.isclose(
                blob["final_octave_delta_pp"], last["delta_gap_pp"], abs_tol=GAP_TOL
            ), f"{app}/{pol}: final_octave_delta_pp"


# ── group 3: onset rule + saturated flag ─────────────────────────────────


def test_saturation_onset_matches_step_down_rule(data):
    for app, app_blob in data["per_app"].items():
        for pol, blob in app_blob.items():
            expected = _onset(blob["octaves"])
            assert blob["saturation_onset"] == expected, (
                f"{app}/{pol}: onset={blob['saturation_onset']} expected={expected}"
            )


def test_saturated_within_paper_l3_iff_onset_not_never(data):
    for app, app_blob in data["per_app"].items():
        for pol, blob in app_blob.items():
            expected = blob["saturation_onset"] != "never"
            assert blob["saturated_within_paper_l3"] is expected, (
                f"{app}/{pol}: saturated flag mismatch"
            )


def test_onset_token_is_known_label(data):
    valid = set(data["meta"]["scope_l3_sizes"]) | {"never"}
    for app, app_blob in data["per_app"].items():
        for pol, blob in app_blob.items():
            assert blob["saturation_onset"] in valid, (
                f"{app}/{pol}: unknown onset token {blob['saturation_onset']!r}"
            )


# ── group 4: per_policy aggregation + ranking ────────────────────────────


def test_per_policy_onset_counts_and_totals(data):
    for pol, view in data["per_policy"].items():
        onsets = [
            data["per_app"][app][pol]["saturation_onset"]
            for app in data["per_app"]
            if pol in data["per_app"][app]
        ]
        expected_counts = dict(Counter(onsets))
        assert view["onset_counts"] == expected_counts, (
            f"{pol}: onset_counts mismatch ({view['onset_counts']} vs {expected_counts})"
        )
        assert view["n_saturated"] == sum(v for k, v in expected_counts.items() if k != "never")
        assert view["n_never_saturated"] == expected_counts.get("never", 0)


def test_per_policy_apps_sorted_with_consistent_fields(data):
    for pol, view in data["per_policy"].items():
        apps_field = view["apps"]
        sorted_names = [a["app"] for a in apps_field]
        assert sorted_names == sorted(sorted_names), f"{pol}: apps not sorted by name"
        for entry in apps_field:
            blob = data["per_app"][entry["app"]][pol]
            assert entry["onset"] == blob["saturation_onset"], (
                f"{pol}/{entry['app']}: apps[].onset mismatch"
            )
            assert math.isclose(
                entry["final_slope"], blob["final_octave_slope_pp"], abs_tol=SLOPE_TOL
            ), f"{pol}/{entry['app']}: apps[].final_slope mismatch"


def test_saturation_rank_by_policy_matches_sort_key(data):
    def key(pol: str):
        view = data["per_policy"][pol]
        return (
            -view["onset_counts"].get("1MB", 0),
            -view["onset_counts"].get("4MB", 0),
            view["n_never_saturated"],
        )

    expected_rank = sorted(data["per_policy"].keys(), key=key)
    assert data["meta"]["saturation_rank_by_policy"] == expected_rank, (
        f"rank={data['meta']['saturation_rank_by_policy']} expected={expected_rank}"
    )
