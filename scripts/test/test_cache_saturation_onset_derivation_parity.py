"""Derivation parity gate for ``wiki/data/cache_saturation_onset.json``.

Single upstream: ``oracle_gap_auc.json#per_app``. This artifact backs
the cross-gate-consistency block of gate 173 OGC-Der (curvature) — it
publishes ``saturation_rank_by_policy`` which OGC-Der mirrors as the
lead_agrees check. Locking the derivation here closes that half-locked
seam.

Locks the saturation predicate so any drift in the saturation onset
walker (smallest i where ALL octaves[i:] satisfy −0.5 < delta_pp ≤ 0),
the per-octave slope sign convention (slope_pp_per_octave = round(−Δgap/Δlog2, 4)),
the round-4dp quantization, the per-policy onset_counts (defaultdict
keyed by onset label), n_saturated arithmetic (sum of all non-'never'
buckets), n_never_saturated reducer, the saturation_rank_by_policy
sort key ((-c['1MB'], -c['4MB'], n_never_saturated) — ties broken by
FEWER never-saturated cells), or the apps-list ordering trips a test
before the dashboard re-publishes the ordering POPT > GRASP > LRU > SRRIP.

Mirrors ``build_payload()`` from
``scripts/experiments/ecg/cache_saturation_onset.py`` verbatim.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "cache_saturation_onset.json"
UPSTREAM_PATH = WIKI_DATA / "oracle_gap_auc.json"

PAPER_L3 = ("1MB", "4MB", "8MB")
L3_MB = {"1MB": 1.0, "4MB": 4.0, "8MB": 8.0}
SATURATION_THRESHOLD_PP = 0.5


def _onset(octaves):
    for i, oct_ in enumerate(octaves):
        if all(
            o["delta_gap_pp"] > -SATURATION_THRESHOLD_PP
            and o["delta_gap_pp"] <= 0
            for o in octaves[i:]
        ):
            return oct_["from"]
    last = octaves[-1]
    if last["delta_gap_pp"] > -SATURATION_THRESHOLD_PP and last["delta_gap_pp"] <= 0:
        return last["from"]
    return "never"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def upstream():
    if not UPSTREAM_PATH.exists():
        pytest.skip(f"missing {UPSTREAM_PATH}")
    return json.loads(UPSTREAM_PATH.read_text())


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_app", "per_policy"}


def test_meta_fields(artifact):
    expected = {
        "source", "scope_l3_sizes", "saturation_threshold_pp_per_octave",
        "n_apps", "n_policies", "apps", "policies",
        "saturation_rank_by_policy",
    }
    assert set(artifact["meta"].keys()) == expected


def test_meta_constants(artifact):
    m = artifact["meta"]
    assert m["scope_l3_sizes"] == list(PAPER_L3)
    assert m["saturation_threshold_pp_per_octave"] == SATURATION_THRESHOLD_PP


def test_meta_source_points_at_upstream(artifact):
    assert artifact["meta"]["source"].endswith("oracle_gap_auc.json")


def test_per_app_cell_shape(artifact):
    expected = {
        "octaves", "saturation_onset", "saturated_within_paper_l3",
        "final_octave_slope_pp", "final_octave_delta_pp",
    }
    for app, cells in artifact["per_app"].items():
        for pol, c in cells.items():
            assert set(c.keys()) == expected, f"{app}/{pol}: cell field drift"


def test_octave_record_shape(artifact):
    expected = {
        "from", "to", "gap_from", "gap_to", "delta_gap_pp",
        "slope_pp_per_octave",
    }
    for app, cells in artifact["per_app"].items():
        for pol, c in cells.items():
            for o in c["octaves"]:
                assert set(o.keys()) == expected, f"{app}/{pol}: octave drift"


def test_per_policy_view_shape(artifact):
    expected = {"onset_counts", "apps", "n_saturated", "n_never_saturated"}
    for pol, v in artifact["per_policy"].items():
        assert set(v.keys()) == expected, f"{pol}: per_policy drift"


def test_per_policy_app_record_shape(artifact):
    expected = {"app", "onset", "final_slope"}
    for pol, v in artifact["per_policy"].items():
        for rec in v["apps"]:
            assert set(rec.keys()) == expected


# ----------------------------------------------------------------------
# Group B: meta counters & scope
# ----------------------------------------------------------------------

def test_meta_apps_sorted(artifact):
    assert artifact["meta"]["apps"] == sorted(artifact["per_app"].keys())


def test_meta_policies_sorted(artifact):
    assert artifact["meta"]["policies"] == sorted(artifact["per_policy"].keys())


def test_meta_n_apps_counter(artifact):
    assert artifact["meta"]["n_apps"] == len(artifact["per_app"])


def test_meta_n_policies_is_max_per_app_breadth(artifact):
    expected = max((len(b) for b in artifact["per_app"].values()), default=0)
    assert artifact["meta"]["n_policies"] == expected


# ----------------------------------------------------------------------
# Group C: octave & onset byte-exact reproduction
# ----------------------------------------------------------------------

def test_octaves_byte_exact_against_upstream(artifact, upstream):
    """Re-derive octaves from upstream trajectory_by_policy and compare
    byte-by-byte (load-bearing: slope sign convention, log2 base,
    round-4dp quantization, sizes-in-PAPER_L3-order filter)."""
    for app, app_blob in upstream["per_app"].items():
        for pol, traj in app_blob["trajectory_by_policy"].items():
            sizes = [s for s in PAPER_L3 if s in traj]
            if len(sizes) < 2:
                assert pol not in artifact["per_app"].get(app, {})
                continue
            expected_octaves = []
            for src, dst in zip(sizes, sizes[1:]):
                d_log = math.log2(L3_MB[dst]) - math.log2(L3_MB[src])
                d_gap = traj[dst] - traj[src]
                expected_octaves.append({
                    "from": src,
                    "to": dst,
                    "gap_from": round(traj[src], 4),
                    "gap_to": round(traj[dst], 4),
                    "delta_gap_pp": round(d_gap, 4),
                    "slope_pp_per_octave": (
                        round(-d_gap / d_log, 4) if d_log > 0 else 0.0
                    ),
                })
            cell = artifact["per_app"][app][pol]
            assert cell["octaves"] == expected_octaves, (
                f"{app}/{pol}: octaves drift"
            )


def test_saturation_onset_byte_exact(artifact):
    for app, cells in artifact["per_app"].items():
        for pol, c in cells.items():
            expected = _onset(c["octaves"])
            assert c["saturation_onset"] == expected, (
                f"{app}/{pol}: onset drift art={c['saturation_onset']} "
                f"exp={expected}"
            )


def test_saturated_within_paper_l3_flag(artifact):
    for app, cells in artifact["per_app"].items():
        for pol, c in cells.items():
            assert c["saturated_within_paper_l3"] is (c["saturation_onset"] != "never")


def test_final_octave_fields_match_last_octave(artifact):
    for app, cells in artifact["per_app"].items():
        for pol, c in cells.items():
            last = c["octaves"][-1]
            assert c["final_octave_slope_pp"] == last["slope_pp_per_octave"]
            assert c["final_octave_delta_pp"] == last["delta_gap_pp"]


# ----------------------------------------------------------------------
# Group D: per-policy reducer & rank parity
# ----------------------------------------------------------------------

def test_per_policy_onset_counts_match_per_app(artifact):
    expected = defaultdict(lambda: defaultdict(int))
    for app, cells in artifact["per_app"].items():
        for pol, c in cells.items():
            expected[pol][c["saturation_onset"]] += 1
    for pol, view in artifact["per_policy"].items():
        assert view["onset_counts"] == dict(expected[pol])


def test_per_policy_apps_sorted_by_app_name(artifact):
    for pol, view in artifact["per_policy"].items():
        names = [r["app"] for r in view["apps"]]
        assert names == sorted(names), f"{pol}: apps not sorted by name"


def test_per_policy_apps_records_match_per_app(artifact):
    for pol, view in artifact["per_policy"].items():
        for rec in view["apps"]:
            cell = artifact["per_app"][rec["app"]][pol]
            assert rec["onset"] == cell["saturation_onset"]
            assert rec["final_slope"] == cell["final_octave_slope_pp"]


def test_n_saturated_arithmetic(artifact):
    """n_saturated = sum of values where key != 'never' (load-bearing —
    NOT total minus never; equivalent when no extra keys exist)."""
    for pol, view in artifact["per_policy"].items():
        expected = sum(v for k, v in view["onset_counts"].items() if k != "never")
        assert view["n_saturated"] == expected


def test_n_never_saturated_lookup(artifact):
    for pol, view in artifact["per_policy"].items():
        assert view["n_never_saturated"] == view["onset_counts"].get("never", 0)


def test_saturation_rank_by_policy_sort_key(artifact):
    """Sort key: (-1MB_count, -4MB_count, n_never_saturated). Ties
    broken by FEWER never-saturated cells (ascending) — load-bearing."""
    pol_views = artifact["per_policy"]
    expected = [k for k, _ in sorted(
        pol_views.items(),
        key=lambda kv: (
            -kv[1]["onset_counts"].get("1MB", 0),
            -kv[1]["onset_counts"].get("4MB", 0),
            kv[1]["n_never_saturated"],
        ),
    )]
    assert artifact["meta"]["saturation_rank_by_policy"] == expected


# ----------------------------------------------------------------------
# Group E: claim invariants
# ----------------------------------------------------------------------

def test_popt_saturates_at_or_before_lru(artifact):
    """The paper claim: oracle-aware POPT saturates earliest. Rank
    list MUST place POPT no later than LRU/SRRIP."""
    rank = artifact["meta"]["saturation_rank_by_policy"]
    if "POPT" in rank and "LRU" in rank:
        assert rank.index("POPT") <= rank.index("LRU")
    if "POPT" in rank and "SRRIP" in rank:
        assert rank.index("POPT") <= rank.index("SRRIP")


def test_onset_label_is_in_paper_l3_or_never(artifact):
    valid = set(PAPER_L3) | {"never"}
    for app, cells in artifact["per_app"].items():
        for pol, c in cells.items():
            assert c["saturation_onset"] in valid


def test_octaves_consecutive_in_paper_l3_order(artifact):
    """Octave from→to chain must walk PAPER_L3 in order (i.e. matches
    one of [(1MB,4MB),(4MB,8MB)] or both)."""
    valid_pairs = {("1MB", "4MB"), ("4MB", "8MB")}
    for app, cells in artifact["per_app"].items():
        for pol, c in cells.items():
            for o in c["octaves"]:
                assert (o["from"], o["to"]) in valid_pairs


def test_no_cell_has_fewer_than_one_octave(artifact):
    for app, cells in artifact["per_app"].items():
        for pol, c in cells.items():
            assert len(c["octaves"]) >= 1
