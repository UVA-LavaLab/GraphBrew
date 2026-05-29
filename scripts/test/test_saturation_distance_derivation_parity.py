"""Derivation parity gate for ``wiki/data/saturation_distance.json``.

Locks the per-app 4MB->8MB saturation-distance report (gate 65)
against its two upstreams — ``oracle_gap.json#rows`` for the
best-policy miss rates and ``wss_relative_l3.json#meta.wss_proxies``
for the per-graph WSS proxy — so any silent drift in the
best-policy min reducer, the 4MB/8MB filter, the per-app
{median, mean, p90, max, min} reducer, the WSS-floor filter for
non-negative-distance violations, the pico-sentinel rule, or the
app-diversity range trips a test before the dashboard re-publishes
the "memory-bound vs compute-bound apps differ" story.

    oracle_gap.json#rows                  wss_relative_l3.json
                  │                                │
                  └──── saturation_distance.py:build() ────┘
                                  │
                                  ▼
    wiki/data/saturation_distance.json    ← gate target

The gated claim: at the 4MB->8MB step, the best-policy miss-rate
improvement (distance_pp) is non-negative whenever the graph's WSS
exceeds the 4MB L3 (no "8MB is worse" anomalies on still-bottlenecked
cells), email-Eu-core saturates within 0.05 pp on every app (the
corpus's pico-sentinel), and per-app medians vary by ≥ 3 pp across
apps (the corpus retains app-level signal).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "saturation_distance.json"
ORACLE_PATH = WIKI_DATA / "oracle_gap.json"
WSS_PATH = WIKI_DATA / "wss_relative_l3.json"

PICO_GRAPH = "email-Eu-core"
PICO_SATURATION_PP = 0.05
APP_DIVERSITY_THRESHOLD_PP = 3.0
WSS_FLOOR_BYTES = 4 * 1024 * 1024


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def oracle_doc() -> dict:
    if not ORACLE_PATH.exists():
        pytest.skip(f"missing {ORACLE_PATH}")
    return json.loads(ORACLE_PATH.read_text())


@pytest.fixture(scope="module")
def wss_doc() -> dict:
    if not WSS_PATH.exists():
        pytest.skip(f"missing {WSS_PATH}")
    return json.loads(WSS_PATH.read_text())


@pytest.fixture(scope="module")
def reconstructed(oracle_doc, wss_doc) -> dict:
    """Mirror build() end-to-end against the same upstreams."""
    wss_proxies = wss_doc["meta"]["wss_proxies"]
    rows = oracle_doc["rows"]
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["app"], r["graph"])][r["l3_size"]].append(
            float(r["miss_rate"])
        )

    per_cell, per_app_distances, nonneg = [], defaultdict(list), []
    for (app, graph), by_l3 in sorted(cells.items()):
        if "4MB" not in by_l3 or "8MB" not in by_l3:
            continue
        best4 = min(by_l3["4MB"]) * 100.0
        best8 = min(by_l3["8MB"]) * 100.0
        dist_pp = round(best4 - best8, 4)
        wss = wss_proxies.get(graph, 0)
        per_cell.append({
            "app": app, "graph": graph, "wss_bytes": wss,
            "best4_miss_pp": round(best4, 4),
            "best8_miss_pp": round(best8, 4),
            "distance_pp": dist_pp,
            "is_pico_sentinel": graph == PICO_GRAPH,
        })
        per_app_distances[app].append(dist_pp)
        if wss > WSS_FLOOR_BYTES and dist_pp < 0:
            nonneg.append({
                "app": app, "graph": graph, "distance_pp": dist_pp,
            })

    per_app, medians = {}, []
    for app, ds in sorted(per_app_distances.items()):
        if not ds:
            continue
        med = round(_median(ds), 4)
        per_app[app] = {
            "n_graphs": len(ds),
            "median_pp": med,
            "mean_pp": round(sum(ds) / len(ds), 4),
            "p90_pp": round(_pct(ds, 0.9), 4),
            "max_pp": round(max(ds), 4),
            "min_pp": round(min(ds), 4),
        }
        medians.append(med)
    diversity_range = round(max(medians) - min(medians), 4) if medians else 0.0

    pico_violations = [
        c for c in per_cell
        if c["is_pico_sentinel"] and c["distance_pp"] > PICO_SATURATION_PP
    ]
    return {
        "per_cell": per_cell,
        "per_app": per_app,
        "diversity_range": diversity_range,
        "non_negative_violations": nonneg,
        "pico_violations": pico_violations,
    }


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_app", "per_cell"}


def test_meta_carries_canonical_fields(artifact):
    expected = {
        "app_count", "cell_count", "pico_graph", "pico_saturation_pp",
        "wss_floor_bytes", "app_diversity_range_pp",
        "app_diversity_threshold", "non_negative_violations",
        "pico_violations", "invariant_non_negative",
        "invariant_pico_saturated", "invariant_app_diversity",
        "verdict", "verdict_invariant",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing fields: {missing}"


def test_thresholds_pinned(artifact):
    m = artifact["meta"]
    assert m["pico_graph"] == PICO_GRAPH
    assert m["pico_saturation_pp"] == PICO_SATURATION_PP
    assert m["wss_floor_bytes"] == WSS_FLOOR_BYTES
    assert m["app_diversity_threshold"] == APP_DIVERSITY_THRESHOLD_PP


def test_verdict_invariant_string_pinned(artifact):
    expected = (
        "PASS iff (1) every cell with WSS > 4 MB has non-negative "
        "4MB->8MB best-policy improvement, (2) email-Eu-core "
        "(pico-sentinel) is saturated for every app within "
        f"{PICO_SATURATION_PP} pp, and (3) per-app median "
        "saturation distance varies by at least "
        f"{APP_DIVERSITY_THRESHOLD_PP} pp across apps."
    )
    assert artifact["meta"]["verdict_invariant"] == expected


def test_per_cell_entry_shape(artifact):
    expected = {
        "app", "graph", "wss_bytes",
        "best4_miss_pp", "best8_miss_pp", "distance_pp",
        "is_pico_sentinel",
    }
    for r in artifact["per_cell"]:
        missing = expected - set(r.keys())
        assert not missing, f"per_cell entry missing {missing}"


def test_per_app_entry_shape(artifact):
    expected = {"n_graphs", "median_pp", "mean_pp", "p90_pp", "max_pp", "min_pp"}
    for app, r in artifact["per_app"].items():
        missing = expected - set(r.keys())
        assert not missing, f"per_app[{app}] missing {missing}"


# ----------------------------------------------------------------------
# Group B: per-cell cross-source parity
# ----------------------------------------------------------------------

def test_cell_count_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["cell_count"] == len(reconstructed["per_cell"])


def test_per_cell_list_size_matches_meta(artifact):
    assert len(artifact["per_cell"]) == artifact["meta"]["cell_count"]


def test_per_cell_keyset_matches_recomputation(artifact, reconstructed):
    a = {(r["app"], r["graph"]) for r in artifact["per_cell"]}
    e = {(r["app"], r["graph"]) for r in reconstructed["per_cell"]}
    assert a == e


def test_per_cell_records_match_recomputation(artifact, reconstructed):
    expected = {(r["app"], r["graph"]): r for r in reconstructed["per_cell"]}
    for r in artifact["per_cell"]:
        key = (r["app"], r["graph"])
        e = expected[key]
        assert r["wss_bytes"] == e["wss_bytes"]
        assert r["best4_miss_pp"] == e["best4_miss_pp"], (
            f"{key}: best4 drift — {r['best4_miss_pp']} vs {e['best4_miss_pp']}"
        )
        assert r["best8_miss_pp"] == e["best8_miss_pp"]
        assert r["distance_pp"] == e["distance_pp"], (
            f"{key}: distance drift — {r['distance_pp']} vs {e['distance_pp']}"
        )
        assert r["is_pico_sentinel"] == e["is_pico_sentinel"]


def test_distance_equals_best4_minus_best8(artifact):
    for r in artifact["per_cell"]:
        assert r["distance_pp"] == round(
            r["best4_miss_pp"] - r["best8_miss_pp"], 4
        ), f"{r['app']}/{r['graph']}: distance_pp formula drift"


def test_pico_sentinel_only_for_pico_graph(artifact):
    for r in artifact["per_cell"]:
        assert r["is_pico_sentinel"] == (r["graph"] == PICO_GRAPH)


def test_pico_sentinel_present_for_at_least_one_app(artifact):
    """The pico graph must appear in per_cell for at least one app — it
    is what enforces the saturation invariant. Not every app has the
    pico graph at both 4MB and 8MB, so we don't require all apps."""
    pico_cells = [r for r in artifact["per_cell"] if r["is_pico_sentinel"]]
    assert pico_cells, (
        "no pico-sentinel cells present — saturation invariant is vacuous"
    )


# ----------------------------------------------------------------------
# Group C: per-app reducer cross-source parity
# ----------------------------------------------------------------------

def test_app_count_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["app_count"] == len(reconstructed["per_app"])


def test_per_app_keyset_matches_recomputation(artifact, reconstructed):
    assert set(artifact["per_app"].keys()) == set(reconstructed["per_app"].keys())


def test_per_app_records_match_recomputation(artifact, reconstructed):
    for app, r in artifact["per_app"].items():
        e = reconstructed["per_app"][app]
        for k in ("n_graphs", "median_pp", "mean_pp", "p90_pp", "max_pp", "min_pp"):
            assert r[k] == e[k], (
                f"per_app[{app}].{k} drift — {r[k]!r} vs {e[k]!r}"
            )


def test_per_app_n_graphs_matches_cell_count(artifact):
    by_app = defaultdict(int)
    for r in artifact["per_cell"]:
        by_app[r["app"]] += 1
    for app, r in artifact["per_app"].items():
        assert r["n_graphs"] == by_app[app], (
            f"per_app[{app}].n_graphs ≠ count of per_cell rows for that app"
        )


def test_per_app_min_le_median_le_max(artifact):
    for app, r in artifact["per_app"].items():
        assert r["min_pp"] <= r["median_pp"] <= r["max_pp"], (
            f"per_app[{app}]: min ≤ median ≤ max ordering broken — "
            f"min={r['min_pp']} median={r['median_pp']} max={r['max_pp']}"
        )


def test_per_app_p90_in_range(artifact):
    for app, r in artifact["per_app"].items():
        assert r["min_pp"] <= r["p90_pp"] <= r["max_pp"], (
            f"per_app[{app}]: p90 {r['p90_pp']} outside [min={r['min_pp']}, "
            f"max={r['max_pp']}]"
        )


# ----------------------------------------------------------------------
# Group D: aggregate reducers + verdict cross-source parity
# ----------------------------------------------------------------------

def test_app_diversity_range_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["app_diversity_range_pp"] == (
        reconstructed["diversity_range"]
    )


def test_app_diversity_range_formula(artifact):
    medians = [r["median_pp"] for r in artifact["per_app"].values()]
    expected = round(max(medians) - min(medians), 4) if medians else 0.0
    assert artifact["meta"]["app_diversity_range_pp"] == expected


def test_non_negative_violations_match_recomputation(artifact, reconstructed):
    assert artifact["meta"]["non_negative_violations"] == (
        reconstructed["non_negative_violations"]
    )


def test_pico_violations_match_recomputation(artifact, reconstructed):
    assert artifact["meta"]["pico_violations"] == reconstructed["pico_violations"]


def test_invariant_non_negative_matches_recomputation(artifact):
    expected = len(artifact["meta"]["non_negative_violations"]) == 0
    assert artifact["meta"]["invariant_non_negative"] == expected


def test_invariant_pico_saturated_matches_recomputation(artifact):
    expected = len(artifact["meta"]["pico_violations"]) == 0
    assert artifact["meta"]["invariant_pico_saturated"] == expected


def test_invariant_app_diversity_matches_recomputation(artifact):
    expected = (
        artifact["meta"]["app_diversity_range_pp"]
        >= APP_DIVERSITY_THRESHOLD_PP
    )
    assert artifact["meta"]["invariant_app_diversity"] == expected


def test_verdict_matches_and_of_invariants(artifact):
    m = artifact["meta"]
    expected = "PASS" if (
        m["invariant_non_negative"]
        and m["invariant_pico_saturated"]
        and m["invariant_app_diversity"]
    ) else "FAIL"
    assert m["verdict"] == expected


def test_current_verdict_is_pass(artifact):
    assert artifact["meta"]["verdict"] == "PASS", (
        "saturation_distance regressed to FAIL — the corpus has lost "
        "either monotonic 4MB->8MB improvement on still-bottlenecked "
        "cells, pico-sentinel saturation, or app-level diversity in "
        "the per-app median saturation distance."
    )
