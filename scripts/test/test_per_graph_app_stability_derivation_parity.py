"""Derivation parity gate for per_graph_app_stability.json (gate 187).

Regenerates per-(graph, app) winner stability classifications from
oracle_gap.json#rows and asserts byte-equality against the committed
artifact. Validates the four-class taxonomy that the paper uses to decide
which cells need a per-L3 disclaimer.

Load-bearing rules:

- PAPER_L3_SIZES = ("1MB", "4MB", "8MB"); rows with other L3s dropped
- Per-(graph, app, l3) winner set = ALL policies tied for min gap_pp with
  tolerance abs(g - min) < 1e-6 (NOT exact equality)
- Winners list sorted alphabetically
- Classification rules (in evaluation order):
    1. <2 L3 sizes present → "insufficient_l3"
    2. empty intersection across L3s → "regime_change"
    3. |intersection|==1 AND union==intersection → "stable_unique"
    4. |intersection|==1 AND union!=intersection → "stable_unique_with_ties"
    5. |intersection|>1 → "stable_partial"
- per_ga list sorted by (graph, app) tuple (insertion order matters)
- meta.n_stable_unique = sum of stable_unique + stable_unique_with_ties
- stability_fraction divides by max(1, total - insufficient) (zero-div guard)
- Line formats: stable cells "g/a -> winner"; partial "g/a -> w1,w2";
  regime/insufficient "g/a"
- per_graph_rollup: stable_unique counter includes ties variant; 'partial'
  key (NOT 'stable_partial') for the rollup counter
- JSON written with sort_keys=True
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "per_graph_app_stability.json"
ORACLE = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")
TIE_TOL = 1e-6
KNOWN_CLASSES = {
    "stable_unique",
    "stable_unique_with_ties",
    "stable_partial",
    "regime_change",
    "insufficient_l3",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.exists(), f"missing {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def oracle() -> dict:
    assert ORACLE.exists(), f"missing {ORACLE}"
    return json.loads(ORACLE.read_text())


def _cell_winners(rows, scope_l3):
    by_cell: dict[tuple, list[tuple[float, str]]] = defaultdict(list)
    for r in rows:
        l3 = r["l3_size"]
        if l3 not in scope_l3:
            continue
        gap = float(r["gap_pp"])
        by_cell[(r["graph"], r["app"], l3)].append((gap, r["policy"]))
    winners = {}
    for k, pairs in by_cell.items():
        min_gap = min(g for g, _ in pairs)
        tied = sorted(p for g, p in pairs if abs(g - min_gap) < TIE_TOL)
        winners[k] = tied
    return winners


def _classify(winners_per_l3):
    l3s = sorted(winners_per_l3.keys())
    if len(l3s) < 2:
        return {
            "l3_sizes_present": l3s,
            "winners_by_l3": winners_per_l3,
            "classification": "insufficient_l3",
            "intersection": [],
            "union": sorted({p for ws in winners_per_l3.values() for p in ws}),
            "unique_in_intersection": False,
        }
    intersection = set(winners_per_l3[l3s[0]])
    for l3 in l3s[1:]:
        intersection &= set(winners_per_l3[l3])
    inter_sorted = sorted(intersection)
    union = sorted({p for ws in winners_per_l3.values() for p in ws})
    if not intersection:
        cls = "regime_change"
    elif len(intersection) == 1 and set(union) == intersection:
        cls = "stable_unique"
    elif len(intersection) == 1:
        cls = "stable_unique_with_ties"
    else:
        cls = "stable_partial"
    return {
        "l3_sizes_present": l3s,
        "winners_by_l3": winners_per_l3,
        "classification": cls,
        "intersection": inter_sorted,
        "union": union,
        "unique_in_intersection": len(intersection) == 1,
    }


@pytest.fixture(scope="module")
def derived(oracle) -> dict:
    rows = oracle["rows"]
    winners = _cell_winners(rows, set(PAPER_L3_SIZES))
    by_ga: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(dict)
    for (g, a, l), wlist in winners.items():
        by_ga[(g, a)][l] = wlist

    per_ga = []
    for (graph, app) in sorted(by_ga.keys()):
        rec = _classify(by_ga[(graph, app)])
        rec["graph"] = graph
        rec["app"] = app
        per_ga.append(rec)

    cls_counts = defaultdict(int)
    for r in per_ga:
        cls_counts[r["classification"]] += 1

    stable = [
        f"{r['graph']}/{r['app']} -> {r['intersection'][0]}"
        for r in per_ga
        if r["classification"] in ("stable_unique", "stable_unique_with_ties")
    ]
    regime = [
        f"{r['graph']}/{r['app']}"
        for r in per_ga
        if r["classification"] == "regime_change"
    ]
    partial = [
        f"{r['graph']}/{r['app']} -> {','.join(r['intersection'])}"
        for r in per_ga
        if r["classification"] == "stable_partial"
    ]
    insufficient = [
        f"{r['graph']}/{r['app']}"
        for r in per_ga
        if r["classification"] == "insufficient_l3"
    ]

    per_graph = defaultdict(
        lambda: {"n_apps": 0, "stable_unique": 0, "regime_change": 0, "partial": 0}
    )
    for r in per_ga:
        rollup = per_graph[r["graph"]]
        rollup["n_apps"] += 1
        if r["classification"] in ("stable_unique", "stable_unique_with_ties"):
            rollup["stable_unique"] += 1
        elif r["classification"] == "regime_change":
            rollup["regime_change"] += 1
        elif r["classification"] == "stable_partial":
            rollup["partial"] += 1

    n_total = len(per_ga)
    n_insuff = cls_counts.get("insufficient_l3", 0)
    n_stab = (
        cls_counts.get("stable_unique", 0)
        + cls_counts.get("stable_unique_with_ties", 0)
    )
    return {
        "meta": {
            "scope_l3_sizes": list(PAPER_L3_SIZES),
            "n_graph_app_pairs": n_total,
            "n_stable_unique": n_stab,
            "n_stable_partial": cls_counts.get("stable_partial", 0),
            "n_regime_change": cls_counts.get("regime_change", 0),
            "n_insufficient_l3": n_insuff,
            "stability_fraction_excl_insufficient": (
                n_stab / max(1, n_total - n_insuff)
            ),
        },
        "stable_unique_cells": stable,
        "regime_change_cells": regime,
        "stable_partial_cells": partial,
        "insufficient_cells": insufficient,
        "per_graph_rollup": dict(per_graph),
        "per_graph_app": per_ga,
    }


# ---------------------------------------------------------------------------
# Group A — meta scope + counts
# ---------------------------------------------------------------------------


def test_meta_scope_l3_sizes(artifact):
    assert artifact["meta"]["scope_l3_sizes"] == list(PAPER_L3_SIZES)


def test_meta_n_graph_app_pairs(artifact, derived):
    assert artifact["meta"]["n_graph_app_pairs"] == derived["meta"]["n_graph_app_pairs"]


def test_meta_n_stable_unique(artifact, derived):
    assert artifact["meta"]["n_stable_unique"] == derived["meta"]["n_stable_unique"]


def test_meta_n_stable_partial(artifact, derived):
    assert artifact["meta"]["n_stable_partial"] == derived["meta"]["n_stable_partial"]


def test_meta_n_regime_change(artifact, derived):
    assert artifact["meta"]["n_regime_change"] == derived["meta"]["n_regime_change"]


def test_meta_n_insufficient_l3(artifact, derived):
    assert (
        artifact["meta"]["n_insufficient_l3"]
        == derived["meta"]["n_insufficient_l3"]
    )


def test_meta_counts_sum_to_pairs(artifact):
    m = artifact["meta"]
    total = (
        m["n_stable_unique"]
        + m["n_stable_partial"]
        + m["n_regime_change"]
        + m["n_insufficient_l3"]
    )
    assert total == m["n_graph_app_pairs"]


def test_meta_stability_fraction_formula(artifact):
    m = artifact["meta"]
    denom = max(1, m["n_graph_app_pairs"] - m["n_insufficient_l3"])
    expected = m["n_stable_unique"] / denom
    assert abs(m["stability_fraction_excl_insufficient"] - expected) <= 1e-12


def test_meta_stability_fraction_matches_derived(artifact, derived):
    assert abs(
        artifact["meta"]["stability_fraction_excl_insufficient"]
        - derived["meta"]["stability_fraction_excl_insufficient"]
    ) <= 1e-12


# ---------------------------------------------------------------------------
# Group B — per_graph_app list shape, classification semantics, sort
# ---------------------------------------------------------------------------


def test_per_graph_app_count_matches(artifact, derived):
    assert len(artifact["per_graph_app"]) == len(derived["per_graph_app"])


def test_per_graph_app_sorted_by_graph_app(artifact):
    keys = [(r["graph"], r["app"]) for r in artifact["per_graph_app"]]
    assert keys == sorted(keys)


def test_per_graph_app_unique_keys(artifact):
    keys = [(r["graph"], r["app"]) for r in artifact["per_graph_app"]]
    assert len(keys) == len(set(keys))


def test_per_graph_app_classifications_known(artifact):
    for r in artifact["per_graph_app"]:
        assert r["classification"] in KNOWN_CLASSES


def test_per_graph_app_l3_sizes_alpha_sorted(artifact):
    for r in artifact["per_graph_app"]:
        l3s = r["l3_sizes_present"]
        assert l3s == sorted(l3s)
        for l in l3s:
            assert l in PAPER_L3_SIZES


def test_per_graph_app_winners_alpha_sorted(artifact):
    for r in artifact["per_graph_app"]:
        for l3, w in r["winners_by_l3"].items():
            assert w == sorted(w), f"{r['graph']}/{r['app']}@{l3}: {w}"


def test_per_graph_app_intersection_sorted(artifact):
    for r in artifact["per_graph_app"]:
        assert r["intersection"] == sorted(r["intersection"])


def test_per_graph_app_union_sorted(artifact):
    for r in artifact["per_graph_app"]:
        assert r["union"] == sorted(r["union"])


def test_per_graph_app_unique_in_intersection_flag(artifact):
    for r in artifact["per_graph_app"]:
        assert r["unique_in_intersection"] == (len(r["intersection"]) == 1)


def test_per_graph_app_classification_rules(artifact):
    """Re-classify from intersection/union and ensure label matches."""
    for r in artifact["per_graph_app"]:
        if len(r["l3_sizes_present"]) < 2:
            assert r["classification"] == "insufficient_l3"
            continue
        inter = set(r["intersection"])
        union = set(r["union"])
        if not inter:
            assert r["classification"] == "regime_change"
        elif len(inter) == 1 and union == inter:
            assert r["classification"] == "stable_unique"
        elif len(inter) == 1:
            assert r["classification"] == "stable_unique_with_ties"
        else:
            assert r["classification"] == "stable_partial"


def test_per_graph_app_derived_parity(artifact, derived):
    assert artifact["per_graph_app"] == derived["per_graph_app"]


# ---------------------------------------------------------------------------
# Group C — headline lists (line format + filter parity)
# ---------------------------------------------------------------------------


def test_stable_unique_cells_count(artifact):
    assert len(artifact["stable_unique_cells"]) == artifact["meta"]["n_stable_unique"]


def test_stable_partial_cells_count(artifact):
    assert (
        len(artifact["stable_partial_cells"]) == artifact["meta"]["n_stable_partial"]
    )


def test_regime_change_cells_count(artifact):
    assert (
        len(artifact["regime_change_cells"]) == artifact["meta"]["n_regime_change"]
    )


def test_insufficient_cells_count(artifact):
    assert (
        len(artifact["insufficient_cells"]) == artifact["meta"]["n_insufficient_l3"]
    )


def test_stable_unique_cells_format(artifact):
    for line in artifact["stable_unique_cells"]:
        assert " -> " in line and "/" in line.split(" -> ")[0]


def test_stable_partial_cells_format(artifact):
    for line in artifact["stable_partial_cells"]:
        head, tail = line.split(" -> ", 1)
        assert "/" in head
        # tail is comma-joined; if multiple winners, they must be alphabetical
        winners = tail.split(",")
        assert winners == sorted(winners)


def test_regime_change_cells_format(artifact):
    for line in artifact["regime_change_cells"]:
        assert "/" in line and " -> " not in line


def test_insufficient_cells_format(artifact):
    for line in artifact["insufficient_cells"]:
        assert "/" in line and " -> " not in line


def test_headline_lists_derived_parity(artifact, derived):
    assert artifact["stable_unique_cells"] == derived["stable_unique_cells"]
    assert artifact["stable_partial_cells"] == derived["stable_partial_cells"]
    assert artifact["regime_change_cells"] == derived["regime_change_cells"]
    assert artifact["insufficient_cells"] == derived["insufficient_cells"]


# ---------------------------------------------------------------------------
# Group D — per_graph_rollup counter semantics
# ---------------------------------------------------------------------------


def test_per_graph_rollup_keys_match_graph_set(artifact):
    expected = {r["graph"] for r in artifact["per_graph_app"]}
    assert set(artifact["per_graph_rollup"].keys()) == expected


def test_per_graph_rollup_n_apps_matches(artifact):
    derived_napp = defaultdict(int)
    for r in artifact["per_graph_app"]:
        derived_napp[r["graph"]] += 1
    for g, rollup in artifact["per_graph_rollup"].items():
        assert rollup["n_apps"] == derived_napp[g]


def test_per_graph_rollup_counters_include_ties(artifact):
    """The 'stable_unique' rollup counter must include both stable_unique and
    stable_unique_with_ties."""
    expected = defaultdict(lambda: {"stable_unique": 0, "regime_change": 0, "partial": 0})
    for r in artifact["per_graph_app"]:
        if r["classification"] in ("stable_unique", "stable_unique_with_ties"):
            expected[r["graph"]]["stable_unique"] += 1
        elif r["classification"] == "regime_change":
            expected[r["graph"]]["regime_change"] += 1
        elif r["classification"] == "stable_partial":
            expected[r["graph"]]["partial"] += 1
    for g, rollup in artifact["per_graph_rollup"].items():
        for k in ("stable_unique", "regime_change", "partial"):
            assert rollup[k] == expected[g][k], f"{g}.{k}: got {rollup[k]} expected {expected[g][k]}"


def test_per_graph_rollup_derived_parity(artifact, derived):
    assert artifact["per_graph_rollup"] == derived["per_graph_rollup"]


def test_per_graph_rollup_no_extra_keys(artifact):
    for rollup in artifact["per_graph_rollup"].values():
        assert set(rollup.keys()) == {"n_apps", "stable_unique", "regime_change", "partial"}


# ---------------------------------------------------------------------------
# Group E — full byte parity + tie-tolerance round-trip
# ---------------------------------------------------------------------------


def test_winner_min_gap_tie_tolerance(oracle, artifact):
    """For each (graph, app, l3) in the artifact, every winner must have
    gap_pp within 1e-6 of the cell minimum."""
    by_cell: dict[tuple, list[tuple[float, str]]] = defaultdict(list)
    for r in oracle["rows"]:
        if r["l3_size"] in PAPER_L3_SIZES:
            by_cell[(r["graph"], r["app"], r["l3_size"])].append(
                (float(r["gap_pp"]), r["policy"])
            )
    for rec in artifact["per_graph_app"]:
        for l3, winners in rec["winners_by_l3"].items():
            pairs = by_cell.get((rec["graph"], rec["app"], l3))
            assert pairs, f"no oracle rows for {rec['graph']}/{rec['app']}@{l3}"
            min_gap = min(g for g, _ in pairs)
            policy_to_gap = {p: g for g, p in pairs}
            for w in winners:
                assert abs(policy_to_gap[w] - min_gap) < TIE_TOL


def test_full_artifact_byte_parity(artifact, derived):
    """End-to-end: artifact == derived (excluding meta.source)."""
    a = dict(artifact)
    a_meta = dict(a["meta"])
    a_meta.pop("source", None)
    a["meta"] = a_meta
    assert a == derived
