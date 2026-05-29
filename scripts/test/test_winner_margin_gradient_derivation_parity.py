"""Derivation parity gate for ``wiki/data/winner_margin_gradient.json``.

Locks the per-(app, L3) winner-margin gradient (gate 48) against
its single upstream — ``oracle_gap.json#rows`` — so any silent
drift in the win counter, the tie-break ordering, the classifier
thresholds, or the strong-cell fraction reducer trips a test
before the dashboard re-publishes the "winners are not decided by
a single graph" defence.

    oracle_gap.json#rows  (filtered to L3 in {1MB, 4MB, 8MB})
                  │
        winner_margin_gradient.py:build_payload()
                  │
                  ▼
    wiki/data/winner_margin_gradient.json    ← gate target

The gated claim: across every (app, L3) cell in the paper's L3
scope, the top policy's lead over the runner-up is auditable, and
the classifier (decisive ≥ 4, moderate ≥ 2, weak == 1, tied == 0)
labels every cell. The strong-cell fraction (decisive + moderate)
sums the two defensible classes; weak and tied cells are surfaced
by name so reviewer pushback "what if a single graph flips it?" is
answered with the exact list.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "winner_margin_gradient.json"
ORACLE_PATH = WIKI_DATA / "oracle_gap.json"

PAPER_L3_SIZES = ("1MB", "4MB", "8MB")
KNOWN_CLASSES = ("decisive", "moderate", "weak", "tied")


def _classify(margin: int) -> str:
    if margin >= 4:
        return "decisive"
    if margin >= 2:
        return "moderate"
    if margin == 1:
        return "weak"
    return "tied"


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
def reconstructed(oracle_doc) -> dict:
    """Mirror build_payload end-to-end against the same upstream rows."""
    rows = oracle_doc["rows"]
    paper_rows = [r for r in rows if r["l3_size"] in PAPER_L3_SIZES]
    apps = sorted({r["app"] for r in paper_rows})
    win_counts: dict = defaultdict(Counter)
    seen_cells: dict = defaultdict(set)
    for r in paper_rows:
        key = (r["app"], r["l3_size"])
        seen_cells[key].add((r["graph"], r["app"], r["l3_size"]))
        if int(r["is_winner"]) == 1:
            win_counts[key][r["policy"]] += 1
    cell_count = {key: len(v) for key, v in seen_cells.items()}

    per_cell: dict = {}
    class_counts: Counter = Counter()
    for app in apps:
        for l3 in PAPER_L3_SIZES:
            key = (app, l3)
            c = win_counts.get(key, Counter())
            if not c:
                continue
            ordered = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
            top_policy, top_wins = ordered[0]
            runner_wins = ordered[1][1] if len(ordered) > 1 else 0
            margin = top_wins - runner_wins
            klass = _classify(margin)
            class_counts[klass] += 1
            tied_with = sorted(
                p for p, w in c.items() if w == top_wins and p != top_policy
            )
            per_cell[f"{app}__{l3}"] = {
                "app": app,
                "l3_size": l3,
                "top_policy": top_policy,
                "top_wins": top_wins,
                "runner_up_wins": runner_wins,
                "margin": margin,
                "class": klass,
                "n_cells_in_scope": cell_count[key],
                "tied_top_policies": tied_with,
                "win_counts": dict(c),
            }

    n_total = sum(class_counts.values())
    n_strong = class_counts["decisive"] + class_counts["moderate"]
    strong_fraction = round(n_strong / n_total, 4) if n_total else 0.0
    weak = sorted(k for k, v in per_cell.items() if v["class"] == "weak")
    tied = sorted(k for k, v in per_cell.items() if v["class"] == "tied")
    return {
        "apps": apps, "per_cell": per_cell, "class_counts": dict(class_counts),
        "n_cells_total": n_total, "strong_fraction": strong_fraction,
        "weak_cells": weak, "tied_cells": tied,
    }


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_cell"}


def test_meta_carries_canonical_fields(artifact):
    expected = {
        "source", "scope_l3_sizes", "n_apps", "apps",
        "n_cells_total", "class_thresholds", "class_counts",
        "strong_cell_fraction", "weak_cells", "tied_cells",
        "n_weak_cells", "n_tied_cells",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing fields: {missing}"


def test_scope_l3_sizes_pinned(artifact):
    assert tuple(artifact["meta"]["scope_l3_sizes"]) == PAPER_L3_SIZES


def test_class_thresholds_pinned(artifact):
    assert artifact["meta"]["class_thresholds"] == {
        "decisive": "margin >= 4",
        "moderate": "2 <= margin < 4",
        "weak": "margin == 1",
        "tied": "margin == 0",
    }


def test_per_cell_entry_shape(artifact):
    expected = {
        "app", "l3_size", "top_policy", "top_wins",
        "runner_up_wins", "margin", "class",
        "n_cells_in_scope", "tied_top_policies", "win_counts",
    }
    for key, r in artifact["per_cell"].items():
        missing = expected - set(r.keys())
        assert not missing, f"per_cell[{key}] missing fields: {missing}"


def test_per_cell_key_format(artifact):
    """Keys are app__l3 with app and l3 inside scope."""
    for key, r in artifact["per_cell"].items():
        assert key == f"{r['app']}__{r['l3_size']}", (
            f"per_cell key {key!r} disagrees with row {r}"
        )
        assert r["l3_size"] in PAPER_L3_SIZES


def test_per_cell_class_only_known(artifact):
    for key, r in artifact["per_cell"].items():
        assert r["class"] in KNOWN_CLASSES, (
            f"per_cell[{key}]: unknown class {r['class']!r}"
        )


# ----------------------------------------------------------------------
# Group B: cell-set cross-source parity
# ----------------------------------------------------------------------

def test_apps_match_recomputation(artifact, reconstructed):
    assert artifact["meta"]["apps"] == reconstructed["apps"]


def test_n_apps_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["n_apps"] == len(reconstructed["apps"])


def test_per_cell_keyset_matches_recomputation(artifact, reconstructed):
    assert set(artifact["per_cell"].keys()) == set(
        reconstructed["per_cell"].keys()
    )


def test_n_cells_total_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["n_cells_total"] == reconstructed["n_cells_total"]


def test_n_cells_total_equals_per_cell_size(artifact):
    assert artifact["meta"]["n_cells_total"] == len(artifact["per_cell"])


# ----------------------------------------------------------------------
# Group C: per-cell reducer cross-source parity
# ----------------------------------------------------------------------

def test_per_cell_records_match_recomputation(artifact, reconstructed):
    expected = reconstructed["per_cell"]
    for key, r in artifact["per_cell"].items():
        assert key in expected, f"unexpected per_cell key {key}"
        e = expected[key]
        assert r["top_policy"] == e["top_policy"], (
            f"{key}: top_policy drift — {r['top_policy']!r} vs {e['top_policy']!r}"
        )
        assert r["top_wins"] == e["top_wins"]
        assert r["runner_up_wins"] == e["runner_up_wins"]
        assert r["margin"] == e["margin"]
        assert r["class"] == e["class"]
        assert r["n_cells_in_scope"] == e["n_cells_in_scope"]
        assert sorted(r["tied_top_policies"]) == sorted(e["tied_top_policies"])
        assert r["win_counts"] == e["win_counts"]


def test_margin_equals_top_wins_minus_runner_up(artifact):
    for key, r in artifact["per_cell"].items():
        assert r["margin"] == r["top_wins"] - r["runner_up_wins"], (
            f"{key}: margin ≠ top_wins − runner_up_wins"
        )


def test_top_wins_equals_max_win_counts(artifact):
    for key, r in artifact["per_cell"].items():
        assert r["top_wins"] == max(r["win_counts"].values()), (
            f"{key}: top_wins ≠ max(win_counts)"
        )


def test_runner_up_strictly_below_or_equal_top(artifact):
    for key, r in artifact["per_cell"].items():
        assert r["runner_up_wins"] <= r["top_wins"], (
            f"{key}: runner_up_wins > top_wins violates ordering"
        )


def test_tie_break_is_alphabetical(artifact):
    """When multiple policies tie at the max, top_policy is the
    alphabetically smallest of them (Counter sorted by (-wins, name))."""
    for key, r in artifact["per_cell"].items():
        tied_at_top = sorted(
            p for p, w in r["win_counts"].items() if w == r["top_wins"]
        )
        assert tied_at_top[0] == r["top_policy"], (
            f"{key}: top_policy {r['top_policy']!r} not the "
            f"alphabetically smallest among tied {tied_at_top}"
        )


def test_class_matches_classifier(artifact):
    for key, r in artifact["per_cell"].items():
        assert r["class"] == _classify(r["margin"]), (
            f"{key}: class {r['class']!r} disagrees with _classify({r['margin']})"
        )


def test_tied_top_policies_excludes_top(artifact):
    for key, r in artifact["per_cell"].items():
        assert r["top_policy"] not in r["tied_top_policies"], (
            f"{key}: tied_top_policies should not include top_policy"
        )


def test_tied_top_policies_all_tied(artifact):
    for key, r in artifact["per_cell"].items():
        for p in r["tied_top_policies"]:
            assert r["win_counts"].get(p) == r["top_wins"], (
                f"{key}: {p!r} listed as tied but win_counts disagree"
            )


def test_tied_class_has_at_least_one_tie(artifact):
    for key, r in artifact["per_cell"].items():
        if r["class"] == "tied":
            assert r["tied_top_policies"], (
                f"{key}: class=tied but tied_top_policies is empty"
            )


# ----------------------------------------------------------------------
# Group D: aggregate reducers + headline cross-source parity
# ----------------------------------------------------------------------

def test_class_counts_match_recomputation(artifact, reconstructed):
    a = dict(artifact["meta"]["class_counts"])
    e = dict(reconstructed["class_counts"])
    for k in KNOWN_CLASSES:
        assert a.get(k, 0) == e.get(k, 0), (
            f"class_counts[{k}] drift — {a.get(k, 0)} vs {e.get(k, 0)}"
        )


def test_class_counts_sum_to_n_cells_total(artifact):
    assert sum(artifact["meta"]["class_counts"].values()) == (
        artifact["meta"]["n_cells_total"]
    )


def test_strong_cell_fraction_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["strong_cell_fraction"] == (
        reconstructed["strong_fraction"]
    )


def test_strong_cell_fraction_formula(artifact):
    cc = artifact["meta"]["class_counts"]
    n = artifact["meta"]["n_cells_total"]
    expected = round((cc.get("decisive", 0) + cc.get("moderate", 0)) / n, 4) if n else 0.0
    assert artifact["meta"]["strong_cell_fraction"] == expected


def test_weak_cells_match_recomputation(artifact, reconstructed):
    assert artifact["meta"]["weak_cells"] == reconstructed["weak_cells"]


def test_tied_cells_match_recomputation(artifact, reconstructed):
    assert artifact["meta"]["tied_cells"] == reconstructed["tied_cells"]


def test_n_weak_cells_matches_list(artifact):
    assert artifact["meta"]["n_weak_cells"] == len(artifact["meta"]["weak_cells"])


def test_n_tied_cells_matches_list(artifact):
    assert artifact["meta"]["n_tied_cells"] == len(artifact["meta"]["tied_cells"])


def test_weak_cells_classified_correctly(artifact):
    for key in artifact["meta"]["weak_cells"]:
        assert artifact["per_cell"][key]["class"] == "weak"


def test_tied_cells_classified_correctly(artifact):
    for key in artifact["meta"]["tied_cells"]:
        assert artifact["per_cell"][key]["class"] == "tied"
