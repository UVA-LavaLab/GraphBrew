"""Gate 125 — winner_margin_gradient.json arithmetic + classification.

Publishes per-(app, L3) winner margins (top_wins - runner_up_wins)
across the paper L3 scope {1MB, 4MB, 8MB} and classifies every cell:

    decisive : margin >= 4
    moderate : 2 <= margin < 4
    weak     : margin == 1
    tied     : margin == 0

This gate reproduces every per_cell entry and the class-count summary
from the upstream oracle_gap rows, so any corpus change that demotes
a 'decisive' cell to 'weak' or 'tied' surfaces immediately rather
than silently.

Source: wiki/data/oracle_gap.json rows (filtered to PAPER_L3_SIZES).
Tie-break for top_policy: alphabetical ascending policy name when
two policies share the highest win count (sort key = (-wins, policy)).

Invariants (16 tests, 4 groups):
- meta + scope (4): source, scope_l3_sizes==('1MB','4MB','8MB'),
  apps = sorted unique apps in paper rows, class_thresholds match
  documented inequalities; n_apps == len(apps).
- per_cell counts from source (4): win_counts equals oracle_gap
  is_winner==1 tallies per (app,l3); n_cells_in_scope equals #distinct
  (graph,app,l3) tuples; top_policy/top_wins/runner_up_wins reproduce
  the sorted order with alphabetical tie-break; margin = top-runner.
- per_cell classification (3): class follows the four classification
  rules; tied_top_policies = sorted other policies tied with top;
  every per_cell key matches f'{app}__{l3}' pattern.
- aggregates + disclosures (5): class_counts equals counter over
  per_cell.class; n_cells_total = sum; strong_cell_fraction = round
  ((decisive+moderate)/n_total, 4); weak_cells and tied_cells are
  sorted lists matching per_cell.class; n_weak_cells and n_tied_cells
  match list lengths.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pytest

ARTIFACT = Path("wiki/data/winner_margin_gradient.json")
SOURCE = Path("wiki/data/oracle_gap.json")

PAPER_L3 = ("1MB", "4MB", "8MB")
FRACTION_TOL = 1e-4


def _classify(margin: int) -> str:
    if margin >= 4:
        return "decisive"
    if margin >= 2:
        return "moderate"
    if margin == 1:
        return "weak"
    return "tied"


@pytest.fixture(scope="module")
def data():
    assert ARTIFACT.exists()
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def src_rows():
    assert SOURCE.exists()
    rows = json.loads(SOURCE.read_text())["rows"]
    return [r for r in rows if r["l3_size"] in PAPER_L3]


@pytest.fixture(scope="module")
def expected_wins(src_rows):
    win_counts: dict = defaultdict(Counter)
    seen_cells: dict = defaultdict(set)
    for r in src_rows:
        key = (r["app"], r["l3_size"])
        seen_cells[key].add((r["graph"], r["app"], r["l3_size"]))
        if int(r["is_winner"]) == 1:
            win_counts[key][r["policy"]] += 1
    return win_counts, {k: len(v) for k, v in seen_cells.items()}


# ── group 1: meta + scope ────────────────────────────────────────────────


def test_meta_source_and_scope(data):
    m = data["meta"]
    assert m["source"].endswith("oracle_gap.json")
    assert tuple(m["scope_l3_sizes"]) == PAPER_L3


def test_meta_apps_is_sorted_unique(data, src_rows):
    expected = sorted({r["app"] for r in src_rows})
    assert data["meta"]["apps"] == expected
    assert data["meta"]["n_apps"] == len(expected)


def test_class_thresholds_strings(data):
    t = data["meta"]["class_thresholds"]
    assert t["decisive"] == "margin >= 4"
    assert t["moderate"] == "2 <= margin < 4"
    assert t["weak"] == "margin == 1"
    assert t["tied"] == "margin == 0"


def test_class_counts_keys_subset_of_four(data):
    valid = {"decisive", "moderate", "weak", "tied"}
    assert set(data["meta"]["class_counts"]).issubset(valid)


# ── group 2: per_cell counts from source ─────────────────────────────────


def test_per_cell_win_counts_from_source(data, expected_wins):
    win_counts, _ = expected_wins
    for key, cell in data["per_cell"].items():
        app, l3 = key.split("__")
        expected = dict(win_counts[(app, l3)])
        assert cell["win_counts"] == expected, f"{key}: win_counts mismatch"


def test_per_cell_n_cells_in_scope(data, expected_wins):
    _, cell_count = expected_wins
    for key, cell in data["per_cell"].items():
        app, l3 = key.split("__")
        assert cell["n_cells_in_scope"] == cell_count[(app, l3)], key


def test_top_and_runner_with_alphabetical_tie_break(data):
    for key, cell in data["per_cell"].items():
        c = cell["win_counts"]
        ordered = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        expected_top, expected_top_wins = ordered[0]
        expected_runner = ordered[1][1] if len(ordered) > 1 else 0
        assert cell["top_policy"] == expected_top, f"{key}: top_policy"
        assert cell["top_wins"] == expected_top_wins, f"{key}: top_wins"
        assert cell["runner_up_wins"] == expected_runner, f"{key}: runner_up_wins"


def test_margin_equals_top_minus_runner(data):
    for key, cell in data["per_cell"].items():
        assert cell["margin"] == cell["top_wins"] - cell["runner_up_wins"], key


# ── group 3: classification + tied_top_policies ──────────────────────────


def test_class_follows_margin_rule(data):
    for key, cell in data["per_cell"].items():
        assert cell["class"] == _classify(cell["margin"]), f"{key}: class"


def test_tied_top_policies_is_sorted_and_excludes_top(data):
    for key, cell in data["per_cell"].items():
        expected = sorted(
            p for p, w in cell["win_counts"].items()
            if w == cell["top_wins"] and p != cell["top_policy"]
        )
        assert cell["tied_top_policies"] == expected, key


def test_per_cell_key_pattern(data):
    apps = set(data["meta"]["apps"])
    for key, cell in data["per_cell"].items():
        assert "__" in key
        a, l3 = key.split("__")
        assert a == cell["app"] and l3 == cell["l3_size"]
        assert a in apps and l3 in PAPER_L3


# ── group 4: aggregates + disclosures ────────────────────────────────────


def test_class_counts_equals_per_cell_counter(data):
    expected = Counter(cell["class"] for cell in data["per_cell"].values())
    for cls, n in data["meta"]["class_counts"].items():
        assert expected[cls] == n, cls
    for cls, n in expected.items():
        assert data["meta"]["class_counts"].get(cls, 0) == n, cls


def test_n_cells_total_matches_sum(data):
    expected = sum(data["meta"]["class_counts"].values())
    assert data["meta"]["n_cells_total"] == expected
    assert data["meta"]["n_cells_total"] == len(data["per_cell"])


def test_strong_cell_fraction_rounded(data):
    counts = data["meta"]["class_counts"]
    strong = counts.get("decisive", 0) + counts.get("moderate", 0)
    total = data["meta"]["n_cells_total"]
    expected = round(strong / total, 4) if total else 0.0
    assert math.isclose(
        data["meta"]["strong_cell_fraction"], expected, abs_tol=FRACTION_TOL
    )


def test_weak_cells_list_matches_per_cell(data):
    expected = sorted(k for k, v in data["per_cell"].items() if v["class"] == "weak")
    assert data["meta"]["weak_cells"] == expected
    assert data["meta"]["n_weak_cells"] == len(expected)


def test_tied_cells_list_matches_per_cell(data):
    expected = sorted(k for k, v in data["per_cell"].items() if v["class"] == "tied")
    assert data["meta"]["tied_cells"] == expected
    assert data["meta"]["n_tied_cells"] == len(expected)
