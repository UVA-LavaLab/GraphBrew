"""Derivation parity gate for ``wiki/data/oracle_gap.json``.

THE upstream of EVERYTHING: ~13 downstream artifacts (gates 36, 41, 169,
172–178 etc.) source rows from this file. Locking its derivation closes
the deepest seam in the dashboard — every aggregation gate above already
sanity-checks its slice, but this gate mirrors the per-cell construction
(empirical-oracle floor) and the summary stat block byte-for-byte against
``literature_faithfulness_postfix.csv``.

Locks:
* Per-cell skip predicate (cells with < 2 policies dropped; unknown
  families dropped).
* Oracle = min over PRESENT policies (NOT all four).
* is_winner predicate uses ``abs(mr − oracle) < 1e-9`` (NOT equality).
* miss_rate / oracle formatted to 6dp strings; gap_pp to 3dp string;
  n_policies_in_cell as int.
* Row sort key (family, graph, app, L3_SIZE_BYTES[l3], policy) — load-bearing
  for downstream consumers that assume stable order.
* Regime classifier NON-STRICT <= boundaries (tiny ≤64KB, small ≤256KB,
  else large).
* Summary p90 index ``sorted[min(n-1, int(round(0.9·(n-1))))]`` —
  bespoke not numpy.percentile.
* Summary mean uses statistics.fmean; median uses statistics.median;
  all stats round 4dp.
* by_policy_family / by_policy_regime keys sorted by (policy, label) tuple.
* overall_by_policy emits ALL of POLICIES (even if no rows; 0-filled).
* wins per-policy = count of is_winner == "1" rows.

Mirrors ``_load_cells()`` / ``_per_cell()`` / ``_summarize()`` from
``scripts/experiments/ecg/oracle_gap_report.py`` verbatim.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "oracle_gap.json"
UPSTREAM_CSV = WIKI_DATA / "literature_faithfulness_postfix.csv"

POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")

GRAPH_FAMILY = {
    "email-Eu-core":   "social",
    "soc-pokec":       "social",
    "soc-LiveJournal1": "social",
    "com-orkut":       "social",
    "cit-Patents":     "citation",
    "web-Google":      "web",
    "roadNet-CA":      "road",
    "delaunay_n19":    "mesh",
}

L3_SIZE_BYTES = {
    "4kB":  4 * 1024,
    "16kB": 16 * 1024,
    "32kB": 32 * 1024,
    "64kB": 64 * 1024,
    "256kB": 256 * 1024,
    "1MB":  1024 * 1024,
    "2MB":  2 * 1024 * 1024,
    "4MB":  4 * 1024 * 1024,
    "8MB":  8 * 1024 * 1024,
}


def _regime(l3):
    b = L3_SIZE_BYTES.get(l3, -1)
    if b < 0:
        return "unknown"
    if b <= 64 * 1024:
        return "tiny"
    if b <= 256 * 1024:
        return "small"
    return "large"


def _load_cells(path):
    out = defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                mr = float(r["miss_rate"])
            except (KeyError, ValueError):
                continue
            if not math.isfinite(mr):
                continue
            pol = (r.get("policy") or "").strip()
            if pol not in POLICIES:
                continue
            key = (r.get("graph", ""), r.get("app", ""), r.get("l3_size", ""))
            out[key][pol] = mr
    return out


def _per_cell(cells):
    rows = []
    for key, miss_by_pol in cells.items():
        if len(miss_by_pol) < 2:
            continue
        graph, app, l3 = key
        fam = GRAPH_FAMILY.get(graph, "unknown")
        if fam == "unknown":
            continue
        oracle = min(miss_by_pol.values())
        for pol in POLICIES:
            mr = miss_by_pol.get(pol)
            if mr is None:
                continue
            gap_pp = (mr - oracle) * 100.0
            rows.append({
                "graph":     graph,
                "app":       app,
                "l3_size":   l3,
                "family":    fam,
                "regime":    _regime(l3),
                "policy":    pol,
                "miss_rate": f"{mr:.6f}",
                "oracle":    f"{oracle:.6f}",
                "gap_pp":    f"{gap_pp:.3f}",
                "is_winner": "1" if abs(mr - oracle) < 1e-9 else "0",
                "n_policies_in_cell": len(miss_by_pol),
            })
    rows.sort(key=lambda r: (
        r["family"], r["graph"], r["app"],
        L3_SIZE_BYTES.get(r["l3_size"], 0), r["policy"],
    ))
    return rows


def _summarize(rows):
    by_policy = defaultdict(list)
    by_policy_family = defaultdict(list)
    by_policy_regime = defaultdict(list)
    win_count = defaultdict(int)
    cells_seen = set()
    for r in rows:
        gap = float(r["gap_pp"])
        pol = r["policy"]
        by_policy[pol].append(gap)
        by_policy_family[(pol, r["family"])].append(gap)
        by_policy_regime[(pol, r["regime"])].append(gap)
        if r["is_winner"] == "1":
            win_count[pol] += 1
        cells_seen.add((r["graph"], r["app"], r["l3_size"]))

    def _stats(xs):
        if not xs:
            return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
        ys = sorted(xs)
        p90 = ys[min(len(ys) - 1, int(round(0.9 * (len(ys) - 1))))]
        return {
            "n": len(xs),
            "mean":   round(statistics.fmean(xs), 4),
            "median": round(statistics.median(xs), 4),
            "p90":    round(p90, 4),
            "max":    round(max(xs), 4),
        }

    return {
        "n_cells": len(cells_seen),
        "n_rows":  len(rows),
        "overall_by_policy": {
            p: {**_stats(by_policy[p]), "wins": win_count[p]}
            for p in POLICIES
        },
        "by_policy_family": {
            f"{p}/{fam}": _stats(xs)
            for (p, fam), xs in sorted(by_policy_family.items())
        },
        "by_policy_regime": {
            f"{p}/{r}": _stats(xs)
            for (p, r), xs in sorted(by_policy_regime.items())
        },
    }


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def expected_rows():
    if not UPSTREAM_CSV.exists():
        pytest.skip(f"missing {UPSTREAM_CSV}")
    return _per_cell(_load_cells(UPSTREAM_CSV))


@pytest.fixture(scope="module")
def expected_summary(expected_rows):
    return _summarize(expected_rows)


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"summary", "rows"}


def test_summary_top_level_keys(artifact):
    expected = {"n_cells", "n_rows",
                "overall_by_policy", "by_policy_family", "by_policy_regime"}
    assert set(artifact["summary"].keys()) == expected


def test_overall_by_policy_keys_exact(artifact):
    """Must emit ALL four POLICIES, not just those with rows
    (load-bearing — downstream consumers index by all 4)."""
    assert set(artifact["summary"]["overall_by_policy"].keys()) == set(POLICIES)


def test_overall_by_policy_entry_shape(artifact):
    expected = {"n", "mean", "median", "p90", "max", "wins"}
    for p, s in artifact["summary"]["overall_by_policy"].items():
        assert set(s.keys()) == expected, f"{p}: overall entry drift"


def test_by_policy_family_entry_shape(artifact):
    expected = {"n", "mean", "median", "p90", "max"}
    for k, s in artifact["summary"]["by_policy_family"].items():
        assert set(s.keys()) == expected, f"{k}: family entry drift"


def test_by_policy_regime_entry_shape(artifact):
    expected = {"n", "mean", "median", "p90", "max"}
    for k, s in artifact["summary"]["by_policy_regime"].items():
        assert set(s.keys()) == expected, f"{k}: regime entry drift"


def test_row_field_shape(artifact):
    expected = {
        "graph", "app", "l3_size", "family", "regime", "policy",
        "miss_rate", "oracle", "gap_pp", "is_winner",
        "n_policies_in_cell",
    }
    for r in artifact["rows"]:
        assert set(r.keys()) == expected, f"row drift: {r}"


# ----------------------------------------------------------------------
# Group B: per-row byte-exact derivation
# ----------------------------------------------------------------------

def test_rows_byte_exact_against_csv(artifact, expected_rows):
    """The full row stream must byte-match _per_cell(_load_cells(CSV))
    — pins skip predicate (<2 policies), unknown-family drop, oracle =
    min over present policies, gap_pp 3dp format, miss_rate/oracle 6dp
    format, is_winner abs-tol predicate, and the (family, graph, app,
    L3_bytes, policy) sort key."""
    assert artifact["rows"] == expected_rows


def test_n_rows_matches_row_stream(artifact):
    assert artifact["summary"]["n_rows"] == len(artifact["rows"])


def test_n_cells_matches_distinct_keys(artifact):
    distinct = {(r["graph"], r["app"], r["l3_size"]) for r in artifact["rows"]}
    assert artifact["summary"]["n_cells"] == len(distinct)


def test_row_string_formats(artifact):
    """miss_rate / oracle MUST be 6dp strings; gap_pp MUST be 3dp string;
    n_policies_in_cell MUST be int — load-bearing for byte-equal regen."""
    for r in artifact["rows"]:
        assert isinstance(r["miss_rate"], str)
        assert isinstance(r["oracle"], str)
        assert isinstance(r["gap_pp"], str)
        assert isinstance(r["n_policies_in_cell"], int)
        # 6dp count
        assert "." in r["miss_rate"] and len(r["miss_rate"].split(".")[1]) == 6
        assert "." in r["oracle"] and len(r["oracle"].split(".")[1]) == 6
        # 3dp count
        assert "." in r["gap_pp"] and len(r["gap_pp"].split(".")[1]) == 3


def test_is_winner_predicate_consistency(artifact):
    """is_winner == '1' iff abs(miss_rate − oracle) < 1e-9 — derived
    from the row-local 6dp-rounded strings."""
    for r in artifact["rows"]:
        mr = float(r["miss_rate"])
        oracle = float(r["oracle"])
        expected = "1" if abs(mr - oracle) < 1e-9 else "0"
        assert r["is_winner"] == expected, (
            f"is_winner drift: {r['graph']}/{r['app']}/{r['l3_size']}/{r['policy']}"
        )


def test_gap_pp_formula(artifact):
    """gap_pp = round((miss_rate − oracle) · 100, 3dp string) — derived
    from row-local 6dp string fields."""
    for r in artifact["rows"]:
        mr = float(r["miss_rate"])
        oracle = float(r["oracle"])
        expected = f"{(mr - oracle) * 100.0:.3f}"
        # Allow ±1 in last decimal place due to original mr/oracle
        # being themselves rounded.
        assert abs(float(r["gap_pp"]) - float(expected)) < 0.01, (
            f"gap_pp drift: {r}"
        )


def test_family_assignment_from_graph(artifact):
    for r in artifact["rows"]:
        assert r["family"] == GRAPH_FAMILY[r["graph"]]


def test_regime_assignment_from_l3(artifact):
    for r in artifact["rows"]:
        assert r["regime"] == _regime(r["l3_size"])


def test_row_sort_key(artifact):
    """Rows sorted by (family, graph, app, L3_bytes, policy)."""
    keys = [
        (r["family"], r["graph"], r["app"],
         L3_SIZE_BYTES.get(r["l3_size"], 0), r["policy"])
        for r in artifact["rows"]
    ]
    assert keys == sorted(keys)


# ----------------------------------------------------------------------
# Group C: regime classifier sanity
# ----------------------------------------------------------------------

def test_regime_boundaries():
    """Boundaries are NON-STRICT <= (load-bearing)."""
    assert _regime("64kB") == "tiny"      # boundary inclusive
    assert _regime("256kB") == "small"    # boundary inclusive
    assert _regime("4kB") == "tiny"
    assert _regime("32kB") == "tiny"
    assert _regime("1MB") == "large"
    assert _regime("8MB") == "large"


def test_regime_unknown_l3():
    assert _regime("totally-unknown") == "unknown"


# ----------------------------------------------------------------------
# Group D: summary stat block byte-exact
# ----------------------------------------------------------------------

def test_overall_by_policy_byte_exact(artifact, expected_summary):
    assert artifact["summary"]["overall_by_policy"] == expected_summary["overall_by_policy"]


def test_by_policy_family_byte_exact(artifact, expected_summary):
    assert artifact["summary"]["by_policy_family"] == expected_summary["by_policy_family"]


def test_by_policy_regime_byte_exact(artifact, expected_summary):
    assert artifact["summary"]["by_policy_regime"] == expected_summary["by_policy_regime"]


def test_n_cells_and_n_rows_byte_exact(artifact, expected_summary):
    assert artifact["summary"]["n_cells"] == expected_summary["n_cells"]
    assert artifact["summary"]["n_rows"] == expected_summary["n_rows"]


# ----------------------------------------------------------------------
# Group E: stat-formula spot checks (independent of upstream)
# ----------------------------------------------------------------------

def test_p90_formula_spot_check():
    """p90 = sorted[min(n-1, int(round(0.9·(n-1))))]
    — for n=10, idx = round(8.1) = 8 → 9th element.
    For n=5, idx = round(3.6) = 4 → 5th (last) element."""
    xs = list(range(10))  # 0..9
    ys = sorted(xs)
    idx = min(len(ys) - 1, int(round(0.9 * (len(ys) - 1))))
    assert idx == 8
    assert ys[idx] == 8

    xs2 = list(range(5))  # 0..4
    ys2 = sorted(xs2)
    idx2 = min(len(ys2) - 1, int(round(0.9 * (len(ys2) - 1))))
    assert idx2 == 4
    assert ys2[idx2] == 4


def test_wins_match_is_winner_count(artifact):
    expected_wins = defaultdict(int)
    for r in artifact["rows"]:
        if r["is_winner"] == "1":
            expected_wins[r["policy"]] += 1
    for p in POLICIES:
        assert artifact["summary"]["overall_by_policy"][p]["wins"] == expected_wins[p]


def test_overall_n_matches_row_count_per_policy(artifact):
    counts = defaultdict(int)
    for r in artifact["rows"]:
        counts[r["policy"]] += 1
    for p in POLICIES:
        assert artifact["summary"]["overall_by_policy"][p]["n"] == counts[p]


def test_family_key_format(artifact):
    for k in artifact["summary"]["by_policy_family"].keys():
        assert "/" in k
        pol, fam = k.split("/", 1)
        assert pol in POLICIES
        assert fam in set(GRAPH_FAMILY.values())


def test_regime_key_format(artifact):
    for k in artifact["summary"]["by_policy_regime"].keys():
        assert "/" in k
        pol, regime = k.split("/", 1)
        assert pol in POLICIES
        assert regime in {"tiny", "small", "large"}


def test_oracle_le_miss_rate(artifact):
    """oracle ≤ miss_rate for every row (oracle is per-cell min)."""
    for r in artifact["rows"]:
        assert float(r["oracle"]) <= float(r["miss_rate"]) + 1e-9


def test_n_policies_in_cell_ge_2(artifact):
    """Cells with <2 policies are dropped — every row should report ≥2."""
    for r in artifact["rows"]:
        assert r["n_policies_in_cell"] >= 2
