"""Cross-artifact integrity gate for ``claim_density.json``.

Gate 108 locks ``wiki/data/claim_density.json`` to two sources:

1.  ``wiki/data/literature_reproduction_summary.csv`` — the canonical
    per-(graph, app, l3_size, policy, citation) verdict table. Every
    per-graph rollup field in claim_density (``n_claims``, ``n_ok``,
    ``n_cells``, ``n_apps``, ``n_policies``, ``n_citations``,
    ``status_counts``, ``ok_pct``) must recompute exactly from the CSV
    for that graph.

2.  ``scripts/experiments/ecg/literature_baselines.py`` — the canonical
    Python module that defines ``INVARIANT_CLAIMS``, ``PER_GRAPH_CLAIMS``
    and ``KNOWN_DEVIATIONS``. Every CSV row must be reachable from
    ``claims_for()`` or registered in ``KNOWN_DEVIATIONS``; every
    deviation key must appear in the CSV with some status; every CSV
    citation string must be one of the citations declared in
    ``INVARIANT_CLAIMS + PER_GRAPH_CLAIMS``.

Together they pin the graph -> claims pipeline end-to-end: a future
contributor who silently drops a graph from the baselines, mutates a
citation string, or breaks the rollup logic in claim_density_report.py
will trip a named invariant here.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"

CLAIM_DENSITY_PATH = REPO_ROOT / "wiki" / "data" / "claim_density.json"
REPRO_CSV_PATH = REPO_ROOT / "wiki" / "data" / "literature_reproduction_summary.csv"

EXPECTED_GRAPHS = {
    "cit-Patents",
    "com-orkut",
    "email-Eu-core",
    "soc-LiveJournal1",
    "soc-pokec",
    "web-Google",
}
EXPECTED_GRAPH_COUNT = 6
EXPECTED_TOTAL_CLAIMS = 279

NUMERIC_PCT_TOL = 1e-6

# Statuses we accept appearing in the CSV. Anything else is a hint that
# a new label was introduced without rolling it into claim_density logic.
EXPECTED_STATUSES = {
    "ok",
    "within_tolerance",
    "known_deviation",
    "missing",
    "no_claim",
    "fail",
    "disagree",
    "insufficient_data",
}


@pytest.fixture(scope="module")
def claim_density() -> dict:
    return json.loads(CLAIM_DENSITY_PATH.read_text())


@pytest.fixture(scope="module")
def repro_rows() -> list[dict]:
    with REPRO_CSV_PATH.open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def lit_baselines():
    sys.path.insert(0, str(ECG_DIR))
    try:
        import literature_baselines as lb  # type: ignore[import-not-found]
    finally:
        try:
            sys.path.remove(str(ECG_DIR))
        except ValueError:
            pass
    return lb


def _by_graph(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["graph"]].append(r)
    return dict(out)


# ---------------------------------------------------------------------------
# Group A — claim_density ↔ reproduction CSV (4)
# ---------------------------------------------------------------------------


def test_graph_universe_matches(claim_density: dict, repro_rows: list[dict]) -> None:
    summary_graphs = {g["graph"] for g in claim_density["graphs"]}
    csv_graphs = {r["graph"] for r in repro_rows}
    assert summary_graphs == EXPECTED_GRAPHS, (
        f"claim_density graphs={sorted(summary_graphs)} expected={sorted(EXPECTED_GRAPHS)}"
    )
    assert csv_graphs == EXPECTED_GRAPHS, (
        f"reproduction CSV graphs={sorted(csv_graphs)} expected={sorted(EXPECTED_GRAPHS)}"
    )
    assert claim_density["summary"]["n_graphs"] == EXPECTED_GRAPH_COUNT, (
        f"summary.n_graphs={claim_density['summary']['n_graphs']} expected={EXPECTED_GRAPH_COUNT}"
    )
    assert len(claim_density["graphs"]) == EXPECTED_GRAPH_COUNT


def test_total_claim_count_matches_csv(claim_density: dict, repro_rows: list[dict]) -> None:
    assert claim_density["summary"]["total_claims"] == len(repro_rows) == EXPECTED_TOTAL_CLAIMS
    per_graph_sum = sum(g["n_claims"] for g in claim_density["graphs"])
    assert per_graph_sum == claim_density["summary"]["total_claims"], (
        f"sum(per-graph n_claims)={per_graph_sum} != summary.total_claims="
        f"{claim_density['summary']['total_claims']}"
    )


def test_total_ok_count_matches_csv(claim_density: dict, repro_rows: list[dict]) -> None:
    csv_ok = sum(1 for r in repro_rows if r["status"] == "ok")
    assert claim_density["summary"]["total_ok"] == csv_ok, (
        f"summary.total_ok={claim_density['summary']['total_ok']} CSV ok={csv_ok}"
    )
    per_graph_ok_sum = sum(g["n_ok"] for g in claim_density["graphs"])
    assert per_graph_ok_sum == csv_ok, (
        f"sum(per-graph n_ok)={per_graph_ok_sum} CSV ok={csv_ok}"
    )


def test_per_graph_fields_recompute_from_csv(claim_density: dict, repro_rows: list[dict]) -> None:
    """Every per-graph numeric field + status_counts dict recomputes exactly
    from the matching CSV slice."""
    by_g = _by_graph(repro_rows)
    bad = []
    for entry in claim_density["graphs"]:
        g = entry["graph"]
        grows = by_g.get(g, [])
        assert grows, f"claim_density has graph {g!r} not present in CSV"
        n = len(grows)
        n_ok = sum(1 for r in grows if r["status"] == "ok")
        cells = {(r["app"], r["l3_size"]) for r in grows}
        apps = sorted({r["app"] for r in grows})
        policies = sorted({r["policy"] for r in grows})
        citations = sorted({r["citation"] for r in grows if r.get("citation")})
        status_counts = dict(Counter(r["status"] for r in grows))
        checks = (
            ("n_claims", n, entry["n_claims"]),
            ("n_ok", n_ok, entry["n_ok"]),
            ("n_cells", len(cells), entry["n_cells"]),
            ("n_apps", len(apps), entry["n_apps"]),
            ("n_policies", len(policies), entry["n_policies"]),
            ("n_citations", len(citations), entry["n_citations"]),
            ("apps", apps, entry["apps"]),
            ("policies", policies, entry["policies"]),
            ("citations", citations, entry["citations"]),
            ("status_counts", status_counts, entry["status_counts"]),
        )
        for label, computed, stated in checks:
            if computed != stated:
                bad.append((g, label, computed, stated))
    assert not bad, f"claim_density per-graph field mismatches: {bad[:5]}"


# ---------------------------------------------------------------------------
# Group B — CSV ↔ literature_baselines.py (4)
# ---------------------------------------------------------------------------


def test_every_csv_row_reachable_from_baselines(repro_rows: list[dict], lit_baselines) -> None:
    """Every CSV row's (graph, app, l3_size, policy) is reachable from
    ``claims_for(graph, app, l3_size)`` or registered in
    ``KNOWN_DEVIATIONS``."""
    bad = []
    for r in repro_rows:
        key4 = (r["graph"], r["app"], r["l3_size"], r["policy"])
        matched = lit_baselines.claims_for(r["graph"], r["app"], r["l3_size"])
        matched_policies = {c.policy for c in matched}
        if r["policy"] not in matched_policies and key4 not in lit_baselines.KNOWN_DEVIATIONS:
            bad.append(key4)
    assert not bad, (
        f"CSV rows not reachable from literature_baselines: {len(bad)} examples={bad[:5]}"
    )


def test_every_known_deviation_appears_in_csv(repro_rows: list[dict], lit_baselines) -> None:
    """Every key in ``KNOWN_DEVIATIONS`` must appear in the CSV with some
    status — a deviation key with no corresponding row means the underlying
    sweep cell was dropped silently."""
    csv_keys = {(r["graph"], r["app"], r["l3_size"], r["policy"]) for r in repro_rows}
    missing = [k for k in lit_baselines.KNOWN_DEVIATIONS if k not in csv_keys]
    assert not missing, (
        f"KNOWN_DEVIATIONS keys missing from CSV: {len(missing)} examples={missing[:5]}"
    )


def test_all_known_graphs_present_in_csv(repro_rows: list[dict], lit_baselines) -> None:
    """``all_known_graphs()`` (graphs that appear in PER_GRAPH_CLAIMS) must
    all show up in the CSV — none silently dropped from the sweep."""
    csv_graphs = {r["graph"] for r in repro_rows}
    missing = [g for g in lit_baselines.all_known_graphs() if g not in csv_graphs]
    assert not missing, f"PER_GRAPH_CLAIMS graphs missing from CSV: {missing}"


def test_csv_citations_subset_of_baselines(repro_rows: list[dict], lit_baselines) -> None:
    """Every citation string in the CSV is exactly one of the citations
    declared in ``INVARIANT_CLAIMS + PER_GRAPH_CLAIMS``."""
    baseline_citations = {
        c.citation
        for c in (lit_baselines.INVARIANT_CLAIMS + lit_baselines.PER_GRAPH_CLAIMS)
    }
    csv_citations = {r["citation"] for r in repro_rows if r.get("citation")}
    rogue = csv_citations - baseline_citations
    assert not rogue, (
        f"CSV citation strings not declared in literature_baselines: {sorted(rogue)[:3]}"
    )


# ---------------------------------------------------------------------------
# Group C — claim_density internal hygiene (4)
# ---------------------------------------------------------------------------


def test_status_counts_sum_to_n_claims(claim_density: dict) -> None:
    bad = []
    for entry in claim_density["graphs"]:
        s = sum(entry["status_counts"].values())
        if s != entry["n_claims"]:
            bad.append((entry["graph"], s, entry["n_claims"]))
    assert not bad, f"status_counts sums ≠ n_claims: {bad}"


def test_n_ok_equals_status_counts_ok(claim_density: dict) -> None:
    bad = []
    for entry in claim_density["graphs"]:
        if entry["n_ok"] != entry["status_counts"].get("ok", 0):
            bad.append((entry["graph"], entry["n_ok"], entry["status_counts"].get("ok", 0)))
    assert not bad, f"n_ok ≠ status_counts['ok']: {bad}"


def test_ok_pct_matches_ratio(claim_density: dict) -> None:
    bad = []
    for entry in claim_density["graphs"]:
        if entry["n_claims"] == 0:
            expected = 0.0
        else:
            expected = entry["n_ok"] / entry["n_claims"] * 100.0
        if abs(entry["ok_pct"] - expected) > NUMERIC_PCT_TOL:
            bad.append((entry["graph"], entry["ok_pct"], expected))
    assert not bad, f"ok_pct mismatches: {bad}"


def test_status_counts_use_known_states_only(claim_density: dict) -> None:
    bad = []
    for entry in claim_density["graphs"]:
        rogue = set(entry["status_counts"].keys()) - EXPECTED_STATUSES
        if rogue:
            bad.append((entry["graph"], sorted(rogue)))
    assert not bad, f"unknown statuses in claim_density: {bad}"


# ---------------------------------------------------------------------------
# Group D — summary math (1)
# ---------------------------------------------------------------------------


def test_summary_aggregates(claim_density: dict, repro_rows: list[dict]) -> None:
    """Summary fields are internally consistent: total_ok_pct is ratio of
    total_ok / total_claims * 100, and every CSV row appears exactly once."""
    summary = claim_density["summary"]
    expected_pct = summary["total_ok"] / summary["total_claims"] * 100.0
    assert abs(summary["total_ok_pct"] - expected_pct) < NUMERIC_PCT_TOL, (
        f"summary.total_ok_pct={summary['total_ok_pct']} expected={expected_pct}"
    )
    # CSV rows must be unique per (graph, app, l3_size, policy, citation):
    # duplicate rows would distort all counts in claim_density.
    keys = [(r["graph"], r["app"], r["l3_size"], r["policy"], r["citation"]) for r in repro_rows]
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    assert not dupes, f"duplicate (graph, app, l3, policy, citation) rows in CSV: {dupes[:5]}"
