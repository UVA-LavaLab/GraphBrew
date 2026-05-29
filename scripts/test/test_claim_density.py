"""Sanity gate for ``wiki/data/claim_density.{csv,json,md}``.

The claim-density mini-report (built by
``scripts/experiments/ecg/claim_density_report.py``) tallies per-graph
literature claims from the reproduction summary. This module pins:

1. All three artifacts exist and are non-empty.
2. CSV and JSON describe the same set of graphs.
3. The graph set is a subset of the lit-faith CSV graph set
   (no orphan graphs).
4. Every graph has ≥ 1 claim and ≥ 1 cell (no zero-density entries).
5. Per-graph sum of status counts equals n_claims.
6. Total claims across graphs equals summary.total_claims.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI = REPO_ROOT / "wiki" / "data"
CSV_PATH = WIKI / "claim_density.csv"
JSON_PATH = WIKI / "claim_density.json"
MD_PATH = WIKI / "claim_density.md"
LIT_FAITH_CSV = WIKI / "literature_faithfulness_postfix.csv"


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"{path.relative_to(REPO_ROOT)} not on disk. "
            f"Run `python3 -m scripts.experiments.ecg.claim_density_report`."
        )


@pytest.fixture(scope="module")
def csv_rows() -> list[dict]:
    _require(CSV_PATH)
    with CSV_PATH.open(newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def payload() -> dict:
    _require(JSON_PATH)
    return json.loads(JSON_PATH.read_text())


def test_artifacts_exist() -> None:
    for p in (CSV_PATH, JSON_PATH, MD_PATH):
        _require(p)
        assert p.stat().st_size > 0, f"{p} is empty"


def test_csv_and_json_have_same_graphs(csv_rows: list[dict], payload: dict) -> None:
    csv_graphs = {r["graph"] for r in csv_rows}
    json_graphs = {r["graph"] for r in payload["graphs"]}
    assert csv_graphs == json_graphs, (
        f"CSV/JSON graph sets disagree. "
        f"only_csv={sorted(csv_graphs - json_graphs)} "
        f"only_json={sorted(json_graphs - csv_graphs)}"
    )


def test_density_graphs_subset_of_lit_faith(payload: dict) -> None:
    _require(LIT_FAITH_CSV)
    with LIT_FAITH_CSV.open(newline="") as fh:
        lit_graphs = {r["graph"] for r in csv.DictReader(fh)}
    density_graphs = {r["graph"] for r in payload["graphs"]}
    orphans = density_graphs - lit_graphs
    assert not orphans, (
        f"claim_density mentions graphs absent from lit-faith CSV: "
        f"{sorted(orphans)}"
    )


def test_every_graph_has_at_least_one_claim_and_cell(payload: dict) -> None:
    bad: list[str] = []
    for r in payload["graphs"]:
        if r["n_claims"] < 1 or r["n_cells"] < 1:
            bad.append(f"{r['graph']} claims={r['n_claims']} cells={r['n_cells']}")
    assert not bad, f"zero-density graph(s): {bad}"


def test_status_counts_sum_to_n_claims(payload: dict) -> None:
    for r in payload["graphs"]:
        s = sum(r["status_counts"].values())
        assert s == r["n_claims"], (
            f"{r['graph']}: sum(status_counts)={s} but n_claims={r['n_claims']}"
        )


def test_summary_total_matches_per_graph_sum(payload: dict) -> None:
    total = sum(r["n_claims"] for r in payload["graphs"])
    assert total == payload["summary"]["total_claims"], (
        f"per-graph sum={total} but summary.total_claims={payload['summary']['total_claims']}"
    )
    total_ok = sum(r["n_ok"] for r in payload["graphs"])
    assert total_ok == payload["summary"]["total_ok"], (
        f"per-graph sum(n_ok)={total_ok} but summary.total_ok={payload['summary']['total_ok']}"
    )


def test_ok_pct_within_unit_range(payload: dict) -> None:
    bad = [(r["graph"], r["ok_pct"]) for r in payload["graphs"]
           if not (0.0 <= r["ok_pct"] <= 100.0)]
    assert not bad, f"ok_pct out of [0, 100]: {bad}"
