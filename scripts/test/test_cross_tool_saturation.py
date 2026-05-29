"""Sanity gate for ``wiki/data/cross_tool_saturation.{csv,json,md}``.

The cross-tool saturation report (built by
``scripts/experiments/ecg/cross_tool_saturation_report.py``) pairs each
(graph, app) cell present in BOTH the cache_sim lit-faith corpus AND a
hardware-anchor (gem5/Sniper) and asks whether the two tools agree at
saturation. This module pins:

1. All three artifacts exist and are non-empty.
2. CSV ↔ JSON cell sets agree.
3. At least one *doubly saturated* cell is present (lose this and the
   anchor data has either gone stale or has stopped overlapping the
   lit-faith corpus).
4. Every doubly-saturated cell agrees on direction within the report's
   own headline tolerance (this is the central soundness claim — if it
   breaks, cache_sim and gem5/Sniper disagree at saturation, which is
   a publication-blocking problem).
5. Every recorded regime is one of the known labels.
6. Spreads are non-negative.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI = REPO_ROOT / "wiki" / "data"
CSV_PATH = WIKI / "cross_tool_saturation.csv"
JSON_PATH = WIKI / "cross_tool_saturation.json"
MD_PATH = WIKI / "cross_tool_saturation.md"

KNOWN_REGIMES = {
    "doubly_saturated", "single_saturated", "neither_saturated", "incomplete",
}
KNOWN_TOOLS = {"gem5", "sniper"}


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"{path.relative_to(REPO_ROOT)} not on disk. "
            f"Run `python3 -m scripts.experiments.ecg.cross_tool_saturation_report`."
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


def test_csv_and_json_have_same_cells(csv_rows: list[dict], payload: dict) -> None:
    csv_cells = {(r["graph"], r["app"], r["tool"]) for r in csv_rows}
    json_cells = {(c["graph"], c["app"], c["tool"]) for c in payload["cells"]}
    assert csv_cells == json_cells, (
        f"CSV/JSON disagree. only_csv={sorted(csv_cells - json_cells)[:3]} "
        f"only_json={sorted(json_cells - csv_cells)[:3]}"
    )


def test_every_tool_is_known(payload: dict) -> None:
    tools = {c["tool"] for c in payload["cells"]}
    unknown = tools - KNOWN_TOOLS
    assert not unknown, f"unknown tool labels: {sorted(unknown)}"


def test_every_regime_is_known(payload: dict) -> None:
    regimes = {c["regime"] for c in payload["cells"]}
    unknown = regimes - KNOWN_REGIMES
    assert not unknown, f"unknown regime labels: {sorted(unknown)}"


def test_spreads_are_non_negative(csv_rows: list[dict]) -> None:
    for r in csv_rows:
        for fld in ("cache_sim_spread_pp", "anchor_spread_pp"):
            v = r.get(fld, "")
            if v in ("", None):
                continue
            assert float(v) >= 0.0, f"negative spread {fld}={v} in {r}"


def test_at_least_one_doubly_saturated_cell(payload: dict) -> None:
    """If this fails the anchor sweeps no longer overlap the lit-faith
    corpus at saturation — usually because gem5/Sniper data went stale.
    """
    n = payload["summary"]["doubly_saturated_total"]
    assert n >= 1, (
        "No (graph, app) cell is doubly saturated in cache_sim AND "
        "either anchor. Re-run gem5/Sniper anchors or expand their "
        "graph scope to overlap the lit-faith corpus."
    )


def test_doubly_saturated_cells_all_agree(payload: dict) -> None:
    """Headline cross-tool soundness claim: at saturation, the three
    simulators must agree within --headline-tol on GRASP−LRU. If this
    fails the paper cannot use cache_sim and the anchors as
    interchangeable evidence of the same architectural phenomenon.
    """
    disagrees = payload["summary"]["disagreements"]
    assert not disagrees, (
        f"{len(disagrees)} doubly-saturated cells disagree on "
        f"GRASP−LRU between cache_sim and the anchor. Examples: "
        f"{disagrees[:3]}"
    )


def test_summary_regime_counts_sum_to_total(payload: dict) -> None:
    total = payload["summary"]["n_cells"]
    s = sum(payload["summary"]["regime_counts"].values())
    assert s == total, f"regime_counts sum={s} but n_cells={total}"
