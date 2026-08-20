"""Bind public ordering recommendations to frozen experiment results."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = PROJECT_ROOT / "docs" / "recommendation-evidence.json"
ARTIFACT_ROOT = Path(
    os.environ.get(
        "GRAPHBREW_PAPER_ARTIFACT_ROOT",
        "/media/Data/00_GraphDatasets/GraphBrew/artifacts",
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def _paired_graph_result(
    rows: list[dict],
    candidate: str,
    rabbit: str,
    kernels: list[str],
) -> tuple[float, float]:
    by_cell = {
        (row["graph"], str(row["algo_id"]), row["benchmark"]):
            float(row["average_time"])
        for row in rows
    }
    graphs = sorted({row["graph"] for row in rows})
    graph_ratios = {
        graph: _geomean([
            by_cell[(graph, rabbit, kernel)]
            / by_cell[(graph, candidate, kernel)]
            for kernel in kernels
        ])
        for graph in graphs
    }
    point = _geomean(list(graph_ratios.values()))
    seed = sum(map(ord, candidate + rabbit + (
        "all" if len(kernels) == 7 else "controlled"
    )))
    rng = random.Random(seed)
    bootstrap = [
        _geomean([
            graph_ratios[graphs[rng.randrange(len(graphs))]]
            for _ in graphs
        ])
        for _ in range(50_000)
    ]
    return point, _percentile(bootstrap, 0.025)


def test_public_recommendations_match_frozen_evidence():
    evidence = json.loads(EVIDENCE_PATH.read_text())
    sources = evidence["source_artifacts"]
    resolved = {
        key: ARTIFACT_ROOT / record["path"]
        for key, record in sources.items()
    }
    missing = [str(path) for path in resolved.values() if not path.is_file()]
    if missing:
        pytest.skip(
            "Frozen recommendation artifacts are unavailable: "
            + ", ".join(missing)
        )
    for key, path in resolved.items():
        assert _sha256(path) == sources[key]["sha256"]

    rows = json.loads(resolved["full_kernel_matrix"].read_text())
    claims = {
        record["spec"]: record
        for record in evidence["validated_recommendations"]
        if "spec" in record
    }
    gorder = (
        "12:leiden:compose:sg_none:"
        "comm_size_desc:intra_gorder:gw8"
    )
    rcmpp = (
        "12:leiden:compose:sg_none:"
        "comm_size_desc:intra_rcmpp"
    )
    all_kernels = [
        "pr", "pr_spmv", "bfs", "cc", "cc_sv", "sssp", "bc",
    ]
    controlled = ["pr", "pr_spmv", "sssp", "bc"]

    for candidate, kernels in (
        (gorder, all_kernels),
        (rcmpp, controlled),
    ):
        claim = claims[candidate]
        for rabbit, point_key, lower_key in (
            ("8:csr", "rabbit_csr_over_graphbrew", "rabbit_csr_lower_95"),
            (
                "8:boost",
                "rabbit_boost_over_graphbrew",
                "rabbit_boost_lower_95",
            ),
        ):
            point, lower = _paired_graph_result(
                rows, candidate, rabbit, kernels,
            )
            assert point == pytest.approx(claim[point_key])
            assert lower == pytest.approx(claim[lower_key])

    rapid = json.loads(
        resolved["low_reuse_rapid_matrix"].read_text()
    )
    low_reuse = next(
        record
        for record in evidence["validated_recommendations"]
        if record["objective"].startswith("Low-reuse")
    )
    comparison = rapid["comparisons"]["bounded_gorder"]
    assert (
        comparison["rabbit_csr"]["reuse1_end_to_end"][
            "rabbit_over_graphbrew_gm"
        ]
        == pytest.approx(low_reuse["rabbit_csr_over_graphbrew"])
    )
    assert (
        comparison["rabbit_boost"]["reuse1_end_to_end"][
            "rabbit_over_graphbrew_gm"
        ]
        == pytest.approx(low_reuse["rabbit_boost_over_graphbrew"])
    )

    selector_claim = next(
        record
        for record in evidence["validated_recommendations"]
        if record["objective"].startswith("Automatic all-kernel")
    )
    selector = json.loads(
        resolved["allkernel_lowreuse_rule2"].read_text()
    )
    assert selector["reuse1"]["boost_over_candidate_gm"] == pytest.approx(
        selector_claim["reuse1_boost_over_candidate_gm"]
    )
    assert selector["reuse1"]["lower_95"] == pytest.approx(
        selector_claim["reuse1_lower_95"]
    )
    assert selector["reuse2"]["boost_over_candidate_gm"] == pytest.approx(
        selector_claim["reuse2_boost_over_candidate_gm"]
    )
    assert selector["reuse2"]["lower_95"] == pytest.approx(
        selector_claim["reuse2_lower_95"]
    )
