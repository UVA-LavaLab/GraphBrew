"""Amortized CPU selector scope tests."""

from scripts.experiments.adaptive.cpu_sprint import (
    CPU_BUDGET_HOURS,
    CPU_DIAGNOSTIC_KERNELS,
    CPU_EXPANSION_TARGET_GRAPHS,
    CPU_HEADLINE_KERNELS,
    CPU_MAPPING_DRAWS,
    CPU_MDE_LIMIT,
    CPU_PRIMARY_REUSE,
    CPU_RAPID_KERNELS,
    CPU_RAPID_PROJECTED_HIGH_HOURS,
    CPU_RESERVE_HOURS,
    CPU_REUSE_REGIMES,
    CPU_SPRINT_GRAPHS,
)
from scripts.experiments.vldb.config import (
    ADAPTIVE_CPU_EXPANSION_GRAPHS,
    EVAL_GRAPHS,
    VLDB_GRAPH_SOURCES,
)
from scripts.lib.pipeline.download import get_graph_info
from scripts.experiments.vldb.runner import select_requested_graphs


def test_cpu_selector_scope_is_unique_and_downloadable():
    names = [graph["name"] for graph in CPU_SPRINT_GRAPHS]
    expansion_names = [
        graph["name"] for graph in ADAPTIVE_CPU_EXPANSION_GRAPHS
    ]
    assert len(names) == len(set(names)) == CPU_EXPANSION_TARGET_GRAPHS
    assert len(EVAL_GRAPHS) == 11
    assert len(expansion_names) == 19
    assert all(
        VLDB_GRAPH_SOURCES[name]["source"] == "catalog"
        and get_graph_info(name) is not None
        for name in expansion_names
    )
    selected = select_requested_graphs([
        "cit-Patents",
        "email-Enron",
        "cit-Patents",
    ])
    assert [graph["name"] for graph in selected] == [
        "cit-Patents",
        "email-Enron",
    ]


def test_cpu_selector_claim_and_budget_scope_is_frozen():
    assert CPU_HEADLINE_KERNELS == (
        "bc", "bfs", "cc", "cc_sv", "pr", "pr_spmv",
    )
    assert CPU_DIAGNOSTIC_KERNELS == ("sssp",)
    assert CPU_REUSE_REGIMES == (20, 50, 100)
    assert CPU_PRIMARY_REUSE == 50
    assert CPU_MAPPING_DRAWS == 3
    assert CPU_RAPID_KERNELS == ("pr_spmv", "cc", "cc_sv")
    assert CPU_RAPID_PROJECTED_HIGH_HOURS == 12.0
    assert CPU_MDE_LIMIT == 0.05
    assert CPU_BUDGET_HOURS == 168.0
    assert CPU_RESERVE_HOURS == 16.8
    assert 1.0 / 0.97 > 1.03
