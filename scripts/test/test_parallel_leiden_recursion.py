"""Parallel Leiden must remain parallel in recursive LAYER subgraphs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_recursive_leiden_inherits_community_thread_policy():
    builder = (
        PROJECT_ROOT
        / "bench/include/external/gapbs/builder.h"
    ).read_text()
    assert (
        "sub_config.deterministicCommunityDetection =\n"
        "                    config.deterministicCommunityDetection;"
    ) in builder
    graphbrew = (
        PROJECT_ROOT
        / "bench/include/graphbrew/reorder/reorder_graphbrew.h"
    ).read_text()
    assert (
        "config.gorderFallback > 0\n"
        "                    && commVertices[c].size()"
    ) in graphbrew
