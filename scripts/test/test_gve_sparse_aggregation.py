"""Guard the sparse GVE aggregation implementation."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPHBREW_HEADER = (
    PROJECT_ROOT
    / "bench/include/graphbrew/reorder/reorder_graphbrew.h"
)


def test_gve_aggregation_uses_sparse_thread_local_scanner():
    source = GRAPHBREW_HEADER.read_text()
    start = source.index("size_t aggregateGVEStyle(")
    end = source.index(
        "//=============================================================================\n"
        "// SECTION 14:",
        start,
    )
    implementation = source[start:end]
    assert "CommunityScanner<K, Weight> scanner(C);" in implementation
    assert "thread_edge_weights" not in implementation
    assert "thread_touched_comms" not in implementation
