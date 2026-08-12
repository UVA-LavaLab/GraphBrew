"""Kernel-specific modeled property working-set sizes."""

from __future__ import annotations

NODE_ID_BYTES = 4
SCORE_BYTES = 4
COUNT_BYTES = 8

KERNEL_CLASS = {
    "generic": 0,
    "pr": 1,
    "bfs": 2,
    "cc": 3,
    "sssp": 4,
    "bc": 5,
    "tc": 6,
    "pr_spmv": 7,
    "cc_sv": 8,
}


def modeled_property_bytes(
    benchmark: str,
    nodes: int,
    directed_edges: int,
) -> int:
    """Mirror C++ ``ModeledPropertyBytes`` exactly."""
    if nodes < 0 or directed_edges < 0:
        raise ValueError("Graph dimensions must be non-negative")
    bitmap_bytes = (nodes + 7) // 8
    if benchmark in {"pr", "pr_spmv"}:
        return nodes * 2 * SCORE_BYTES
    if benchmark == "bfs":
        return nodes * 2 * NODE_ID_BYTES + 2 * bitmap_bytes
    if benchmark in {"cc", "cc_sv"}:
        return nodes * NODE_ID_BYTES + bitmap_bytes
    if benchmark == "sssp":
        return (
            nodes * NODE_ID_BYTES
            + directed_edges * NODE_ID_BYTES
        )
    if benchmark == "bc":
        return (
            nodes * (SCORE_BYTES + COUNT_BYTES + 2 * NODE_ID_BYTES)
            + (directed_edges + 7) // 8
        )
    if benchmark == "tc":
        return nodes * NODE_ID_BYTES
    raise ValueError(
        f"Kernel-specific modeled property bytes unavailable for {benchmark}"
    )


def property_wsr_llc(
    benchmark: str,
    nodes: int,
    directed_edges: int,
    llc_bytes: int,
) -> float:
    if llc_bytes <= 0:
        raise ValueError("LLC capacity must be positive")
    return modeled_property_bytes(
        benchmark, nodes, directed_edges,
    ) / llc_bytes
