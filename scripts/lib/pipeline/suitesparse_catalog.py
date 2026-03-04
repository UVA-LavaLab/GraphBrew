#!/usr/bin/env python3
"""
SuiteSparse Matrix Collection auto-discovery for GraphBrew.

Uses the ``ssgetpy`` library to query the cached SuiteSparse CSV index
and automatically build graph catalogs.  Falls back gracefully when
``ssgetpy`` is unavailable (e.g.  offline / minimal installs).

Size categories are by **edge count** (nnz), not file size:
    Small   :   10 K –  500 K edges   (~225 available)
    Medium  :  500 K –    5 M edges   (~134 available)
    Large   :    5 M –   50 M edges   (~ 70 available)
    XLarge  :   50 M –  500 M edges   (~ 37 available)

Discovery searches multiple SuiteSparse ``kind`` keywords
(graph, network, multigraph) to maximise the pool.

Usage:
    from scripts.lib.pipeline.suitesparse_catalog import discover_graphs

    graphs = discover_graphs("medium", limit=100)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("graphbrew.suitesparse_catalog")

# ── size-category boundaries (by nnz / edge count) ───────────────────────
SIZE_BOUNDS: Dict[str, Tuple[int, int]] = {
    "SMALL":  (10_000, 500_000),
    "MEDIUM": (500_000, 5_000_000),
    "LARGE":  (5_000_000, 50_000_000),
    "XLARGE": (50_000_000, 500_000_000),
}

# SuiteSparse kind keywords that represent graph-like matrices.
# Searching all three maximises the discoverable pool.
_DISCOVERY_KINDS: Tuple[str, ...] = ("graph", "network", "multigraph")

# Canonical download URL pattern  (replaces old Heroku URL)
SUITESPARSE_MM_URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"

# Groups that are known to contain graph-like matrices
_GRAPH_GROUPS = {
    "SNAP", "DIMACS10", "LAW", "Gleich", "Newman", "Arenas",
    "Pajek", "Gset", "ML_Graph", "Mycielski", "vanHeukelum",
    "GAP", "MAWI", "GenBank", "Belcastro", "AG-Monien",
    "Gaertner", "Lucifora", "FlowIPM22", "HB", "Rossi",
}

# Kind substrings that indicate a graph/network matrix
_GRAPH_KIND_KEYWORDS = {"graph", "network", "multigraph"}


# ── helpers ───────────────────────────────────────────────────────────────

def _estimate_size_mb(nnz: int) -> int:
    """Estimate compressed download size (MB) from nnz.

    Empirical: SuiteSparse MM .tar.gz compresses to roughly 3–6 bytes/nnz.
    We use 5 as a conservative midpoint.
    """
    return max(1, int(math.ceil(nnz * 5 / (1024 * 1024))))


def _kind_to_category(kind: str) -> str:
    """Map SuiteSparse ``kind`` string to a GraphBrew category."""
    kl = kind.lower()
    if "social" in kl or "community" in kl:
        return "social"
    if "web" in kl:
        return "web"
    if "road" in kl:
        return "road"
    if "citation" in kl:
        return "citation"
    if "random" in kl or "synthetic" in kl:
        return "synthetic"
    if "temporal" in kl:
        return "temporal"
    if "multigraph" in kl:
        return "multigraph"
    if "communication" in kl:
        return "communication"
    if "weighted" in kl:
        return "weighted-graph"
    if "undirected" in kl:
        return "undirected-graph"
    if "directed" in kl:
        return "directed-graph"
    return "graph"


def _is_symmetric(psym: float) -> bool:
    """Consider a matrix symmetric if pattern symmetry >= 0.9."""
    return psym is not None and psym >= 0.9


# ── main discovery function ──────────────────────────────────────────────

def discover_graphs(
    size_category: str,
    limit: int = 100,
    *,
    prefer_diverse: bool = True,
    exclude_names: Optional[set] = None,
) -> List["DownloadableGraph"]:
    """Auto-discover graph matrices from SuiteSparse for a given size category.

    Parameters
    ----------
    size_category : str
        One of ``"SMALL"``, ``"MEDIUM"``, ``"LARGE"``, ``"XLARGE"`` (case-insensitive).
    limit : int
        Maximum number of graphs to return (default 100).
    prefer_diverse : bool
        When True, spread selections across different groups and kinds for
        structural diversity (round-robin sampling).
    exclude_names : set, optional
        Graph names to skip (e.g. already in the hardcoded catalog).

    Returns
    -------
    list of DownloadableGraph
        Discovered graphs, sorted by group then name.
    """
    # Lazy import — module is usable even when ssgetpy is missing
    try:
        import ssgetpy  # type: ignore
    except ImportError:
        log.warning(
            "ssgetpy not installed — cannot auto-discover graphs.  "
            "Install with: pip install ssgetpy"
        )
        return []

    # Avoid circular import at module level
    from .download import DownloadableGraph

    size_category = size_category.upper()
    if size_category not in SIZE_BOUNDS:
        log.error(f"Unknown size '{size_category}'. Choose from: {list(SIZE_BOUNDS)}")
        return []

    lo, hi = SIZE_BOUNDS[size_category]
    exclude_names = exclude_names or set()

    # Query for graph-kind matrices in the nnz range.
    # Search multiple kind keywords to maximise the discovery pool:
    #   "graph"      — directed/undirected/weighted graph, etc.
    #   "network"    — power network, communication network, etc.
    #   "multigraph" — multi-edge graphs
    seen_ids: set = set()
    candidates = []
    for kind_kw in _DISCOVERY_KINDS:
        for m in ssgetpy.search(kind=kind_kw, nzbounds=(lo, hi), limit=10_000):
            if m.id not in seen_ids and m.rows == m.cols:
                seen_ids.add(m.id)
                candidates.append(m)

    # Exclude already-catalogued names
    candidates = [m for m in candidates if m.name not in exclude_names]

    if not candidates:
        log.warning(f"No candidate graphs found for size {size_category} ({lo:,}–{hi:,} nnz)")
        return []

    # ── diversity-aware sampling ──────────────────────────────────────
    if prefer_diverse and len(candidates) > limit:
        # Bucket by (group, rough_kind) and round-robin to maximise variety
        from collections import defaultdict
        buckets: Dict[str, list] = defaultdict(list)
        for m in candidates:
            bucket_key = m.group
            buckets[bucket_key].append(m)

        # Sort each bucket by nnz for reproducible selection
        for v in buckets.values():
            v.sort(key=lambda m: m.nnz)

        selected = []
        bucket_keys = sorted(buckets.keys())
        idx = {k: 0 for k in bucket_keys}
        while len(selected) < limit:
            added_any = False
            for k in bucket_keys:
                if idx[k] < len(buckets[k]):
                    selected.append(buckets[k][idx[k]])
                    idx[k] += 1
                    added_any = True
                    if len(selected) >= limit:
                        break
            if not added_any:
                break
        candidates = selected
    else:
        # Simple truncation, sorted by nnz
        candidates.sort(key=lambda m: m.nnz)
        candidates = candidates[:limit]

    # ── convert to DownloadableGraph ──────────────────────────────────
    result: List[DownloadableGraph] = []
    for m in candidates:
        url = SUITESPARSE_MM_URL.format(group=m.group, name=m.name)
        dg = DownloadableGraph(
            name=m.name,
            url=url,
            size_mb=_estimate_size_mb(m.nnz),
            nodes=m.rows,
            edges=m.nnz,
            symmetric=_is_symmetric(m.psym),
            category=_kind_to_category(m.kind),
            description=f"{m.kind} (group={m.group})",
        )
        result.append(dg)

    result.sort(key=lambda g: (g.category, g.name))
    log.info(
        f"SuiteSparse auto-discovery: {len(result)} graphs for size {size_category} "
        f"({lo:,}–{hi:,} nnz) from {len(set(m.group for m in candidates))} groups"
    )
    return result


def discover_all_sizes(
    limit_per_size: int = 100,
    *,
    exclude_names: Optional[set] = None,
) -> Dict[str, List["DownloadableGraph"]]:
    """Discover graphs for all four size categories at once.

    Returns
    -------
    dict mapping size name → list of DownloadableGraph
    """
    return {
        sc: discover_graphs(sc, limit=limit_per_size, exclude_names=exclude_names)
        for sc in SIZE_BOUNDS
    }


def size_for_graph(nnz: int) -> Optional[str]:
    """Return the size category name for a graph with the given nnz, or None."""
    for sc, (lo, hi) in SIZE_BOUNDS.items():
        if lo <= nnz < hi:
            return sc
    return None


# ── backward-compatible aliases ───────────────────────────────────────────
TIER_BOUNDS = SIZE_BOUNDS  # deprecated alias
discover_all_tiers = discover_all_sizes  # deprecated alias
tier_for_graph = size_for_graph  # deprecated alias
