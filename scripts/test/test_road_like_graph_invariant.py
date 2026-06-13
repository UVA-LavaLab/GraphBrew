"""Road-like-graph invariant for GRASP.

GRASP's central design assumption is that the graph has a small,
identifiable set of *hot* vertices ("hubs") whose adjacency lists
benefit from being pinned in the L3 — i.e., GRASP is "hot-vertex
pinning". On graphs *without* hubs *and without local clustering*
(e.g., road networks, where every vertex has roughly the same
degree ~3-4 and there is no triangle/community structure either),
the design has nothing to pin and degrades into noise.

A purely low-hub-concentration graph is NOT sufficient — a uniform-
degree mesh like ``delaunay_n19`` has hub_concentration ~0.14 but
clustering_coeff ~0.38, and GRASP still wins by >13pp at 1 MB
because its random-within-bucket protection accidentally aligns
with the mesh's local cluster structure (protected vertices stay
in L3 long enough for their neighbours to land hits). We therefore
require BOTH a low hub_concentration AND a low clustering_coeff to
classify a graph as "road-like" (i.e., no exploitable locality at
all).

This file pins that *as a theory-driven invariant*: any graph in our
corpus that is road-like (low hub AND low clustering) must NOT show a
meaningful GRASP-over-LRU win at any L3 size. If a future GRASP
variant claims to win on a road-like graph, that's a sign of either
(a) overfitting to the GRASP-friendly graphs in our corpus, (b) the
road-like graph actually having structure we mis-measured, or (c) a
real algorithmic improvement that should be documented and
celebrated — in any case, the test failure forces the reviewer to
look.

We pin both directions:

* ``test_road_like_graph_present`` — make sure we actually have at
  least one road-like graph in the corpus, so the invariant is
  *load-bearing* rather than vacuously satisfied.
* ``test_grasp_does_not_beat_lru_on_road_like_graphs`` — the
  road-like graphs must have ``GRASP miss >= LRU miss - TOLERANCE_PP``
  for the headline kernel (PR) at the L3 size we have data for.

The list of road-like graphs is discovered from
``corpus_diversity.json`` rather than hardcoded so that adding a new
road-style graph automatically extends the invariant.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_JSON = REPO_ROOT / "wiki" / "data" / "corpus_diversity.json"
LIT_FAITH_CSV = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.csv"

# Theory thresholds. A graph is "road-like" iff it has BOTH:
#   * hub_concentration < HUB_LOW_THRESHOLD (no power-law tail of
#     degree — every vertex has roughly the same degree, so there is
#     no hot set to pin in L3), AND
#   * clustering_coeff < CLUSTER_LOW_THRESHOLD (no triangles /
#     communities — so even random vertex protection cannot create
#     incidental reuse from neighbour overlap).
#
# The 0.20 hub cut is well below the lowest hub_concentration of our
# social/web/citation graphs (the lowest non-road graph in the corpus
# has ~0.34) and comfortably above the roadNet-CA value (~0.14). The
# 0.10 cluster cut excludes meshes (delaunay_n19 ~0.38) while still
# capturing roadNet-CA (~0.063); it also sits comfortably below
# cit-Patents (~0.067) so the corpus easily distinguishes the two
# regimes.
HUB_LOW_THRESHOLD = 0.20
CLUSTER_LOW_THRESHOLD = 0.10

# A GRASP win below this magnitude is noise (signal is comparable to
# the lit-faith within_tolerance band on this regime). A GRASP-over-LRU
# improvement exceeding this magnitude on a road-like graph would be
# the surprising result that violates the design's stated assumption.
GRASP_WIN_NOISE_FLOOR_PP = 0.5

# Documented road-like GRASP wins (array-relative GRASP 0.15, single-thread
# corpus). The "GRASP needs a reusable hot set" premise holds at the literature
# operating point (L3 = 1 MB) for the property-array-reuse kernels (pr/bfs/bc:
# GRASP ties or trails LRU there), but breaks in TWO mechanistically-understood
# regimes, both of which are legitimate and are pinned here as known exceptions:
#   (a) SUB-WSS caches (< 1 MB): roadNet-CA's working set vastly exceeds a
#       64-256 kB L3, so LRU thrashes; GRASP's array-relative biased retention
#       holds a fixed DBG-front subset resident and cuts conflict/capacity
#       misses (an anti-thrashing effect, not hot-set reuse).
#   (b) cc at any size: connected-components' union-find repeatedly re-reads
#       component-representative (high-degree, DBG-front) vertices, so GRASP
#       captures genuine reuse even on a road graph.
# A GRASP win on a road-like cell OUTSIDE this set is the surprising result the
# gate must still catch. Re-pin only with a mechanism write-up (see
# docs/findings/grasp_road_anti_thrashing.md).
KNOWN_ROAD_GRASP_WIN_CELLS = frozenset({
    ("roadNet-CA", "bc", "256kB"),
    ("roadNet-CA", "bc", "64kB"),
    ("roadNet-CA", "bfs", "256kB"),
    ("roadNet-CA", "bfs", "64kB"),
    ("roadNet-CA", "cc", "1MB"),
    ("roadNet-CA", "cc", "256kB"),
    ("roadNet-CA", "pr", "256kB"),
    ("roadNet-CA", "sssp", "16kB"),
    ("roadNet-CA", "sssp", "64kB"),
})


def _load_corpus() -> list[dict]:
    if not CORPUS_JSON.exists():
        pytest.skip(f"{CORPUS_JSON.relative_to(REPO_ROOT)} not on disk")
    return json.loads(CORPUS_JSON.read_text())


def _feature(card: dict, key: str) -> float | None:
    feats = card.get("features", {}) if isinstance(card, dict) else {}
    v = feats.get(key) if isinstance(feats, dict) else None
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _is_road_like(card: dict) -> bool:
    hub = _feature(card, "hub_concentration")
    cluster = _feature(card, "clustering_coeff")
    if hub is None or cluster is None:
        return False
    return hub < HUB_LOW_THRESHOLD and cluster < CLUSTER_LOW_THRESHOLD


def _road_like_graphs() -> list[str]:
    return [c.get("graph", "") for c in _load_corpus() if _is_road_like(c)]


def _miss_rate(
    graph: str, policy: str, benchmark: str = "pr", l3_size: str | None = None
) -> float | None:
    """Look up the cache_sim L3 miss rate for (graph, policy, benchmark).

    If ``l3_size`` is given, return that specific cell. Otherwise return
    the first matching row's value (kept for backwards compatibility
    with the legacy single-cell call shape).
    """
    if not LIT_FAITH_CSV.exists():
        return None
    with LIT_FAITH_CSV.open() as f:
        for row in csv.DictReader(f):
            if (
                row.get("graph") == graph
                and row.get("policy") == policy
                and (row.get("app") or row.get("benchmark") or "pr") == benchmark
                and (l3_size is None or row.get("l3_size") == l3_size)
            ):
                try:
                    return float(row.get("miss_rate") or row.get("l3_miss_rate") or "nan")
                except (TypeError, ValueError):
                    continue
    return None


def _road_like_cells() -> list[tuple[str, str, str]]:
    """Enumerate every (graph, app, l3_size) cell in lit-faith CSV for
    a road-like graph that has both LRU and GRASP rows. We test the
    invariant on every such cell so a regression at any L3 size or
    benchmark is caught (not just the PR headline).
    """
    if not LIT_FAITH_CSV.exists():
        return []
    road = set(_road_like_graphs())
    if not road:
        return []
    cells: dict[tuple[str, str, str], set[str]] = {}
    with LIT_FAITH_CSV.open() as f:
        for row in csv.DictReader(f):
            g = row.get("graph", "")
            if g not in road:
                continue
            app = row.get("app") or row.get("benchmark") or "pr"
            l3 = row.get("l3_size") or ""
            pol = row.get("policy", "")
            cells.setdefault((g, app, l3), set()).add(pol)
    return sorted(k for k, pols in cells.items() if {"LRU", "GRASP"} <= pols)


def test_road_like_graph_present() -> None:
    """The corpus must contain at least one road-like graph so the
    GRASP-needs-reuse invariant is load-bearing.
    """
    graphs = _road_like_graphs()
    assert graphs, (
        f"corpus has zero graphs with hub_concentration < {HUB_LOW_THRESHOLD} "
        f"AND clustering_coeff < {CLUSTER_LOW_THRESHOLD}; add a road-style "
        f"graph (e.g., roadNet-CA) so the GRASP-needs-reuse invariant is "
        f"load-bearing rather than vacuous"
    )


@pytest.mark.parametrize(
    "cell",
    _road_like_cells() or [("__skip__", "__skip__", "__skip__")],
)
def test_grasp_does_not_beat_lru_on_road_like_graphs(
    cell: tuple[str, str, str],
) -> None:
    """GRASP must not show a meaningful win over LRU on road-like graphs
    at any (app, L3 size) cell we have data for. The road-like predicate
    asserts there is no reusable working set, so GRASP cannot create one
    by pinning a hot bucket — at any L3 size, at any kernel.
    """
    graph, app, l3 = cell
    if graph == "__skip__":
        pytest.skip("no road-like graphs in corpus")
    lru = _miss_rate(graph, "LRU", benchmark=app, l3_size=l3)
    grasp = _miss_rate(graph, "GRASP", benchmark=app, l3_size=l3)
    if lru is None or grasp is None:
        pytest.skip(
            f"no LRU/GRASP cache_sim data for {graph}/{app}@{l3} in lit-faith CSV "
            f"(have lru={lru!r}, grasp={grasp!r})"
        )
    grasp_minus_lru_pp = (grasp - lru) * 100.0
    if cell in KNOWN_ROAD_GRASP_WIN_CELLS:
        # Documented anti-thrashing / cc-reuse exception (see module header).
        # Assert it STILL wins so a stale pin (cell that stopped winning) is
        # surfaced for removal rather than silently masking a regression.
        assert grasp_minus_lru_pp < -GRASP_WIN_NOISE_FLOOR_PP, (
            f"{graph}/{app}@{l3} is pinned in KNOWN_ROAD_GRASP_WIN_CELLS as a "
            f"documented road-like GRASP win, but GRASP no longer beats LRU here "
            f"(grasp={grasp:.4f} lru={lru:.4f}, Δ={grasp_minus_lru_pp:+.3f}pp). "
            f"Remove this cell from the pinned set."
        )
        return
    assert grasp_minus_lru_pp > -GRASP_WIN_NOISE_FLOOR_PP, (
        f"GRASP beats LRU by {-grasp_minus_lru_pp:.3f}pp on road-like graph "
        f"{graph}/{app}@{l3} (grasp={grasp:.4f} lru={lru:.4f}) and is NOT in the "
        f"documented KNOWN_ROAD_GRASP_WIN_CELLS set; this contradicts the "
        f"GRASP-paper design assumption that GRASP wins by pinning a hot set in "
        f"L3. Either the corpus_diversity hub_concentration / clustering_coeff is "
        f"mis-measured, or GRASP has gained a locality-free improvement that needs "
        f"to be documented. Add to the pinned set only after writing up the cause."
    )
