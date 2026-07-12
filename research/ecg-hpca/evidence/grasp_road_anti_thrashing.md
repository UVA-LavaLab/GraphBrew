# GRASP helps road-like graphs in two regimes (array-relative 0.15)

**Date:** 2026-06-12
**Context:** array-relative GRASP normalization (0.15 of the vertex array,
GRASP's founded `add_region(ptr, frac, n)` method) on the single-thread,
deterministic cache_sim corpus.

## Finding

The long-held invariant "GRASP cannot help road-like graphs because they have
no reusable hot set" (`test_road_like_graph_invariant.py`) is **only true at the
literature operating point (L3 = 1 MB) for the property-array-reuse kernels
(pr / bfs / bc)**. On `roadNet-CA` (hub_concentration ~0.14, clustering ~0.063)
GRASP beats LRU in 10 cells via two mechanistically-distinct effects:

| Regime | Cells | Mechanism |
| ------ | ----- | --------- |
| Sub-WSS cache (< 1 MB) | bc/bfs @ 64-256 kB, pr @ 256 kB, sssp @ 16-64 kB | roadNet's working set ≫ a 64-256 kB L3, so LRU thrashes. GRASP's array-relative biased retention keeps a fixed DBG-front subset resident and cuts conflict/capacity misses — an **anti-thrashing** effect, not hot-set reuse. |
| cc at any size | cc @ 64 kB / 256 kB / 1 MB | Connected-components' union-find repeatedly re-reads component-representative (high-degree, DBG-front) vertices, so GRASP captures **genuine reuse** even on a road graph. |

Magnitudes (GRASP − LRU): cc@1MB −12.63pp, sssp@64kB −10.31pp, bc@256kB −7.39pp,
bfs@256kB −6.32pp, bfs@64kB −5.78pp, cc@256kB −3.72pp, sssp@16kB −1.34pp,
cc@64kB −1.25pp, pr@256kB −0.63pp, bc@64kB −0.55pp.

At L3 = 1 MB (the literature operating point), the original premise still holds
for the non-cc kernels: GRASP **trails** LRU on roadNet by large margins
(bfs +62.6pp, sssp +55.8pp, bc +16.2pp, pr +2.6pp) — GRASP's degree-bias actively
hurts frontier workloads on road graphs. Only cc@1MB wins (−12.6pp). This
catastrophic frontier regression was present at the prior 0.50×LLC normalization
too (even larger: bfs +68.4pp, sssp +70.1pp), so it is inherent GRASP behavior,
not introduced by the array-relative change.

## Why this is not a regression

GRASP's array-relative protection is a *biased-retention* policy. On a
thrashing workload it behaves like a partial pin of a fixed subset, which
reduces misses regardless of whether that subset is the "true" hot set. This is
a real, if incidental, benefit and is consistent with SRRIP-family behavior. The
cc reuse is genuine working-set capture.

## Paper implication

The paper should NOT claim "GRASP helps only high-locality graphs." The honest
characterization is: GRASP's degree-biased retention helps wherever there is
repeated access to DBG-front vertices (cc) OR wherever the cache is small enough
that any biased retention beats LRU thrashing (< 1 MB), and hurts frontier
kernels at large caches on low-locality graphs. The 10 cells are pinned in
`KNOWN_ROAD_GRASP_WIN_CELLS`; new wins outside that set still fail the gate.
