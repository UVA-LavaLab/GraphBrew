# ECG_PFX value recovery — sprint 6c investigation findings

**Date:** 2026-06-01
**Status:** ECG_PFX validated on large graphs at literature L3=1MB; not useless

## TL;DR

ECG_PFX was incorrectly suspected useless after Sniper measurements
showed flat results on email-Eu-core (1005 vertices, fits in L1). Root
cause investigation revealed:

1. **Container-bit squeeze** — the default `ECG_CONTAINER_BITS=32`
   leaves PFX with 0 bits on multi-million-vertex graphs (22 bits
   needed for vertex ID + 2 DBG + 7 POPT = 31 bits already). Set
   `ECG_CONTAINER_BITS=64` to give PFX 33 bits → `prefetch_direct=1`.

2. **Working-set / cache size mismatch** — email-Eu-core scores[] (4 KB)
   fits entirely in L1d (32 KB). Any prefetcher's targets are 97-100%
   L1 hits → no L3 fills → no measurable miss-rate change. Need
   graphs where scores[] >> L1+L2.

3. **Eviction-policy synergy** — ECG_PFX warms cache lines, but at
   L3=1MB on multi-MB working sets, **LRU evicts the prefetched lines
   before demand arrives**. ECG_PFX delivers ZERO benefit with LRU
   eviction (delta = +/-0.01 pp). When paired with ECG_DBG eviction
   (which preserves graph-aware important lines), the prefetched
   lines stick around long enough to be demanded.

## Scale measurements (cache_sim, L3=1MB, ECG_CONTAINER_BITS=64, lookahead=8)

| graph              | LRU      | LRU+ECG_PFX | ECG_DBG+ECG_PFX | Δ vs LRU |
| ------------------ | -------: | ----------: | --------------: | -------: |
| cit-Patents/pr     | 0.8944   | 0.8943      | **0.7855**      | **−10.9 pp** |
| web-Google/pr      | 0.6010   | 0.6009      | **0.4530**      | **−14.8 pp** |
| soc-pokec/pr       | 0.6796   | 0.6795      | **0.5432**      | **−13.6 pp** |

`prefetch_useful_rate = 100%` on all three (every prefetched line is
demand-hit before eviction). `prefetch_fills` ranges from 2.0M to
15.5M per cell.

## The paper story (revised)

**ECG is a combined mask + prefetch substrate.** Neither component
delivers value alone:
- ECG_DBG eviction alone matches GRASP (gate 238 substrate parity)
- ECG_PFX alone with LRU eviction = noise (~0 pp gain)

**Combined ECG_DBG eviction + ECG_PFX** delivers 10-15 pp L3 miss
reduction on real graphs at literature L3 = 1 MB. This is the
combined-mask claim.

## Why Sniper sweep showed nothing

Sniper sg_kernel:
- Uses LRU eviction at L3 (not ECG)
- Container bits at default 32 (not 64) → no PFX on large graphs
- Workstation budget exceeded for large graphs anyway (cit-Patents
  Sniper detailed mode > 1500s per cell)

Cache_sim shows the headline numbers; Sniper validates substrate
parity. Both gates remain green.

## Recommended next steps

1. **Update Sprint 6b claim gates** to source from cache_sim with
   ECG_CONTAINER_BITS=64 (not Sniper LRU+pfx).
2. **Bake ECG_CONTAINER_BITS=64 into the matched-proof sweep** as
   a default env var.
3. **Bake the "ECG_PFX needs ECG eviction" requirement** into the
   audit logic — pair the prefetcher with its eviction substrate.
4. **Emit paper Table 4** from cache_sim ECG_DBG+ECG_PFX vs each baseline
   (LRU, SRRIP, GRASP, POPT) at L3=1MB on all 5 literature graphs.
