# Finding — Sniper delaunay_n19/pr 4-arm DRAM comparison

**Date:** 2026-06-06 (sprint 6f-6 follow-up)
**Discovered during:** User push-back on "why is Sniper slow?" → diagnosed chatty-kernel anti-pattern → ran full 4-arm cycle-accurate comparison

## The 4-arm cycle-accurate measurement

After fixing the chatty-kernel anti-pattern (bug 5A: kernel-side LRU
dedup of mode-6 hints, commit `de3bcdd6`), all 4 arms of
delaunay_n19/pr at -i 1 iterations completed cycle-accurately on
Sniper. Cache hierarchy: L1d=32KB/8w, L2=256KB/8w, L3=1MB/16w,
DDR4-2400. Property array = 2 MB (exceeds all cache levels).

| Arm | DRAM requests | l3_miss_rate | pf_useful | DRAM ratio | Wall |
|---|---:|---:|---:|---:|---:|
| **none** (baseline) | 594,187 | 0.9621 | 0 | 1.00× | 33min |
| **DROPLET** L2 | **593,708** | 0.9609 | **223,165** | **1.00× neutral** | 45min |
| **ECG_PFX mode 6** CHARGED=1 | 4,843,870 | 0.9311 | 26,563 | **8.15× WORSE** | 86min |
| **ECG_PFX mode 6** CHARGED=0 | 4,840,057 | 0.9315 | 26,615 | **8.15× WORSE** | 90min |

## Key findings

### 1. DROPLET wins absolutely on cycle-accurate Sniper

DROPLET issues **223,165 useful L2 prefetches** with 99.9% accuracy
while keeping DRAM demand requests at baseline level (593,708 ≈
594,187). The prefetches land in L2 and prevent L2→L3 escalations
without adding to DRAM traffic. ECG_PFX mode 6 issues only 26,563
useful L2 prefetches (8.4× fewer) AND adds 8.15× more DRAM traffic.

### 2. ECG_PFX mode 6's mask reads dominate DRAM traffic

Mode 6's per-edge mask array is 8 bytes × 3.15M edges = 25 MB,
~25× larger than the 1 MB L3. Reading it sequentially per PR
iteration generates 25 MB of L3→DRAM traffic that ECG_PFX has to
pay for the prefetch hints it issues. Net DRAM cost per useful
prefetch:

  DROPLET:   (593K - baseline) / 223K = ~0 DRAM/useful (free)
  ECG_PFX:   (4.84M - 594K) / 26K   ≈ 162 DRAM/useful (very expensive)

### 3. The `ECG_EDGE_MASK_CHARGED` env var has NO EFFECT on Sniper

In cache_sim, `ECG_EDGE_MASK_CHARGED=0` skips the `SIM_CACHE_READ`
calls on the mask array, hiding the mask cost from the demand-memory
metric. In Sniper, however, every memory access is cycle-accurately
counted regardless of the env var — there is no way to "uncharge"
the mask reads. Confirmed empirically: CHARGED=1 and CHARGED=0
produce nearly identical DRAM (4.84M vs 4.84M) and pf_useful
(26,563 vs 26,615).

This means: the cache_sim mode 6 corpus result (Table 7) is in part
an artifact of how cache_sim accounts for mask-read traffic. On
real hardware (and on cycle-accurate Sniper), the mask cost is
unavoidable.

### 4. cache_sim "+14% per-Mreq efficiency vs mode 2" is REGIME-SPECIFIC

The cache_sim corpus result was framed as bandwidth-efficient
Pareto-frontier (pp/Mreq) where mode 6 beats mode 2 by +14% on
demand-memory reduction per prefetch request. Sniper data show that
under cycle-accurate accounting where the mask reads are
unavoidable, mode 6's per-request efficiency story doesn't survive
— DROPLET wins on every metric except the cache_sim-specific
denominator-normalized one.

## Implications for the paper

The paper currently claims:
- "Mode 6 is the most bandwidth-efficient graph-aware prefetcher we measured" (§5.3)
- "+14.2% pp/Mreq vs mode 2, +34.9% vs DROPLET"

These are TRUE for cache_sim. They are NOT TRUE for cycle-accurate
Sniper on the one cell we've measured (delaunay_n19/pr).

Honest re-framing options:
1. Acknowledge the cache_sim-specific accounting and report Sniper
   numbers separately
2. Re-evaluate mode 6 on cache_sim using a demand-memory metric
   that includes mask reads (i.e., CHARGED=1 with the full DRAM
   accounting, not pp/Mreq)
3. Reposition mode 6 from "more bandwidth-efficient" to a
   different design point (e.g., "lower hint-stream overhead than
   DROPLET" — fewer prefetches issued, easier to gate)

The cache_sim corpus result is not WRONG, but the cycle-accurate
Sniper measurement on one cell suggests the bandwidth-efficiency
framing depends on how mask reads are accounted. Sections 5.3, 5.4,
and 6.1 need to be re-examined with this evidence.

## Confidence notes

- Sample size: 1 cell (delaunay_n19/pr) on 1 simulator (Sniper).
  Cache_sim has the 4-cell corpus result. Honest cross-sim story
  requires N≥2 Sniper cells but the budget barely accommodates one.
- This finding does NOT invalidate the substrate-level
  cache_sim corpus claim (mode 6 vs mode 2 trade-off). It shows
  that the trade-off looks DIFFERENT under cycle-accurate
  cost accounting.
- Bug 5A (chatty-kernel dedup) is real and shipped: without it,
  the Sniper run cannot complete in any reasonable budget.

## Provenance

- Cell directory: `/tmp/graphbrew-pfx-sniper-delaunay-i1-{none,droplet,charged0}/`
- Mode 6 CHARGED=1 cell: `/tmp/graphbrew-pfx-sniper-delaunay-i1/`
- All 4 cells used same Sniper config (LRU L3=1MB, 256-entry kernel dedup window)
- sg_kernel binary: post-commit `de3bcdd6` (with kernel-side dedup)
