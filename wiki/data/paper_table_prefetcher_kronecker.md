# Paper Table 4 — ECG_DBG + ECG_PFX vs literature baselines at L3=1MB

Cache simulator with `ECG_CONTAINER_BITS=64` and runtime
`ECG_PREFETCH_LOOKAHEAD=8` (`ECG_PREFETCH_MODE=2`, popt-ranked).
DROPLET-combined column uses the same lookahead window with
sequential target selection (`ECG_PREFETCH_MODE=3`) — best-case
oracle comparator to literature DROPLET (Basak HPCA'19); the
real DROPLET stride detector would add mis-prediction overhead.

## How to read this table

Two metrics are reported. **The prefetcher-aware metric is the
`demand-memory` rate**, not L3 miss-rate:

- **`l3_miss_rate`** = `l3.misses / l3.accesses`. cache_sim's
  l3.misses counter is incremented on every L3 lookup that misses,
  including prefetch-triggered lookups (`prefetch()` calls
  `l3->access()` which increments `misses_++`; see
  bench/include/cache_sim/cache_sim.h:465 + 1480). When a
  prefetcher is active, the prefetcher itself triggers L3 misses
  (the fetch from memory IS an L3 miss), so L3 miss-rate barely
  moves even when the prefetcher eliminates demand misses 1-for-1.
- **`demand-memory` rate** = `memory_accesses / total_accesses`.
  `memory_accesses_++` only fires on the demand path (cache_sim.h:1450)
  and `prefetch()` explicitly does NOT increment it (cache_sim.h:1463
  comment). This is demand misses to memory per demand access —
  the metric the DROPLET paper's claims map onto.

## Headline summary — demand-memory metric (prefetcher-aware)

- Cells with full data: **3** of 3
- Mean Δ ECG_combined demand-memory vs LRU: **-8.75 pp**
- **Marginal ECG_PFX gain on top of ECG_DBG eviction: `-4.15` pp**  ← the honest prefetcher value
- **Marginal DROPLET gain on top of ECG_DBG eviction: `-7.67` pp**
- Active prefetcher cells (≥1k requests issued): ECG_PFX **3**, DROPLET **3** of 3
- Active-cell mean marginal: ECG_PFX **-4.15** pp, DROPLET **-7.67** pp
- Prefetcher efficiency (pp demand-memory reduction per million requests, active cells):
  - ECG_PFX: **0.0175** pp/Mreq
  - DROPLET: **0.0180** pp/Mreq

## L3 miss-rate (pre-prefetch-aware metric; eviction story only)

- Mean Δ ECG_combined L3 miss vs LRU: **-13.19 pp** ← eviction component dominates
- Mean Δ ECG_combined L3 miss vs GRASP: **+0.32 pp**
- Mean Δ ECG_combined L3 miss vs POPT: **-2.94 pp**
- Mean Δ ECG_PFX L3 miss vs DROPLET (same baseline): **+0.03 pp** ← misleading: see demand-memory metric above
- Mean prefetch useful-rate: **99.95%**
- Total prefetch requests issued: ECG_PFX **713,687,416** vs DROPLET **1,275,610,925** (DROPLET issues 1.79× more)

## Per-cell demand-memory rate (prefetcher-aware)

| graph | app | LRU | ECG_DBG | ECG+PFX | ECG+DROP | Δ DBG vs LRU | Marg. PFX | Marg. DROP |
|---|---|---|---|---|---|---|---|---|
| kron-s22 | bfs | 0.1103 | 0.1086 | 0.1082 | 0.1078 | -0.17 pp | -0.04 pp | -0.08 pp |
| kron-s22 | pr | 0.1572 | 0.1105 | 0.0754 | 0.0330 | -4.67 pp | -3.52 pp | -7.76 pp |
| kron-s24 | pr | 0.2747 | 0.1851 | 0.0961 | 0.0334 | -8.97 pp | -8.90 pp | -15.16 pp |

## Per-cell L3 miss-rates (legacy — kept for cross-reference)

| graph | app | LRU | GRASP | POPT | ECG_DBG | ECG+PFX | ECG+DROP | Δ LRU | Δ GRASP | Δ POPT | Δ DROPLET |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kron-s22 | bfs | 0.9973 | 0.9809 | 0.9881 | 0.9828 | 0.9828 | 0.9828 | -1.45 pp | +0.19 pp | -0.53 pp | +0.00 pp |
| kron-s22 | pr | 0.5468 | 0.3780 | 0.4212 | 0.3816 | 0.3835 | 0.3829 | -16.33 pp | +0.56 pp | -3.77 pp | +0.07 pp |
| kron-s24 | pr | 0.6702 | 0.4502 | 0.4975 | 0.4507 | 0.4524 | 0.4522 | -21.78 pp | +0.22 pp | -4.51 pp | +0.02 pp |

## Prefetcher efficiency (ECG_PFX vs DROPLET on same baseline)

`req/useful` = total prefetch requests issued per useful prefetch.
Lower is better (fewer wasted predictions per cache-hit benefit).
`ratio` = ECG_PFX(req/useful) / DROPLET(req/useful). < 1.0 means ECG_PFX
is more efficient than DROPLET.

| graph | app | ECG_PFX requests | DROPLET requests | ECG_PFX req/useful | DROPLET req/useful | ratio |
|---|---|---:|---:|---:|---:|---:|
| kron-s22 | bfs | 17,075 | 31,656 | 2.307 | 2.079 | 1.110 |
| kron-s22 | pr | 110,612,168 | 251,823,879 | 5.773 | 5.946 | 0.971 |
| kron-s24 | pr | 603,058,173 | 1,023,755,390 | 3.069 | 3.053 | 1.005 |

## Prefetcher activity (ECG_PFX)

| graph | app | requests | fills | useful | useful_rate |
|---|---|---:|---:|---:|---:|
| kron-s22 | bfs | 17,075 | 7,402 | 7,402 | 100.00% |
| kron-s22 | pr | 110,612,168 | 19,189,498 | 19,160,714 | 99.85% |
| kron-s24 | pr | 603,058,173 | 196,480,428 | 196,468,098 | 99.99% |
