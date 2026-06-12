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

- Cells with full data: **0** of 0
- Active prefetcher cells (≥1k requests issued): ECG_PFX **0**, DROPLET **0** of 0

## L3 miss-rate (pre-prefetch-aware metric; eviction story only)

- Total prefetch requests issued: ECG_PFX **0** vs DROPLET **0** (DROPLET issues 0.00× more)

## Per-cell demand-memory rate (prefetcher-aware)

| graph | app | LRU | ECG_DBG | ECG+PFX | ECG+DROP | Δ DBG vs LRU | Marg. PFX | Marg. DROP |
|---|---|---|---|---|---|---|---|---|

## Per-cell L3 miss-rates (legacy — kept for cross-reference)

| graph | app | LRU | GRASP | POPT | ECG_DBG | ECG+PFX | ECG+DROP | Δ LRU | Δ GRASP | Δ POPT | Δ DROPLET |
|---|---|---|---|---|---|---|---|---|---|---|---|

## Prefetcher efficiency (ECG_PFX vs DROPLET on same baseline)

`req/useful` = total prefetch requests issued per useful prefetch.
Lower is better (fewer wasted predictions per cache-hit benefit).
`ratio` = ECG_PFX(req/useful) / DROPLET(req/useful). < 1.0 means ECG_PFX
is more efficient than DROPLET.

| graph | app | ECG_PFX requests | DROPLET requests | ECG_PFX req/useful | DROPLET req/useful | ratio |
|---|---|---:|---:|---:|---:|---:|

## Prefetcher activity (ECG_PFX)

| graph | app | requests | fills | useful | useful_rate |
|---|---|---:|---:|---:|---:|
