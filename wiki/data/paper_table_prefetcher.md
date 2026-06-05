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

- Cells with full data: **16** of 16
- Mean Δ ECG_combined demand-memory vs LRU: **-7.24 pp**
- **Marginal ECG_PFX gain on top of ECG_DBG eviction: `-4.72` pp**  ← the honest prefetcher value
- **Marginal DROPLET gain on top of ECG_DBG eviction: `-13.35` pp**
- Active prefetcher cells (≥1k requests issued): ECG_PFX **12**, DROPLET **12** of 16
- Active-cell mean marginal: ECG_PFX **-6.31** pp, DROPLET **-17.80** pp
- Prefetcher efficiency (pp demand-memory reduction per million requests, active cells):
  - ECG_PFX: **0.2780** pp/Mreq
  - DROPLET: **0.2278** pp/Mreq

## L3 miss-rate (pre-prefetch-aware metric; eviction story only)

- Mean Δ ECG_combined L3 miss vs LRU: **-5.52 pp** ← eviction component dominates
- Mean Δ ECG_combined L3 miss vs GRASP: **+0.01 pp**
- Mean Δ ECG_combined L3 miss vs POPT: **+0.55 pp**
- Mean Δ ECG_PFX L3 miss vs DROPLET (same baseline): **+0.00 pp** ← misleading: see demand-memory metric above
- Mean prefetch useful-rate: **99.99%**
- Total prefetch requests issued: ECG_PFX **272,555,455** vs DROPLET **937,407,051** (DROPLET issues 3.44× more)

## Per-cell demand-memory rate (prefetcher-aware)

| graph | app | LRU | ECG_DBG | ECG+PFX | ECG+DROP | Δ DBG vs LRU | Marg. PFX | Marg. DROP |
|---|---|---|---|---|---|---|---|---|
| cit-Patents | bc | 0.6922 | 0.7034 | 0.7032 | 0.7023 | +1.12 pp | -0.02 pp | -0.11 pp |
| cit-Patents | bfs | 0.1025 | 0.1013 | 0.0981 | 0.0922 | -0.12 pp | -0.31 pp | -0.91 pp |
| cit-Patents | pr | 0.3790 | 0.3330 | 0.2370 | 0.0566 | -4.60 pp | -9.60 pp | -27.64 pp |
| cit-Patents | sssp | 0.5755 | 0.5551 | 0.3923 | 0.0995 | -2.04 pp | -16.28 pp | -45.56 pp |
| com-orkut | pr | 0.2670 | 0.2593 | 0.1950 | 0.0325 | -0.77 pp | -6.44 pp | -22.68 pp |
| email-Eu-core | pr | 0.0156 | 0.0156 | 0.0156 | 0.0156 | +0.00 pp | +0.00 pp | +0.00 pp |
| soc-LiveJournal1 | bc | 0.5360 | 0.5081 | 0.5085 | 0.5089 | -2.79 pp | +0.04 pp | +0.08 pp |
| soc-LiveJournal1 | bfs | 0.1434 | 0.1299 | 0.1132 | 0.0787 | -1.35 pp | -1.66 pp | -5.12 pp |
| soc-LiveJournal1 | pr | 0.2045 | 0.1911 | 0.1364 | 0.0356 | -1.34 pp | -5.47 pp | -15.55 pp |
| soc-LiveJournal1 | sssp | 0.3612 | 0.3370 | 0.2279 | 0.0369 | -2.41 pp | -10.92 pp | -30.02 pp |
| soc-pokec | bc | 0.6974 | 0.6259 | 0.6259 | 0.6255 | -7.15 pp | -0.00 pp | -0.04 pp |
| soc-pokec | pr | 0.2716 | 0.2172 | 0.1498 | 0.0335 | -5.45 pp | -6.74 pp | -18.37 pp |
| soc-pokec | sssp | 0.4664 | 0.3856 | 0.2526 | 0.0264 | -8.09 pp | -13.29 pp | -35.91 pp |
| web-Google | bc | 0.4232 | 0.4245 | 0.4259 | 0.4249 | +0.12 pp | +0.15 pp | +0.05 pp |
| web-Google | bfs | 0.1021 | 0.0986 | 0.0969 | 0.0945 | -0.35 pp | -0.16 pp | -0.41 pp |
| web-Google | pr | 0.2023 | 0.1525 | 0.1037 | 0.0385 | -4.98 pp | -4.88 pp | -11.40 pp |

## Per-cell L3 miss-rates (legacy — kept for cross-reference)

| graph | app | LRU | GRASP | POPT | ECG_DBG | ECG+PFX | ECG+DROP | Δ LRU | Δ GRASP | Δ POPT | Δ DROPLET |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cit-Patents | bc | 0.8843 | 0.8985 | 0.8823 | 0.8986 | 0.8984 | 0.8990 | +1.41 pp | -0.01 pp | +1.61 pp | -0.06 pp |
| cit-Patents | bfs | 0.9702 | 0.9590 | 0.9602 | 0.9596 | 0.9597 | 0.9595 | -1.05 pp | +0.07 pp | -0.05 pp | +0.02 pp |
| cit-Patents | pr | 0.8937 | 0.7857 | 0.7744 | 0.7857 | 0.7855 | 0.7855 | -10.82 pp | -0.01 pp | +1.11 pp | +0.01 pp |
| cit-Patents | sssp | 0.8483 | 0.8177 | 0.8216 | 0.8179 | 0.8177 | 0.8175 | -3.06 pp | +0.00 pp | -0.39 pp | +0.02 pp |
| com-orkut | pr | 0.7050 | 0.6848 | 0.6125 | 0.6849 | 0.6848 | 0.6847 | -2.02 pp | -0.01 pp | +7.23 pp | +0.00 pp |
| email-Eu-core | pr | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| soc-LiveJournal1 | bc | 0.8390 | 0.7949 | 0.8129 | 0.7948 | 0.7948 | 0.7950 | -4.42 pp | -0.01 pp | -1.80 pp | -0.02 pp |
| soc-LiveJournal1 | bfs | 0.8182 | 0.7402 | 0.7525 | 0.7397 | 0.7412 | 0.7407 | -7.70 pp | +0.10 pp | -1.13 pp | +0.05 pp |
| soc-LiveJournal1 | pr | 0.7318 | 0.6839 | 0.6225 | 0.6839 | 0.6839 | 0.6839 | -4.79 pp | -0.00 pp | +6.14 pp | +0.01 pp |
| soc-LiveJournal1 | sssp | 0.7063 | 0.6586 | 0.6582 | 0.6573 | 0.6580 | 0.6582 | -4.83 pp | -0.06 pp | -0.02 pp | -0.02 pp |
| soc-pokec | bc | 0.8522 | 0.7648 | 0.7945 | 0.7648 | 0.7648 | 0.7645 | -8.74 pp | -0.00 pp | -2.97 pp | +0.03 pp |
| soc-pokec | pr | 0.6796 | 0.5433 | 0.5467 | 0.5433 | 0.5432 | 0.5431 | -13.63 pp | -0.01 pp | -0.34 pp | +0.02 pp |
| soc-pokec | sssp | 0.6366 | 0.5277 | 0.5639 | 0.5282 | 0.5277 | 0.5272 | -10.89 pp | -0.00 pp | -3.62 pp | +0.05 pp |
| web-Google | bc | 0.7044 | 0.7071 | 0.7027 | 0.7092 | 0.7084 | 0.7085 | +0.41 pp | +0.13 pp | +0.57 pp | -0.00 pp |
| web-Google | bfs | 0.9690 | 0.9357 | 0.9469 | 0.9354 | 0.9353 | 0.9360 | -3.37 pp | -0.03 pp | -1.16 pp | -0.07 pp |
| web-Google | pr | 0.6009 | 0.4531 | 0.4168 | 0.4532 | 0.4531 | 0.4529 | -14.78 pp | +0.00 pp | +3.63 pp | +0.02 pp |

## Prefetcher efficiency (ECG_PFX vs DROPLET on same baseline)

`req/useful` = total prefetch requests issued per useful prefetch.
Lower is better (fewer wasted predictions per cache-hit benefit).
`ratio` = ECG_PFX(req/useful) / DROPLET(req/useful). < 1.0 means ECG_PFX
is more efficient than DROPLET.

| graph | app | ECG_PFX requests | DROPLET requests | ECG_PFX req/useful | DROPLET req/useful | ratio |
|---|---|---:|---:|---:|---:|---:|
| cit-Patents | bfs | 224,884 | 807,800 | 1.297 | 1.623 | 0.799 |
| cit-Patents | pr | 17,605,247 | 58,526,023 | 1.131 | 1.303 | 0.868 |
| cit-Patents | sssp | 8,148,536 | 27,803,278 | 1.234 | 1.506 | 0.819 |
| com-orkut | pr | 127,471,222 | 462,261,926 | 2.057 | 2.117 | 0.972 |
| email-Eu-core | pr | 48,509 | 62,197 | — | — | — |
| soc-LiveJournal1 | bfs | 1,683,527 | 5,759,963 | 2.504 | 2.771 | 0.904 |
| soc-LiveJournal1 | pr | 51,349,413 | 159,066,424 | 2.460 | 2.680 | 0.918 |
| soc-LiveJournal1 | sssp | 25,431,799 | 78,908,163 | 2.445 | 2.756 | 0.887 |
| soc-pokec | pr | 24,009,855 | 85,822,033 | 1.861 | 2.440 | 0.763 |
| soc-pokec | sssp | 11,931,756 | 42,715,169 | 1.870 | 2.480 | 0.754 |
| web-Google | bfs | 43,000 | 143,546 | 2.140 | 2.897 | 0.739 |
| web-Google | pr | 4,607,707 | 15,530,529 | 2.253 | 3.252 | 0.693 |

## Prefetcher activity (ECG_PFX)

| graph | app | requests | fills | useful | useful_rate |
|---|---|---:|---:|---:|---:|
| cit-Patents | bc | 0 | 0 | 0 | 0.00% |
| cit-Patents | bfs | 224,884 | 173,356 | 173,355 | 100.00% |
| cit-Patents | pr | 17,605,247 | 15,564,263 | 15,559,302 | 99.97% |
| cit-Patents | sssp | 8,148,536 | 6,605,749 | 6,605,649 | 100.00% |
| com-orkut | pr | 127,471,222 | 61,977,867 | 61,973,677 | 99.99% |
| email-Eu-core | pr | 48,509 | 0 | 0 | 0.00% |
| soc-LiveJournal1 | bc | 0 | 0 | 0 | 0.00% |
| soc-LiveJournal1 | bfs | 1,683,527 | 672,287 | 672,281 | 100.00% |
| soc-LiveJournal1 | pr | 51,349,413 | 20,877,192 | 20,876,962 | 100.00% |
| soc-LiveJournal1 | sssp | 25,431,799 | 10,401,629 | 10,401,357 | 100.00% |
| soc-pokec | bc | 0 | 0 | 0 | 0.00% |
| soc-pokec | pr | 24,009,855 | 12,900,265 | 12,898,548 | 99.99% |
| soc-pokec | sssp | 11,931,756 | 6,381,922 | 6,381,722 | 100.00% |
| web-Google | bc | 0 | 0 | 0 | 0.00% |
| web-Google | bfs | 43,000 | 20,098 | 20,094 | 99.98% |
| web-Google | pr | 4,607,707 | 2,045,656 | 2,045,546 | 99.99% |
