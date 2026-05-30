# ECG substrate-parity audit

Gate 238 — ECG-Parity. Locks the cache_sim component-proof matrix's
load-bearing invariants: ECG re-implementations match the policies
they shadow, the prefetch path is actually firing, and the dedup
bookkeeping is consistent. This is the confidence-floor that must
hold before any cluster-scale ECG sweep is launched.

## Rules
- **E1** — every required ablation present with status=ok per benchmark
- **E2** — |miss_rate(ECG_DBG_only) - miss_rate(GRASP_DBG_only)| <= 0.0005
- **E3** — |miss_rate(ECG_POPT_primary) - miss_rate(POPT_only)| <= 0.0005
- **E4** — every PFX ablation has ecg_runtime_issued >= 1 per benchmark
- **E5** — PFX ablation has prefetch_useful > 0 on PR AND prefetch_useful <= prefetch_requests on all benchmarks
- **E6** — ecg_pfx_encoded <= ecg_pfx_candidates for any row with candidates; all PFX counters are non-negative (dedup_skips is intentionally unbounded vs issued — dedup is a runtime per-access counter)
- **E7** — baselines (LRU/SRRIP/GRASP/POPT) have strictly positive memory_accesses and l3_misses
- **E8** — distinct backend count >= 1

## Constants
- ε(DBG parity): `0.0005`
- ε(POPT parity): `0.0005`
- PFX issued floor: `1`
- backend floor: `1`
- required ablations: `DBG_PFX, ECG_DBG_only, ECG_POPT_primary, GRASP_DBG_only, LRU_cache_only, PFX_POPT_only, PFX_degree_only, POPT_PFX, POPT_only, SRRIP_cache_only`
- PFX ablations: `DBG_PFX, PFX_POPT_only, PFX_degree_only, POPT_PFX`

## Totals
- observations: **54**
- benchmarks: `bfs, pr, sssp`
- backends: `cache_sim`
- ablations present: `DBG_PFX, DBG_POPT_PFX, ECG_ADAPTIVE_NO_FULL_POPT, ECG_ADAPTIVE_ORACLE, ECG_COMBINED, ECG_DBG_POPT, ECG_DBG_only, ECG_EMBEDDED, ECG_EPOCH_EMBEDDED, ECG_POPT_TIE, ECG_POPT_primary, GRASP_DBG_only, LRU_cache_only, PFX_POPT_only, PFX_degree_only, POPT_PFX, POPT_only, SRRIP_cache_only`

## DBG parity

| benchmark | GRASP_DBG_only | ECG_DBG_only | |Δ| |
| --- | ---: | ---: | ---: |
| bfs | 0.943862 | 0.943862 | 0.0 |
| pr | 0.38075800000000004 | 0.38075800000000004 | 0.0 |
| sssp | 0.008516999999999997 | 0.008516999999999997 | 0.0 |

## POPT parity

| benchmark | POPT_only | ECG_POPT_primary | |Δ| |
| --- | ---: | ---: | ---: |
| bfs | 0.887725 | 0.887725 | 0.0 |
| pr | 0.21436999999999995 | 0.214125 | 0.0002449999999999397 |
| sssp | 0.008516999999999997 | 0.008516999999999997 | 0.0 |

## PFX activation

| benchmark | ablation | issued | useful | requests |
| --- | --- | ---: | ---: | ---: |
| bfs | DBG_PFX | 24 | 18 | 24 |
| bfs | PFX_POPT_only | 24 | 17 | 24 |
| bfs | PFX_degree_only | 20 | 17 | 20 |
| bfs | POPT_PFX | 24 | 17 | 24 |
| pr | DBG_PFX | 36642 | 3257 | 36642 |
| pr | PFX_POPT_only | 36642 | 2107 | 36642 |
| pr | PFX_degree_only | 29676 | 2427 | 29676 |
| pr | POPT_PFX | 36642 | 625 | 36642 |
| sssp | DBG_PFX | 18005 | 42 | 18005 |
| sssp | PFX_POPT_only | 18005 | 42 | 18005 |
| sssp | PFX_degree_only | 14450 | 40 | 14450 |
| sssp | POPT_PFX | 18005 | 42 | 18005 |

## Violations

_None._
