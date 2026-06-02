# Paper Table 4 — ECG_DBG + ECG_PFX vs literature baselines at L3=1MB

Cache simulator with `ECG_CONTAINER_BITS=64` and runtime
`ECG_PREFETCH_LOOKAHEAD=8` (`ECG_PREFETCH_MODE=2`, popt-ranked).

## Headline summary

- Cells with full data: **2** of 3
- Mean Δ ECG_combined vs LRU: **-6.24 pp**
- Mean Δ ECG_combined vs GRASP: **+0.05 pp**
- Mean Δ ECG_combined vs POPT: **+2.51 pp**
- Mean prefetch useful-rate: **100.00%**

## Per-cell miss-rates

| graph | app | LRU | SRRIP | GRASP | POPT | ECG_DBG_ONLY | ECG+PFX | Δ vs LRU | Δ vs GRASP | Δ vs POPT |
|---|---|---|---|---|---|---|---|---|---|---|
| soc-LiveJournal1 | bfs | 0.8182 | 0.7836 | 0.7402 | 0.7525 | 0.7397 | 0.7412 | -7.70 pp | +0.10 pp | -1.13 pp |
| soc-LiveJournal1 | pr | 0.7318 | 0.6986 | 0.6839 | 0.6225 | 0.6839 | 0.6839 | -4.79 pp | -0.00 pp | +6.14 pp |
| soc-LiveJournal1 | sssp | — | — | — | — | — | — | — | — | — |

## Prefetcher activity

| graph | app | requests | fills | useful | useful_rate |
|---|---|---:|---:|---:|---:|
| soc-LiveJournal1 | bfs | 1,683,527 | 672,287 | 672,281 | 100.00% |
| soc-LiveJournal1 | pr | 51,349,413 | 20,877,192 | 20,876,962 | 100.00% |
| soc-LiveJournal1 | sssp | 0 | 0 | 0 | 0.00% |
