# Paper Table 4 — ECG_DBG + ECG_PFX vs literature baselines at L3=1MB

Cache simulator with `ECG_CONTAINER_BITS=64` and runtime
`ECG_PREFETCH_LOOKAHEAD=8` (`ECG_PREFETCH_MODE=2`, popt-ranked).
DROPLET-combined column uses the same lookahead window with
sequential target selection (`ECG_PREFETCH_MODE=3`) — faithful
comparator to the literature DROPLET edge-stream stride prefetcher.

## Headline summary

- Cells with full data: **3** of 3
- Mean Δ ECG_combined vs LRU: **-13.19 pp**
- Mean Δ ECG_combined vs GRASP: **+0.32 pp**
- Mean Δ ECG_combined vs POPT: **-2.94 pp**
- Mean Δ ECG_PFX vs DROPLET (same baseline): **+0.03 pp**
- Mean prefetch useful-rate: **99.95%**
- Total prefetch requests issued: ECG_PFX **713,687,416** vs DROPLET **1,275,610,925** (DROPLET issues 1.79× more)
- Total useful prefetches: ECG_PFX **215,636,214** vs DROPLET **377,665,322** (DROPLET useful is 1.75× ECG_PFX's, both 99.99% useful_rate)
- Requests per useful prefetch: ECG_PFX **3.310** vs DROPLET **3.378** (ECG_PFX is more efficient — fewer wasted predictions per cache-hit benefit)

## Per-cell miss-rates

| graph | app | LRU | GRASP | POPT | ECG_DBG | ECG+PFX | ECG+DROP | Δ LRU | Δ GRASP | Δ POPT | Δ DROPLET |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kron-s22 | bfs | 0.9973 | 0.9809 | 0.9881 | — | 0.9828 | 0.9828 | -1.45 pp | +0.19 pp | -0.53 pp | +0.00 pp |
| kron-s22 | pr | 0.5468 | 0.3780 | 0.4212 | — | 0.3835 | 0.3829 | -16.33 pp | +0.56 pp | -3.77 pp | +0.07 pp |
| kron-s24 | pr | 0.6702 | 0.4502 | 0.4975 | — | 0.4524 | 0.4522 | -21.78 pp | +0.22 pp | -4.51 pp | +0.02 pp |

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
