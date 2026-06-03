# Paper Table 4 — ECG_DBG + ECG_PFX vs literature baselines at L3=1MB

Cache simulator with `ECG_CONTAINER_BITS=64` and runtime
`ECG_PREFETCH_LOOKAHEAD=8` (`ECG_PREFETCH_MODE=2`, popt-ranked).
DROPLET-combined column uses the same lookahead window with
sequential target selection (`ECG_PREFETCH_MODE=3`) — faithful
comparator to the literature DROPLET edge-stream stride prefetcher.

## Headline summary

- Cells with full data: **14** of 16
- Mean Δ ECG_combined vs LRU: **-5.84 pp**
- Mean Δ ECG_combined vs GRASP: **+0.01 pp**
- Mean Δ ECG_combined vs POPT: **+0.24 pp**
- Mean Δ ECG_PFX vs DROPLET (same baseline): **+0.00 pp**
- Mean prefetch useful-rate: **99.99%**
- Total prefetch requests issued: ECG_PFX **145,084,233** vs DROPLET **475,145,125** (DROPLET issues 3.27× more)
- Total useful prefetches: ECG_PFX **75,634,816** vs DROPLET **211,157,873** (DROPLET useful is 2.79× ECG_PFX's, both 99.99% useful_rate)
- Requests per useful prefetch: ECG_PFX **1.918** vs DROPLET **2.250** (ECG_PFX is more efficient — fewer wasted predictions per cache-hit benefit)

## Per-cell miss-rates

| graph | app | LRU | GRASP | POPT | ECG_DBG | ECG+PFX | ECG+DROP | Δ LRU | Δ GRASP | Δ POPT | Δ DROPLET |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cit-Patents | bc | 0.8843 | 0.8985 | 0.8823 | 0.8986 | 0.8984 | 0.8990 | +1.41 pp | -0.01 pp | +1.61 pp | -0.06 pp |
| cit-Patents | bfs | 0.9702 | 0.9590 | 0.9602 | 0.9596 | 0.9597 | 0.9595 | -1.05 pp | +0.07 pp | -0.05 pp | +0.02 pp |
| cit-Patents | pr | 0.8937 | 0.7857 | 0.7744 | 0.7857 | 0.7855 | 0.7855 | -10.82 pp | -0.01 pp | +1.11 pp | +0.01 pp |
| cit-Patents | sssp | 0.8483 | 0.8177 | 0.8216 | 0.8179 | 0.8177 | 0.8175 | -3.06 pp | +0.00 pp | -0.39 pp | +0.02 pp |
| com-orkut | pr | — | — | — | — | — | — | — | — | — | — |
| email-Eu-core | pr | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.00 pp | +0.00 pp | +0.00 pp | +0.00 pp |
| soc-LiveJournal1 | bc | — | — | — | — | — | — | — | — | — | — |
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
| com-orkut | pr | 0 | 0 | 0 | 0.00% |
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
