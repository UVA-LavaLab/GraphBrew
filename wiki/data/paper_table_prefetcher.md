# Paper Table 4 — ECG_DBG + ECG_PFX vs literature baselines at L3=1MB

Cache simulator with `ECG_CONTAINER_BITS=64` and runtime
`ECG_PREFETCH_LOOKAHEAD=8` (`ECG_PREFETCH_MODE=2`, popt-ranked).

## Headline summary

- Cells with full data: **14** of 16
- Mean Δ ECG_combined vs LRU: **-5.84 pp**
- Mean Δ ECG_combined vs GRASP: **+0.01 pp**
- Mean Δ ECG_combined vs POPT: **+0.24 pp**
- Mean prefetch useful-rate: **99.99%**

## Per-cell miss-rates

| graph | app | LRU | SRRIP | GRASP | POPT | ECG_DBG_ONLY | ECG+PFX | Δ vs LRU | Δ vs GRASP | Δ vs POPT |
|---|---|---|---|---|---|---|---|---|---|---|
| cit-Patents | bc | 0.8843 | 0.8799 | 0.8985 | 0.8823 | 0.8986 | 0.8984 | +1.41 pp | -0.01 pp | +1.61 pp |
| cit-Patents | bfs | 0.9702 | 0.9669 | 0.9590 | 0.9602 | 0.9596 | 0.9597 | -1.05 pp | +0.07 pp | -0.05 pp |
| cit-Patents | pr | 0.8937 | 0.8790 | 0.7857 | 0.7744 | 0.7857 | 0.7855 | -10.82 pp | -0.01 pp | +1.11 pp |
| cit-Patents | sssp | 0.8483 | 0.8444 | 0.8177 | 0.8216 | 0.8179 | 0.8177 | -3.06 pp | +0.00 pp | -0.39 pp |
| com-orkut | pr | — | — | — | — | — | — | — | — | — |
| email-Eu-core | pr | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | +0.00 pp | +0.00 pp | +0.00 pp |
| soc-LiveJournal1 | bc | — | — | — | — | — | — | — | — | — |
| soc-LiveJournal1 | bfs | 0.8182 | 0.7836 | 0.7402 | 0.7525 | 0.7397 | 0.7412 | -7.70 pp | +0.10 pp | -1.13 pp |
| soc-LiveJournal1 | pr | 0.7318 | 0.6986 | 0.6839 | 0.6225 | 0.6839 | 0.6839 | -4.79 pp | -0.00 pp | +6.14 pp |
| soc-LiveJournal1 | sssp | 0.7063 | 0.6814 | 0.6586 | 0.6582 | 0.6573 | 0.6580 | -4.83 pp | -0.06 pp | -0.02 pp |
| soc-pokec | bc | 0.8522 | 0.8288 | 0.7648 | 0.7945 | 0.7648 | 0.7648 | -8.74 pp | -0.00 pp | -2.97 pp |
| soc-pokec | pr | 0.6796 | 0.6343 | 0.5433 | 0.5467 | 0.5433 | 0.5432 | -13.63 pp | -0.01 pp | -0.34 pp |
| soc-pokec | sssp | 0.6366 | 0.6026 | 0.5277 | 0.5639 | 0.5282 | 0.5277 | -10.89 pp | -0.00 pp | -3.62 pp |
| web-Google | bc | 0.7044 | 0.6883 | 0.7071 | 0.7027 | 0.7092 | 0.7084 | +0.41 pp | +0.13 pp | +0.57 pp |
| web-Google | bfs | 0.9690 | 0.9697 | 0.9357 | 0.9469 | 0.9354 | 0.9353 | -3.37 pp | -0.03 pp | -1.16 pp |
| web-Google | pr | 0.6009 | 0.5440 | 0.4531 | 0.4168 | 0.4532 | 0.4531 | -14.78 pp | +0.00 pp | +3.63 pp |

## Prefetcher activity

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
