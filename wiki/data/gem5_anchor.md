# gem5 literature anchor

Source sweep: `/tmp/graphbrew-grasp-gem5-sweep/<graph>-<app>/DBG`

## Invariants

| invariant | status | detail |
|---|:---:|---|
| `GRASP_LE_LRU_headline:email-Eu-core/pr@256kB` | ✅ | grasp=0.1106 lru=0.1320 Δ=-2.142pp (tolerance ≤ +0.50pp) |
| `GRASP_LE_LRU_headline:email-Eu-core/bc@256kB` | ✅ | grasp=0.0253 lru=0.0286 Δ=-0.328pp (tolerance ≤ +0.50pp) |
| `asymptote_within_1.0pp:email-Eu-core/pr@2MB` | ✅ | spread=0.042pp across ['GRASP', 'LRU', 'SRRIP'] (tolerance ≤ 1.00pp) |
| `asymptote_within_1.0pp:email-Eu-core/bc@2MB` | ✅ | spread=0.025pp across ['GRASP', 'LRU', 'SRRIP'] (tolerance ≤ 1.00pp) |
| `small_cache_divergence:email-Eu-core/pr@4kB` | ✅ | spread=30.807pp across ['GRASP', 'LRU', 'SRRIP'] (min ≥ 2.00pp; L-shape holds) |
| `small_cache_divergence:email-Eu-core/bc@4kB` | ✅ | spread=14.309pp across ['GRASP', 'LRU', 'SRRIP'] (min ≥ 2.00pp; L-shape holds) |
| `no_error_rows` | ✅ | 48 ok rows across 8 cells |

## Per-cell summary

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | ok | err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| email-Eu-core | bc | 4kB | 0.7115 | 0.8026 | 0.6595 | — | 6 | 0 |
| email-Eu-core | bc | 32kB | 0.1104 | 0.1094 | 0.1115 | — | 6 | 0 |
| email-Eu-core | bc | 256kB | 0.0286 | 0.0244 | 0.0253 | — | 6 | 0 |
| email-Eu-core | bc | 2MB | 0.0019 | 0.0021 | 0.0018 | — | 6 | 0 |
| email-Eu-core | pr | 4kB | 0.2916 | 0.5996 | 0.4997 | — | 6 | 0 |
| email-Eu-core | pr | 32kB | 0.1352 | 0.1516 | 0.1445 | — | 6 | 0 |
| email-Eu-core | pr | 256kB | 0.1320 | 0.0940 | 0.1106 | — | 6 | 0 |
| email-Eu-core | pr | 2MB | 0.0050 | 0.0054 | 0.0053 | — | 6 | 0 |
