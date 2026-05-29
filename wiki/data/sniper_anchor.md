# Sniper literature anchor

Source sweep: `/tmp/graphbrew-grasp-sniper-sweep/<graph>-<app>/DBG`

## Invariants

| invariant | status | detail |
|---|:---:|---|
| `GRASP_LE_LRU_headline:email-Eu-core/pr@256kB` | ✅ | grasp=0.1209 lru=0.1176 Δ=+0.328pp (tolerance ≤ +0.50pp) |
| `GRASP_LE_LRU_headline:cit-Patents/pr@256kB` | ✅ | grasp=0.1176 lru=0.1208 Δ=-0.319pp (tolerance ≤ +0.50pp) |
| `asymptote_within_1.0pp:email-Eu-core/pr@2MB` | ✅ | spread=0.446pp across ['GRASP', 'LRU', 'SRRIP'] (tolerance ≤ 1.00pp) |
| `asymptote_within_1.0pp:cit-Patents/pr@2MB` | ✅ | spread=0.123pp across ['GRASP', 'LRU', 'SRRIP'] (tolerance ≤ 1.00pp) |
| `small_cache_divergence:email-Eu-core/pr@4kB` | ✅ | spread=3.759pp across ['GRASP', 'LRU', 'SRRIP'] (min ≥ 2.00pp; L-shape holds) |
| `small_cache_divergence:cit-Patents/pr@4kB` | ✅ | spread=11.386pp across ['GRASP', 'LRU', 'SRRIP'] (min ≥ 2.00pp; L-shape holds) |
| `no_error_rows` | ✅ | 24 ok rows across 8 cells |

## Per-cell summary

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | ok | err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cit-Patents | pr | 4kB | 0.8014 | 0.9153 | 0.8865 | — | 3 | 0 |
| cit-Patents | pr | 32kB | 0.1196 | 0.1144 | 0.1198 | — | 3 | 0 |
| cit-Patents | pr | 256kB | 0.1208 | 0.1184 | 0.1176 | — | 3 | 0 |
| cit-Patents | pr | 2MB | 0.1210 | 0.1209 | 0.1197 | — | 3 | 0 |
| email-Eu-core | pr | 4kB | 0.9169 | 0.9150 | 0.8793 | — | 3 | 0 |
| email-Eu-core | pr | 32kB | 0.1200 | 0.1197 | 0.1239 | — | 3 | 0 |
| email-Eu-core | pr | 256kB | 0.1176 | 0.1219 | 0.1209 | — | 3 | 0 |
| email-Eu-core | pr | 2MB | 0.1174 | 0.1144 | 0.1189 | — | 3 | 0 |
