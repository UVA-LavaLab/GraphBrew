# Sniper headline-1MB anchor (literature canonical L3)

Source sweep: `[PosixPath('/tmp/graphbrew-headline-1mb-sniper')]/<graph>-<app>/DBG`

## Invariants

| invariant | status | detail |
|---|:---:|---|
| `GRASP_LE_LRU_headline:email-Eu-core/pr@256kB` | ⚠️ | cell not in sweep |
| `GRASP_LE_LRU_headline:email-Eu-core/bfs@256kB` | ⚠️ | cell not in sweep |
| `GRASP_LE_LRU_headline:email-Eu-core/sssp@256kB` | ⚠️ | cell not in sweep |
| `GRASP_LE_LRU_headline:cit-Patents/pr@256kB` | ⚠️ | cell not in sweep |
| `GRASP_LE_LRU_headline:cit-Patents/bfs@256kB` | ⚠️ | cell not in sweep |
| `GRASP_LE_LRU_headline:cit-Patents/sssp@256kB` | ⚠️ | cell not in sweep |
| `asymptote_within_1.0pp:email-Eu-core/pr@2MB` | ⚠️ | cell not in sweep |
| `asymptote_within_1.0pp:email-Eu-core/bfs@2MB` | ⚠️ | cell not in sweep |
| `asymptote_within_1.0pp:email-Eu-core/sssp@2MB` | ⚠️ | cell not in sweep |
| `asymptote_within_1.0pp:cit-Patents/pr@2MB` | ⚠️ | cell not in sweep |
| `asymptote_within_1.0pp:cit-Patents/bfs@2MB` | ⚠️ | cell not in sweep |
| `asymptote_within_1.0pp:cit-Patents/sssp@2MB` | ⚠️ | cell not in sweep |
| `small_cache_divergence:email-Eu-core/pr@4kB` | ⚠️ | cell not in sweep |
| `small_cache_divergence:email-Eu-core/bfs@4kB` | ⚠️ | cell not in sweep |
| `small_cache_divergence:email-Eu-core/sssp@4kB` | ⚠️ | cell not in sweep |
| `small_cache_divergence:cit-Patents/pr@4kB` | ⚠️ | cell not in sweep |
| `small_cache_divergence:cit-Patents/bfs@4kB` | ⚠️ | cell not in sweep |
| `small_cache_divergence:cit-Patents/sssp@4kB` | ⚠️ | cell not in sweep |
| `no_error_rows` | ✅ | 15 ok rows across 3 cells |

## Per-cell summary

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | ok | err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| email-Eu-core | bfs | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 5 | 0 |
| email-Eu-core | pr | 1MB | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 5 | 0 |
| email-Eu-core | sssp | 1MB | 0.9669 | 0.9601 | 0.9630 | 0.9622 | 5 | 0 |
