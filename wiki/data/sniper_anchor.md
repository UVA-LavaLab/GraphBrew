# Sniper literature anchor

Source sweep: `/tmp/graphbrew-grasp-sniper-sweep/<graph>-<app>/DBG`

## Invariants

| invariant | status | detail |
|---|:---:|---|
| `GRASP_LE_LRU_headline:email-Eu-core/pr@256kB` | ✅ | grasp=0.1209 lru=0.1176 Δ=+0.328pp (tolerance ≤ +0.45pp) |
| `GRASP_LE_LRU_headline:email-Eu-core/sssp@256kB` | ✅ | grasp=0.0803 lru=0.0806 Δ=-0.029pp (tolerance ≤ +0.45pp) |
| `GRASP_LE_LRU_headline:cit-Patents/pr@256kB` | ✅ | grasp=0.1176 lru=0.1208 Δ=-0.319pp (tolerance ≤ +0.45pp) |
| `GRASP_LE_LRU_headline:cit-Patents/sssp@256kB` | ✅ | grasp=0.0805 lru=0.0805 Δ=-0.004pp (tolerance ≤ +0.45pp) |
| `asymptote_within_1.0pp:email-Eu-core/pr@2MB` | ✅ | spread=0.446pp across ['GRASP', 'LRU', 'SRRIP'] (tolerance ≤ 1.00pp) |
| `asymptote_within_1.0pp:email-Eu-core/sssp@2MB` | ✅ | spread=0.145pp across ['GRASP', 'LRU', 'SRRIP'] (tolerance ≤ 1.00pp) |
| `asymptote_within_1.0pp:cit-Patents/pr@2MB` | ✅ | spread=0.123pp across ['GRASP', 'LRU', 'SRRIP'] (tolerance ≤ 1.00pp) |
| `asymptote_within_1.0pp:cit-Patents/sssp@2MB` | ✅ | spread=0.159pp across ['GRASP', 'LRU', 'SRRIP'] (tolerance ≤ 1.00pp) |
| `small_cache_divergence:email-Eu-core/pr@4kB` | ✅ | spread=3.759pp across ['GRASP', 'LRU', 'SRRIP'] (min ≥ 2.00pp; L-shape holds) |
| `small_cache_divergence:email-Eu-core/sssp@4kB` | ✅ | spread=3.341pp across ['GRASP', 'LRU', 'SRRIP'] (min ≥ 2.00pp; L-shape holds) |
| `small_cache_divergence:cit-Patents/pr@4kB` | ✅ | spread=11.386pp across ['GRASP', 'LRU', 'SRRIP'] (min ≥ 2.00pp; L-shape holds) |
| `small_cache_divergence:cit-Patents/sssp@4kB` | ✅ | spread=6.357pp across ['GRASP', 'LRU', 'SRRIP'] (min ≥ 2.00pp; L-shape holds) |
| `no_error_rows` | ✅ | 72 ok rows across 24 cells |

## Per-cell summary

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | ok | err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cit-Patents | bfs | 4kB | 0.9608 | 0.9537 | 0.9428 | — | 3 | 0 |
| cit-Patents | bfs | 32kB | 0.4663 | 0.4604 | 0.4657 | — | 3 | 0 |
| cit-Patents | bfs | 256kB | 0.4644 | 0.4534 | 0.4678 | — | 3 | 0 |
| cit-Patents | bfs | 2MB | 0.4682 | 0.4653 | 0.4654 | — | 3 | 0 |
| cit-Patents | pr | 4kB | 0.8014 | 0.9153 | 0.8865 | — | 3 | 0 |
| cit-Patents | pr | 32kB | 0.1196 | 0.1144 | 0.1198 | — | 3 | 0 |
| cit-Patents | pr | 256kB | 0.1208 | 0.1184 | 0.1176 | — | 3 | 0 |
| cit-Patents | pr | 2MB | 0.1210 | 0.1209 | 0.1197 | — | 3 | 0 |
| cit-Patents | sssp | 4kB | 0.9278 | 0.8911 | 0.8642 | — | 3 | 0 |
| cit-Patents | sssp | 32kB | 0.0816 | 0.0809 | 0.0826 | — | 3 | 0 |
| cit-Patents | sssp | 256kB | 0.0805 | 0.0804 | 0.0805 | — | 3 | 0 |
| cit-Patents | sssp | 2MB | 0.0814 | 0.0798 | 0.0809 | — | 3 | 0 |
| email-Eu-core | bfs | 4kB | 0.9683 | 0.9577 | 0.9505 | — | 3 | 0 |
| email-Eu-core | bfs | 32kB | 0.4609 | 0.4631 | 0.4758 | — | 3 | 0 |
| email-Eu-core | bfs | 256kB | 0.4672 | 0.4639 | 0.4624 | — | 3 | 0 |
| email-Eu-core | bfs | 2MB | 0.4592 | 0.4662 | 0.4639 | — | 3 | 0 |
| email-Eu-core | pr | 4kB | 0.9169 | 0.9150 | 0.8793 | — | 3 | 0 |
| email-Eu-core | pr | 32kB | 0.1200 | 0.1197 | 0.1239 | — | 3 | 0 |
| email-Eu-core | pr | 256kB | 0.1176 | 0.1219 | 0.1209 | — | 3 | 0 |
| email-Eu-core | pr | 2MB | 0.1174 | 0.1144 | 0.1189 | — | 3 | 0 |
| email-Eu-core | sssp | 4kB | 0.9264 | 0.8972 | 0.8930 | — | 3 | 0 |
| email-Eu-core | sssp | 32kB | 0.0801 | 0.0782 | 0.0808 | — | 3 | 0 |
| email-Eu-core | sssp | 256kB | 0.0806 | 0.0786 | 0.0803 | — | 3 | 0 |
| email-Eu-core | sssp | 2MB | 0.0812 | 0.0798 | 0.0802 | — | 3 | 0 |
