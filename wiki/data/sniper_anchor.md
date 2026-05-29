# Sniper literature anchor

Source sweep: `/tmp/graphbrew-grasp-sniper-sweep/<graph>-<app>/DBG`

## Invariants

| invariant | status | detail |
|---|:---:|---|
| `GRASP_LE_LRU_headline:pr@256kB` | ✅ | grasp=0.1209 lru=0.1176 Δ=+0.328pp (tolerance ≤ +0.50pp) |
| `asymptote_within_1.0pp:pr@2MB` | ✅ | spread=0.446pp across ['GRASP', 'LRU', 'SRRIP'] (tolerance ≤ 1.00pp) |
| `no_error_rows` | ✅ | 12 ok rows across 4 cells |

## Per-cell summary

| graph | app | L3 | LRU | SRRIP | GRASP | POPT | ok | err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| email-Eu-core | pr | 4kB | 0.9169 | 0.9150 | 0.8793 | — | 3 | 0 |
| email-Eu-core | pr | 32kB | 0.1200 | 0.1197 | 0.1239 | — | 3 | 0 |
| email-Eu-core | pr | 256kB | 0.1176 | 0.1219 | 0.1209 | — | 3 | 0 |
| email-Eu-core | pr | 2MB | 0.1174 | 0.1144 | 0.1189 | — | 3 | 0 |
