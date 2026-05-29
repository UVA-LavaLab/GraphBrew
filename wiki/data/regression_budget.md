# Regression budget — distance to disagree

Per-cell distance (pp) the observed Δ must drift in the
adverse direction before the lit-faith status flips to
`disagree`. Larger margins = more robust headline.

## Headline

- Cells in distribution: **221** (of 270 total claim rows)
- Minimum margin: **0.102 pp**
- p10 margin: 1.496 pp
- Median margin: 6.137 pp
- p90 margin: 10.696 pp
- Max margin: 19.415 pp

### By claim kind

| kind | n cells | min margin (pp) | median margin (pp) |
|---|---:|---:|---:|
| cache_policy | 101 | 1.305 | 6.875 |
| popt_ge_grasp | 49 | 0.102 | 1.783 |
| popt_near_grasp_active | 11 | 0.813 | 7.386 |
| popt_near_grasp_inactive | 60 | 2.273 | 6.715 |

Cache-policy claims (LRU/SRRIP/GRASP/POPT individually) are
the **primary** load-bearing checks. POPT_GE_GRASP enforces
the P-OPT-vs-GRASP ordering and POPT_NEAR_GRASP_IF_BIG_GAP
only fires when GRASP outperforms LRU by >10 pp; the latter
is reported in this table as a *what-if* margin should the
gate fire later.

## 10 most fragile cache-policy cells

| graph | app | l3 | policy | status | Δ (pp) | margin (pp) | citation |
|---|---|---|---|---|---:|---:|---|
| web-Google | pr | 1MB | SRRIP | ok | -5.698 | 1.305 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| soc-LiveJournal1 | sssp | 1MB | POPT | within_tolerance | -1.466 | 1.470 | Balaji & Lucia HPCA 2021 Fig 10 |
| cit-Patents | bc | 1MB | GRASP | within_tolerance | +0.337 | 1.665 | Faldu et al. HPCA 2020 Fig 11 |
| com-orkut | cc | 8MB | SRRIP | ok | -10.943 | 2.060 | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| cit-Patents | pr | 8MB | SRRIP | ok | -4.915 | 2.085 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| soc-pokec | pr | 1MB | SRRIP | ok | -4.517 | 2.485 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| web-Google | pr | 8MB | GRASP | ok | -1.253 | 2.750 | Faldu et al. HPCA 2020 Fig 10 |
| cit-Patents | pr | 4MB | SRRIP | ok | -4.181 | 2.820 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| com-orkut | pr | 1MB | SRRIP | ok | -4.177 | 2.825 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| com-orkut | pr | 4MB | SRRIP | ok | -3.651 | 3.350 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |

## 10 most fragile cells (any kind)

| graph | app | l3 | policy | kind | status | Δ (pp) | margin (pp) |
|---|---|---|---|---|---|---:|---:|
| soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.398 | 0.102 |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.814 | 0.186 |
| web-Google | cc | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.268 | 0.233 |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.262 | 0.238 |
| web-Google | bfs | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.098 | 0.402 |
| soc-pokec | pr | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.421 | 0.579 |
| soc-pokec | sssp | 4MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.872 | 0.628 |
| cit-Patents | sssp | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.863 | 0.637 |
| cit-Patents | bfs | 8MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.689 | 0.811 |
| com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | popt_near_grasp_active | ok | +6.187 | 0.813 |

