# Regression budget — distance to disagree

Per-cell distance (pp) the observed Δ must drift in the
adverse direction before the lit-faith status flips to
`disagree`. Larger margins = more robust headline.

## Headline

- Cells in distribution: **299** (of 330 total claim rows)
- Minimum margin: **0.018 pp**
- p10 margin: 1.228 pp
- Median margin: 6.116 pp
- p90 margin: 11.870 pp
- Max margin: 67.023 pp

### By claim kind

| kind | n cells | min margin (pp) | median margin (pp) |
|---|---:|---:|---:|
| cache_policy | 101 | 0.580 | 6.715 |
| popt_ge_grasp | 91 | 0.018 | 1.500 |
| popt_near_grasp_active | 12 | 0.813 | 8.102 |
| popt_near_grasp_inactive | 95 | 2.273 | 6.998 |

Cache-policy claims (LRU/SRRIP/GRASP/POPT individually) are
the **primary** load-bearing checks. POPT_GE_GRASP enforces
the P-OPT-vs-GRASP ordering and POPT_NEAR_GRASP_IF_BIG_GAP
only fires when GRASP outperforms LRU by >10 pp; the latter
is reported in this table as a *what-if* margin should the
gate fire later.

## 10 most fragile cache-policy cells

| graph | app | l3 | policy | status | Δ (pp) | margin (pp) | citation |
|---|---|---|---|---|---:|---:|---|
| cit-Patents | bc | 1MB | GRASP | within_tolerance | +1.420 | 0.580 | Faldu et al. HPCA 2020 Fig 11 |
| web-Google | pr | 1MB | SRRIP | ok | -5.668 | 1.335 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| cit-Patents | pr | 8MB | SRRIP | ok | -5.045 | 1.960 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| com-orkut | cc | 8MB | SRRIP | ok | -10.943 | 2.060 | Jaleel et al. ISCA 2010 §5.2 (scan-resistance argument extended to CC) |
| soc-pokec | pr | 1MB | SRRIP | ok | -4.529 | 2.475 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| cit-Patents | pr | 4MB | SRRIP | ok | -4.277 | 2.725 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| web-Google | pr | 8MB | GRASP | ok | -1.256 | 2.745 | Faldu et al. HPCA 2020 Fig 10 |
| com-orkut | pr | 1MB | SRRIP | ok | -4.225 | 2.780 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| com-orkut | pr | 4MB | SRRIP | ok | -3.681 | 3.320 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| soc-LiveJournal1 | pr | 4MB | SRRIP | ok | -3.632 | 3.370 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |

## 10 most fragile cells (any kind)

| graph | app | l3 | policy | kind | status | Δ (pp) | margin (pp) |
|---|---|---|---|---|---|---:|---:|
| soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.482 | 0.018 |
| soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.445 | 0.055 |
| cit-Patents | sssp | 8MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.437 | 0.063 |
| cit-Patents | sssp | 4MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.411 | 0.089 |
| soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.398 | 0.102 |
| cit-Patents | bc | 8MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.342 | 0.158 |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.819 | 0.181 |
| web-Google | cc | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.268 | 0.233 |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.262 | 0.238 |
| soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.240 | 0.260 |
