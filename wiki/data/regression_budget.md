# Regression budget — distance to disagree

Per-cell distance (pp) the observed Δ must drift in the
adverse direction before the lit-faith status flips to
`disagree`. Larger margins = more robust headline.

## Headline

- Cells in distribution: **234** (of 279 total claim rows)
- Minimum margin: **0.001 pp**
- p10 margin: 1.412 pp
- Median margin: 6.386 pp
- p90 margin: 11.640 pp
- Max margin: 24.478 pp

### By claim kind

| kind | n cells | min margin (pp) | median margin (pp) |
|---|---:|---:|---:|
| cache_policy | 101 | 1.255 | 6.820 |
| popt_ge_grasp | 60 | 0.001 | 1.933 |
| popt_near_grasp_active | 12 | 0.852 | 7.297 |
| popt_near_grasp_inactive | 61 | 1.202 | 7.057 |

Cache-policy claims (LRU/SRRIP/GRASP/POPT individually) are
the **primary** load-bearing checks. POPT_GE_GRASP enforces
the P-OPT-vs-GRASP ordering and POPT_NEAR_GRASP_IF_BIG_GAP
only fires when GRASP outperforms LRU by >10 pp; the latter
is reported in this table as a *what-if* margin should the
gate fire later.

## 10 most fragile cache-policy cells

| graph | app | l3 | policy | status | Δ (pp) | margin (pp) | citation |
|---|---|---|---|---|---:|---:|---|
| web-Google | pr | 1MB | SRRIP | ok | -5.749 | 1.255 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| cit-Patents | pr | 8MB | SRRIP | ok | -5.169 | 1.835 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| soc-pokec | pr | 1MB | SRRIP | ok | -4.442 | 2.560 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| cit-Patents | pr | 4MB | SRRIP | ok | -4.341 | 2.660 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| com-orkut | pr | 4MB | SRRIP | ok | -4.205 | 2.795 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| soc-LiveJournal1 | pr | 4MB | SRRIP | ok | -4.151 | 2.850 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| soc-LiveJournal1 | pr | 8MB | SRRIP | ok | -4.048 | 2.955 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| web-Google | pr | 8MB | GRASP | ok | +0.969 | 3.035 | Faldu et al. HPCA 2020 Fig 10 |
| soc-pokec | pr | 4MB | SRRIP | ok | -3.913 | 3.090 | Jaleel et al. ISCA 2010 §5.2; Faldu et al. HPCA 2020 §6.1 |
| cit-Patents | bfs | 1MB | GRASP | ok | -0.762 | 3.265 | Faldu et al. HPCA 2020 Fig 11 |

## 10 most fragile cells (any kind)

| graph | app | l3 | policy | kind | status | Δ (pp) | margin (pp) |
|---|---|---|---|---|---|---:|---:|
| soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.499 | 0.001 |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.359 | 0.141 |
| cit-Patents | bc | 4MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.225 | 0.275 |
| web-Google | bfs | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +1.122 | 0.378 |
| cit-Patents | sssp | 4MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.948 | 0.552 |
| com-orkut | bc | 1MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.840 | 0.660 |
| com-orkut | bfs | 4MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.793 | 0.707 |
| cit-Patents | bfs | 4MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.779 | 0.721 |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | popt_ge_grasp | ok | +0.170 | 0.830 |
| soc-pokec | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | popt_near_grasp_active | ok | +6.148 | 0.852 |
