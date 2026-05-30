# Literature-faithfulness tolerance-calibration audit

For every literature claim whose comparator-asserted bound is actually exercised, this report computes the **slack** — how many pp the observed `|delta_pct|` could move in the wrong direction before the verdict flips from `ok` → `disagree`. Small slack means a future regen could easily break the gate; large slack means the bound is over-permissive.

## Summary

- Total per_claim rows: **330**
- Audited rows (real assertion fired): **202**
- Median slack: **4.9436 pp**
- p10 slack: **1.211 pp** · p90 slack: **11.1196 pp**
- Min / max slack: **0.1024** / **61.5226** pp
- Fragile rows (slack < 1.0 pp): **16** (7.9%)
- Very comfortable rows (slack ≥ 5.0 pp): **101**
- Negative-slack rows (audit bug if non-zero): **0**

## Audit-status breakdown

| audit_status | count |
|---|---:|
| `audited` | 202 |
| `deviation` | 30 |
| `not_triggered` | 98 |

## Slack histogram (audited rows)

| bin (pp) | count |
|---|---:|
| [-∞, 0.0) | 0 |
| [0.0, 0.5) | 5 |
| [0.5, 1.0) | 11 |
| [1.0, 2.0) | 39 |
| [2.0, 5.0) | 46 |
| [5.0, 10.0) | 74 |
| [10.0, 20.0) | 23 |
| [20.0, +∞) | 4 |

## Per-policy slack distribution

| policy | n | min | p10 | median | p90 | max | fragile |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRASP | 19 | 1.6629 | 3.348 | 5.9519 | 12.7432 | 15.7657 | 0 |
| POPT | 8 | 1.466 | 2.9517 | 9.7203 | 15.0611 | 19.4131 | 0 |
| POPT_GE_GRASP | 88 | 0.1024 | 0.7683 | 1.6126 | 7.7961 | 61.5226 | 15 |
| POPT_NEAR_GRASP_IF_BIG_GAP | 12 | 0.8129 | 1.5429 | 7.719 | 10.8454 | 15.2458 | 1 |
| SRRIP | 75 | 1.3021 | 3.3998 | 6.7966 | 10.7893 | 12.9997 | 0 |

## Per-app slack distribution

| app | n | min | median | fragile |
|---|---:|---:|---:|---:|
| bc | 30 | 1.2154 | 5.5334 | 0 |
| bfs | 43 | 0.2382 | 3.9806 | 3 |
| cc | 28 | 0.1024 | 5.9736 | 3 |
| pr | 64 | 0.1857 | 4.5077 | 7 |
| sssp | 37 | 0.6276 | 5.1961 | 3 |

## Top-15 most fragile rows

| graph | app | L3 | policy | sign | tol | slack pp | status |
|---|---|---|---|---|---:|---:|---|
| soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | - | 1.5 | 0.1024 | ok |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | - | 1.0 | 0.1857 | ok |
| web-Google | cc | 1MB | POPT_GE_GRASP | - | 1.5 | 0.2325 | ok |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | - | 1.5 | 0.2382 | ok |
| web-Google | bfs | 1MB | POPT_GE_GRASP | - | 1.5 | 0.4018 | ok |
| soc-pokec | pr | 1MB | POPT_GE_GRASP | - | 1.0 | 0.5786 | ok |
| soc-pokec | sssp | 4MB | POPT_GE_GRASP | - | 1.5 | 0.6276 | ok |
| cit-Patents | sssp | 1MB | POPT_GE_GRASP | - | 1.5 | 0.6366 | ok |
| delaunay_n19 | pr | 16kB | POPT_GE_GRASP | - | 1.0 | 0.6681 | ok |
| cit-Patents | bfs | 8MB | POPT_GE_GRASP | - | 1.5 | 0.8113 | ok |
| com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 2.0 | 0.8129 | ok |
| delaunay_n19 | pr | 4kB | POPT_GE_GRASP | - | 1.0 | 0.9211 | ok |
| soc-LiveJournal1 | sssp | 8MB | POPT_GE_GRASP | - | 1.5 | 0.9225 | ok |
| roadNet-CA | pr | 16kB | POPT_GE_GRASP | - | 1.0 | 0.9686 | ok |
| email-Eu-core | pr | 4MB | POPT_GE_GRASP | - | 1.0 | 0.975 | ok |
