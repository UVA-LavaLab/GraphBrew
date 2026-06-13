# Literature-faithfulness tolerance-calibration audit

For every literature claim whose comparator-asserted bound is actually exercised, this report computes the **slack** — how many pp the observed `|delta_pct|` could move in the wrong direction before the verdict flips from `ok` → `disagree`. Small slack means a future regen could easily break the gate; large slack means the bound is over-permissive.

## Summary

- Total per_claim rows: **279**
- Audited rows (real assertion fired): **147**
- Median slack: **5.9644 pp**
- p10 slack: **1.8919 pp** · p90 slack: **12.1646 pp**
- Min / max slack: **0.0953** / **32.5153** pp
- Fragile rows (slack < 1.0 pp): **9** (6.1%)
- Very comfortable rows (slack ≥ 5.0 pp): **91**
- Negative-slack rows (audit bug if non-zero): **0**

## Audit-status breakdown

| audit_status | count |
|---|---:|
| `audited` | 147 |
| `deviation` | 37 |
| `missing_data` | 28 |
| `not_triggered` | 67 |

## Slack histogram (audited rows)

| bin (pp) | count |
|---|---:|
| [-∞, 0.0) | 0 |
| [0.0, 0.5) | 7 |
| [0.5, 1.0) | 2 |
| [1.0, 2.0) | 8 |
| [2.0, 5.0) | 39 |
| [5.0, 10.0) | 67 |
| [10.0, 20.0) | 22 |
| [20.0, +∞) | 2 |

## Per-policy slack distribution

| policy | n | min | p10 | median | p90 | max | fragile |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRASP | 18 | 3.0308 | 3.4699 | 7.0512 | 12.9123 | 15.7453 | 0 |
| POPT | 8 | 2.7539 | 3.8278 | 8.497 | 14.0372 | 17.8502 | 0 |
| POPT_GE_GRASP | 40 | 0.0953 | 0.3628 | 2.9851 | 13.31 | 32.5153 | 9 |
| POPT_NEAR_GRASP_IF_BIG_GAP | 6 | 2.7242 | 3.0882 | 6.9285 | 11.18 | 13.255 | 0 |
| SRRIP | 75 | 1.251 | 3.2112 | 6.5257 | 10.4746 | 13.0 | 0 |

## Per-app slack distribution

| app | n | min | median | fragile |
|---|---:|---:|---:|---:|
| bc | 22 | 0.0953 | 5.6714 | 3 |
| bfs | 28 | 0.4941 | 7.4119 | 2 |
| cc | 26 | 0.2523 | 9.8811 | 2 |
| pr | 48 | 0.4291 | 4.6665 | 1 |
| sssp | 23 | 0.2519 | 5.8392 | 1 |

## Top-15 most fragile rows

| graph | app | L3 | policy | sign | tol | slack pp | status |
|---|---|---|---|---|---:|---:|---|
| web-Google | bc | 1MB | POPT_GE_GRASP | - | 1.5 | 0.0953 | ok |
| com-orkut | sssp | 8MB | POPT_GE_GRASP | - | 1.5 | 0.2519 | ok |
| soc-pokec | cc | 1MB | POPT_GE_GRASP | - | 1.5 | 0.2523 | ok |
| cit-Patents | cc | 8MB | POPT_GE_GRASP | - | 1.5 | 0.3401 | ok |
| soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | - | 1.5 | 0.3653 | ok |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | - | 1.0 | 0.4291 | ok |
| soc-LiveJournal1 | bfs | 8MB | POPT_GE_GRASP | - | 1.5 | 0.4941 | ok |
| com-orkut | bfs | 4MB | POPT_GE_GRASP | - | 1.5 | 0.5112 | ok |
| cit-Patents | bc | 1MB | POPT_GE_GRASP | - | 1.5 | 0.7644 | ok |
| soc-pokec | pr | 4MB | POPT_GE_GRASP | - | 1.0 | 1.1135 | ok |
| web-Google | pr | 1MB | SRRIP | ~ | 2.0 | 1.251 | ok |
| cit-Patents | bfs | 1MB | POPT_GE_GRASP | - | 1.5 | 1.2848 | ok |
| web-Google | pr | 4MB | POPT_GE_GRASP | - | 1.0 | 1.3298 | ok |
| com-orkut | bfs | 1MB | POPT_GE_GRASP | - | 1.5 | 1.4731 | ok |
| cit-Patents | pr | 8MB | SRRIP | ~ | 2.0 | 1.8311 | ok |
