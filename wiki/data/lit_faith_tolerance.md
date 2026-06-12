# Literature-faithfulness tolerance-calibration audit

For every literature claim whose comparator-asserted bound is actually exercised, this report computes the **slack** — how many pp the observed `|delta_pct|` could move in the wrong direction before the verdict flips from `ok` → `disagree`. Small slack means a future regen could easily break the gate; large slack means the bound is over-permissive.

## Summary

- Total per_claim rows: **279**
- Audited rows (real assertion fired): **173**
- Median slack: **5.5996 pp**
- p10 slack: **1.2778 pp** · p90 slack: **11.576 pp**
- Min / max slack: **0.001** / **22.4332** pp
- Fragile rows (slack < 1.0 pp): **13** (7.5%)
- Very comfortable rows (slack ≥ 5.0 pp): **100**
- Negative-slack rows (audit bug if non-zero): **0**

## Audit-status breakdown

| audit_status | count |
|---|---:|
| `audited` | 173 |
| `deviation` | 17 |
| `missing_data` | 28 |
| `not_triggered` | 61 |

## Slack histogram (audited rows)

| bin (pp) | count |
|---|---:|
| [-∞, 0.0) | 0 |
| [0.0, 0.5) | 4 |
| [0.5, 1.0) | 9 |
| [1.0, 2.0) | 24 |
| [2.0, 5.0) | 36 |
| [5.0, 10.0) | 74 |
| [10.0, 20.0) | 25 |
| [20.0, +∞) | 1 |

## Per-policy slack distribution

| policy | n | min | p10 | median | p90 | max | fragile |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRASP | 18 | 3.0308 | 3.4699 | 7.0512 | 12.9123 | 15.7453 | 0 |
| POPT | 8 | 4.7145 | 6.5389 | 12.0716 | 18.5985 | 22.4332 | 0 |
| POPT_GE_GRASP | 60 | 0.001 | 0.7024 | 1.8729 | 7.914 | 18.9778 | 12 |
| POPT_NEAR_GRASP_IF_BIG_GAP | 12 | 0.8523 | 1.0705 | 6.1516 | 13.4791 | 17.5804 | 1 |
| SRRIP | 75 | 1.251 | 3.2112 | 6.5257 | 10.4746 | 13.0 | 0 |

## Per-app slack distribution

| app | n | min | median | fragile |
|---|---:|---:|---:|---:|
| bc | 29 | 0.2746 | 5.3479 | 3 |
| bfs | 33 | 0.001 | 6.593 | 5 |
| cc | 28 | 0.8523 | 7.0914 | 2 |
| pr | 50 | 0.8298 | 5.6538 | 1 |
| sssp | 33 | 0.5518 | 5.0216 | 2 |

## Top-15 most fragile rows

| graph | app | L3 | policy | sign | tol | slack pp | status |
|---|---|---|---|---|---:|---:|---|
| soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | - | 1.5 | 0.001 | ok |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | - | 1.5 | 0.1413 | ok |
| cit-Patents | bc | 4MB | POPT_GE_GRASP | - | 1.5 | 0.2746 | ok |
| web-Google | bfs | 1MB | POPT_GE_GRASP | - | 1.5 | 0.3779 | ok |
| cit-Patents | sssp | 4MB | POPT_GE_GRASP | - | 1.5 | 0.5518 | ok |
| com-orkut | bc | 1MB | POPT_GE_GRASP | - | 1.5 | 0.6601 | ok |
| com-orkut | bfs | 4MB | POPT_GE_GRASP | - | 1.5 | 0.7071 | ok |
| cit-Patents | bfs | 4MB | POPT_GE_GRASP | - | 1.5 | 0.7209 | ok |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | - | 1.0 | 0.8298 | ok |
| soc-pokec | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 2.0 | 0.8523 | ok |
| soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | - | 1.5 | 0.886 | ok |
| cit-Patents | sssp | 1MB | POPT_GE_GRASP | - | 1.5 | 0.8964 | ok |
| cit-Patents | cc | 4MB | POPT_GE_GRASP | - | 1.5 | 0.9772 | ok |
| com-orkut | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 2.0 | 1.057 | ok |
| soc-LiveJournal1 | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 2.0 | 1.1917 | ok |
