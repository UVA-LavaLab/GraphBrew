# Literature-faithfulness tolerance-calibration audit

For every literature claim whose comparator-asserted bound is actually exercised, this report computes the **slack** — how many pp the observed `|delta_pct|` could move in the wrong direction before the verdict flips from `ok` → `disagree`. Small slack means a future regen could easily break the gate; large slack means the bound is over-permissive.

## Summary

- Total per_claim rows: **330**
- Audited rows (real assertion fired): **204**
- Median slack: **4.8606 pp**
- p10 slack: **0.7798 pp** · p90 slack: **11.1577 pp**
- Min / max slack: **0.0181** / **61.5226** pp
- Fragile rows (slack < 1.0 pp): **25** (12.2%)
- Very comfortable rows (slack ≥ 5.0 pp): **100**
- Negative-slack rows (audit bug if non-zero): **0**

## Audit-status breakdown

| audit_status | count |
|---|---:|
| `audited` | 204 |
| `deviation` | 24 |
| `missing_data` | 7 |
| `not_triggered` | 95 |

## Slack histogram (audited rows)

| bin (pp) | count |
|---|---:|
| [-∞, 0.0) | 0 |
| [0.0, 0.5) | 14 |
| [0.5, 1.0) | 11 |
| [1.0, 2.0) | 37 |
| [2.0, 5.0) | 42 |
| [5.0, 10.0) | 71 |
| [10.0, 20.0) | 25 |
| [20.0, +∞) | 4 |

## Per-policy slack distribution

| policy | n | min | p10 | median | p90 | max | fragile |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRASP | 18 | 0.5799 | 3.3776 | 6.8035 | 12.9422 | 15.7704 | 1 |
| POPT | 8 | 3.7818 | 4.4653 | 10.1217 | 15.1156 | 19.4003 | 0 |
| POPT_GE_GRASP | 91 | 0.0181 | 0.2595 | 1.5003 | 7.7817 | 61.5226 | 23 |
| POPT_NEAR_GRASP_IF_BIG_GAP | 12 | 0.8129 | 1.5717 | 7.6447 | 10.8437 | 15.6517 | 1 |
| SRRIP | 75 | 1.3321 | 3.3876 | 6.6314 | 10.7893 | 12.9997 | 0 |

## Per-app slack distribution

| app | n | min | median | fragile |
|---|---:|---:|---:|---:|
| bc | 34 | 0.0181 | 5.2771 | 5 |
| bfs | 43 | 0.2382 | 4.4815 | 7 |
| cc | 28 | 0.1024 | 5.9736 | 3 |
| pr | 60 | 0.1813 | 4.742 | 5 |
| sssp | 39 | 0.0632 | 4.8538 | 5 |

## Top-15 most fragile rows

| graph | app | L3 | policy | sign | tol | slack pp | status |
|---|---|---|---|---|---:|---:|---|
| soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | - | 1.5 | 0.0181 | ok |
| soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | - | 1.5 | 0.0551 | ok |
| cit-Patents | sssp | 8MB | POPT_GE_GRASP | - | 1.5 | 0.0632 | ok |
| cit-Patents | sssp | 4MB | POPT_GE_GRASP | - | 1.5 | 0.0889 | ok |
| soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | - | 1.5 | 0.1024 | ok |
| cit-Patents | bc | 8MB | POPT_GE_GRASP | - | 1.5 | 0.1582 | ok |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | - | 1.0 | 0.1813 | ok |
| web-Google | cc | 1MB | POPT_GE_GRASP | - | 1.5 | 0.2325 | ok |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | - | 1.5 | 0.2382 | ok |
| soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | - | 1.5 | 0.2595 | ok |
| soc-LiveJournal1 | sssp | 4MB | POPT_GE_GRASP | - | 1.5 | 0.2801 | ok |
| soc-LiveJournal1 | bfs | 8MB | POPT_GE_GRASP | - | 1.5 | 0.2997 | ok |
| soc-LiveJournal1 | bfs | 1MB | POPT_GE_GRASP | - | 1.5 | 0.4101 | ok |
| web-Google | bfs | 1MB | POPT_GE_GRASP | - | 1.5 | 0.4538 | ok |
| cit-Patents | bc | 1MB | GRASP | - | 3.0 | 0.5799 | within_tolerance |
