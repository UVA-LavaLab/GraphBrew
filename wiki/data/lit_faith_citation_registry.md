# lit-faith citation registry purity (gate 246)

**Status:** active  •  registered works: 3  •  per_claim rows: 279  •  violations: 0

## Rules
- **C1** — every per_claim citation matches >=1 registered canonical work
- **C2** — every registered canonical work is referenced >=1 time
- **C3** — within (policy, app, expected_sign) bucket, all rows share >=1 canonical key
- **C4** — every registry entry has non-empty venue + year
- **C5** — every per_claim row carries a non-empty citation

## Registered canonical works

| key | venue | year | per_claim references |
|---|---|---:|---:|
| `Faldu-HPCA-2020` | HPCA | 2020 | 156 |
| `Balaji-HPCA-2021` | HPCA | 2021 | 192 |
| `Jaleel-ISCA-2010` | ISCA | 2010 | 84 |

## Shared canonical keys per (policy, app, sign) bucket

| bucket | shared canonical keys |
|---|---|
| `GRASP|bc|-` | `Faldu-HPCA-2020` |
| `GRASP|bc|~` | `Faldu-HPCA-2020` |
| `GRASP|bfs|-` | `Faldu-HPCA-2020` |
| `GRASP|bfs|~` | `Faldu-HPCA-2020` |
| `GRASP|pr|-` | `Faldu-HPCA-2020` |
| `GRASP|pr|~` | `Faldu-HPCA-2020` |
| `GRASP|sssp|-` | `Balaji-HPCA-2021` |
| `POPT_GE_GRASP|bc|-` | `Balaji-HPCA-2021` |
| `POPT_GE_GRASP|bfs|-` | `Balaji-HPCA-2021` |
| `POPT_GE_GRASP|cc|-` | `Balaji-HPCA-2021` |
| `POPT_GE_GRASP|pr|-` | `Balaji-HPCA-2021` |
| `POPT_GE_GRASP|sssp|-` | `Balaji-HPCA-2021` |
| `POPT_NEAR_GRASP_IF_BIG_GAP|bc|~` | `Balaji-HPCA-2021`, `Faldu-HPCA-2020` |
| `POPT_NEAR_GRASP_IF_BIG_GAP|bfs|~` | `Balaji-HPCA-2021`, `Faldu-HPCA-2020` |
| `POPT_NEAR_GRASP_IF_BIG_GAP|cc|~` | `Balaji-HPCA-2021`, `Faldu-HPCA-2020` |
| `POPT_NEAR_GRASP_IF_BIG_GAP|pr|~` | `Balaji-HPCA-2021`, `Faldu-HPCA-2020` |
| `POPT_NEAR_GRASP_IF_BIG_GAP|sssp|~` | `Balaji-HPCA-2021`, `Faldu-HPCA-2020` |
| `POPT|pr|-` | `Balaji-HPCA-2021` |
| `POPT|sssp|-` | `Balaji-HPCA-2021` |
| `SRRIP|bc|~` | `Faldu-HPCA-2020`, `Jaleel-ISCA-2010` |
| `SRRIP|bfs|~` | `Faldu-HPCA-2020`, `Jaleel-ISCA-2010` |
| `SRRIP|cc|~` | `Jaleel-ISCA-2010` |
| `SRRIP|pr|~` | `Faldu-HPCA-2020`, `Jaleel-ISCA-2010` |
| `SRRIP|sssp|~` | `Balaji-HPCA-2021`, `Jaleel-ISCA-2010` |

**0 violations** — every citation maps to a registered canonical work, every registered work is referenced, every (policy, app, sign) bucket is internally consistent.
