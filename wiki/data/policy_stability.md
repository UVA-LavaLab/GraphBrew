# Per-policy stability index across apps

Source: `wiki/data/oracle_gap_auc.json`  •  apps=5  •  policies=4

Stability metric: **coefficient of variation (stdev / mean) of AUC across apps**. Lower CV = more predictable across workloads.

## Headline

- Safest (lowest CV): **LRU**
- Highest-variance: **POPT**
- Best mean AUC (lowest avg gap): **POPT**

## Per-policy stability table

| policy | mean AUC | stdev AUC | CV | best app | worst app | worst/best | mean rank | wins | lasts | always top-2 |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| **GRASP** | 10.0641 | 4.2561 | 0.4229 | bc (3.547) | bfs (15.2143) | 4.2893 | 2.2 | 1 | 1 | no |
| **LRU** | 16.4346 | 6.1985 | 0.3772 | bc (9.3557) | pr (25.5568) | 2.7317 | 3.6 | 0 | 3 | no |
| **POPT** | 8.1314 | 4.3569 | 0.5358 | pr (0.9115) | bc (13.3901) | 14.6902 | 2.0 | 3 | 1 | no |
| **SRRIP** | 11.0845 | 4.3833 | 0.3954 | bc (5.0207) | pr (17.2249) | 3.4308 | 2.2 | 1 | 0 | no |

## Per-policy rank per app (1 = AUC winner; 4 = AUC worst)

| policy | bc | bfs | cc | pr | sssp | rank mean | rank stdev |
|---|---|---|---|---|---|---|---|
| **GRASP** | 1 | 4 | 2 | 2 | 2 | 2.2 | 0.9798 |
| **LRU** | 3 | 3 | 4 | 4 | 4 | 3.6 | 0.4899 |
| **POPT** | 4 | 1 | 1 | 1 | 3 | 2.0 | 1.2649 |
| **SRRIP** | 2 | 2 | 3 | 3 | 1 | 2.2 | 0.7483 |

## Interpretation

- Coefficient of variation isolates *behavior dispersion* from *absolute scale*. A policy with CV near zero behaves the same way regardless of workload.
- 'Always top-2' is a defensible 'safe default' claim: such a policy never finishes worse than runner-up on any app.
- LRU pairs the highest mean AUC with low CV: it is *predictably bad*. POPT pairs the lowest mean AUC with the highest variance: it is the high-reward / high-variance choice.
