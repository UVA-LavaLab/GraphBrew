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
| **GRASP** | 8.686 | 4.5871 | 0.5281 | cc (1.5205) | bfs (13.539) | 8.9043 | 2.4 | 2 | 2 | no |
| **LRU** | 17.3031 | 7.1957 | 0.4159 | sssp (10.5331) | pr (27.3615) | 2.5977 | 3.6 | 0 | 3 | no |
| **POPT** | 6.3784 | 4.5734 | 0.717 | pr (0.3965) | cc (13.7625) | 34.71 | 1.6 | 3 | 0 | no |
| **SRRIP** | 12.3839 | 5.3134 | 0.4291 | bc (5.9964) | pr (19.4213) | 3.2388 | 2.4 | 0 | 0 | no |

## Per-policy rank per app (1 = AUC winner; 4 = AUC worst)

| policy | bc | bfs | cc | pr | sssp | rank mean | rank stdev |
|---|---|---|---|---|---|---|---|
| **GRASP** | 1 | 4 | 1 | 2 | 4 | 2.4 | 1.3565 |
| **LRU** | 4 | 3 | 4 | 4 | 3 | 3.6 | 0.4899 |
| **POPT** | 3 | 1 | 2 | 1 | 1 | 1.6 | 0.8 |
| **SRRIP** | 2 | 2 | 3 | 3 | 2 | 2.4 | 0.4899 |

## Interpretation

- Coefficient of variation isolates *behavior dispersion* from *absolute scale*. A policy with CV near zero behaves the same way regardless of workload.
- 'Always top-2' is a defensible 'safe default' claim: such a policy never finishes worse than runner-up on any app.
- LRU pairs the highest mean AUC with low CV: it is *predictably bad*. POPT pairs the lowest mean AUC with the highest variance: it is the high-reward / high-variance choice.
