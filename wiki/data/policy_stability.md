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
| **GRASP** | 8.5293 | 4.3245 | 0.507 | cc (1.5205) | bfs (12.8861) | 8.4749 | 2.2 | 1 | 1 | no |
| **LRU** | 17.9282 | 6.9603 | 0.3882 | bc (11.0047) | pr (27.7572) | 2.5223 | 3.8 | 0 | 4 | no |
| **POPT** | 6.3747 | 4.4935 | 0.7049 | pr (0.3774) | cc (13.7625) | 36.4666 | 1.6 | 3 | 0 | no |
| **SRRIP** | 12.6992 | 5.4561 | 0.4296 | bc (5.7773) | pr (19.7583) | 3.42 | 2.4 | 1 | 0 | no |

## Per-policy rank per app (1 = AUC winner; 4 = AUC worst)

| policy | bc | bfs | cc | pr | sssp | rank mean | rank stdev |
|---|---|---|---|---|---|---|---|
| **GRASP** | 2 | 2 | 1 | 2 | 4 | 2.2 | 0.9798 |
| **LRU** | 4 | 4 | 4 | 4 | 3 | 3.8 | 0.4 |
| **POPT** | 3 | 1 | 2 | 1 | 1 | 1.6 | 0.8 |
| **SRRIP** | 1 | 3 | 3 | 3 | 2 | 2.4 | 0.8 |

## Interpretation

- Coefficient of variation isolates *behavior dispersion* from *absolute scale*. A policy with CV near zero behaves the same way regardless of workload.
- 'Always top-2' is a defensible 'safe default' claim: such a policy never finishes worse than runner-up on any app.
- LRU pairs the highest mean AUC with low CV: it is *predictably bad*. POPT pairs the lowest mean AUC with the highest variance: it is the high-reward / high-variance choice.
