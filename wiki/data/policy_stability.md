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
| **GRASP** | 9.3595 | 5.7486 | 0.6142 | cc (1.6668) | bfs (16.8516) | 10.1102 | 2.2 | 2 | 1 | no |
| **LRU** | 20.1684 | 9.8132 | 0.4866 | bc (9.5084) | cc (32.4693) | 3.4148 | 3.8 | 0 | 4 | no |
| **POPT** | 5.1632 | 3.5671 | 0.6909 | pr (0.0142) | cc (10.4865) | 738.4859 | 1.6 | 3 | 0 | no |
| **SRRIP** | 14.8182 | 7.7679 | 0.5242 | bc (5.1733) | cc (24.1584) | 4.6698 | 2.4 | 0 | 0 | no |

## Per-policy rank per app (1 = AUC winner; 4 = AUC worst)

| policy | bc | bfs | cc | pr | sssp | rank mean | rank stdev |
|---|---|---|---|---|---|---|---|
| **GRASP** | 1 | 4 | 1 | 2 | 3 | 2.2 | 1.1662 |
| **LRU** | 4 | 3 | 4 | 4 | 4 | 3.8 | 0.4 |
| **POPT** | 3 | 1 | 2 | 1 | 1 | 1.6 | 0.8 |
| **SRRIP** | 2 | 2 | 3 | 3 | 2 | 2.4 | 0.4899 |

## Interpretation

- Coefficient of variation isolates *behavior dispersion* from *absolute scale*. A policy with CV near zero behaves the same way regardless of workload.
- 'Always top-2' is a defensible 'safe default' claim: such a policy never finishes worse than runner-up on any app.
- LRU pairs the highest mean AUC with low CV: it is *predictably bad*. POPT pairs the lowest mean AUC with the highest variance: it is the high-reward / high-variance choice.
