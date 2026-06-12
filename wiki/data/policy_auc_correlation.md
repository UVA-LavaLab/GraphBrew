# Cross-app policy-AUC correlation matrix

Source: `wiki/data/oracle_gap_auc.json`  •  apps=5  •  policies=4

Each app's AUC vector across 4 policies is z-normalized within the app, then pairwise Pearson correlations are taken across the 4 policy dimensions. r=+1 ⇒ two apps rank policies in the same order; r=-1 ⇒ exact opposite ordering.

## Clusters by AUC winner

| winner policy | apps |
|---|---|
| **GRASP** | bc, cc |
| **POPT** | bfs, pr, sssp |

## Correlation matrix (Pearson r on z-normalized AUC vectors)

| app | bc | bfs | cc | pr | sssp |
|---|---|---|---|---|---|
| **bc** | +1.000 | -0.170 | +0.781 | +0.455 | +0.097 |
| **bfs** | -0.170 | +1.000 | +0.130 | +0.712 | +0.945 |
| **cc** | +0.781 | +0.130 | +1.000 | +0.789 | +0.224 |
| **pr** | +0.455 | +0.712 | +0.789 | +1.000 | +0.748 |
| **sssp** | +0.097 | +0.945 | +0.224 | +0.748 | +1.000 |

## In-cluster vs out-cluster mean correlation

| app | cluster | intra mean r | inter mean r | intra-inter gap |
|---|---|---:|---:|---:|
| bc | GRASP | 0.7814 | 0.1274 | 0.654 |
| bfs | POPT | 0.8283 | -0.0201 | 0.8484 |
| cc | GRASP | 0.7814 | 0.3808 | 0.4006 |
| pr | POPT | 0.7301 | 0.6218 | 0.1084 |
| sssp | POPT | 0.8465 | 0.1608 | 0.6858 |

## Top pairs by similarity

| rank | app A | app B | Pearson r |
|---:|---|---|---:|
| 1 | bfs | sssp | +0.945 |
| 2 | cc | pr | +0.789 |
| 3 | bc | cc | +0.781 |
| 4 | pr | sssp | +0.748 |
| 5 | bfs | pr | +0.712 |
| 6 | bc | pr | +0.455 |
| 7 | cc | sssp | +0.224 |
| 8 | bfs | cc | +0.130 |
| 9 | bc | sssp | +0.097 |
| 10 | bc | bfs | -0.170 |

## Interpretation

- A positive intra-inter gap means apps inside the same 'AUC-winner' cluster are more correlated than apps across clusters — strong evidence that AUC winners are not idiosyncratic per-app artifacts.
- A negative gap would mean the AUC clustering is noise and the per-app winners do not reflect a shared structural preference.
