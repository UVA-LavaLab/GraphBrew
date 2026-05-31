# Cross-app policy-AUC correlation matrix

Source: `wiki/data/oracle_gap_auc.json`  •  apps=5  •  policies=4

Each app's AUC vector across 4 policies is z-normalized within the app, then pairwise Pearson correlations are taken across the 4 policy dimensions. r=+1 ⇒ two apps rank policies in the same order; r=-1 ⇒ exact opposite ordering.

## Clusters by AUC winner

| winner policy | apps |
|---|---|
| **GRASP** | cc |
| **POPT** | bfs, pr, sssp |
| **SRRIP** | bc |

## Correlation matrix (Pearson r on z-normalized AUC vectors)

| app | bc | bfs | cc | pr | sssp |
|---|---|---|---|---|---|
| **bc** | +1.000 | +0.077 | +0.718 | +0.426 | +0.103 |
| **bfs** | +0.077 | +1.000 | +0.224 | +0.886 | +0.781 |
| **cc** | +0.718 | +0.224 | +1.000 | +0.649 | -0.215 |
| **pr** | +0.426 | +0.886 | +0.649 | +1.000 | +0.524 |
| **sssp** | +0.103 | +0.781 | -0.215 | +0.524 | +1.000 |

## In-cluster vs out-cluster mean correlation

| app | cluster | intra mean r | inter mean r | intra-inter gap |
|---|---|---:|---:|---:|
| bc | SRRIP | None | 0.3312 | None |
| bfs | POPT | 0.8333 | 0.1504 | 0.6829 |
| cc | GRASP | None | 0.3441 | None |
| pr | POPT | 0.705 | 0.5378 | 0.1672 |
| sssp | POPT | 0.6525 | -0.056 | 0.7086 |

## Top pairs by similarity

| rank | app A | app B | Pearson r |
|---:|---|---|---:|
| 1 | bfs | pr | +0.886 |
| 2 | bfs | sssp | +0.781 |
| 3 | bc | cc | +0.718 |
| 4 | cc | pr | +0.649 |
| 5 | pr | sssp | +0.524 |
| 6 | bc | pr | +0.426 |
| 7 | bfs | cc | +0.224 |
| 8 | bc | sssp | +0.103 |
| 9 | bc | bfs | +0.077 |
| 10 | cc | sssp | -0.215 |

## Interpretation

- A positive intra-inter gap means apps inside the same 'AUC-winner' cluster are more correlated than apps across clusters — strong evidence that AUC winners are not idiosyncratic per-app artifacts.
- A negative gap would mean the AUC clustering is noise and the per-app winners do not reflect a shared structural preference.
