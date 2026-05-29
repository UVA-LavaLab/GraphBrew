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
| **bc** | +1.000 | -0.242 | +0.744 | +0.363 | -0.138 |
| **bfs** | -0.242 | +1.000 | -0.075 | +0.704 | +0.855 |
| **cc** | +0.744 | -0.075 | +1.000 | +0.654 | -0.350 |
| **pr** | +0.363 | +0.704 | +0.654 | +1.000 | +0.410 |
| **sssp** | -0.138 | +0.855 | -0.350 | +0.410 | +1.000 |

## In-cluster vs out-cluster mean correlation

| app | cluster | intra mean r | inter mean r | intra-inter gap |
|---|---|---:|---:|---:|
| bc | GRASP | 0.7437 | -0.0058 | 0.7495 |
| bfs | POPT | 0.7795 | -0.1588 | 0.9384 |
| cc | GRASP | 0.7437 | 0.0763 | 0.6674 |
| pr | POPT | 0.5572 | 0.5088 | 0.0484 |
| sssp | POPT | 0.6327 | -0.2442 | 0.8769 |

## Top pairs by similarity

| rank | app A | app B | Pearson r |
|---:|---|---|---:|
| 1 | bfs | sssp | +0.855 |
| 2 | bc | cc | +0.744 |
| 3 | bfs | pr | +0.704 |
| 4 | cc | pr | +0.654 |
| 5 | pr | sssp | +0.410 |
| 6 | bc | pr | +0.363 |
| 7 | bfs | cc | -0.075 |
| 8 | bc | sssp | -0.138 |
| 9 | bc | bfs | -0.242 |
| 10 | cc | sssp | -0.350 |

## Interpretation

- A positive intra-inter gap means apps inside the same 'AUC-winner' cluster are more correlated than apps across clusters — strong evidence that AUC winners are not idiosyncratic per-app artifacts.
- A negative gap would mean the AUC clustering is noise and the per-app winners do not reflect a shared structural preference.
