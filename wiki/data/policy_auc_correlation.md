# Cross-app policy-AUC correlation matrix

Source: `wiki/data/oracle_gap_auc.json`  •  apps=5  •  policies=4

Each app's AUC vector across 4 policies is z-normalized within the app, then pairwise Pearson correlations are taken across the 4 policy dimensions. r=+1 ⇒ two apps rank policies in the same order; r=-1 ⇒ exact opposite ordering.

## Clusters by AUC winner

| winner policy | apps |
|---|---|
| **GRASP** | bc |
| **POPT** | bfs, cc, pr |
| **SRRIP** | sssp |

## Correlation matrix (Pearson r on z-normalized AUC vectors)

| app | bc | bfs | cc | pr | sssp |
|---|---|---|---|---|---|
| **bc** | +1.000 | -0.844 | -0.279 | -0.233 | +0.500 |
| **bfs** | -0.844 | +1.000 | +0.672 | +0.502 | -0.085 |
| **cc** | -0.279 | +0.672 | +1.000 | +0.930 | +0.067 |
| **pr** | -0.233 | +0.502 | +0.930 | +1.000 | -0.229 |
| **sssp** | +0.500 | -0.085 | +0.067 | -0.229 | +1.000 |

## In-cluster vs out-cluster mean correlation

| app | cluster | intra mean r | inter mean r | intra-inter gap |
|---|---|---:|---:|---:|
| bc | GRASP | None | -0.214 | None |
| bfs | POPT | 0.5874 | -0.4642 | 1.0516 |
| cc | POPT | 0.8014 | -0.1061 | 0.9075 |
| pr | POPT | 0.7164 | -0.2311 | 0.9474 |
| sssp | SRRIP | None | 0.0633 | None |

## Top pairs by similarity

| rank | app A | app B | Pearson r |
|---:|---|---|---:|
| 1 | cc | pr | +0.930 |
| 2 | bfs | cc | +0.672 |
| 3 | bfs | pr | +0.502 |
| 4 | bc | sssp | +0.500 |
| 5 | cc | sssp | +0.067 |
| 6 | bfs | sssp | -0.085 |
| 7 | pr | sssp | -0.229 |
| 8 | bc | pr | -0.233 |
| 9 | bc | cc | -0.279 |
| 10 | bc | bfs | -0.844 |

## Interpretation

- A positive intra-inter gap means apps inside the same 'AUC-winner' cluster are more correlated than apps across clusters — strong evidence that AUC winners are not idiosyncratic per-app artifacts.
- A negative gap would mean the AUC clustering is noise and the per-app winners do not reflect a shared structural preference.
