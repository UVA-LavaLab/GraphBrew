# Per-family policy-AUC clustering replay (gate 57)

Per-family re-derivation of the AUC winner and Pearson correlation clusters from gate 50. Tests whether the global POPT-friendly / GRASP-friendly clustering is intrinsic to the apps or merely a side-effect of the corpus's family mix.

- source: `wiki/data/oracle_gap.json`
- qualifying families (full L3 coverage): ['citation', 'social', 'web']
- global clusters: GRASP-friendly=['bc', 'cc'], POPT-friendly=['bfs', 'pr', 'sssp']
- min winners matching across qualifying families: **3** / 5
- qualifying families where intra > inter correlation: **2** / 3
- verdict: **PASS**
- pinned deviations: 4 / observed: 4 / max allowed: 4

## Per-family winner replay

| family | n_graphs | qualified | winners matching | intra-r | inter-r | gap |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| citation | 1 | ✅ | 4/5 | 0.9362 | 0.9176 | 0.0186 |
| mesh | 1 | ❌ | — | — | — | — |
| road | 1 | ❌ | — | — | — | — |
| social | 4 | ✅ | 4/5 | 0.9618 | 0.8916 | 0.0702 |
| web | 1 | ✅ | 3/5 | -0.0851 | 0.0569 | -0.142 |

### citation — winner-by-app vs global

| app | family winner | global winner | match |
| --- | --- | --- | :---: |
| bc | GRASP | GRASP | ✅ |
| bfs | POPT | POPT | ✅ |
| cc | GRASP | GRASP | ✅ |
| pr | POPT | POPT | ✅ |
| sssp | GRASP | POPT | ❌ |

### social — winner-by-app vs global

| app | family winner | global winner | match |
| --- | --- | --- | :---: |
| bc | GRASP | GRASP | ✅ |
| bfs | POPT | POPT | ✅ |
| cc | GRASP | GRASP | ✅ |
| pr | POPT | POPT | ✅ |
| sssp | GRASP | POPT | ❌ |

### web — winner-by-app vs global

| app | family winner | global winner | match |
| --- | --- | --- | :---: |
| bc | SRRIP | GRASP | ❌ |
| bfs | POPT | POPT | ✅ |
| cc | POPT | GRASP | ❌ |
| pr | POPT | POPT | ✅ |
| sssp | POPT | POPT | ✅ |
