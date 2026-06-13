# Per-family policy-AUC clustering replay (gate 57)

Per-family re-derivation of the AUC winner and Pearson correlation clusters from gate 50. Tests whether the global POPT-friendly / GRASP-friendly clustering is intrinsic to the apps or merely a side-effect of the corpus's family mix.

- source: `wiki/data/oracle_gap.json`
- qualifying families (full L3 coverage): ['citation', 'social', 'web']
- global clusters: GRASP-friendly=['bc', 'cc'], POPT-friendly=['bfs', 'pr', 'sssp']
- min winners matching across qualifying families: **2** / 5
- qualifying families where intra > inter correlation: **1** / 3
- verdict: **FAIL**
- pinned deviations: 4 / observed: 7 / max allowed: 4
- NEW deviations vs pin: [{'family': 'citation', 'app': 'bfs'}, {'family': 'social', 'app': 'bfs'}, {'family': 'web', 'app': 'sssp'}]

## Per-family winner replay

| family | n_graphs | qualified | winners matching | intra-r | inter-r | gap |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| citation | 1 | ✅ | 3/5 | 0.6769 | 0.7585 | -0.0816 |
| mesh | 1 | ❌ | — | — | — | — |
| road | 1 | ❌ | — | — | — | — |
| social | 4 | ✅ | 3/5 | 0.7157 | 0.8033 | -0.0876 |
| web | 1 | ✅ | 2/5 | -0.056 | -0.0748 | 0.0188 |

### citation — winner-by-app vs global

| app | family winner | global winner | match |
| --- | --- | --- | :---: |
| bc | GRASP | GRASP | ✅ |
| bfs | GRASP | POPT | ❌ |
| cc | GRASP | GRASP | ✅ |
| pr | POPT | POPT | ✅ |
| sssp | GRASP | POPT | ❌ |

### social — winner-by-app vs global

| app | family winner | global winner | match |
| --- | --- | --- | :---: |
| bc | GRASP | GRASP | ✅ |
| bfs | GRASP | POPT | ❌ |
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
| sssp | SRRIP | POPT | ❌ |
