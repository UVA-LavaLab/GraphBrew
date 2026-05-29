# Per-family policy-AUC clustering replay (gate 57)

Per-family re-derivation of the AUC winner and Pearson correlation clusters from gate 50. Tests whether the global POPT-friendly / GRASP-friendly clustering is intrinsic to the apps or merely a side-effect of the corpus's family mix.

- source: `wiki/data/oracle_gap.json`
- qualifying families (full L3 coverage): ['citation', 'social', 'web']
- global clusters: GRASP-friendly=['bc', 'cc'], POPT-friendly=['bfs', 'pr', 'sssp']
- min winners matching across qualifying families: **3** / 5
- qualifying families where intra > inter correlation: **3** / 3
- verdict: **PASS**
- pinned deviations: 2 / observed: 2 / max allowed: 2

## Per-family winner replay

| family | n_graphs | qualified | winners matching | intra-r | inter-r | gap |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| citation | 1 | ✅ | 3/5 | 0.933 | 0.686 | 0.247 |
| mesh | 1 | ❌ | — | — | — | — |
| road | 1 | ❌ | — | — | — | — |
| social | 4 | ✅ | 5/5 | 0.9207 | 0.6028 | 0.3179 |
| web | 1 | ✅ | 5/5 | 0.5164 | 0.3112 | 0.2052 |

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
| bfs | POPT | POPT | ✅ |
| cc | GRASP | GRASP | ✅ |
| pr | POPT | POPT | ✅ |
| sssp | POPT | POPT | ✅ |

### web — winner-by-app vs global

| app | family winner | global winner | match |
| --- | --- | --- | :---: |
| bc | GRASP | GRASP | ✅ |
| bfs | POPT | POPT | ✅ |
| cc | GRASP | GRASP | ✅ |
| pr | POPT | POPT | ✅ |
| sssp | POPT | POPT | ✅ |

