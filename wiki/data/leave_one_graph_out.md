# Leave-one-graph-out (LOGO) winner robustness

For each application, drop each graph in turn and re-rank policy
winners by cell count. A LOGO-robust claim has the same top
policy after every drop — no single graph is driving the headline.

- Graphs in corpus: **8** (`cit-Patents`, `com-orkut`, `delaunay_n19`, `email-Eu-core`, `roadNet-CA`, `soc-LiveJournal1`, `soc-pokec`, `web-Google`)
- LOGO-robust applications: **4** (`bc, bfs, pr, sssp`)
- LOGO-fragile applications: **1** (`cc`)

## Per-app summary

| App | Full-corpus top | Wins | Robust? | Fragile drops |
| :-- | :-------------- | ---: | :------ | :------------ |
| `bc` | `GRASP` | 17 | ✓ ROBUST | — |
| `bfs` | `GRASP` | 15 | ✓ ROBUST | — |
| `cc` | `POPT` | 11 | ✗ FRAGILE | `web-Google` |
| `pr` | `POPT` | 21 | ✓ ROBUST | — |
| `sssp` | `GRASP` | 14 | ✓ ROBUST | — |

## Per-(app, drop) detail

### `bc` — full top: `GRASP` (17 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `GRASP` | 14 | 9 | ✓ |
| `com-orkut` | `GRASP` | 14 | 9 | ✓ |
| `delaunay_n19` | `GRASP` | 17 | 12 | ✓ |
| `email-Eu-core` | `GRASP` | 14 | 12 | ✓ |
| `roadNet-CA` | `GRASP` | 15 | 10 | ✓ |
| `soc-LiveJournal1` | `GRASP` | 14 | 9 | ✓ |
| `soc-pokec` | `GRASP` | 14 | 9 | ✓ |
| `web-Google` | `GRASP` | 17 | 12 | ✓ |

### `bfs` — full top: `GRASP` (15 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `GRASP` | 13 | 4 | ✓ |
| `com-orkut` | `GRASP` | 13 | 4 | ✓ |
| `delaunay_n19` | `GRASP` | 15 | 5 | ✓ |
| `email-Eu-core` | `GRASP` | 12 | 5 | ✓ |
| `roadNet-CA` | `GRASP` | 12 | 3 | ✓ |
| `soc-LiveJournal1` | `GRASP` | 12 | 2 | ✓ |
| `soc-pokec` | `GRASP` | 14 | 6 | ✓ |
| `web-Google` | `GRASP` | 14 | 6 | ✓ |

### `cc` — full top: `POPT` (11 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `POPT` | 10 | 3 | ✓ |
| `com-orkut` | `POPT` | 10 | 3 | ✓ |
| `delaunay_n19` | `POPT` | 11 | 2 | ✓ |
| `email-Eu-core` | `POPT` | 11 | 2 | ✓ |
| `roadNet-CA` | `POPT` | 8 | 1 | ✓ |
| `soc-LiveJournal1` | `POPT` | 10 | 3 | ✓ |
| `soc-pokec` | `POPT` | 9 | 1 | ✓ |
| `web-Google` | `GRASP` | 9 | 1 | ✗ |

### `pr` — full top: `POPT` (21 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `POPT` | 18 | 9 | ✓ |
| `com-orkut` | `POPT` | 19 | 11 | ✓ |
| `delaunay_n19` | `POPT` | 17 | 9 | ✓ |
| `email-Eu-core` | `POPT` | 18 | 12 | ✓ |
| `roadNet-CA` | `POPT` | 18 | 11 | ✓ |
| `soc-LiveJournal1` | `POPT` | 18 | 9 | ✓ |
| `soc-pokec` | `POPT` | 20 | 13 | ✓ |
| `web-Google` | `POPT` | 19 | 10 | ✓ |

### `sssp` — full top: `GRASP` (14 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `GRASP` | 11 | 6 | ✓ |
| `com-orkut` | `GRASP` | 12 | 8 | ✓ |
| `delaunay_n19` | `GRASP` | 14 | 9 | ✓ |
| `email-Eu-core` | `GRASP` | 14 | 9 | ✓ |
| `roadNet-CA` | `GRASP` | 11 | 7 | ✓ |
| `soc-LiveJournal1` | `GRASP` | 11 | 6 | ✓ |
| `soc-pokec` | `GRASP` | 12 | 8 | ✓ |
| `web-Google` | `GRASP` | 13 | 10 | ✓ |
