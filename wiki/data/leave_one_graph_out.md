# Leave-one-graph-out (LOGO) winner robustness

For each application, drop each graph in turn and re-rank policy
winners by cell count. A LOGO-robust claim has the same top
policy after every drop — no single graph is driving the headline.

- Graphs in corpus: **8** (`cit-Patents`, `com-orkut`, `delaunay_n19`, `email-Eu-core`, `roadNet-CA`, `soc-LiveJournal1`, `soc-pokec`, `web-Google`)
- LOGO-robust applications: **3** (`bc, cc, pr`)
- LOGO-fragile applications: **2** (`bfs, sssp`)

## Per-app summary

| App | Full-corpus top | Wins | Robust? | Fragile drops |
| :-- | :-------------- | ---: | :------ | :------------ |
| `bc` | `GRASP` | 15 | ✓ ROBUST | — |
| `bfs` | `GRASP` | 12 | ✗ FRAGILE | `cit-Patents`, `soc-LiveJournal1` |
| `cc` | `GRASP` | 17 | ✓ ROBUST | — |
| `pr` | `POPT` | 22 | ✓ ROBUST | — |
| `sssp` | `POPT` | 8 | ✗ FRAGILE | `com-orkut`, `roadNet-CA`, `web-Google` |

## Per-(app, drop) detail

### `bc` — full top: `GRASP` (15 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `GRASP` | 13 | 7 | ✓ |
| `com-orkut` | `GRASP` | 12 | 5 | ✓ |
| `delaunay_n19` | `GRASP` | 15 | 8 | ✓ |
| `email-Eu-core` | `GRASP` | 13 | 9 | ✓ |
| `roadNet-CA` | `GRASP` | 15 | 10 | ✓ |
| `soc-LiveJournal1` | `GRASP` | 12 | 5 | ✓ |
| `soc-pokec` | `GRASP` | 12 | 5 | ✓ |
| `web-Google` | `GRASP` | 13 | 7 | ✓ |

### `bfs` — full top: `GRASP` (12 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `GRASP` | 9 | 0 | ✗ |
| `com-orkut` | `GRASP` | 11 | 4 | ✓ |
| `delaunay_n19` | `GRASP` | 12 | 3 | ✓ |
| `email-Eu-core` | `GRASP` | 11 | 3 | ✓ |
| `roadNet-CA` | `GRASP` | 10 | 3 | ✓ |
| `soc-LiveJournal1` | `GRASP` | 9 | 0 | ✗ |
| `soc-pokec` | `GRASP` | 11 | 4 | ✓ |
| `web-Google` | `GRASP` | 11 | 4 | ✓ |

### `cc` — full top: `GRASP` (17 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `GRASP` | 14 | 12 | ✓ |
| `com-orkut` | `GRASP` | 14 | 12 | ✓ |
| `delaunay_n19` | `GRASP` | 17 | 15 | ✓ |
| `email-Eu-core` | `GRASP` | 17 | 15 | ✓ |
| `roadNet-CA` | `GRASP` | 14 | 13 | ✓ |
| `soc-LiveJournal1` | `GRASP` | 14 | 12 | ✓ |
| `soc-pokec` | `GRASP` | 14 | 12 | ✓ |
| `web-Google` | `GRASP` | 15 | 14 | ✓ |

### `pr` — full top: `POPT` (22 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `POPT` | 19 | 10 | ✓ |
| `com-orkut` | `POPT` | 19 | 10 | ✓ |
| `delaunay_n19` | `POPT` | 19 | 12 | ✓ |
| `email-Eu-core` | `POPT` | 19 | 13 | ✓ |
| `roadNet-CA` | `POPT` | 18 | 10 | ✓ |
| `soc-LiveJournal1` | `POPT` | 19 | 10 | ✓ |
| `soc-pokec` | `POPT` | 21 | 14 | ✓ |
| `web-Google` | `POPT` | 20 | 12 | ✓ |

### `sssp` — full top: `POPT` (8 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `POPT` | 8 | 4 | ✓ |
| `com-orkut` | `GRASP` | 7 | 1 | ✗ |
| `delaunay_n19` | `POPT` | 8 | 1 | ✓ |
| `email-Eu-core` | `POPT` | 8 | 1 | ✓ |
| `roadNet-CA` | `GRASP` | 7 | 2 | ✗ |
| `soc-LiveJournal1` | `POPT` | 7 | 2 | ✓ |
| `soc-pokec` | `POPT` | 7 | 2 | ✓ |
| `web-Google` | `GRASP` | 7 | 0 | ✗ |
