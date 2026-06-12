# Leave-one-graph-out (LOGO) winner robustness

For each application, drop each graph in turn and re-rank policy
winners by cell count. A LOGO-robust claim has the same top
policy after every drop — no single graph is driving the headline.

- Graphs in corpus: **8** (`cit-Patents`, `com-orkut`, `delaunay_n19`, `email-Eu-core`, `roadNet-CA`, `soc-LiveJournal1`, `soc-pokec`, `web-Google`)
- LOGO-robust applications: **2** (`cc, pr`)
- LOGO-fragile applications: **3** (`bc, bfs, sssp`)

## Per-app summary

| App | Full-corpus top | Wins | Robust? | Fragile drops |
| :-- | :-------------- | ---: | :------ | :------------ |
| `bc` | `GRASP` | 12 | ✗ FRAGILE | `com-orkut` |
| `bfs` | `POPT` | 13 | ✗ FRAGILE | `cit-Patents`, `com-orkut`, `soc-pokec`, `web-Google` |
| `cc` | `GRASP` | 17 | ✓ ROBUST | — |
| `pr` | `POPT` | 24 | ✓ ROBUST | — |
| `sssp` | `GRASP` | 10 | ✗ FRAGILE | `roadNet-CA` |

## Per-(app, drop) detail

### `bc` — full top: `GRASP` (12 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `GRASP` | 10 | 1 | ✓ |
| `com-orkut` | `POPT` | 10 | 1 | ✗ |
| `delaunay_n19` | `GRASP` | 12 | 2 | ✓ |
| `email-Eu-core` | `GRASP` | 9 | 2 | ✓ |
| `roadNet-CA` | `GRASP` | 11 | 4 | ✓ |
| `soc-LiveJournal1` | `GRASP` | 10 | 1 | ✓ |
| `soc-pokec` | `GRASP` | 11 | 3 | ✓ |
| `web-Google` | `GRASP` | 12 | 2 | ✓ |

### `bfs` — full top: `POPT` (13 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `GRASP` | 11 | 0 | ✗ |
| `com-orkut` | `GRASP` | 11 | 0 | ✗ |
| `delaunay_n19` | `POPT` | 13 | 1 | ✓ |
| `email-Eu-core` | `POPT` | 10 | 1 | ✓ |
| `roadNet-CA` | `POPT` | 12 | 3 | ✓ |
| `soc-LiveJournal1` | `POPT` | 12 | 2 | ✓ |
| `soc-pokec` | `GRASP` | 11 | 0 | ✗ |
| `web-Google` | `GRASP` | 11 | 0 | ✗ |

### `cc` — full top: `GRASP` (17 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `GRASP` | 15 | 10 | ✓ |
| `com-orkut` | `GRASP` | 14 | 8 | ✓ |
| `delaunay_n19` | `GRASP` | 17 | 11 | ✓ |
| `email-Eu-core` | `GRASP` | 17 | 11 | ✓ |
| `roadNet-CA` | `GRASP` | 12 | 6 | ✓ |
| `soc-LiveJournal1` | `GRASP` | 15 | 10 | ✓ |
| `soc-pokec` | `GRASP` | 14 | 9 | ✓ |
| `web-Google` | `GRASP` | 15 | 12 | ✓ |

### `pr` — full top: `POPT` (24 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `POPT` | 21 | 14 | ✓ |
| `com-orkut` | `POPT` | 21 | 14 | ✓ |
| `delaunay_n19` | `POPT` | 20 | 14 | ✓ |
| `email-Eu-core` | `POPT` | 21 | 17 | ✓ |
| `roadNet-CA` | `POPT` | 21 | 16 | ✓ |
| `soc-LiveJournal1` | `POPT` | 21 | 14 | ✓ |
| `soc-pokec` | `POPT` | 22 | 16 | ✓ |
| `web-Google` | `POPT` | 21 | 14 | ✓ |

### `sssp` — full top: `GRASP` (10 wins)

| Dropped graph | Top after drop | Wins | Margin | Same? |
| :------------ | :------------- | ---: | -----: | :---- |
| `cit-Patents` | `GRASP` | 8 | 2 | ✓ |
| `com-orkut` | `GRASP` | 9 | 3 | ✓ |
| `delaunay_n19` | `GRASP` | 10 | 3 | ✓ |
| `email-Eu-core` | `GRASP` | 10 | 3 | ✓ |
| `roadNet-CA` | `GRASP` | 7 | 0 | ✗ |
| `soc-LiveJournal1` | `GRASP` | 9 | 4 | ✓ |
| `soc-pokec` | `GRASP` | 8 | 2 | ✓ |
| `web-Google` | `GRASP` | 9 | 4 | ✓ |
