# Per-(app, L3) winner-margin gradient

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

**Headline:** 11/15 (73%) of (app, L3) cells have a margin >= 2; 4 weak, 0 tied (honestly disclosed).

## Classification thresholds

| class | rule | count |
|---|---|---:|
| decisive | margin >= 4 | 2 |
| moderate | 2 <= margin < 4 | 9 |
| weak | margin == 1 | 4 |
| tied | margin == 0 | 0 |

## Per-(app, L3) cell verdict

| app | L3 | top | top_wins | runner_up | margin | class | tied with |
|---|---|---|---:|---:|---:|---|---|
| bc | 1MB | GRASP | 5 | 3 | 2 | moderate | — |
| bc | 4MB | GRASP | 5 | 2 | 3 | moderate | — |
| bc | 8MB | GRASP | 5 | 2 | 3 | moderate | — |
| bfs | 1MB | GRASP | 6 | 2 | 4 | decisive | — |
| bfs | 4MB | GRASP | 4 | 3 | 1 | weak | — |
| bfs | 8MB | POPT | 5 | 2 | 3 | moderate | — |
| cc | 1MB | POPT | 4 | 2 | 2 | moderate | — |
| cc | 4MB | GRASP | 3 | 2 | 1 | weak | — |
| cc | 8MB | POPT | 3 | 2 | 1 | weak | — |
| pr | 1MB | POPT | 6 | 3 | 3 | moderate | — |
| pr | 4MB | POPT | 5 | 2 | 3 | moderate | — |
| pr | 8MB | POPT | 5 | 2 | 3 | moderate | — |
| sssp | 1MB | GRASP | 5 | 1 | 4 | decisive | — |
| sssp | 4MB | GRASP | 4 | 1 | 3 | moderate | — |
| sssp | 8MB | SRRIP | 3 | 2 | 1 | weak | — |

## Honest disclosures

- Weak cells (margin == 1, single-graph flip risk): ['bfs__4MB', 'cc__4MB', 'cc__8MB', 'sssp__8MB']
- No tied cells.
