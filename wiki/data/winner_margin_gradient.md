# Per-(app, L3) winner-margin gradient

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

**Headline:** 9/15 (60%) of (app, L3) cells have a margin >= 2; 5 weak, 1 tied (honestly disclosed).

## Classification thresholds

| class | rule | count |
|---|---|---:|
| decisive | margin >= 4 | 4 |
| moderate | 2 <= margin < 4 | 5 |
| weak | margin == 1 | 5 |
| tied | margin == 0 | 1 |

## Per-(app, L3) cell verdict

| app | L3 | top | top_wins | runner_up | margin | class | tied with |
|---|---|---|---:|---:|---:|---|---|
| bc | 1MB | POPT | 4 | 3 | 1 | weak | — |
| bc | 4MB | GRASP | 5 | 2 | 3 | moderate | — |
| bc | 8MB | GRASP | 4 | 2 | 2 | moderate | — |
| bfs | 1MB | GRASP | 4 | 3 | 1 | weak | — |
| bfs | 4MB | GRASP | 4 | 3 | 1 | weak | — |
| bfs | 8MB | POPT | 6 | 1 | 5 | decisive | — |
| cc | 1MB | GRASP | 3 | 3 | 0 | tied | POPT |
| cc | 4MB | GRASP | 5 | 1 | 4 | decisive | — |
| cc | 8MB | GRASP | 5 | 2 | 3 | moderate | — |
| pr | 1MB | POPT | 8 | 1 | 7 | decisive | — |
| pr | 4MB | POPT | 6 | 1 | 5 | decisive | — |
| pr | 8MB | POPT | 5 | 2 | 3 | moderate | — |
| sssp | 1MB | GRASP | 4 | 1 | 3 | moderate | — |
| sssp | 4MB | GRASP | 3 | 2 | 1 | weak | — |
| sssp | 8MB | POPT | 4 | 3 | 1 | weak | — |

## Honest disclosures

- Weak cells (margin == 1, single-graph flip risk): ['bc__1MB', 'bfs__1MB', 'bfs__4MB', 'sssp__4MB', 'sssp__8MB']
- Tied cells (margin == 0, report as multi-policy tie): ['cc__1MB']
