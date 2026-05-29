# Per-(app, L3) winner-margin gradient

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

**Headline:** 12/15 (80%) of (app, L3) cells have a margin >= 2; 1 weak, 2 tied (honestly disclosed).

## Classification thresholds

| class | rule | count |
|---|---|---:|
| decisive | margin >= 4 | 6 |
| moderate | 2 <= margin < 4 | 6 |
| weak | margin == 1 | 1 |
| tied | margin == 0 | 2 |

## Per-(app, L3) cell verdict

| app | L3 | top | top_wins | runner_up | margin | class | tied with |
|---|---|---|---:|---:|---:|---|---|
| bc | 1MB | GRASP | 4 | 4 | 0 | tied | SRRIP |
| bc | 4MB | GRASP | 6 | 1 | 5 | decisive | — |
| bc | 8MB | GRASP | 5 | 1 | 4 | decisive | — |
| bfs | 1MB | GRASP | 4 | 1 | 3 | moderate | — |
| bfs | 4MB | POPT | 5 | 1 | 4 | decisive | — |
| bfs | 8MB | POPT | 4 | 2 | 2 | moderate | — |
| cc | 1MB | GRASP | 5 | 1 | 4 | decisive | — |
| cc | 4MB | GRASP | 5 | 0 | 5 | decisive | — |
| cc | 8MB | GRASP | 4 | 1 | 3 | moderate | — |
| pr | 1MB | POPT | 7 | 1 | 6 | decisive | — |
| pr | 4MB | POPT | 4 | 2 | 2 | moderate | — |
| pr | 8MB | POPT | 4 | 2 | 2 | moderate | — |
| sssp | 1MB | POPT | 3 | 2 | 1 | weak | — |
| sssp | 4MB | GRASP | 3 | 1 | 2 | moderate | — |
| sssp | 8MB | GRASP | 2 | 2 | 0 | tied | SRRIP |

## Honest disclosures

- Weak cells (margin == 1, single-graph flip risk): ['sssp__1MB']
- Tied cells (margin == 0, report as multi-policy tie): ['bc__1MB', 'sssp__8MB']

