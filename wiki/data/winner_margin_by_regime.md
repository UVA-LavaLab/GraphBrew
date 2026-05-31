# Gate 62 — Winner-margin distribution per WSS regime

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff every regime has at least one win AND at least one oracle-aware policy has median winner-margin strictly larger at under_wss than at over_wss

cells classified: 114 (skipped 0)

## Per-(policy, regime) winner margin in pp of miss-rate

| policy | regime | cells won | median pp | mean pp | p90 pp | max pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GRASP | under_wss | 24 | 0.719 | 2.211 | 7.478 | 10.055 |
| GRASP | near_wss | 29 | 1.268 | 1.571 | 3.083 | 6.187 |
| GRASP | over_wss | 7 | 0.000 | 0.108 | 0.002 | 0.754 |
| LRU | under_wss | 1 | 2.756 | 2.756 | 2.756 | 2.756 |
| LRU | near_wss | 4 | 1.021 | 1.671 | 4.643 | 4.643 |
| LRU | over_wss | 1 | 0.002 | 0.002 | 0.002 | 0.002 |
| POPT | under_wss | 20 | 1.226 | 2.183 | 5.291 | 8.755 |
| POPT | near_wss | 16 | 1.556 | 2.400 | 6.195 | 8.652 |
| POPT | over_wss | 4 | 0.018 | 0.636 | 2.508 | 2.508 |
| SRRIP | under_wss | 3 | 0.334 | 0.387 | 0.533 | 0.533 |
| SRRIP | near_wss | 3 | 1.163 | 1.091 | 2.097 | 2.097 |
| SRRIP | over_wss | 2 | 0.001 | 0.001 | 0.002 | 0.002 |

## Margin-shrink evidence

Oracle-aware policies whose under_wss median margin exceeds their over_wss median margin (the paper's central claim).

| policy | under median pp | over median pp |
| --- | ---: | ---: |
| GRASP | 0.719 | 0.000 |
| POPT | 1.226 | 0.018 |
