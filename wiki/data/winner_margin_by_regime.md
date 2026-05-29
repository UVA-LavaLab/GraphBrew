# Gate 62 — Winner-margin distribution per WSS regime

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff every regime has at least one win AND at least one oracle-aware policy has median winner-margin strictly larger at under_wss than at over_wss

cells classified: 114 (skipped 0)

## Per-(policy, regime) winner margin in pp of miss-rate

| policy | regime | cells won | median pp | mean pp | p90 pp | max pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GRASP | under_wss | 23 | 0.863 | 2.326 | 7.478 | 10.055 |
| GRASP | near_wss | 27 | 1.528 | 1.651 | 3.083 | 6.187 |
| GRASP | over_wss | 6 | 0.001 | 0.149 | 0.019 | 0.872 |
| LRU | under_wss | 1 | 2.756 | 2.756 | 2.756 | 2.756 |
| LRU | near_wss | 4 | 1.021 | 1.671 | 4.643 | 4.643 |
| LRU | over_wss | 1 | 0.002 | 0.002 | 0.002 | 0.002 |
| POPT | under_wss | 21 | 1.157 | 2.163 | 5.213 | 8.755 |
| POPT | near_wss | 18 | 1.157 | 2.212 | 4.758 | 8.246 |
| POPT | over_wss | 5 | 0.020 | 0.508 | 2.481 | 2.481 |
| SRRIP | under_wss | 3 | 0.488 | 0.452 | 0.533 | 0.533 |
| SRRIP | near_wss | 3 | 1.345 | 1.152 | 2.097 | 2.097 |
| SRRIP | over_wss | 2 | 0.001 | 0.001 | 0.002 | 0.002 |

## Margin-shrink evidence

Oracle-aware policies whose under_wss median margin exceeds their over_wss median margin (the paper's central claim).

| policy | under median pp | over median pp |
| --- | ---: | ---: |
| GRASP | 0.863 | 0.001 |
| POPT | 1.157 | 0.020 |
