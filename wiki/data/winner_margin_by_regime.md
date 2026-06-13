# Gate 62 — Winner-margin distribution per WSS regime

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff every regime has at least one win AND at least one oracle-aware policy has median winner-margin strictly larger at under_wss than at over_wss

cells classified: 114 (skipped 0)

## Per-(policy, regime) winner margin in pp of miss-rate

| policy | regime | cells won | median pp | mean pp | p90 pp | max pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GRASP | under_wss | 33 | 1.248 | 2.307 | 6.077 | 12.887 |
| GRASP | near_wss | 22 | 2.275 | 2.314 | 3.850 | 4.832 |
| GRASP | over_wss | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| LRU | under_wss | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| LRU | near_wss | 8 | 0.212 | 0.824 | 1.141 | 4.381 |
| LRU | over_wss | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| POPT | under_wss | 14 | 1.813 | 2.043 | 5.128 | 5.712 |
| POPT | near_wss | 19 | 2.105 | 3.370 | 6.255 | 12.018 |
| POPT | over_wss | 2 | 3.719 | 3.719 | 6.868 | 6.868 |
| SRRIP | under_wss | 1 | 1.520 | 1.520 | 1.520 | 1.520 |
| SRRIP | near_wss | 3 | 0.661 | 0.829 | 1.729 | 1.729 |
| SRRIP | over_wss | 1 | 0.961 | 0.961 | 0.961 | 0.961 |

## Margin-shrink evidence

Oracle-aware policies whose under_wss median margin exceeds their over_wss median margin (the paper's central claim).

| policy | under median pp | over median pp |
| --- | ---: | ---: |
| GRASP | 1.248 | 0.000 |
