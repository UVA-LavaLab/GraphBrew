# Gate 62 — Winner-margin distribution per WSS regime

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff every regime has at least one win AND at least one oracle-aware policy has median winner-margin strictly larger at under_wss than at over_wss

cells classified: 114 (skipped 0)

## Per-(policy, regime) winner margin in pp of miss-rate

| policy | regime | cells won | median pp | mean pp | p90 pp | max pp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GRASP | under_wss | 25 | 1.020 | 2.073 | 5.943 | 11.717 |
| GRASP | near_wss | 23 | 0.927 | 1.839 | 6.148 | 8.980 |
| GRASP | over_wss | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| LRU | under_wss | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| LRU | near_wss | 6 | 0.535 | 1.099 | 1.141 | 4.381 |
| LRU | over_wss | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| POPT | under_wss | 22 | 0.846 | 2.149 | 6.084 | 8.808 |
| POPT | near_wss | 21 | 4.318 | 4.609 | 10.580 | 15.479 |
| POPT | over_wss | 2 | 4.244 | 4.244 | 7.587 | 7.587 |
| SRRIP | under_wss | 1 | 1.520 | 1.520 | 1.520 | 1.520 |
| SRRIP | near_wss | 2 | 0.913 | 0.913 | 1.729 | 1.729 |
| SRRIP | over_wss | 1 | 0.961 | 0.961 | 0.961 | 0.961 |

## Margin-shrink evidence

Oracle-aware policies whose under_wss median margin exceeds their over_wss median margin (the paper's central claim).

| policy | under median pp | over median pp |
| --- | ---: | ---: |
| GRASP | 1.021 | 0.000 |
