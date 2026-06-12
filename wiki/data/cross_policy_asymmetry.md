# Gate 64 — Cross-policy mean-margin asymmetry

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff every (A,B) policy pair has at least one cell where each side wins head-to-head AND the largest observed asymmetry ratio (max(meanA,meanB) / min(meanA,meanB)) is strictly less than the sanity ceiling of 20.0

max asymmetry ratio observed: 3.0335 (ceiling 20.0)

## Head-to-head per policy pair (margin in pp of miss-rate)

| pair | A wins | B wins | ties | A mean pp | B mean pp | asymmetry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRASP vs LRU | 88 | 14 | 12 | 5.509 | 13.306 | 2.415 |
| GRASP vs POPT | 50 | 52 | 12 | 2.115 | 5.200 | 2.459 |
| GRASP vs SRRIP | 86 | 16 | 12 | 3.831 | 11.621 | 3.034 |
| LRU vs POPT | 9 | 87 | 18 | 8.659 | 6.220 | 1.392 |
| LRU vs SRRIP | 5 | 80 | 29 | 1.329 | 2.021 | 1.520 |
| POPT vs SRRIP | 86 | 10 | 18 | 4.511 | 7.974 | 1.768 |
