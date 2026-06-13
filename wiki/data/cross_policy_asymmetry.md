# Gate 64 — Cross-policy mean-margin asymmetry

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff every (A,B) policy pair has at least one cell where each side wins head-to-head AND the largest observed asymmetry ratio (max(meanA,meanB) / min(meanA,meanB)) is strictly less than the sanity ceiling of 20.0

max asymmetry ratio observed: 4.0742 (ceiling 20.0)

## Head-to-head per policy pair (margin in pp of miss-rate)

| pair | A wins | B wins | ties | A mean pp | B mean pp | asymmetry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRASP vs LRU | 88 | 17 | 9 | 4.720 | 14.674 | 3.109 |
| GRASP vs POPT | 60 | 45 | 9 | 3.045 | 6.137 | 2.015 |
| GRASP vs SRRIP | 85 | 20 | 9 | 3.095 | 12.610 | 4.074 |
| LRU vs POPT | 15 | 81 | 18 | 7.398 | 4.571 | 1.618 |
| LRU vs SRRIP | 5 | 80 | 29 | 1.329 | 2.021 | 1.520 |
| POPT vs SRRIP | 69 | 27 | 18 | 3.422 | 4.883 | 1.427 |
