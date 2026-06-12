# Gate 60 — WSS-relative knee location

source: `wiki/data/wss_relative_l3.json`

verdict: **PASS**

  invariant: PASS iff max(knee_rank of oracle-aware policies) < min(knee_rank of non-oracle policies)

knee threshold: median gap-to-oracle ≤ **0.50 pp**

regime ladder (increasing capacity): under_wss → near_wss → over_wss

## Per-policy knee location

| policy | type | knee regime | knee rank | median@under | median@near | median@over |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| GRASP | oracle-aware | under_wss | 0 | 0.000 | 0.440 | 0.000 |
| LRU | non-oracle | over_wss | 2 | 4.012 | 6.024 | 0.000 |
| POPT | oracle-aware | under_wss | 0 | 0.030 | 0.103 | 0.000 |
| SRRIP | non-oracle | over_wss | 2 | 2.518 | 3.942 | 0.000 |

## Per-policy win rate by regime

| policy | win@under | win@near | win@over |
| --- | ---: | ---: | ---: |
| GRASP | 0.521 | 0.442 | 0.714 |
| LRU | 0.000 | 0.115 | 0.071 |
| POPT | 0.458 | 0.404 | 0.143 |
| SRRIP | 0.021 | 0.038 | 0.071 |
