# Gate 60 — WSS-relative knee location

source: `wiki/data/wss_relative_l3.json`

verdict: **PASS**

  invariant: PASS iff max(knee_rank of oracle-aware policies) < min(knee_rank of non-oracle policies)

knee threshold: median gap-to-oracle ≤ **0.50 pp**

regime ladder (increasing capacity): under_wss → near_wss → over_wss

## Per-policy knee location

| policy | type | knee regime | knee rank | median@under | median@near | median@over |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| GRASP | oracle-aware | under_wss | 0 | 0.006 | 0.000 | 0.000 |
| LRU | non-oracle | over_wss | 2 | 2.144 | 5.040 | 0.004 |
| POPT | oracle-aware | under_wss | 0 | 0.070 | 0.633 | 0.002 |
| SRRIP | non-oracle | over_wss | 2 | 1.797 | 3.337 | 0.002 |

## Per-policy win rate by regime

| policy | win@under | win@near | win@over |
| --- | ---: | ---: | ---: |
| GRASP | 0.479 | 0.519 | 0.571 |
| LRU | 0.021 | 0.077 | 0.071 |
| POPT | 0.438 | 0.346 | 0.214 |
| SRRIP | 0.062 | 0.058 | 0.143 |
