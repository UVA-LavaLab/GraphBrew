# Gate 60 — WSS-relative knee location

source: `wiki/data/wss_relative_l3.json`

verdict: **PASS**

  invariant: PASS iff max(knee_rank of oracle-aware policies) < min(knee_rank of non-oracle policies)

knee threshold: median gap-to-oracle ≤ **0.50 pp**

regime ladder (increasing capacity): under_wss → near_wss → over_wss

## Per-policy knee location

| policy | type | knee regime | knee rank | median@under | median@near | median@over |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| GRASP | oracle-aware | under_wss | 0 | 0.001 | 0.000 | 0.000 |
| LRU | non-oracle | over_wss | 2 | 2.333 | 5.410 | 0.000 |
| POPT | oracle-aware | under_wss | 0 | 0.090 | 0.890 | 0.000 |
| SRRIP | non-oracle | over_wss | 2 | 2.136 | 3.842 | 0.000 |

## Per-policy win rate by regime

| policy | win@under | win@near | win@over |
| --- | ---: | ---: | ---: |
| GRASP | 0.500 | 0.577 | 0.643 |
| LRU | 0.021 | 0.077 | 0.071 |
| POPT | 0.417 | 0.288 | 0.143 |
| SRRIP | 0.062 | 0.058 | 0.143 |
