# Gate 61 — Per-family oracle-gap curvature replay

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff at least one family replays the pattern AND no NEW deviating family appears beyond the pinned set

qualifying families (full 1MB/4MB/8MB coverage, all 4 policies, at least one app): citation, social, web

replay count: **2 / 3**
deviating families (pinned): ['citation', 'social']
deviating families (new):    []

## Per-family mean curvature by policy (pp/octave²)

Positive = trajectory bending toward plateau (knee). Negative = still accelerating descent.

| family | GRASP | POPT | LRU | SRRIP | replays? |
| --- | ---: | ---: | ---: | ---: | :---: |
| citation | -0.329 | -0.811 | -2.283 | -2.224 | no |
| social | 1.161 | -0.850 | -1.024 | -0.558 | yes |
| web | -2.900 | 0.591 | -0.667 | -0.698 | yes |
