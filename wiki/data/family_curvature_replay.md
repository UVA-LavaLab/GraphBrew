# Gate 61 — Per-family oracle-gap curvature replay

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff at least one family replays the pattern AND no NEW deviating family appears beyond the pinned set

qualifying families (full 1MB/4MB/8MB coverage, all 4 policies, at least one app): citation, social, web

replay count: **1 / 3**
deviating families (pinned): ['citation', 'social']
deviating families (new):    []

## Per-family mean curvature by policy (pp/octave²)

Positive = trajectory bending toward plateau (knee). Negative = still accelerating descent.

| family | GRASP | POPT | LRU | SRRIP | replays? |
| --- | ---: | ---: | ---: | ---: | :---: |
| citation | -0.398 | -0.100 | -1.744 | -1.685 | no |
| social | -0.078 | -0.365 | -1.192 | -0.727 | no |
| web | -0.417 | 0.358 | -0.767 | -0.798 | yes |
