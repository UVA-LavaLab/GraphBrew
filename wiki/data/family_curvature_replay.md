# Gate 61 — Per-family oracle-gap curvature replay

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff at least one family replays the pattern AND no NEW deviating family appears beyond the pinned set

qualifying families (full 1MB/4MB/8MB coverage, all 4 policies, at least one app): citation, social, web

replay count: **3 / 3**
deviating families (pinned): []
deviating families (new):    []

## Per-family mean curvature by policy (pp/octave²)

Positive = trajectory bending toward plateau (knee). Negative = still accelerating descent.

| family | GRASP | POPT | LRU | SRRIP | replays? |
| --- | ---: | ---: | ---: | ---: | :---: |
| citation | -1.094 | 0.004 | -1.382 | -1.310 | yes |
| social | 0.099 | -0.213 | -0.805 | -0.708 | yes |
| web | 0.737 | -0.133 | -1.431 | -1.651 | yes |
