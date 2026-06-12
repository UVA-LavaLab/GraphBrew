# Gate 63 — Per-family winner-margin replay

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff at least one qualifying family replays the global margin-shrink pattern (some oracle-aware policy median under_wss > over_wss) AND no NEW family deviates beyond the pinned set

qualifying families: 1 (social)

replaying families: 1 (social)

deviating families: 0 (-)

pinned deviating: 0 (-)

## Per-family margin-shrink evidence

| family | qualifying | replays | oracle-aware shrink summary |
| --- | :---: | :---: | --- |
| citation | no | no | — |
| mesh | no | no | — |
| road | no | no | — |
| social | yes | yes | GRASP: 2.084→0.000 pp |
| web | no | no | — |

## Per-family cell counts

| family | classified | skipped |
| --- | ---: | ---: |
| citation | 15 | 0 |
| mesh | 5 | 0 |
| road | 25 | 0 |
| social | 54 | 0 |
| web | 15 | 0 |
