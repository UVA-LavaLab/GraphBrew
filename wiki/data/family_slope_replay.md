# Gate 67 — Per-family capacity-sensitivity slope replay

source: `wiki/data/oracle_gap.json`

verdict: **PASS**

  invariant: PASS iff at least one family replays the pattern (LRU and SRRIP both strictly steeper than GRASP, every policy median < -5.0 pp/octave) AND no NEW deviating family appears beyond the pinned set.

qualifying families (full 1MB/4MB/8MB coverage, all 4 policies, at least one app): citation, social, web

replay count: **3 / 3**
deviating families (pinned): ['social']
deviating families (new):    []

## Per-family median slope by policy (pp / log2(L3 MB))

Smaller (more negative) = more cache-hungry. Help floor: -5.0 pp/octave.

| family | GRASP | POPT | LRU | SRRIP | n cells | replays? |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| citation | -14.822 | -13.652 | -15.752 | -15.105 | 5 | yes |
| social | -13.012 | -12.388 | -13.450 | -13.891 | 18 | yes |
| web | -12.701 | -15.975 | -18.666 | -18.033 | 5 | yes |
