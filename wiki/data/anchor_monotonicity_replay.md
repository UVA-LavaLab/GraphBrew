# Anchor cell L3-sweep monotonicity replay

**Verdict:** PASS (catastrophic_bumps=0)

## Per-tool summary

| tool | cells | steps | bumps | bump_rate_% | hard_bumps | max_bump_pp | verdict |
|---|---:|---:|---:|---:|---:|---:|:---|
| gem5 | 6 | 18 | 0 | 0.00 | 0 | 0.0000 | PASS |
| sniper | 18 | 54 | 19 | 35.19 | 2 | 1.1834 | PASS |

## Per-tool tolerances (locked)

| tool | bump_rate_max_% | hard_bumps_max | max_bump_pp_max |
|---|---:|---:|---:|
| gem5 | 0.00 | 0 | 0.0000 |
| sniper | 40.00 | 5 | 2.0000 |

## Worst bumps per tool (up to 6)

### gem5

_no bumps observed_

### sniper

| graph | app | policy | L3 from -> to | delta_pp |
|---|---|---|---|---:|
| cit-Patents | bfs | SRRIP | 256kB -> 2MB | +1.1834 |
| email-Eu-core | bfs | LRU | 32kB -> 256kB | +0.6333 |
| cit-Patents | pr | SRRIP | 32kB -> 256kB | +0.4005 |
| cit-Patents | bfs | LRU | 256kB -> 2MB | +0.3785 |
| cit-Patents | pr | SRRIP | 256kB -> 2MB | +0.2485 |
| email-Eu-core | bfs | SRRIP | 256kB -> 2MB | +0.2371 |

## Constants

- hard_bump_threshold_pp = 0.5
- catastrophic_bump_pp   = 3.0
