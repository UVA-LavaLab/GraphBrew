# ECG PFX vs DROPLET head-to-head

Gate 241 — ECG-Pfx-vs-DROPLET. Sibling family to the substrate-
parity trinity (gates 238/239/240) but evaluates a different
axis: ECG's PFX prefetcher vs DROPLET on the same baseline.

**Status:** `active`

## Rules
- **G1** — every required arm {LRU, DROPLET, ECG_PFX} present with status=ok per (benchmark, section, l3_size) cell
- **G2** — if pf_issued==0 then arm_miss_rate - lru_miss_rate <= 0.005 (a quiet prefetcher must not degrade baseline)
- **G3** — if pf_issued > 0 then pf_useful / pf_issued >= 0.05
- **G4** — logged-only: head-to-head wins below the neutral floor are comparable (no violation)
- **G5** — if postfix.expected_backend set, every row has matching backend AND simulator
- **G6** — len(per_observation) >= postfix.expected_minimum_observations

## Constants
- ε(neutral floor): `0.005`
- ε(useful floor): `0.05`
- required arms: `DROPLET, ECG_PFX, LRU`

## Totals
- observations: **9**
- cells (benchmark × section × L3): **3**
- benchmarks: `bfs, pr, sssp`
- backends: `sniper`
- sections: `0`
- arms present: `DROPLET, ECG_PFX, LRU`

## Head-to-head

| benchmark | section | L3 | LRU | DROPLET | ECG_PFX | DROPLET-LRU | ECG-LRU | ECG-DROPLET | DROPLET useful | ECG useful |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bfs | 0 | 1MB | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | None |
| pr | 0 | 1MB | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1.0 | None |
| sssp | 0 | 1MB | 0.959983831851253 | 0.8562669071235347 | 0.9580152671755725 | 0.10371692472771832 | 0.0019685646756805175 | 0.1017483600520378 | 0.8837368661489264 | 1.0 |

## Violations

_None._
