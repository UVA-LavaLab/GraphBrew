# ECG substrate-parity audit (Sniper)

Gate 240 — ECG-Sniper-Parity. Sibling to gate 238 (cache_sim)
and gate 239 (gem5). Locks the POPT-arm (and optionally DBG-arm)
substrate faithfulness under Sniper. PFX activation and DROPLET
comparison remain out of scope here.

**Status:** `deferred`

## Deferred

- defer reason: No ECG_DBG_ONLY or ECG_POPT_PRIMARY rows present in any Sniper roi_matrix.csv under /tmp/graphbrew-*sniper*. The existing /tmp/graphbrew-grasp-sniper-sweep provides LRU/SRRIP/GRASP across {pr, bfs, sssp, bc} × {cit-Patents, email-Eu-core} × {4kB, 32kB, 256kB, 2MB} L3 sizes — useful for the sniper_anchor and sniper_slope_replay gates but not for ECG substrate-parity. A future ECG-enabled Sniper sweep (target location: /tmp/graphbrew-ecg-sniper-matched-proof-*) will populate this fixture.
- expected source pattern: `/tmp/graphbrew-ecg-sniper-matched-proof-*/{graph}-{app}/roi_matrix.csv`
- expected minimum observations: 6
- expected POPT-arm policies: `LRU, POPT, ECG_POPT_PRIMARY`
- expected DBG-arm policies: `GRASP, ECG_DBG_ONLY`

## Rules
- **G1** — every required POPT-arm policy present with status=ok per (benchmark, section, l3_size) cell
- **G1b** — if ECG_DBG_ONLY present on a cell, GRASP must also be present
- **G2** — |miss_rate(ECG_POPT_PRIMARY) - miss_rate(POPT)| <= 0.002
- **G2b** — if both GRASP and ECG_DBG_ONLY present, |miss_rate(ECG_DBG_ONLY) - miss_rate(GRASP)| <= 0.002
- **G3** — every row has backend=sniper AND simulator=sniper (no silent cache_sim/gem5 ingestion)
- **G4** — ipc > 0.0 AND instructions >= 1 on every row
- **G5** — LRU baseline has strictly positive l3_accesses and l3_misses on every cell
- **G6** — l3_misses <= l3_accesses and l3_miss_rate in [0,1] on every row
- **G7** — len(per_observation) >= postfix.expected_minimum_observations

## Constants
- ε(POPT parity): `0.002`
- ε(DBG parity): `0.002`
- IPC floor: `> 0.0`
- instructions floor: `>= 1`
- POPT-arm required: `ECG_POPT_PRIMARY, LRU, POPT`
- DBG-arm optional: `ECG_DBG_ONLY, GRASP`
- baseline policies: `LRU`

## Totals
- observations: **0**
- cells (benchmark × section × L3): **0**
- benchmarks: ``
- backends: ``
- sections: ``
- policies present: ``

## POPT parity (Sniper)

_No POPT-arm rows present (deferred or empty)._

## DBG parity (Sniper)

_No DBG-arm rows present (deferred or DBG arm not yet curated)._

## Violations

_None._
