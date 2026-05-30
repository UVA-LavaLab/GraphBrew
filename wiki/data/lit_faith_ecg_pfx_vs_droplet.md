# ECG PFX vs DROPLET head-to-head

Gate 241 — ECG-Pfx-vs-DROPLET. Sibling family to the substrate-
parity trinity (gates 238/239/240) but evaluates a different
axis: ECG's PFX prefetcher vs DROPLET on the same baseline.

**Status:** `deferred`

## Deferred

- defer reason: No runtime prefetcher activity is present in any /tmp/graphbrew-*/roi_matrix.csv corpus today. Audit of all droplet_* and ecg_pfx_* counter columns across {gem5 bracket sweep, sniper grasp sweep, popt geometry matches, dbg ordering proofs} shows only configuration columns (droplet_indirect_degree, droplet_prefetch_degree, droplet_stride_table_size, ecg_pfx_delivery, ecg_pfx_hint_filter) populated and identical across rows; runtime counters (droplet_indirect_issued, droplet_stride_issued, droplet_edge_accesses, ecg_pfx_issued, ecg_pfx_target_hints_seen, etc.) are zero everywhere. Either prefetcher was activated for any of the existing matched sweeps. A future matched-proof comparison sweep with two arms (prefetcher=ecg_pfx, prefetcher=droplet) on the same baseline will activate this gate.
- expected source pattern: `/tmp/graphbrew-ecg-vs-droplet-matched-proof-*/{graph}-{app}/roi_matrix.csv`
- expected minimum observations: 8
- expected required arms: `LRU, DROPLET, ECG_PFX`

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
- observations: **0**
- cells (benchmark × section × L3): **0**
- benchmarks: ``
- backends: ``
- sections: ``
- arms present: ``

## Head-to-head

_No head-to-head rows (deferred or empty)._

## Violations

_None._
