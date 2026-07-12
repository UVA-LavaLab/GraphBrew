# HPCA Mode 6 — Experimental Arm Naming Convention

**Status:** Adopted 2026-06-07 (post-sprint-6f-7 cleanup)
**Applies to:** All HPCA mode 6 evaluation artifacts (manifest stages, output
directories, CSV columns, paper tables, finding docs)

## The problem this solves

Sprint 6f-1..6f-7 accumulated arm names like `mode 6 amp=1 CHARGED=0` which
require deep ECG-codebase knowledge to interpret. Reviewers cannot tell
`mode 2` from `mode 6` from looking at a paper table. This convention
replaces opaque integer modes with a structured, decomposed name.

## The three-axis name

Every prefetcher arm has a three-segment name:

```
{selection}__{delivery}__{bandwidth}
```

(Double underscore separator; lowercase; no spaces.)

### Axis 1: SELECTION — how is the prefetch target chosen?

| token | meaning | maps to legacy |
|---|---|---|
| `none` | no prefetching (baseline arm) | mode 0 |
| `seq` | sequential next-K destinations; no selection | mode 3 (DROPLET-style) |
| `degree` | runtime degree-ranked pick from a lookahead window | mode 1 |
| `popt_rt` | runtime POPT-ranked pick from a lookahead window | mode 2 |
| `popt_off` | offline POPT-ranked target encoded per-edge in fat-mask | **mode 6 (paper headline)** |
| `popt_off_xi` | offline POPT, cross-iteration variant | mode 7 |
| `farfuture` | runtime pick from global hot-table | mode 4 |

### Axis 2: DELIVERY — how does the prefetcher get its metadata?

| token | meaning | maps to legacy |
|---|---|---|
| `_` | not applicable (baseline) | — |
| `hw` | pure hardware (stride detector, no software metadata) | (paper-faithful DROPLET ideal) |
| `sw` | software-resident: mask/metadata read from a memory array per edge | CHARGED=1 |
| `isa` | ISA-extension: payload arrives via custom-0 / register, zero memory cost | CHARGED=0 (gem5's `ecg_extract`) |
| `oracle` | simulator magic — pure upper bound, no architecturally-defined mechanism | (cache_sim's all-level free fill) |

### Axis 3: BANDWIDTH — how many prefetches per edge does this arm issue?

| token | meaning | maps to legacy |
|---|---|---|
| `k0` | 0 prefetches (baseline) | — |
| `k1` | 1 prefetch per edge | LH=1 / amp=0 |
| `k2` | 2 prefetches per edge (1 selected + 1 sequential) | amp=1 |
| `k4` | 4 prefetches per edge | LH=4 / amp=3 |
| `k8` | 8 prefetches per edge | **LH=8 (DROPLET default in our cache_sim)** |
| `k16` | 16 prefetches per edge | **LH=16 (DROPLET paper default)** |
| `k32`, `k64` | sweep stress points | — |

### Composite name examples

| legacy notation | new name | meaning |
|---|---|---|
| baseline | `baseline` (alias for `none___k0`) | no prefetcher |
| DROPLET LH=8 (ours) | `seq__sw__k8` | sequential next-K, software-resident CSR, 8 prefetches/edge |
| DROPLET LH=16 (paper default) | `seq__sw__k16` | same, paper-default lookahead |
| mode 2 K=1 LH=8 | `popt_rt__sw__k1` | runtime POPT-ranked, software, 1 best target |
| mode 6 amp=0 CHARGED=1 (negative) | `popt_off__sw__k1` | offline POPT, software-loaded mask, 1 target |
| mode 6 amp=1 CHARGED=1 (negative) | `popt_off__sw__k2` | offline POPT, software, 1 selected + 1 sequential |
| mode 6 amp=0 CHARGED=0 | `popt_off__isa__k1` | offline POPT, ISA-delivered, 1 target |
| **mode 6 amp=1 CHARGED=0 (HEADLINE)** | **`popt_off__isa__k2`** | **offline POPT, ISA-delivered, 2 prefetches/edge** |
| mode 6 amp=1 oracle | `popt_off__oracle__k2` | (upper bound — same as ISA in cache_sim) |

## Eviction policy is SEPARATE

The prefetcher name describes only the prefetcher mechanism. Eviction policy
is recorded as a separate field. Standard eviction labels:

| eviction token | meaning |
|---|---|
| `lru` | classical LRU (baseline) |
| `srrip` | static RRIP (HPCA standard) |
| `drrip` | dynamic RRIP (P-OPT uses this) |
| `grasp` | GRASP pin-and-protect ABR |
| `popt` | P-OPT (Belady-MIN approximation from rereference matrix) |
| `ecg_dbg` | ECG eviction using DBG ranks only |
| `ecg_popt` | ECG eviction with POPT-primary signal |

## Output directory structure

```
results/ecg_experiments/hpca_mode6/<run_id>/
  matrices/<phase>/<eviction>/<prefetcher_name>/<graph>/<algorithm>/
    roi_matrix.csv
    roi_matrix.json
    logs/
```

Example:
```
results/ecg_experiments/hpca_mode6/run_v1/
  matrices/phase_3/ecg_dbg/popt_off__isa__k2/com-orkut/pr/
    roi_matrix.csv      ← the HEADLINE result for com-orkut
```

## CSV columns to add for self-describing rows

Every row in `roi_matrix.csv` should record:

| column | example value |
|---|---|
| `arm_selection` | `popt_off` |
| `arm_delivery` | `isa` |
| `arm_bandwidth_k` | `2` |
| `arm_eviction` | `ecg_dbg` |
| `arm_short_name` | `popt_off__isa__k2` |
| (existing) `prefetcher` | `ECG_PFX` |
| (existing) `policy` | `ECG:DBG_ONLY` |

This way the CSV is self-explanatory; downstream table emitters can group
by `arm_short_name` directly.

## Legend section for paper tables

Paper tables that use these names must include a one-line legend below the
table, e.g.:

> Arm naming: `{selection}__{delivery}__{bandwidth_k}`. Selection: `none`,
> `seq` (sequential, no selection), `popt_rt` (runtime POPT-ranked),
> `popt_off` (offline POPT per-edge mask). Delivery: `sw` (software memory
> load), `isa` (ISA-extension free). Bandwidth: `kN` = N prefetches/edge.
> See `research/ecg-hpca/evidence/hpca_naming_convention_v1.md` for full conventions.

## Migration: legacy → new names cheatsheet

For anyone reading old finding docs / sprint 6f-7 commits:

| old | new |
|---|---|
| "mode 0 / none" | `baseline` |
| "mode 1" | `degree__sw__kN` |
| "mode 2" | `popt_rt__sw__k1` (K=1 default) |
| "mode 3" / "DROPLET-style" | `seq__sw__kN` (N = lookahead) |
| "mode 6 CHARGED=1 amp=0" | `popt_off__sw__k1` |
| "mode 6 CHARGED=1 amp=1" | `popt_off__sw__k2` |
| "mode 6 CHARGED=0 amp=0" | `popt_off__isa__k1` |
| **"mode 6 CHARGED=0 amp=1"** | **`popt_off__isa__k2` (paper headline)** |
| "mode 7" / "cross-iter" | `popt_off_xi__{delivery}__{bandwidth}` |
| "mode 4" / "far-future" | `farfuture__sw__kN` |
| "POPT lookahead K=1 LH=8" | `popt_rt__sw__k1` |
| "POPT lookahead K=4 LH=8" | `popt_rt__sw__k4` |

## What this does NOT cover (yet)

These remain free-form parameters per arm (recorded as CSV columns, not
in the name):

- Lookahead window size (`ECG_EDGE_MASK_LOOKAHEAD`, separate from `bandwidth_k`)
- Dedup window (`ECG_PFX_KERNEL_DEDUP`)
- DROPLET specific knobs (`droplet_prefetch_degree`, `droplet_indirect_degree`,
  `droplet_stride_table_size`)
- Cache hierarchy (recorded as separate columns: `l1d_size`, `l2_size`,
  `l3_size`, `prefetcher_level`)
- Graph + algorithm (recorded as `graph`, `benchmark` columns)

For a full reproducibility record, every CSV row carries: arm name +
eviction + lookahead + dedup + cache config + graph + benchmark + git hash.
