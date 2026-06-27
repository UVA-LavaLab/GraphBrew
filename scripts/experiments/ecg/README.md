# GraphBrew ECG cache experiments

Single-source-of-truth (SSOT), config-driven experiment package for the ECG /
PULSE cache-replacement work. This folder was reduced from 177 ad-hoc scripts to
the small, structured module set below. **Every experiment is defined in
`experiments.json` and run through `experiments.py`** — do not add new top-level
scripts.

## The story this testbed proves (three axes)

ECG unifies three prior, separate graph-cache mechanisms into ONE per-edge
*epoch* carried in the edge stream (`ECG_EXTRACT`), reserving zero cache ways:

| axis | prior mechanism | ECG's claim | experiment |
|------|-----------------|-------------|------------|
| **A1 replacement vs GRASP** | degree-based retention | epoch encodes degree priority; ECG >= GRASP, wins on community/mesh | `headline-scale`, `multi-kernel` |
| **A1 replacement vs P-OPT** | next-reference matrix in reserved ways | epoch encodes next-ref order at **0 ways**; ECG >= P-OPT iso-area | `headline-scale`, `prefetch-off-control` |
| **A2 prefetch vs DROPLET** | structure-aware prefetch engine | the SAME epoch filters edge-stream lookahead | `prefetch-axis` |
| **A3 combined** | GRASP/P-OPT + DROPLET (two engines) | one epoch drives both replacement + prefetch | `combined-axis` |

*Carry the epoch, not the matrix -- and not a second prefetcher.* Honest caveat:
on the prefetch axis DROPLET is latency-optimal (cuts demand most) while ECG_PFX
is efficiency-optimal (~3x fewer fills, all useful) -- a Pareto trade, not a rout.

## Layout

```
experiments.py      ENTRY POINT — the orchestrator (list | show | run | verify | analyze)
experiments.json    SSOT CONFIG — defaults + graph_sets + named experiments
roi_matrix.py       THE ENGINE — the one driver that runs a cell on cache_sim | gem5 | Sniper

flows/              experiment generators (produce raw data; each wraps roi_matrix)
  scale_sweep.py        cache_sim pressure x graph x policy sweep
  eviction_matrix.py    ECG variant eviction matrix
  proof_matrix.py       component-ablation proof matrix
  paper_run.py          manifest-driven paper run
  paper_pipeline.py     one-command paper pipeline

verify/             correctness gates (3-sim equivalence)
  ecg.py                eviction-policy spec-compliance (cache_sim/gem5/Sniper) + runs equiv.py gates
  pfx.py                prefetch-path equivalence
  equiv.py              BEHAVIORAL cross-sim equivalence + reorder guard + insertion-RRPV invariant

analysis/           turn results into tables/figures
  scale.py              aggregate the scale sweep (regime map)
  parity.py             cross-sim miss-rate join
  coverage.py           headline coverage report
  anchor_summary.py     summarize a gem5/Sniper anchor sweep-root

lib/                shared library modules (imported, never run directly)
  literature_baselines.py / literature_faithfulness.py / literature_preflight.py

sweeps/             bash sweep drivers (cross-sim 1MB anchors, PFX sweeps, ...)
slurm/              cluster sbatch templates
```

## Running experiments

```bash
# what is defined
python3 experiments.py list

# the resolved cells of one experiment (no run)
python3 experiments.py show headline-scale

# run it (resumable: re-running skips finished cells)
python3 experiments.py run headline-scale
python3 experiments.py run headline-scale --dry-run          # print commands only
python3 experiments.py run headline-scale --only cit-Patents # filter cells

# correctness + analysis
python3 experiments.py verify          # spec-compliance + behavioral equivalence + insertion invariant
python3 experiments.py verify --equiv  # ONLY the behavioral equivalence/insertion gate (fast, cache_sim)
python3 experiments.py verify --pfx    # verify/pfx.py  (prefetch equivalence)
python3 experiments.py analyze         # analysis/scale.py (regime map)
# deep cross-sim behavioral equivalence (slow; pressured kron cell):
python3 verify/equiv.py --gem5         #   add gem5;  --sniper add Sniper
```

### What the equivalence gate guards (and why spec-compliance alone did not)
`verify/ecg.py` asserts every *eviction* obeys its policy spec. That is necessary
but cannot catch: a backwards *insertion* RRPV (gem5 once inserted non-property
data near-MRU → GRASP backfired), an *unreordered* workload (Sniper's sg_kernel
once ignored `-o`), or cross-sim *direction* disagreement. `verify/equiv.py` adds
behavioral gates on a **pressured** cell (property > L3) where all three sims must
agree GRASP/ECG help, plus a static invariant that non-property inserts stay
SRRIP-distant. These run automatically at the end of `experiments.py verify`.


## Adding an experiment (the only allowed way to grow this folder)

Add a named entry under `"experiments"` in `experiments.json`. An experiment is a
cartesian product expanded into resumable `roi_matrix` cells:

```
graphs x l3_sizes x policies x benchmarks x prefetchers
```

Every field falls back to `"defaults"`; `"graphs": "@eval"` resolves a named
graph set; `"prefetchers"` defaults to `["none"]` (replacement axes) and is set
to `["none", "DROPLET", "ECG_PFX"]` for the prefetch/combined axes. The headline
config (8B epoch, structure prefetcher, iso-area charged P-OPT) lives in
`"defaults"`, so a new experiment only overrides what differs.

If a genuinely new *kind* of experiment is needed, add a thin module under
`flows/` and a dispatch line in `experiments.py` — never a loose top-level script.
