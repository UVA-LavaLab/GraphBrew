# GraphBrew ECG cache experiments

Single-source-of-truth (SSOT), config-driven experiment package for the ECG /
PULSE cache-replacement work. This folder was reduced from 177 ad-hoc scripts to
the small, structured module set below. **Every experiment is defined in
`experiments.json` and run through `experiments.py`** — do not add new top-level
scripts.

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
  ecg.py                eviction-policy equivalence (cache_sim/gem5/Sniper)
  pfx.py                prefetch-path equivalence

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
python3 experiments.py verify          # verify/ecg.py  (eviction equivalence)
python3 experiments.py verify --pfx    # verify/pfx.py  (prefetch equivalence)
python3 experiments.py analyze         # analysis/scale.py (regime map)
```

## Adding an experiment (the only allowed way to grow this folder)

Add a named entry under `"experiments"` in `experiments.json`. An experiment is a
cartesian product expanded into resumable `roi_matrix` cells:

```
graphs x l3_sizes x policies x benchmarks
```

Every field falls back to `"defaults"`; `"graphs": "@eval"` resolves a named
graph set. The headline config (8B epoch, structure prefetcher, iso-area charged
P-OPT) lives in `"defaults"`, so a new experiment only overrides what differs.

If a genuinely new *kind* of experiment is needed, add a thin module under
`flows/` and a dispatch line in `experiments.py` — never a loose top-level script.
