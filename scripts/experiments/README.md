# `scripts/experiments/` — paper experiment runners

Each paper / research thread has its OWN folder. No mixing.

```
experiments/
├── vldb/                     ★ VLDB 2026 — single source of truth for VLDB
│   ├── config.py              graph sets, baselines, COMPOSE_VARIANTS, BENCHMARKS, trial counts
│   ├── runner.py              monolithic legacy all-in-one runner
│   ├── figures.py             LaTeX + PNG emitter (called by stage 05)
│   ├── stages/                ★ RECOMMENDED: 5 independent stage runners
│   │   ├── 01_prep.py          download + .mtx → .sg     [needs internet]
│   │   ├── 02_reorder.py       pre-generate .lo cache    [CPU]
│   │   ├── 03_cpu_perf.py      wall-clock kernel sweep   [real CPU]
│   │   ├── 04_cache_sim.py     cache simulator           [host CPU irrelevant]
│   │   ├── 05_aggregate.py     JSON → tables/figures
│   │   ├── slurm/*.sbatch      one sbatch per stage
│   │   └── README.md
│   ├── experiments/           per-experiment recipe READMEs
│   │   ├── exp1_cache/README.md
│   │   ├── exp2_speedup/README.md
│   │   ├── exp3_overhead/README.md
│   │   ├── exp4_endtoend/README.md
│   │   ├── exp5_ablation/README.md
│   │   ├── exp6_sensitivity/README.md
│   │   ├── exp7_chained/README.md
│   │   └── exp8_scalability/README.md
│   └── slurm/
│       └── monolithic.sbatch  legacy SLURM (used by wiki docs)
│
├── ecg/                      ★ ECG / GrAPL paper — cache replacement policies
│   ├── config.py
│   └── runner.py
│
├── adaptive_ml/              ★ adaptive-ordering ML model work
│   └── exp3_model_ablation.py
│
└── legacy/                   archived; no live imports
```

## How to find things

| You want to… | Go to |
|---|---|
| Run a VLDB experiment, stage-by-stage      | `vldb/stages/0[1-5]_*.py`   |
| Run a VLDB experiment, monolithic         | `vldb/runner.py --exp N --local` |
| Change the canonical VLDB config         | `vldb/config.py` |
| Read what a given VLDB experiment does   | `vldb/experiments/exp<N>_*/README.md` |
| Submit a VLDB SLURM job (per-stage)      | `vldb/stages/slurm/0[1-5]_*.sbatch` |
| Submit a VLDB SLURM job (monolithic)     | `vldb/slurm/monolithic.sbatch` |
| Run the ECG paper                        | `ecg/runner.py` |
| Run ECG cache_sim/gem5 ROI parity matrix | `ecg/roi_matrix.py` |
| Run the adaptive-ML ablation             | `adaptive_ml/exp3_model_ablation.py` |

## ECG ROI Parity Runner

Use `ecg/roi_matrix.py` for the step-by-step cache_sim to gem5 validation path.
It keeps L1/L2 on LRU and varies the L3 policy in both simulators, so the fast
cache simulator and gem5 are comparing the same policy scope.

```bash
python3 scripts/experiments/ecg/roi_matrix.py --dry-run
python3 scripts/experiments/ecg/roi_matrix.py --suite cache-sim --policies LRU GRASP POPT ECG:POPT_PRIMARY
python3 scripts/experiments/ecg/roi_matrix.py --suite gem5 --policies LRU GRASP POPT ECG:POPT_PRIMARY
```

The final replacement baseline policy set is:
`LRU`, `SRRIP`, `GRASP`, `POPT_CHARGED`, `POPT`, `ECG_DBG_ONLY`,
`ECG_DBG_PRIMARY_CHARGED`, `ECG_DBG_PRIMARY`, and `ECG_POPT_PRIMARY`.
`*_CHARGED` rows use the same dynamic P-OPT lookup but reduce effective L3
capacity by the reserved ways needed for current+next rereference matrix
columns. They also report estimated matrix-stream traffic fields in the output
CSV. The DROPLET baseline set is: `LRU+DROPLET`, `GRASP+DROPLET`,
`POPT_CHARGED+DROPLET`, `POPT+DROPLET`, `ECG_DBG_PRIMARY_CHARGED+DROPLET`,
`ECG_DBG_PRIMARY+DROPLET`, and `ECG_POPT_PRIMARY+DROPLET`.

## ECG Final Paper Run Orchestrator

Use `ecg/final_paper_run.py` for multi-hour or multi-day final paper matrices.
It expands the checked-in JSON manifest, writes a run snapshot, logs every job,
and resumes by skipping jobs whose output CSV already contains only `ok` rows.
The runner is sequential by design because GraphBrew gem5 uses shared runtime
sideband files under `/tmp`. Any `final_*` profile also checks that the focused
faithfulness CSVs for GRASP, P-OPT, and DROPLET exist and contain only `ok` rows
before launching expensive jobs.
Completed jobs are also merged into `combined_roi_matrix.csv` and/or
`combined_proof_matrix.csv` at the run directory root.

```bash
python3 scripts/experiments/ecg/final_paper_run.py --profile rehearsal --dry-run
python3 scripts/experiments/ecg/final_paper_run.py --profile rehearsal
python3 scripts/experiments/ecg/final_paper_run.py --profile final_replacement --list --dry-run --allow-missing-graphs
python3 scripts/experiments/ecg/final_paper_run.py --profile final_replacement --check-graphs --allow-missing-graphs
python3 scripts/experiments/ecg/final_paper_run.py --status --run-dir results/ecg_experiments/final_paper_runs/replacement_final
python3 scripts/experiments/ecg/final_paper_run.py --profile available_replacement --list --dry-run
python3 scripts/experiments/ecg/final_paper_run.py --profile final_droplet --run-dir results/ecg_experiments/final_paper_runs/droplet_final
python3 scripts/experiments/ecg/final_paper_run.py --profile final_replacement --run-dir results/ecg_experiments/final_paper_runs/replacement_final --from-job soc-LiveJournal1
python3 scripts/experiments/ecg/final_paper_run.py --profile final_replacement --graph soc-pokec --benchmark pr --policy LRU --list --dry-run --allow-missing-graphs --skip-validation-gate
```

For Slurm, split with `--graph`, `--benchmark`, and `--policy`, run each shard in
its own run directory, then aggregate with `paper_pipeline.py --skip-run
--input-run-glob "results/ecg_experiments/final_paper_runs/slurm/<tag>/*"`.
See `wiki/ECG-Slurm-Runs.md`.

For a one-command workflow that launches selected profiles, aggregates CSVs, and
generates paper figures/tables, use `ecg/paper_pipeline.py`. The default figure
set is SVG-first and reports speedup, LLC miss reduction, memory-traffic
reduction, and charged-overhead metrics rather than raw tick/miss bars:

```bash
python3 scripts/experiments/ecg/paper_pipeline.py --profiles rehearsal --dry-run
python3 scripts/experiments/ecg/paper_pipeline.py --profiles available_replacement --run-root results/ecg_experiments/paper_pipeline/available_001
python3 scripts/experiments/ecg/paper_pipeline.py --skip-run --input-csv results/ecg_experiments/roi_matrix/popt_charged_gem5_smoke/roi_matrix.csv
python3 scripts/experiments/ecg/paper_pipeline.py --skip-run --input-run-glob "results/ecg_experiments/final_paper_runs/slurm/<tag>/*"
```

Slurm template: `scripts/experiments/ecg/slurm_final_shard.sbatch`.

Default profiles in `ecg/final_paper_manifest.json`:
- `rehearsal`: synthetic g12 component proof, replacement gem5, and DROPLET
  actual-edge smokes before starting expensive jobs.
- `available_replacement`: replacement-only gem5 sweep on graph files currently
  present under `results/graphs`.
- `final_cache_sim`: large-graph cache simulator replacement-only sweep.
- `final_replacement`: large-graph gem5 replacement-only sweep for PR/BFS/SSSP.
- `final_droplet`: large-graph gem5 DROPLET baseline sweep after non-PR
  DROPLET smokes pass.

## Quick start (recommended path)

```bash
source .venv/bin/activate
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --preview
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview
```

See [vldb/stages/README.md](vldb/stages/README.md) for the full stage doc
and [scripts/README.md](../README.md) for the canonical paths table.
