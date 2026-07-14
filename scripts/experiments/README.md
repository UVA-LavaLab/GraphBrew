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
| Run the adaptive-ML ablation             | `adaptive_ml/exp3_model_ablation.py` |

## Quick start (recommended path)

```bash
source .venv/bin/activate
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --preview
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview
```

See [vldb/stages/README.md](vldb/stages/README.md) for the full stage doc
and [scripts/README.md](../README.md) for the canonical paths table.
