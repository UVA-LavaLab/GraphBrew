# `vldb/` — VLDB 2026 paper

Single source of truth for the VLDB paper. Everything VLDB-related lives
under this folder.

| File / dir | Purpose |
|---|---|
| [`config.py`](config.py)       | ★ Graph sets, baselines, COMPOSE_VARIANTS, benchmark list, trial counts. ONE source of truth. |
| [`runner.py`](runner.py)       | Monolithic all-in-one runner (legacy; still works) |
| [`figures.py`](figures.py)     | LaTeX + PNG emitter (called by stage 05) |
| [`stages/`](stages/README.md)  | ★ Recommended: 5 independent stage runners + per-stage SLURM templates |
| [`experiments/`](experiments/) | Per-experiment recipe READMEs (`exp1_cache/`, ..., `exp8_scalability/`) |
| [`slurm/monolithic.sbatch`](slurm/monolithic.sbatch) | Legacy single-job SLURM template |

## Recommended workflow (stage-based)

```bash
source .venv/bin/activate
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --preview
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --preview  # optional
python3 scripts/experiments/vldb/stages/05_aggregate.py --exp 0           # optional
```

## Monolithic workflow (legacy)

```bash
python3 scripts/experiments/vldb/runner.py --all --local
python3 scripts/experiments/vldb/runner.py --exp 2 --preview
```

## Outputs

| Stage | Path |
|---|---|
| 01 | `results/graphs/<name>/<name>.{sg,mtx,el}` |
| 02 | `results/vldb_mappings/<graph>/<algo_key>.{lo,time}` |
| 03 | `results/vldb_paper/exp<N>_*/...json` |
| 04 | `results/vldb_paper/exp1_cache/cache_results.json` |
| 05 | `paper/figures/`, `paper/dataCharts/` |
