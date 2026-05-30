# Slurm SBATCH schema registry — gate 252

Status: `active`

Totals: files=9  canonical=14  required=7  violations=0

## Required directives

- `--cpus-per-task` — OMP thread count
- `--job-name` — human-readable label
- `--mem` — per-node memory; G|M
- `--nodes` — node count
- `--ntasks` — MPI task count
- `--output` — stdout log template
- `--time` — HH:MM:SS or D-HH:MM:SS

## Files

| File | directive_count | missing_required |
|------|-----------------|------------------|
| scripts/experiments/ecg/slurm_ecg_pfx_scale_proof.sbatch | 8 | — |
| scripts/experiments/ecg/slurm_final_shard.sbatch | 9 | — |
| scripts/experiments/vldb/slurm/monolithic.sbatch | 10 | — |
| scripts/experiments/vldb/stages/slurm/01_prep.sbatch | 8 | — |
| scripts/experiments/vldb/stages/slurm/02_reorder.sbatch | 8 | — |
| scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch | 8 | — |
| scripts/experiments/vldb/stages/slurm/04_cache_sim.sbatch | 8 | — |
| scripts/experiments/vldb/stages/slurm/05_aggregate.sbatch | 8 | — |
| scripts/experiments/vldb/stages/slurm/smoke.sbatch | 8 | — |
