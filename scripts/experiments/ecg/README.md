# `ecg/` — ECG / GrAPL paper

Graph-aware cache replacement policies. Self-contained: config + runner.

| File | Purpose |
|---|---|
| [`config.py`](config.py) | 9 cache policies, 11 reorder×policy pairs, 6 graphs |
| [`runner.py`](runner.py) | 6 experiments (policy comparison, reorder interaction, cache sweep, fat-ID) |
| [`ecg_pfx_scale_proof.py`](ecg_pfx_scale_proof.py) | One-root BFS ECG_PFX scale-proof runner for local or Slurm RISC-V/Sniper shards |

## Run

```bash
python3 scripts/experiments/ecg/runner.py --all --graph-dir results/graphs
python3 scripts/experiments/ecg/runner.py --exp 6                # analytical only
python3 scripts/experiments/ecg/runner.py --exp 1 --preview --dry-run
```

Outputs land in `results/ecg_experiments/`.

## ECG_PFX Scale Proof

Use this helper after the small matched BFS proof is working and before running
larger graph/file-backed matrices:

```bash
python3 scripts/experiments/ecg/ecg_pfx_scale_proof.py \
	--scale 10 \
	--roots 0 \
	--backend both \
	--out-root /tmp/graphbrew-ecg-pfx-scale-g10-r0
```

For larger scales, generate one Slurm row per root/backend and submit with
`scripts/experiments/ecg/slurm_ecg_pfx_scale_proof.sbatch`:

```bash
mkdir -p results/ecg_experiments/slurm
RUN_TAG=ecg_pfx_scale_$(date +%Y%m%d_%H%M%S)
SHARDS=results/ecg_experiments/slurm/${RUN_TAG}_scale.tsv
printf '11\t0\tboth\tresults/ecg_experiments/ecg_pfx_scale_proof/%s_g11_r0\n' "$RUN_TAG" > "$SHARDS"
export SHARDS
sbatch --array=0-0 scripts/experiments/ecg/slurm_ecg_pfx_scale_proof.sbatch
```
