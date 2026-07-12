# Reproduction Runbook

## Required graph

The correctness gates and headline profile expect:

```text
results/graphs/email-Eu-core/email-Eu-core.sg
results/graphs/web-Google/web-Google.sg
results/graphs/soc-pokec/soc-pokec.sg
results/graphs/cit-Patents/cit-Patents.sg
```

Graph datasets and converted `.sg` files are ignored. Build the converter with
`make converter` when staging a new graph. One reproducible SNAP staging recipe:

```bash
mkdir -p \
  results/graphs/email-Eu-core \
  results/graphs/web-Google \
  results/graphs/soc-pokec \
  results/graphs/cit-Patents

curl -L https://snap.stanford.edu/data/email-Eu-core.txt.gz |
  gzip -dc > results/graphs/email-Eu-core/email-Eu-core.el
curl -L https://snap.stanford.edu/data/web-Google.txt.gz |
  gzip -dc > results/graphs/web-Google/web-Google.el
curl -L https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz |
  gzip -dc > results/graphs/soc-pokec/soc-pokec.el
curl -L https://snap.stanford.edu/data/cit-Patents.txt.gz |
  gzip -dc > results/graphs/cit-Patents/cit-Patents.el

make converter
bench/bin/converter \
  -f results/graphs/email-Eu-core/email-Eu-core.el \
  -b results/graphs/email-Eu-core/email-Eu-core.sg
bench/bin/converter \
  -f results/graphs/web-Google/web-Google.el \
  -b results/graphs/web-Google/web-Google.sg
bench/bin/converter \
  -f results/graphs/soc-pokec/soc-pokec.el \
  -b results/graphs/soc-pokec/soc-pokec.sg
bench/bin/converter \
  -f results/graphs/cit-Patents/cit-Patents.el \
  -b results/graphs/cit-Patents/cit-Patents.sg
```

## Build correctness-gate binaries

```bash
make sim-pr sim-bfs
make setup-gem5
make gem5-riscv-m5ops-pr gem5-riscv-m5ops-bfs
make setup-sniper
make sniper-sg_kernel
```

## Validate the resolved paper job

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile streamshield_sniper_realgraph \
  --run-dir /tmp/ecg-successor-webgoogle-dryrun \
  --list --dry-run --no-build
```

The command must contain exactly:

```text
LRU SRRIP GRASP POPT ECG:K2 ECG:K2_STREAMSHIELD
```

## Reproduce the real-graph cache_sim factorial

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_cache_sim_factorial \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_factorial \
  --no-build
```

The factorial adds `ECG:K1` and `ECG:K1_STREAMSHIELD` to the full baseline
set. Use `--allow-missing-graphs --list --dry-run` to inspect the complete job
set before staging all three graphs.

## Reproduce the detailed-simulator mechanism cells

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile gem5_streamshield_mechanism \
  --run-dir results/ecg_experiments/final_paper_runs/gem5_mechanism \
  --no-build

python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile sniper_streamshield_mechanism \
  --run-dir results/ecg_experiments/final_paper_runs/sniper_mechanism \
  --no-build
```

## Run the bounded local matrix

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile streamshield_sniper_realgraph \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_successor_webgoogle \
  --no-build
```

## Generate one-policy Slurm shards

```bash
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
mkdir -p results/slurm_logs results/ecg_experiments/slurm

python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile streamshield_sniper_realgraph \
  --run-tag ecg_successor_webgoogle \
  --out results/ecg_experiments/slurm/ecg_successor_webgoogle.tsv
```

Submit on a configured cluster:

```bash
SHARDS=results/ecg_experiments/slurm/ecg_successor_webgoogle.tsv \
sbatch --array=0-5 scripts/experiments/ecg/slurm/slurm_final_shard.sbatch
```

## Aggregate

```bash
python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-glob \
    "results/ecg_experiments/final_paper_runs/slurm/ecg_successor_webgoogle/*" \
  --run-root results/ecg_experiments/paper_pipeline/ecg_successor_webgoogle
```

## Correctness gates

```bash
python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs --schedule-k 2

python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr --schedule-k 2 --stream-bypass
```
