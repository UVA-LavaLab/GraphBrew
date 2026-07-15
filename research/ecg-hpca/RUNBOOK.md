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
LRU SRRIP GRASP POPT ECG:K2 ECG:K2_ONLINE
ECG:K2_STREAMSHIELD ECG:K2_ONLINE_STREAMSHIELD
```

## Reproduce the real-graph cache_sim factorial

First run the bounded five-algorithm diagnostic matrix:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_preliminary_5alg_3sim \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_preliminary_5alg \
  --no-build
```

This runs LRU, SRRIP, GRASP, charged P-OPT, static K2, and online K2 for
PR/BFS/SSSP/BC/CC on the common `kron_s15_k4` cell in cache_sim, gem5, and
Sniper. Compare policy direction and rank **within** each simulator. Do not
compare absolute gem5 and Sniper miss rates. Canonical Schedule-2 reruns use
fused delivery for all five kernels; gem5 O3 remains prohibited until the
request-bound pair extension is complete.

Then run the matched structure-prefetch sensitivity:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_preliminary_5alg_stride \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_preliminary_5alg_stride \
  --no-build
```

STRIDE8 is enabled for every policy. A lower demand miss rate does not imply
lower bandwidth: compare `total_memory_traffic_with_overhead` and prefetch fills
alongside demand misses.

The current Sniper simple-prefetcher implementation does not export a
demand/prefetch NUCA miss split and expands total LLC read misses by 9x--596x
on this diagnostic. Treat its output as a rejected prefetch configuration, not
as demand-miss or speedup evidence.

When rerunning only one simulator or stage, use a distinct `--run-dir`.
`paper_run.py` refuses to replace a broader resolved manifest with an
`--only`/filtered subset. Aggregate shard directories together with
`paper_pipeline.py --input-run-dirs ...`.

First isolate replacement quality and online regret:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_replacement_baseline \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_replacement \
  --no-build
```

This PR/BFS/SSSP/BC/CC stage disables prefetching and uncharges ECG record delivery. It
reports LRU, SRRIP, GRASP, uncharged and charged P-OPT, K1, all five static K2
arms, and `ECG:K2_ONLINE`.

Before launching real graphs, certify all five algorithms:

```bash
python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper \
  --kernels pr bfs sssp bc cc \
  --schedule-k 2
```

BC certification applies K2 to the forward Brandes edge traversal only; its
runtime successor-DAG backward phase is not a static record stream. CC uses the
existing undirected/symmetric graph contract.

Then run the hardware-faithful placement factorial:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_cache_sim_factorial \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_factorial \
  --no-build
```

The factorial includes uncharged and charged P-OPT, K1/K2, StreamShield, and
online K2 with record traffic charged. Use
`--allow-missing-graphs --list --dry-run` to inspect the complete job set before
staging all three graphs.

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

## Run the full-iteration local matrix

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
sbatch --array=0-7 scripts/experiments/ecg/slurm/slurm_final_shard.sbatch
```

## Aggregate

```bash
python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-glob \
    "results/ecg_experiments/final_paper_runs/ecg_replacement" \
    "results/ecg_experiments/final_paper_runs/slurm/ecg_successor_webgoogle/*" \
  --run-root results/ecg_experiments/paper_pipeline/ecg_successor_webgoogle

test -f \
  results/ecg_experiments/paper_pipeline/ecg_successor_webgoogle/aggregate/online_dueling_regret.csv
```

The replacement profile emits
`aggregate/online_dueling_regret.csv`, which reports online K2's delta from the
best static arm using total LLC misses, plus a separate property-miss diagnostic
and deltas from uncharged and overhead-aware charged P-OPT.

## Correctness gates

```bash
python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs sssp bc cc --schedule-k 2

python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr --schedule-k 2 --stream-bypass
```
