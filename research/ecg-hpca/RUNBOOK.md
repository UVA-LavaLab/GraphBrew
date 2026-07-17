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
make setup-gem5
make setup-sniper
make all-sim
make gem5-riscv-m5ops-pr gem5-riscv-m5ops-bfs \
  gem5-riscv-m5ops-sssp gem5-riscv-m5ops-bc gem5-riscv-m5ops-cc
make sniper-sg_kernel
```

## Final run order

1. Run the three correctness gates.
2. Run and validate `ecg_3sim_allalg_smoke`.
3. Run `ecg_replacement_baseline`.
4. Run `ecg_cache_sim_factorial`.
5. Run `ecg_streamshield_generality` as the placement ablation.
6. Run gem5 and Sniper mechanism profiles.
7. Aggregate only complete, hash-consistent runs.
8. Run the blocked Sniper headline profile only after prefetch calibration.

## Full 3-simulator/all-algorithm smoke

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_3sim_allalg_smoke \
  --run-tag ecg_3sim_smoke \
  --out results/ecg_experiments/slurm/ecg_3sim_smoke.tsv

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_3sim_smoke.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 8 --cache-sim-jobs 5 --gem5-jobs 1 --sniper-jobs 1

python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-glob \
    "results/ecg_experiments/final_paper_runs/local/ecg_3sim_smoke/*" \
  --run-root results/ecg_experiments/paper_pipeline/ecg_3sim_smoke

python3 scripts/experiments/ecg/verify/smoke_coverage.py \
  --csv results/ecg_experiments/paper_pipeline/ecg_3sim_smoke/aggregate/roi_matrix_all.csv
```

Acceptance is exactly 120 valid rows: 3 simulators x 5 algorithms x 8 policies.

## Three-real-graph cross-simulator matrix

This no-prefetch comparison runs web-Google, soc-pokec, and cit-Patents across
cache_sim, gem5, and Sniper for PR/BFS/SSSP/BC/CC and the eight final policies.

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_3sim_realgraph_allalg \
  --run-tag ecg_3sim_realgraph_allalg \
  --out results/ecg_experiments/slurm/ecg_3sim_realgraph_allalg.tsv

python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_3sim_realgraph_allalg \
  --only 24_sniper --benchmark pr --policy LRU \
  --run-dir results/ecg_experiments/final_paper_runs/sniper_realgraph_calibration \
  --no-build

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_3sim_realgraph_allalg.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 9 --cache-sim-jobs 4 --gem5-jobs 4 --sniper-jobs 1

python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-glob \
    "results/ecg_experiments/final_paper_runs/local/ecg_3sim_realgraph_allalg/*" \
  --run-root results/ecg_experiments/paper_pipeline/ecg_3sim_realgraph_allalg

python3 scripts/experiments/ecg/verify/smoke_coverage.py \
  --csv \
    results/ecg_experiments/paper_pipeline/ecg_3sim_realgraph_allalg/aggregate/roi_matrix_all.csv \
  --graph web-Google soc-pokec cit-Patents
```

Acceptance is exactly 360 valid rows. Use
`aggregate/roi_relative_metrics.csv` for within-simulator LRU-normalized
miss/timing comparisons; do not compare absolute miss rates across simulators.

The calibration command must complete three rows before the full launch. The
single-node defaults above are for this 32-core/62-GiB host: four gem5 jobs are
safe because every shard has isolated sidebands, while Sniper remains at one
job under its 20-GiB address-space cap. Reduce concurrency if memory pressure
appears. BC K2 covers the forward Brandes traversal only; CC retains the
artifact's undirected/symmetric graph contract.

### Quick 1B-instruction diagnostic

Use the already-complete full-work cache_sim rows and rerun only gem5/Sniper
with a one-billion-instruction detailed-ROI cap:

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_3sim_realgraph_allalg_1b \
  --run-tag ecg_3sim_realgraph_allalg_1b \
  --out results/ecg_experiments/slurm/ecg_3sim_realgraph_allalg_1b.tsv

awk -F '\t' '$2 ~ /^25_|^26_/' \
  results/ecg_experiments/slurm/ecg_3sim_realgraph_allalg_1b.tsv \
  > results/ecg_experiments/slurm/ecg_3sim_realgraph_detailed_1b.tsv

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_3sim_realgraph_detailed_1b.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 8 --gem5-jobs 4 --sniper-jobs 4
```

Gem5 schedules the cap from the compute ROI work-begin marker, so graph loading
does not consume the one-billion-instruction budget. Capped rows set
`timing_valid_for_speedup=0`; compare cache metrics only and label every table
as instruction-capped diagnostic evidence.

### Fast full-work sampled matrix

The reproducible sample sizes are web-Google 65,536 vertices/502,529 edges,
soc-pokec 65,536/1,089,520, and symmetrized cit-Patents
262,144/340,054 undirected edges. Generate samples with
`flows/sample_realgraph.py`, serialize them with `bench/bin/converter` (`-s`
for cit-Patents), then run:

```bash
python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/web-Google/web-Google.el \
  --output results/graphs/web-Google-n16/web-Google-n16.el \
  --vertices results/graphs/web-Google-n16/web-Google-n16.vertices.tsv \
  --metadata results/graphs/web-Google-n16/web-Google-n16.sample.json \
  --target-vertices 65536

python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/soc-pokec/soc-pokec.el \
  --output results/graphs/soc-pokec-n16/soc-pokec-n16.el \
  --vertices results/graphs/soc-pokec-n16/soc-pokec-n16.vertices.tsv \
  --metadata results/graphs/soc-pokec-n16/soc-pokec-n16.sample.json \
  --target-vertices 65536

python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/cit-Patents/cit-Patents.mtx \
  --output results/graphs/cit-Patents-n18/cit-Patents-n18.el \
  --vertices results/graphs/cit-Patents-n18/cit-Patents-n18.vertices.tsv \
  --metadata results/graphs/cit-Patents-n18/cit-Patents-n18.sample.json \
  --target-vertices 262144

bench/bin/converter \
  -f results/graphs/web-Google-n16/web-Google-n16.el \
  -b results/graphs/web-Google-n16/web-Google-n16.sg
bench/bin/converter \
  -f results/graphs/soc-pokec-n16/soc-pokec-n16.el \
  -b results/graphs/soc-pokec-n16/soc-pokec-n16.sg
bench/bin/converter -s \
  -f results/graphs/cit-Patents-n18/cit-Patents-n18.el \
  -b results/graphs/cit-Patents-n18/cit-Patents-n18-sym.sg

python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_3sim_sampled_allalg \
  --run-tag ecg_3sim_sampled_allalg \
  --out results/ecg_experiments/slurm/ecg_3sim_sampled_allalg.tsv

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_3sim_sampled_allalg.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 12 --cache-sim-jobs 4 --gem5-jobs 4 --sniper-jobs 4
```

All rows run to semantic completion. The samples are deterministic diagnostic
proxies for the named real graphs, not replacements for full-graph authority.
The soc-pokec sample has 2.0 LLC bytes/vertex versus 1.28 for the full graph,
so its sampled cache pressure is lower. Sample metadata counts pre-converter
directed arcs; the symmetrized cit-Patents `.sg` contains both directions.

### Paper-faithful full-graph Sniper ROI

DROPLET warmed graph loading and collected 600 million ROI instructions.
GRASP simulated one representative high-activity iteration, and P-OPT used one
PageRank iteration or sampled pull iterations. GraphBrew follows that precedent
with full graphs and a bounded detailed ROI:

**Blocked:** do not launch this profile. The pinned Sniper tree explicitly
disables the original Pin frontend, so `run-sniper` defaults to SIFT even when
the runner requests `live`. Warm SIFT LRU completes, but warm SIFT K2 aborts in
`queue_model_history` on web-Google before ROI statistics. Unblock by porting
and building the Pin frontend or repairing the warm SIFT queue model.

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_sniper_realgraph_600m \
  --run-tag ecg_sniper_realgraph_600m \
  --out results/ecg_experiments/slurm/ecg_sniper_realgraph_600m.tsv \
  --allow-blocked
```

The generated TSV is inspection-only. Do not pass it to local or Slurm runners
until the manifest blocker is removed.

The planned profile uses the live frontend rather than SIFT trace generation.
Pre-ROI execution is not part of the 600M detailed budget. Explicit property replay
immediately before ROI supplements Sniper's normal cache-warming pass. Fused
transport is validated separately by the strict 120-row smoke gate, avoiding a
cold-start mechanism-proof mode in the performance profile. Capped rows set
`timing_valid_for_speedup=0` because K2 and the baselines do not execute
identical instruction streams; use miss, traffic, and direction metrics only.
The sampled full-completion profile remains the equal-work detailed timing
comparison. Existing calibration shows 600M instructions cover about 18% of
one web-Google PR iteration and 9% of one soc-pokec PR iteration; report this
bounded-window scope explicitly.

## Inspect the blocked headline job

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

To test adaptive placement on reused kernels with the full baseline set:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_streamshield_generality \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_streamshield_generality \
  --no-build
```

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

## Full-iteration headline matrix (blocked)

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile streamshield_sniper_realgraph \
  --run-dir /tmp/ecg-successor-webgoogle-dryrun \
  --list --dry-run --no-build
```

Do not launch this profile until the manifest's `blocked_reason` is removed by
the prefetch-calibration milestone.

## Run local shards in parallel

All binaries must be prebuilt. The launcher gives every shard a unique run
directory and lock; roi_matrix derives isolated fixed-length gem5/Sniper
sideband directories from that output path.

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_streamshield_generality \
  --run-tag ecg_generality_parallel \
  --out results/ecg_experiments/slurm/ecg_generality_parallel.tsv

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_generality_parallel.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 8 \
  --cache-sim-jobs 8 \
  --gem5-jobs 1 \
  --sniper-jobs 1
```

`--jobs` is the global process cap. Per-simulator caps prevent gem5/Sniper
memory overcommit; raise them only on a machine sized for multiple simulators.
Interrupted shards are resumable because each shard is a normal `paper_run.py`
run with completion and content hashes.

## Generate one-policy Slurm shards after calibration

```bash
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt
mkdir -p results/slurm_logs results/ecg_experiments/slurm

python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile streamshield_sniper_realgraph \
  --run-tag ecg_successor_webgoogle \
  --out results/ecg_experiments/slurm/ecg_successor_webgoogle.tsv \
  --allow-blocked
```

Submit on a configured cluster:

```bash
SHARDS=results/ecg_experiments/slurm/ecg_successor_webgoogle.tsv
COUNT=$(wc -l < "$SHARDS")
export SHARDS
sbatch --array=0-$((COUNT - 1))%16 \
  scripts/experiments/ecg/slurm/slurm_final_shard.sbatch
```

## Aggregate

Local completed runs:

```bash
python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-dirs \
    results/ecg_experiments/final_paper_runs/ecg_replacement \
    results/ecg_experiments/final_paper_runs/ecg_factorial \
    results/ecg_experiments/final_paper_runs/ecg_streamshield_generality \
  --run-root results/ecg_experiments/paper_pipeline/ecg_final

test -f \
  results/ecg_experiments/paper_pipeline/ecg_final/aggregate/online_dueling_regret.csv
```

Parallel local or Slurm shards:

```bash
python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-glob \
    "results/ecg_experiments/final_paper_runs/local/ecg_generality_parallel/*" \
  --run-root results/ecg_experiments/paper_pipeline/ecg_generality_parallel
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
  --gem5 --sniper --kernels pr bfs sssp bc cc \
  --schedule-k 2 --stream-bypass

python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs sssp bc cc \
  --schedule-k 2 --stream-bypass --adaptive-stream-bypass
```
