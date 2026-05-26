# ECG Slurm Final Runs

This page describes the reproducible UVA Slurm workflow for ECG final-paper runs. The goal is to split slow gem5 or Sniper simulations into independent shards, run each shard on a separate machine, and aggregate all CSVs later.

The key rule: one GraphBrew gem5 process per node unless you have explicitly isolated the runtime sideband files. The gem5 path uses shared sideband files under `/tmp`, so Slurm jobs should request exclusive nodes or otherwise ensure only one GraphBrew gem5 shard runs per node.

For autonomous execution, follow
[the ECG autonomous final-paper completion task](../plans/ecg-autonomous-final-paper-completion.md).
It keeps build checks, dry-runs, tiny smokes, and shard generation local, while
large file-backed gem5/Sniper/final profiles are saved as Slurm shards instead
of being launched on the workstation.

## Overview

The workflow has four phases:

1. Prepare a shared checkout, Python environment, gem5 build, benchmark binaries, and graph data.
2. Generate one shard per graph / benchmark / policy.
3. Submit Slurm array jobs, where each task runs exactly one shard.
4. Aggregate completed shard directories with `paper_pipeline.py --skip-run`.

The final-run wrapper supports these sharding filters:

```bash
--graph <name>          # exact normalized manifest graph name
--benchmark <name>      # pr, bfs, or sssp
--policy <label>        # LRU, POPT_CHARGED, ECG_DBG_PRIMARY, ...
--only <stage-token>    # optional stage filter
```

For example, this lists exactly one replacement shard:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile final_replacement \
  --run-dir results/ecg_experiments/final_paper_runs/slurm_dryrun \
  --graph soc-pokec \
  --benchmark pr \
  --policy ECG_DBG_PRIMARY_CHARGED \
  --list \
  --dry-run \
  --allow-missing-graphs \
  --skip-validation-gate
```

## 1. Prepare the UVA Slurm Workspace

Use shared storage visible to all compute nodes, such as `$SCRATCH` or a project allocation. Replace module names with the currently available UVA modules.

```bash
# Example layout. Adjust to your allocation.
export GRAPHBREW_ROOT=$SCRATCH/GraphBrew
export GRAPHBREW_RUNS=$SCRATCH/graphbrew_ecg_runs
export GRAPHBREW_GRAPHS=$GRAPHBREW_ROOT/results/graphs

mkdir -p "$GRAPHBREW_RUNS" "$GRAPHBREW_GRAPHS" results/slurm_logs
cd "$GRAPHBREW_ROOT"

module purge
module avail gcc python
module load gcc/<version> python/<version>

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r scripts/requirements.txt
pip install 'scons==4.6.0'
```

Build gem5 once, not inside every Slurm task:

```bash
source .venv/bin/activate
python3 scripts/setup_gem5.py --isa X86
make -j8 all-sim
make gem5-m5ops-pr gem5-m5ops-bfs gem5-m5ops-sssp
```

If gem5 has already been built on shared storage, do not rebuild it in worker jobs. The final-run manifest defaults to `--no-build`.

## 2. Stage Graph Data

Final profiles expect these paths:

```text
results/graphs/soc-pokec/soc-pokec.sg
results/graphs/soc-LiveJournal1/soc-LiveJournal1.sg
results/graphs/com-orkut/com-orkut.sg
results/graphs/cit-Patents/cit-Patents.sg
```

For available-local smoke runs, these paths are used:

```text
results/graphs/email-Eu-core/email-Eu-core.sg
results/graphs/cit-Patents/cit-Patents.sg
```

Preferred graph-staging pattern:

```bash
mkdir -p results/graphs/<graph-name>
# Put raw graph as .el, .mtx, or .txt after removing source-specific headers.
# Then convert to serialized GAP/GraphBrew format.
make converter
bench/bin/converter \
  -f results/graphs/<graph-name>/<raw-file>.el \
  -s \
  -b results/graphs/<graph-name>/<graph-name>.sg
```

For compressed edge-list downloads:

```bash
mkdir -p results/graphs/cit-Patents
zcat /path/to/cit-Patents.txt.gz | grep -v '^#' > results/graphs/cit-Patents/cit-Patents.el
bench/bin/converter \
  -f results/graphs/cit-Patents/cit-Patents.el \
  -s \
  -b results/graphs/cit-Patents/cit-Patents.sg
```

For Matrix Market downloads:

```bash
mkdir -p results/graphs/<graph-name>
bench/bin/converter \
  -f /path/to/<graph-name>.mtx \
  -s \
  -b results/graphs/<graph-name>/<graph-name>.sg
```

After staging, verify exactly what the manifest will see:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile final_replacement \
  --check-graphs \
  --allow-missing-graphs

python3 scripts/experiments/ecg/final_paper_run.py \
  --profile final_droplet \
  --check-graphs \
  --allow-missing-graphs
```

Do not start the multi-day run until missing graphs are resolved or intentionally excluded with `--graph` filters.

## 3. Generate Slurm Shard List

Create one tab-separated row per shard from the checked-in final-run manifest.
The generated TSV is headerless because Slurm array task `0` reads row `1`.

```bash
mkdir -p results/ecg_experiments/slurm
RUN_TAG=final_$(date +%Y%m%d_%H%M%S)
SHARDS=results/ecg_experiments/slurm/${RUN_TAG}_shards.tsv

python3 scripts/experiments/ecg/make_slurm_shards.py \
  --profile final_replacement final_droplet \
  --run-tag "$RUN_TAG" \
  --out "$SHARDS"

wc -l "$SHARDS"
sed -n '1,5p' "$SHARDS"
```

The full four-graph PR/BFS/SSSP replacement+DROPLET set above creates:

```text
4 graphs * 3 benchmarks * (9 replacement policies + 7 DROPLET policies) = 192 Slurm shards
```

For a smaller first pass, use the generator filters.
For example:

```bash
python3 scripts/experiments/ecg/make_slurm_shards.py \
  --profile final_replacement \
  --run-tag "$RUN_TAG" \
  --graph cit-Patents \
  --benchmark pr \
  --policy LRU ECG_DBG_PRIMARY \
  --out "$SHARDS"
```

For the current long Sniper scale-out gate, generate a one-row Slurm shard with:

```bash
python3 scripts/experiments/ecg/make_slurm_shards.py \
  --profile sniper_sift_cit_patents_long \
  --run-tag "$RUN_TAG" \
  --out "$SHARDS"
```

## 4. Slurm Array Script

The repository includes `scripts/experiments/ecg/slurm_final_shard.sbatch`. Replace account, partition, time, and memory with the UVA allocation you are using.

Review it before submitting:

```bash
sed -n '1,120p' scripts/experiments/ecg/slurm_final_shard.sbatch
```

Submit the array:

```bash
RUN_TAG=final_20260524_000000
SHARDS=results/ecg_experiments/slurm/${RUN_TAG}_shards.tsv
N=$(( $(wc -l < "$SHARDS") - 1 ))

export GRAPHBREW_ROOT=$SCRATCH/GraphBrew
export SHARDS
sbatch --array=0-${N} scripts/experiments/ecg/slurm_final_shard.sbatch
```

Use `--exclusive` or an equivalent UVA policy so Slurm does not place two GraphBrew gem5 shards on the same node. If your allocation does not allow exclusive nodes, limit the array to one task per node by partition policy or run cache_sim shards separately from gem5 shards.

## 5. Monitor and Resume

Check Slurm status:

```bash
squeue -u "$USER" -n ecg-gem5
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode
```

Check a shard run directory:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --status \
  --run-dir results/ecg_experiments/final_paper_runs/slurm/${RUN_TAG}/final_replacement_soc-pokec_pr_LRU
```

Summarize every row from the shard TSV without launching simulations:

```bash
python3 scripts/experiments/ecg/slurm_shard_status.py \
  --shards "$SHARDS" \
  --out results/ecg_experiments/slurm/${RUN_TAG}_status.csv

column -s, -t < results/ecg_experiments/slurm/${RUN_TAG}_status.csv | sed -n '1,20p'
```

Use `--fail-on-failed` in CI/check scripts when any failed shard should make the
status command return nonzero. Use `--fail-on-missing` when pending or
not-started rows should also fail the check.

Rerun a failed shard by resubmitting the same row. The wrapper is resumable: existing `ok` CSV rows are skipped unless `--force` is passed.

## 6. Aggregate Shards Later

After Slurm jobs finish, aggregate all shard directories without launching simulations:

```bash
python3 scripts/experiments/ecg/paper_pipeline.py \
  --skip-run \
  --input-run-glob "results/ecg_experiments/final_paper_runs/slurm/${RUN_TAG}/*" \
  --run-root "results/ecg_experiments/paper_pipeline/${RUN_TAG}_aggregate"
```

The aggregate run writes:

```text
aggregate/roi_matrix_all.csv
aggregate/roi_relative_metrics.csv
aggregate/faithfulness_summary.csv
aggregate/popt_storage_overhead_summary.csv
aggregate/prefetch_quality_summary.csv
figures/*.svg
figures/*.png
tables/*.tex
```

If some shards are still running or failed, aggregation still works on completed shard directories. The missing policies/graphs simply will not appear in the combined CSVs.

## 7. Recommended UVA Run Order

1. Run `rehearsal` locally or in a short Slurm job.
2. Run `available_replacement` on the two locally available graph files.
3. Stage all final graph `.sg` files.
4. Run `final_cache_sim` shards if you want fast policy-direction checks.
5. Run `final_replacement` shards.
6. Run `final_droplet` shards after replacement shards are stable.
7. Aggregate with `paper_pipeline.py --skip-run --input-run-glob ...`.

## 8. ECG_PFX Scale-Proof Shards

For root-selected BFS ECG_PFX scale proofs beyond the local g10 gate, use the
dedicated shard wrapper. TSV columns are:

```text
scale<TAB>root<TAB>backend<TAB>out_root
```

Example:

```bash
RUN_TAG=ecg_pfx_scale_$(date +%Y%m%d_%H%M%S)
SHARDS=results/ecg_experiments/slurm/${RUN_TAG}_scale.tsv
mkdir -p results/ecg_experiments/slurm

python3 scripts/experiments/ecg/make_ecg_pfx_scale_shards.py \
  --scale 11:12 \
  --root 0:31 \
  --backend sniper \
  --run-tag "$RUN_TAG" \
  --out "$SHARDS"

export SHARDS
N=$(( $(wc -l < "$SHARDS") - 1 ))
sbatch --array=0-${N} scripts/experiments/ecg/slurm_ecg_pfx_scale_proof.sbatch
```

Each shard runs `scripts/experiments/ecg/ecg_pfx_scale_proof.py`, writes a
`summary.csv`, and keeps the RISC-V gem5 and Sniper outputs under the requested
`out_root`.

Summarize shard status and combine completed `summary.csv` rows with:

```bash
python3 scripts/experiments/ecg/ecg_pfx_scale_status.py \
  --shards "$SHARDS" \
  --out results/ecg_experiments/slurm/${RUN_TAG}_scale_status.csv \
  --combined results/ecg_experiments/slurm/${RUN_TAG}_scale_combined.csv

# Generate matched RISC-V gem5 follow-up shards for useful Sniper roots.
FOLLOWUP_TAG=${RUN_TAG}_gem5_followup
FOLLOWUP_SHARDS=results/ecg_experiments/slurm/${FOLLOWUP_TAG}_scale.tsv
python3 scripts/experiments/ecg/make_ecg_pfx_scale_shards.py \
  --from-combined results/ecg_experiments/slurm/${RUN_TAG}_scale_combined.csv \
  --backend gem5-riscv \
  --run-tag "$FOLLOWUP_TAG" \
  --out "$FOLLOWUP_SHARDS"
```

Current prepared local shard file, generated 2026-05-26:

```bash
RUN_TAG=ecg_pfx_scale_20260526_133517
SHARDS=results/ecg_experiments/slurm/${RUN_TAG}_scale.tsv

# 64 rows: scales 11-12, roots 0-31, backend sniper
wc -l "$SHARDS"

# Submit on the cluster, not on the workstation.
export GRAPHBREW_ROOT=$SCRATCH/GraphBrew
export SHARDS
N=$(( $(wc -l < "$SHARDS") - 1 ))
sbatch --array=0-${N} scripts/experiments/ecg/slurm_ecg_pfx_scale_proof.sbatch

# Monitor and combine completed rows.
python3 scripts/experiments/ecg/ecg_pfx_scale_status.py \
  --shards "$SHARDS" \
  --out results/ecg_experiments/slurm/${RUN_TAG}_scale_status.csv \
  --combined results/ecg_experiments/slurm/${RUN_TAG}_scale_combined.csv
```

## Guardrails

- Do not run multiple GraphBrew gem5 shards on the same node unless sideband files are isolated.
- Keep graph data and run outputs on shared storage visible to all nodes.
- Build gem5 and benchmark binaries once before submitting the array.
- Use exact manifest graph names: `soc-pokec`, `soc-LiveJournal1`, `com-orkut`, `cit-Patents`.
- Use exact normalized policy labels for `--policy`: `LRU`, `SRRIP`, `GRASP`, `POPT_CHARGED`, `POPT`, `ECG_DBG_ONLY`, `ECG_DBG_PRIMARY_CHARGED`, `ECG_DBG_PRIMARY`, `ECG_POPT_PRIMARY`.
- `POPT` is the oracle ceiling; `POPT_CHARGED` is the honest prior-method baseline.
