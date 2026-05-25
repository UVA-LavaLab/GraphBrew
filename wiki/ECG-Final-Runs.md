# ECG Final Runs

This page records the current ECG / gem5 final-paper run workflow. It is the
place to check before starting long cache-policy runs.

For UVA Slurm runs where each graph/benchmark/policy shard runs independently
and results are aggregated later, see [ECG Slurm runs](ECG-Slurm-Runs.md).

## Current Scope

Supported final-run benchmarks:

```text
pr
bfs
sssp
```

These are the kernels with the best current replacement-policy and P-OPT
current-vertex validation. Other gem5 wrappers may have graph hints, but do not
use them for final P-OPT claims until their matrix export and sideband paths are
audited.

## Policy Labels

Replacement-only final profile:

```text
LRU
SRRIP
GRASP
POPT_CHARGED
POPT
ECG_DBG_ONLY
ECG_DBG_PRIMARY_CHARGED
ECG_DBG_PRIMARY
ECG_POPT_PRIMARY
```

DROPLET final profile:

```text
LRU + DROPLET
GRASP + DROPLET
POPT_CHARGED + DROPLET
POPT + DROPLET
ECG_DBG_PRIMARY_CHARGED + DROPLET
ECG_DBG_PRIMARY + DROPLET
ECG_POPT_PRIMARY + DROPLET
```

Interpretation:

| Label | Meaning |
|---|---|
| `POPT` | Uncharged oracle P-OPT ceiling. Rereference matrix lookup is host-side metadata. |
| `POPT_CHARGED` | Honest P-OPT prior-method baseline. Same dynamic lookup, but effective L3 data ways/size are reduced to reserve current+next rereference matrix columns. |
| `ECG_DBG_ONLY` | GRASP-equivalence mode. Same insertion/hit behavior and plain SRRIP victim selection. |
| `ECG_DBG_PRIMARY` | Main oracle-assisted ECG hybrid: DBG first, dynamic P-OPT as tiebreak. |
| `ECG_DBG_PRIMARY_CHARGED` | Same dynamic ECG mode, but with P-OPT reserved-way overhead charged. |
| `ECG_POPT_PRIMARY` | P-OPT-equivalence / oracle validation mode: dynamic P-OPT first, DBG as tiebreak. |

`POPT_CHARGED` and `*_CHARGED` are runner-level labels in
`scripts/experiments/ecg/roi_matrix.py`. They map to the underlying gem5/cache_sim
P-OPT or ECG policies while changing effective L3 geometry and output metadata.
Do not pass `CACHE_POLICY=POPT_CHARGED` directly to a sim binary.

## Charged P-OPT Model

The P-OPT paper stores current and next rereference matrix columns in reserved
LLC ways. GraphBrew models that overhead for charged rows by:

- estimating the vertex-property cache-line count,
- reserving enough LLC ways for `2 * num_cache_lines` bytes by default,
- keeping the same number of cache sets,
- reducing effective L3 data associativity and size,
- recording estimated matrix-stream traffic in the output CSV.

Important CSV fields:

```text
popt_overhead_charged
popt_requested_l3_size
popt_effective_l3_size
popt_effective_l3_ways
popt_reserved_ways
popt_reserved_bytes
popt_matrix_stream_bytes
popt_matrix_stream_cache_lines
popt_charged_total_memory_traffic              # cache_sim rows
popt_charged_l3_misses_plus_matrix_stream      # gem5 rows
```

Current smoke validation:

```text
results/ecg_experiments/roi_matrix/popt_charged_gem5_smoke/roi_matrix.csv
```

On PR g10, L3=4kB, `POPT_CHARGED` reserved one way, used effective L3
`3840B` / 15 ways, and was slower / higher-miss than uncharged `POPT`, as
expected.

## Commands

Check graph availability:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile final_replacement \
  --check-graphs \
  --allow-missing-graphs
```

Dry-run the available local graph profile:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile available_replacement \
  --list \
  --dry-run
```

List one Slurm-friendly shard:

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

Run the charged P-OPT smoke:

```bash
python3 scripts/experiments/ecg/roi_matrix.py \
  --suite gem5 \
  --benchmark pr \
  --options "-g 10 -k 16 -o 5 -n 1 -i 1" \
  --policies POPT_CHARGED POPT ECG:DBG_PRIMARY_CHARGED ECG:DBG_PRIMARY \
  --l3-sizes 4kB \
  --out-dir results/ecg_experiments/roi_matrix/popt_charged_gem5_smoke \
  --timeout-gem5 900 \
  --no-build
```

Run the full rehearsal before any multi-day job:

```bash
python3 scripts/experiments/ecg/final_paper_run.py --profile rehearsal
```

Run a real profile on currently available local graphs:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile available_replacement \
  --run-dir results/ecg_experiments/final_paper_runs/available_replacement_001
```

Run final replacement once graph checks pass:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile final_replacement \
  --run-dir results/ecg_experiments/final_paper_runs/replacement_final_001
```

Run final DROPLET after replacement is stable:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile final_droplet \
  --run-dir results/ecg_experiments/final_paper_runs/droplet_final_001
```

Check an existing run directory:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --status \
  --run-dir results/ecg_experiments/final_paper_runs/replacement_final_001
```

Run the one-command paper pipeline:

```bash
python3 scripts/experiments/ecg/paper_pipeline.py \
  --profiles rehearsal \
  --run-root results/ecg_experiments/paper_pipeline/rehearsal_001
```

Aggregate existing CSVs and generate figures/tables without launching runs:

```bash
python3 scripts/experiments/ecg/paper_pipeline.py \
  --skip-run \
  --input-csv results/ecg_experiments/roi_matrix/popt_charged_gem5_smoke/roi_matrix.csv \
  --run-root results/ecg_experiments/paper_pipeline/charged_smoke_figures
```

Aggregate Slurm shard run directories:

```bash
python3 scripts/experiments/ecg/paper_pipeline.py \
  --skip-run \
  --input-run-glob "results/ecg_experiments/final_paper_runs/slurm/<run_tag>/*" \
  --run-root "results/ecg_experiments/paper_pipeline/<run_tag>_aggregate"
```

Pipeline outputs:

```text
aggregate/roi_matrix_all.csv
aggregate/roi_policy_summary.csv
aggregate/roi_relative_metrics.csv
aggregate/roi_relative_policy_summary.csv
aggregate/popt_charged_overhead.csv
aggregate/faithfulness_summary.csv
aggregate/popt_storage_overhead_summary.csv
aggregate/ecg_mode_overhead_summary.csv
aggregate/prefetch_quality_summary.csv
aggregate/policy_label_map.csv
figures/replacement_speedup_vs_lru.svg        # PNG preview also written
figures/replacement_l3_miss_reduction_vs_lru.svg
figures/replacement_speedup_by_benchmark.svg
figures/replacement_l3_miss_reduction_by_benchmark.svg
figures/droplet_speedup_vs_lru.svg            # written when DROPLET rows exist
figures/droplet_l3_miss_reduction_vs_lru.svg  # written when DROPLET rows exist
figures/droplet_speedup_by_benchmark.svg
figures/droplet_l3_miss_reduction_by_benchmark.svg
figures/component_memory_traffic_reduction_vs_lru.svg
figures/component_memory_traffic_reduction_by_benchmark.svg
figures/droplet_prefetch_accuracy_by_benchmark.svg
figures/charged_overhead.svg                  # charged vs oracle overhead
tables/roi_policy_summary.tex
tables/popt_charged_overhead.tex
tables/faithfulness_summary.tex
tables/popt_storage_overhead_summary.tex
tables/ecg_mode_overhead_summary.tex
tables/prefetch_quality_summary.tex
tables/cache_sim_prefetch_quality_summary.tex
```

## Guardrails

- Do not run multiple GraphBrew gem5 jobs concurrently; runtime sideband files
  under `/tmp` are shared.
- `final_paper_run.py` serializes jobs and writes a lock file.
- Final profiles require focused faithfulness CSVs for GRASP, P-OPT, and
  DROPLET unless `--skip-validation-gate` is explicitly used.
- ECG PFX timing is not a final gem5 claim yet. Cache_sim PFX proves the
  mechanism; gem5 timing-visible PFX still needs the x86/RISC-V hint path.
- Missing graph files must be fixed before full `final_replacement` or
  `final_droplet` runs.