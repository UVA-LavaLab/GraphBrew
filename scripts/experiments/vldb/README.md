# Frozen evaluation reproduction

This compatibility package reproduces the frozen evaluation campaign. New
experiments use the top-level orchestrator and shared modules under
`scripts/lib/`.

| File / dir | Purpose |
|---|---|
| [`config.py`](config.py)       | Frozen graph sets, baselines, variants, benchmarks, and trial counts |
| [`runner.py`](runner.py)       | Monolithic all-in-one runner (legacy; still works) |
| [`figures.py`](figures.py)     | LaTeX + PNG emitter (called by stage 05) |
| [`stages/`](stages/README.md)  | Five independent stage runners + SLURM templates |
| [`experiments/`](experiments/) | Per-experiment recipe READMEs (`exp1_cache/`, ..., `exp8_scalability/`) |
| [`slurm/monolithic.sbatch`](slurm/monolithic.sbatch) | Legacy single-job SLURM template |

## Recommended workflow (stage-based)

```bash
source .venv/bin/activate
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --preview
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview --verify-gate
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --preview  # optional
python3 scripts/experiments/vldb/stages/05_aggregate.py --exp 0 \
  --paper-dir /path/to/private-paper \
  --publish-paper-figures --package-paper                               # optional
```

For reproducible timing on the dedicated 16-core host, pass
`--threads 16 --cpu-list 0-15`. Rapid preview runs use
`--threads 4 --cpu-list 24-27`. Large graph and artifact roots should be
provided with `--graph-dir` and `--artifact-root`; do not place them on the
repository filesystem.

Experiment 1 performs an explicit cache-capacity sweep. Cache simulation is
single-threaded with one cold process trial per cell. The final policy uses
every-access `ultrafast` simulation on four representative graphs at
2/8/22/32/64 MiB; `--graphs`, `--cache-mode`, and `--cache-sizes-kib` create
explicit exploratory cohorts.

## Final weighted SSSP policy freeze

After the final corpus refresh, tune on the SHUFFLED baseline and validate
`<artifact-root>/vldb_paper/sssp_delta_tuning.json`. Then freeze the exact
artifact and recommendations into the repository SSOT.

The final tuner uses `fastest-source-median-tie/v2`: three independent
process invocations per `(graph, delta)`, each with the same three deterministic
sources and one trial per source. Delta order is cyclically shifted by
replicate so elapsed-time drift is not confounded with an ascending sweep.
The fastest candidate minimizes the median over the nine invocation/source
blocks. A candidate enters the tie set only when its paired slowdown is not
positive at the one-sided 95% t threshold (`df=8`) and its median is within 2%
of the fastest median; the smallest delta in that tie set is selected. The
artifact records invocation order, paired intervals, tie sets, and whether a
delta-1 choice is the generated-weight domain floor. Measurement protocol and
selection rule IDs are separate, so a future analysis-only rule change can
re-derive recommendations without rerunning kernels.

First measure without the freeze flag:

```bash
python3 scripts/experiments/vldb/stages/03_cpu_perf.py \
  --exp 2 --tune-sssp-delta \
  --graph-dir /media/Data/00_GraphDatasets/GraphBrew \
  --artifact-root /media/Data/00_GraphDatasets/GraphBrew/artifacts \
  --threads 16 --cpu-list 0-15 --timeout 21600
```

After validation approves that exact artifact, run the validation-only freeze:

```bash
python3 scripts/experiments/vldb/stages/03_cpu_perf.py \
  --exp 2 --tune-sssp-delta --freeze-sssp-policy \
  --graph-dir /media/Data/00_GraphDatasets/GraphBrew \
  --artifact-root /media/Data/00_GraphDatasets/GraphBrew/artifacts \
  --threads 16 --cpu-list 0-15 --timeout 21600
```

This writes `sssp_delta_tuning.json` and `sssp_policy.json` beside
`config.py`. Runtime preflight verifies the reviewed recommendations, exact
policy types and values, thread/affinity policy, graph dimensions, and
semantic graph provenance before any SSSP measurement.

## Monolithic workflow (legacy)

```bash
python3 scripts/experiments/vldb/runner.py --all --local
python3 scripts/experiments/vldb/runner.py --exp 2 --preview
```

## Outputs

| Stage | Path |
|---|---|
| 01 | `<graph-root>/<name>/<name>.{sg,mtx,el}` |
| 02 | `<artifact-root>/vldb_mappings/<graph>/<algo_key>.{lo,json}` |
| 03 | `<artifact-root>/vldb_paper/exp<N>_*/...json` |
| 04 | `<artifact-root>/vldb_paper/exp1_cache/cache_results.json` |
| 05 | `<artifact-root>/vldb_paper/{figures,tables}/` |
