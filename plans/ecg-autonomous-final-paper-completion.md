# ECG Autonomous Final-Paper Completion Task

## Purpose

Finish the ECG/cache_sim/gem5/Sniper final-paper requirement stack without
watering down the claims. Work autonomously through the next unblocked item,
validate each slice, and keep going by switching to another useful stream when a
single stream is blocked.

This task supersedes ad hoc one-off run notes for the ECG final-paper effort.
Use it with `.github/prompts/autonomous-map-follow.prompt.md`.

## Operating Contract

- No compromise on paper faithfulness, accounting, or claim validity.
- Do not ask whether to continue after a successful slice; continue to the next
  unblocked item.
- Stop only for system-crash risk, destructive actions, missing secrets,
  unavailable required data, or mutually exclusive technical choices that cannot
  be inferred from the repo.
- If a local job becomes too large, do not downsize the requirement. Generate a
  Slurm shard instead and continue with local validation, aggregation, docs, or
  another implementation stream.
- Keep generated `results/` artifacts out of commits unless the user explicitly
  asks for them.

## Local vs Slurm Routing

Run locally only when the work is bounded and useful on the workstation:

- Code edits, static inspection, docs, and manifest updates.
- `py_compile`, focused `pytest`, `bash -n`, `git diff --check`.
- Dry-runs, `--list`, `--status`, graph checks, and shard TSV generation.
- Synthetic or tiny smokes that are expected to finish in under 30 minutes and
  under 8 GiB RSS.
- cache_sim available-local experiments when the selected command has a bounded
  timeout and does not block implementation work.

Send to Slurm instead of the local PC:

- Any final profile over large file-backed graphs.
- gem5/Sniper file-backed matrices expected to exceed 30 minutes or 8 GiB RSS.
- `final_replacement`, `final_droplet`, `final_cache_sim`,
  `final_cache_sim_ecg_pfx`, and `sniper_sift_cit_patents_long`.
- Multi-graph or multi-policy gem5/Sniper sweeps.
- Any rerun that previously timed out locally.

Slurm guardrails:

- Use `scripts/experiments/ecg/make_slurm_shards.py`; do not hand-write shard
  loops.
- Run one GraphBrew gem5 shard per node unless sideband isolation has been
  explicitly verified for that backend and run mode.
- Use `scripts/experiments/ecg/slurm_final_shard.sbatch` with `--exclusive` or
  an equivalent allocation policy.
- Build gem5, Sniper, and benchmark wrappers once on shared storage before
  submitting worker jobs.
- Aggregate with `paper_pipeline.py --skip-run --input-run-glob ...`; never
  rerun completed shard directories just to aggregate.

## Required Completion Evidence

The task is done only when the following evidence exists and passes review:

1. ECG replacement faithfulness:
   - GRASP vs `ECG_DBG_ONLY` parity is preserved.
   - P-OPT vs `ECG_POPT_PRIMARY` parity is preserved.
   - Charged P-OPT / charged ECG overhead rows remain explicit.
2. ECG_PFX mechanism proof:
   - cache_sim component proof shows DBG, POPT, PFX, and combined behavior.
   - gem5/Sniper PFX rows have nonzero issued/useful counters before any timing
     discussion.
   - `timing_valid_for_speedup=0` is preserved for explicit/prototype hint rows.
3. Prior-method comparison:
   - LRU, SRRIP, GRASP, P-OPT, DROPLET, charged P-OPT, and ECG rows are present
     where the manifest expects them.
   - DROPLET rows record active/useful prefetch behavior before being used as a
     prior prefetch baseline.
4. Instruction path:
   - RISC-V `ecg.extract` remains buildable and documented.
   - x86 instruction-delivery work either lands as a tested gem5 instruction or
     is documented as an x86 pseudo-op prototype with timing caveats intact.
5. Final infrastructure:
   - Local smokes pass.
   - Slurm shards are generated for large final jobs.
   - Completed shards aggregate into CSVs, figures, and LaTeX tables.
6. Paper hygiene:
   - Docs and wiki reflect exactly which rows are timing-valid.
   - Final answer/commit notes report what passed, what is running on Slurm, and
     what is blocked by missing graph data or cluster credentials.

## Autonomous Backlog

### Phase 0: Preflight and Safety

- [ ] Check for active GraphBrew gem5/Sniper/ROI processes before launching any
  simulator work.
- [ ] Record `git status --short` and work with existing dirty files without
  reverting unrelated user changes.
- [ ] Check disk space for `/tmp`, repo, and target run directories.
- [ ] Verify the Python environment and required tools.
- [ ] Run focused static validation for currently touched ECG files.

Suggested local commands:

```bash
pgrep -af 'gem5.opt|run-sniper|record-trace|roi_matrix.py|final_paper_run.py' || true
git --no-pager status --short
df -h /tmp . results 2>/dev/null || df -h /tmp .
python3 -m py_compile \
  scripts/experiments/ecg/roi_matrix.py \
  scripts/experiments/ecg/final_paper_run.py \
  scripts/experiments/ecg/paper_pipeline.py \
  scripts/experiments/ecg/make_slurm_shards.py \
  scripts/experiments/ecg/slurm_shard_status.py
```

### Phase 1: Finish the Root Instruction Path

- [ ] Preserve the existing RISC-V `ecg.extract` custom-0 path.
- [ ] Implement the cleanest x86 instruction-delivery path supported by gem5.
  Prefer an explicit GraphBrew x86 instruction/pseudo-instruction wrapper over a
  C function-call hint path. Do not claim hardware timing validity until the path
  is measured and the caveat is removed deliberately.
- [ ] Add static tests that prove the x86 path emits an instruction-encoded
  sequence and that `--ecg-pfx-delivery instruction` is plumbed through the
  runner.
- [ ] Keep explicit-hint fallback available for debugging, with timing invalid.

Validation:

```bash
python3 -m pytest -q \
  scripts/test/test_gem5_ecg_pfx_scaffold.py \
  scripts/test/test_final_paper_run.py
```

### Phase 2: Rebuild and Local Smoke

- [ ] Re-apply gem5 overlays after instruction-path edits.
- [ ] Rebuild only the required gem5 ISA/backend unless a broader rebuild is
  necessary.
- [ ] Rebuild Sniper wrappers and overlays after harness or overlay edits.
- [ ] Run bounded synthetic smokes for cache_sim, gem5, and Sniper.
- [ ] Do not expand to file-backed or final-scale simulations locally.

Suggested local smoke commands:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile rehearsal \
  --dry-run \
  --run-dir /tmp/graphbrew-autonomous-rehearsal-dryrun \
  --skip-validation-gate

python3 scripts/experiments/ecg/final_paper_run.py \
  --profile gem5_ecg_pfx_tiny_smoke \
  --list \
  --dry-run \
  --run-dir /tmp/graphbrew-autonomous-gem5-pfx-tiny-dryrun \
  --skip-validation-gate
```

### Phase 3: Local Mechanism Evidence

- [ ] Keep cache_sim as the fast mechanism validator.
- [ ] Run or refresh bounded cache_sim ECG_PFX/component proof rows only when the
  local runtime is manageable.
- [ ] Use `local_cache_sim_diversity_smoke`,
  `local_cache_sim_diversity_medium`, and
  `local_cache_sim_pfx_diversity_smoke` to screen additional manageable graph
  kernels and small/medium graph behavior before detailed simulation.
- [ ] Summarize completed local cache screens with
  `scripts/experiments/ecg/local_cache_screen_summary.py` before choosing
  follow-up gem5/Sniper or Slurm shards.
- [ ] Track cache-size sensitivity before promoting a local win: current local
  triage says email-Eu-core BC, BFS, and PR are positive tiny-cache mechanism
  points but collapse to near ties or reversals at 32kB/256kB, while
  cit-Patents PR/CC remain negative/control points.
- [ ] Treat email-Eu-core BFS ECG_PFX as activation evidence only unless a
  tuning sweep finds a real miss reduction: it issues useful prefetches, but
  current cache-size rows do not materially improve L3 misses versus no-PFX.
- [ ] Aggregate available-local cache_sim and smoke results with
  `paper_pipeline.py --skip-run` where possible.
- [ ] Confirm no ECG_PFX prototype row contributes to speedup figures.
- [ ] For RISC-V-vs-Sniper ECG_PFX comparison, run the PR/BFS/SSSP g6 local
  gate, then root-selected BFS g7/g8/g9/g10. Current matched proof rows are BFS
  g6, g7/r1, g8/r9, g9/r20, and g10/r0: RISC-V gem5 `ecg.extract` and Sniper
  SIFT both report nonzero issued and useful prefetch counters. Treat rows with
  `pf_issued=0` as activation-only, not fill/useful-prefetch evidence; move
  larger BFS sweeps beyond g10 to a long local window or Slurm.

### Phase 4: Slurm Shard Preparation

- [ ] Check graph availability for final profiles.
- [ ] For missing large final graphs, follow
  `plans/ecg-graph-staging-plan.md` on shared cluster storage before strict
  preflight or full-array submission.
- [ ] Generate shard TSVs for every large job rather than launching locally.
- [ ] Use a descriptive `RUN_TAG` that includes date and purpose.
- [ ] Generate a one-row smoke shard first for any new cluster environment.
- [ ] Record shard TSV path, row count, and expected run root.

One-row cluster smoke:

```bash
mkdir -p results/ecg_experiments/slurm
RUN_TAG=autonomous_smoke_$(date +%Y%m%d_%H%M%S)
SHARDS=results/ecg_experiments/slurm/${RUN_TAG}_shards.tsv

python3 scripts/experiments/ecg/make_slurm_shards.py \
  --smoke \
  --run-tag "$RUN_TAG" \
  --out "$SHARDS" \
  --allow-missing-graphs
```

Full final shards:

```bash
mkdir -p results/ecg_experiments/slurm
RUN_TAG=final_$(date +%Y%m%d_%H%M%S)
SHARDS=results/ecg_experiments/slurm/${RUN_TAG}_shards.tsv

python3 scripts/experiments/ecg/make_slurm_shards.py \
  --profile final_replacement final_droplet final_cache_sim final_cache_sim_ecg_pfx \
  --run-tag "$RUN_TAG" \
  --out "$SHARDS"

wc -l "$SHARDS"
```

### Phase 5: Slurm Submission and Monitoring

- [ ] Submit the smoke shard first.
- [ ] Submit the full array only after the smoke shard reaches `ok`.
- [ ] Monitor with Slurm tools and `slurm_shard_status.py`.
- [ ] Rerun failed shards by resubmitting the same row; do not invalidate ok
  shards with `--force` unless the implementation changed.

Submission template:

```bash
export GRAPHBREW_ROOT=$SCRATCH/GraphBrew
export SHARDS=results/ecg_experiments/slurm/<run_tag>_shards.tsv
N=$(( $(wc -l < "$SHARDS") - 1 ))
sbatch --array=0-${N} scripts/experiments/ecg/slurm_final_shard.sbatch
```

Status:

```bash
python3 scripts/experiments/ecg/slurm_shard_status.py \
  --shards "$SHARDS" \
  --out results/ecg_experiments/slurm/${RUN_TAG}_status.csv

# Use this before promoting a smoke shard to a full array.
python3 scripts/experiments/ecg/slurm_shard_status.py \
  --shards "$SHARDS" \
  --require-ok
```

### Phase 6: Aggregation and Figures

- [ ] Aggregate completed Slurm shards without launching new simulations.
- [ ] Confirm aggregate CSV row counts and status counts.
- [ ] Confirm timing-invalid rows do not produce speedup metrics.
- [ ] Refresh paper figures/tables only from completed and valid inputs.

Aggregation:

```bash
python3 scripts/experiments/ecg/paper_pipeline.py \
  --skip-run \
  --input-run-glob "results/ecg_experiments/final_paper_runs/slurm/${RUN_TAG}/*" \
  --run-root "results/ecg_experiments/paper_pipeline/${RUN_TAG}_aggregate"
```

### Phase 7: Documentation and Review

- [ ] Update wiki/runbooks with new behavior, caveats, run directories, and
  validation outputs.
- [ ] Update tests when implementation behavior changes.
- [ ] Run final local checks.
- [ ] Summarize: changed files, local checks, Slurm jobs submitted/running,
  aggregation outputs, and remaining blockers.

Final local checks:

```bash
python3 -m pytest -q \
  scripts/test/test_gem5_ecg_pfx_scaffold.py \
  scripts/test/test_sniper_ecg_pfx_scaffold.py \
  scripts/test/test_final_paper_run.py \
  scripts/test/test_paper_pipeline_sniper.py

python3 -m py_compile \
  scripts/experiments/ecg/roi_matrix.py \
  scripts/experiments/ecg/final_paper_run.py \
  scripts/experiments/ecg/paper_pipeline.py \
  scripts/experiments/ecg/make_slurm_shards.py \
  scripts/experiments/ecg/slurm_shard_status.py

git --no-pager diff --check
```

## Blocker Handling

- Missing graph file: record the exact expected path, generate shards with
  filters for available graphs, and continue local validation.
- Missing Slurm credentials or allocation: generate shard TSVs and document the
  exact submission command; continue with local code/tests/docs.
- Missing cross compiler: keep RISC-V build documentation current, validate x86
  and static RISC-V scaffolding locally, and mark benchmark execution blocked.
- Simulator timeout: route the same shard to Slurm, preserve failed/local logs,
  and continue with another unblocked task.
- System pressure: stop that process, record why, lower local scope, and use
  Slurm for the original requirement.

## Definition of Done

- Focused local checks pass.
- Every large final job is either complete, aggregated, or represented by a
  generated Slurm shard with a monitor command.
- No timing-invalid ECG_PFX row is used for speedup claims.
- Docs identify exact run directories and caveats.
- The user has a concise status: what is done, what is on Slurm, what is blocked,
  and the next automatic action.