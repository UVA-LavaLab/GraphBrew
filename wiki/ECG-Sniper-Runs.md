# ECG Sniper Runs

This page documents the current GraphBrew Sniper backend. Sniper sits between
`cache_sim` and gem5: it is intended for scalable multicore/cache-timing
evidence, while gem5 remains the detailed ISA/custom-hint validation path.

## Current Status

Validated single-thread Sniper paths now include:

- safe kernel profiles for PR/BFS/SSSP and DROPLET PR/BFS,
- bounded SIFT full-wrapper synthetic PR/BFS/SSSP,
- bounded SIFT full-wrapper graph-aware replacement on synthetic PR/BFS/SSSP,
- bounded SIFT full-wrapper email-Eu-core `.sg` PR/BFS/SSSP,
- bounded SIFT full-wrapper email-Eu-core `.sg` replacement rows for `LRU`,
  `GRASP`, `POPT`, and `ECG:DBG_PRIMARY`,
- bounded SIFT full-wrapper email-Eu-core `.sg` DROPLET rows for PR/BFS/SSSP,
  including useful SSSP prefetches,
- an experimental ECG_PFX path in the Sniper harness/context, PR/BFS/SSSP
  wrappers, MagicServer dispatch, and cache prefetcher,
- Sniper-only paper aggregation with CSVs, tables, and SVG/PNG figures.

ECG_PFX status: Sniper wrappers now expose `SNIPER_ECG_PFX_TARGET(vertex)` and
emit lookahead target hints using `SNIPER_ECG_PFX_LOOKAHEAD` or
`ECG_PREFETCH_LOOKAHEAD`. The graph-cache context overlay has per-core
`set/has/get/consume/clearPrefetchTargetHint` storage, and
`scripts/setup_sniper.py --apply-overlays` patches Sniper `magic_server.cc` so
`SimUser` `GRVT`/`GPFX` commands store current-vertex and ECG_PFX target hints.
`EcgPfxPrefetcher` consumes those stored targets, maps them to the exported
property region, deduplicates recent cache lines, and reports ECG_PFX counters.
Wrappers also apply a recent property-cache-line target filter before emitting
`GPFX` (`--ecg-pfx-hint-filter`, default `16`) so obvious duplicate candidates
do not cross the simulator magic-call boundary.
The updated overlays have been copied into the ignored Sniper checkout and
Sniper common/standalone have been rebuilt locally. Treat this as experimental:
the first bounded PR SIFT smoke proves hint consumption, request generation, and
one useful Sniper cache prefetch. Final timing claims still need broader
graph/benchmark sweeps with stable fill/useful counters.

Validated synthetic final-run profile:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_sift_ecg_pfx_smoke \
  --run-dir /tmp/graphbrew-sniper-ecg-pfx-profile \
  --no-build --force
```

Result at `/tmp/graphbrew-sniper-ecg-pfx-fixed-hint-smoke`: 3/3 ok rows for
PR/BFS/SSSP at 32kB L3 after rebuilding wrappers and fixing pre-sideband hint
consumption. Counters: PR `4` target hints / `4` ECG_PFX requests /
`pf_issued=3` / `pf_useful=3`; BFS `4` / `4` / `pf_issued=1` /
`pf_useful=1`; SSSP `4` / `4` / `pf_issued=1` / `pf_useful=1`.

The same run aggregates cleanly through the paper pipeline at
`/tmp/graphbrew-paper-pipeline-ecg-pfx-fixed-hint-smoke`. Timing speedup figures
are intentionally skipped because current Sniper ECG_PFX still uses explicit
`GPFX` hint delivery, but cache and prefetch-quality figures are emitted.

Validated file-backed email-Eu-core profile:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_sift_file_ecg_pfx_smoke \
  --run-dir /tmp/graphbrew-sniper-file-ecg-pfx-profile \
  --no-build --force
```

Result at `/tmp/graphbrew-sniper-file-ecg-pfx-profile`: 3/3 ok rows. PR and BFS
use lookahead 4; SSSP uses lookahead 0 because lookahead 1/4 generated requests
but no cache fills for SSSP. Counters: PR `7438` target hints / `1` ECG_PFX
request / `pf_issued=1` / `pf_useful=1`; BFS `31142` / `63` / `pf_issued=3` /
`pf_useful=2`; SSSP `32128` / `63` / `pf_issued=7` / `pf_useful=7`.

The synthetic and email ECG_PFX profiles aggregate cleanly through the paper
pipeline:

```bash
.venv/bin/python3 scripts/experiments/ecg/paper_pipeline.py \
  --skip-run \
  --input-run-dirs /tmp/graphbrew-sniper-ecg-pfx-profile /tmp/graphbrew-sniper-file-ecg-pfx-profile \
  --run-root /tmp/graphbrew-paper-pipeline-sniper-ecg-pfx
```

Validated aggregate: 6/6 ok rows, all `prefetcher=ECG_PFX`, and figures are
written under generic `prefetch_*` names including
`prefetch_prefetch_accuracy_by_benchmark.svg`.

Matched RISC-V comparison, 2026-05-26: after rebuilding wrappers, reapplying
overlays, and relinking Sniper common/standalone, the bounded SIFT synthetic g6
profile gives a matched useful-prefetch proof for BFS:

```text
/tmp/graphbrew-sniper-ecg-pfx-current-proof
PR:   hints=4, ecg_pfx_issued=4, pf_issued=3, pf_useful=3
BFS:  hints=4, ecg_pfx_issued=4, pf_issued=1, pf_useful=1
SSSP: hints=4, ecg_pfx_issued=4, pf_issued=0, pf_useful=0 (active_no_fill)
```

The paired RISC-V gem5 run at `/tmp/graphbrew-riscv-ecg-pfx-kernel-proof` also
reports useful fills for BFS (`pfIssued=1`, `pfUseful=1` in both ROI sections),
so BFS is the current local matched proof point. PR and SSSP remain diagnostic:
PR is useful in Sniper but not gem5 RISC-V on this tiny point, while SSSP is
useful in gem5 RISC-V but `active_no_fill` in Sniper.

Current local constraint:

```text
full-wrapper multicore claims beyond email-Eu-core 1/2-thread smoke still need validation
cit-Patents PR/LRU SIFT is memory-stable locally but exceeded both 1800s and 7200s limits
```

Next unblocked Sniper work:

- run `sniper_sift_cit_patents_long` on a dedicated local terminal or Slurm
  shard before expanding cit-Patents beyond PR/LRU,
- if that passes, expand to cit-Patents PR/BFS/SSSP LRU, then selected
  replacement policies,
- aggregate cit-Patents rows into the paper pipeline,
- expand full-wrapper thread validation beyond the email-Eu-core 1/2-thread
  smoke before making broad multicore claims.

## Current Safe Path

Use the PR-kernel smoke path for the fastest Sniper validation. It exercises
Sniper's ROI flow, cache hierarchy, sideband loading, and replacement-policy
overlays without running the full GraphBrew graph builder. For full-wrapper
single-thread validation, use the bounded SIFT profiles below.

```bash
python3 scripts/setup_sniper.py --skip-build --apply-overlays
USE_SDE=1 make -C bench/include/sniper_sim/snipersim/common -j1
USE_SDE=1 make -C bench/include/sniper_sim/snipersim/standalone -j1

python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_smoke \
  --run-dir /tmp/graphbrew-final-sniper-ecg-smoke \
  --no-build --force
```

Expected result:

```text
[ok] 04_sniper_pr_kernel_smoke_synthetic_g12_pr: 9 ok rows
```

ECG_PFX validation:

```bash
python3 -m pytest -q scripts/test/test_sniper_ecg_pfx_scaffold.py
make sniper-pr_kernel_smoke sniper-bfs_kernel_smoke sniper-sssp_kernel_smoke
python3 scripts/setup_sniper.py --skip-build --apply-overlays
USE_SDE=1 make -C bench/include/sniper_sim/snipersim/common -j1
USE_SDE=1 make -C bench/include/sniper_sim/snipersim/standalone -j1

python3 scripts/experiments/ecg/roi_matrix.py \
  --suite sniper --benchmark pr --sniper-workload benchmark \
  --allow-sniper-benchmark-workload --sniper-frontend sift \
  --sniper-memory-limit-gb 4 --policies LRU \
  --prefetcher ECG_PFX --prefetcher-level l2 \
  --options "-g 6 -k 8 -o 2 -n 1 -i 1" \
  --l3-sizes 32kB --out-dir /tmp/graphbrew-sniper-ecg-pfx-pr-smoke \
  --timeout-sniper 300 --no-build
```

Validated result: 5/5 static scaffold tests passed, native Sniper PR/BFS/SSSP
smoke wrappers built, overlays copied, Sniper common/standalone relinked, and
the bounded PR SIFT ECG_PFX row passed with `ecg_pfx_target_hints_seen=338`,
`ecg_pfx_issued=1`, `pf_issued=1`, and `pf_useful=1`.

## Safety Proof Snapshot

Validated on 2026-05-24 with the tracked custom Sniper config:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_smoke sniper_thread_scaling sniper_kernel_smoke sniper_droplet_smoke \
  --run-dir /tmp/graphbrew-sniper-all-safe-paths \
  --no-build --force
```

Proof result:

```text
rows: 46
statuses: {'ok': 46}
stages: {'04_sniper_pr_kernel_smoke': 9, '05_sniper_thread_scaling_smoke': 8, '06_sniper_kernel_smoke_suite': 27, '07_sniper_droplet_kernel_smoke': 2}
config: graphbrew/graph_sniper
MimicOS memory/kernel: 4096 MB / 128 MB
workloads: kernel_smoke, pr_kernel_smoke
prefetchers: DROPLET, none
threads: 1, 2
configuration errors: none
```

Post-run process/memory checks found no lingering `run-sniper`, `sde64`,
`bench/bin_sniper`, `sg_kernel`, standalone Sniper, or gem5 process, and system
memory stayed stable. The default unsafe-workload gates were also checked:
`--sniper-workload benchmark` and `--sniper-workload sg_kernel` return
`unsupported` unless their explicit `--allow-*` flags are present; explicit
debug dry-runs are wrapped with `prlimit --as=17179869184`.

The first bounded full-wrapper frontend probe also passed with SIFT and a lower
4 GiB address-space cap:

```bash
python3 scripts/experiments/ecg/roi_matrix.py \
  --suite sniper --benchmark pr --sniper-workload benchmark \
  --allow-sniper-benchmark-workload --sniper-frontend sift \
  --sniper-memory-limit-gb 4 --policies LRU --l3-sizes 32kB \
  --out-dir /tmp/graphbrew-sniper-sift-full-wrapper-probe \
  --timeout-sniper 300 --no-build
```

Result: `status=ok`, `sniper_frontend=sift`, `sniper_memory_limit_gb=4.0`, no
lingering simulator process, and stable memory. BFS and SSSP synthetic full
wrappers were then validated with the same SIFT frontend and 4 GiB cap:

```text
/tmp/graphbrew-sniper-sift-full-wrapper-bfs-probe:  status=ok
/tmp/graphbrew-sniper-sift-full-wrapper-sssp-probe: status=ok
```

Use `sniper_sift_benchmark_suite` to reproduce the bounded PR/BFS/SSSP synthetic
full-wrapper suite. Real email-Eu-core `.sg` full-wrapper profiles are validated
below; larger `.sg` graphs remain gated until individually proven under the same
bounded SIFT path.

The validated bounded replacement-policy full-wrapper profile is:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_sift_replacement_smoke \
  --run-dir /tmp/graphbrew-sniper-sift-replacement-smoke \
  --no-build --force
```

It runs PR/BFS/SSSP full wrappers with `LRU`, `GRASP`, `POPT`, and
`ECG:DBG_PRIMARY` using `--sniper-frontend sift`, the explicit benchmark allow
flag, and a 4 GiB address-space cap.

The validated real file-backed `.sg` SIFT smoke is:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_sift_file_smoke \
  --run-dir /tmp/graphbrew-sniper-sift-file-smoke \
  --no-build --force
```

It uses `results/graphs/email-Eu-core/email-Eu-core.sg`, PR/BFS/SSSP, `LRU`,
`--sniper-frontend sift`, and the same 4 GiB address-space cap.

Validated result at `/tmp/graphbrew-sniper-sift-file-smoke`: 3/3 ok rows for
PR/BFS/SSSP, `sniper_frontend=sift`, `sniper_memory_limit_gb=4.0`, and combined
CSV rows include `final_graph=email-Eu-core` plus `final_graph_path`.

The validated file-backed replacement profile is:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_sift_file_replacement_smoke \
  --run-dir /tmp/graphbrew-sniper-sift-file-replacement-smoke \
  --no-build --force
```

It uses the same email `.sg` graph with `LRU`, `GRASP`, `POPT`, and
`ECG:DBG_PRIMARY` under the bounded SIFT frontend.

File-backed SSSP DROPLET is no longer only a no-fill probe when run on the email
`.sg` full wrapper under SIFT. The probe at
`/tmp/graphbrew-sniper-sift-file-droplet-sssp-probe` produced:

```text
status=ok
droplet_activity=issued
pf_issued=1160
pf_fillups=1160
pf_useful=494
droplet_edge_accesses=6128
droplet_indirect_issued=997
```

Use `sniper_sift_file_droplet_smoke` to reproduce file-backed PR/BFS/SSSP
DROPLET validation on `email-Eu-core.sg` under SIFT with the 4 GiB cap.

Validated result at `/tmp/graphbrew-sniper-sift-file-droplet-smoke`:

```text
rows: 3/3 ok
droplet_activity: issued for PR, BFS, SSSP
prefetch issued/useful:
  PR:   1990 / 1009
  BFS:   349 / 191
  SSSP:  942 / 432
```

The next scale-out smoke is cit-Patents PR/LRU:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_sift_cit_patents_smoke \
  --run-dir /tmp/graphbrew-sniper-sift-cit-patents-smoke \
  --no-build --force
```

It uses `results/graphs/cit-Patents/cit-Patents.sg`, `PR`, `LRU`,
`--sniper-frontend sift`, and an 8 GiB address-space cap. Use this as the gate
before expanding cit-Patents to BFS/SSSP or graph-aware replacement policies.

Local result: the first cit-Patents PR/LRU run reached detailed ROI, stayed
memory-stable at about 2.7 GiB RSS, and timed out at 1800 seconds before writing
CSV output. This was not a memory regression. The timeout exposed a runner bug
where `TimeoutExpired` escaped and left SIFT children alive; `roi_matrix.py` now
kills timed-out process groups and writes an `exit_code=124` error row. The
profile now uses `timeout_sniper=7200`.

The 7200-second local retry at
`/tmp/graphbrew-sniper-sift-cit-patents-smoke-long` also reached detailed ROI,
stayed memory-stable at about 2.7 GiB RSS, and timed out cleanly with
`status=error` and `error=exit_code=124`. No simulator children survived cleanup.
`final_paper_run.py` preserves that failed matrix row in
`combined_roi_matrix.csv` so downstream tooling can see the failed gate while the
run still exits nonzero when the job is executed normally. Treat this as a local
runtime blocker, not a cache/Sniper memory failure.

Use the longer dedicated profile for the next attempt:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_sift_cit_patents_long \
  --run-dir /tmp/graphbrew-sniper-sift-cit-patents-long \
  --no-build --force
```

Do not expand cit-Patents to BFS/SSSP or graph-aware replacement policies until
PR/LRU completes under this longer local/Slurm path.

Real full-wrapper thread scaling initially failed locally. The runner now emits
`-g general/total_cores=<threads>` after loading `graphbrew/graph_sniper`; this
is required because the base config defaults to one core and would otherwise
override `run-sniper -n 2`. With only that fixed, a bounded email-Eu-core PR
2-thread probe starts two Sniper trace threads, then aborts with:

```text
barrier_sync_server: *ERROR* No threads running, no timeout. Application has deadlocked...
ValueError: Invalid prefix roi-begin
exit_code=134
```

Sniper's own OpenMP example uses `OMP_WAIT_POLICY=passive`. `roi_matrix.py` now
sets `OMP_WAIT_POLICY=passive` for Sniper benchmark processes by default, and
records it as `sniper_omp_wait_policy`. With passive waits, email-Eu-core PR
2-thread rows pass for both `LRU` and `GRASP` under bounded SIFT:

```text
/tmp/graphbrew-sniper-thread-passive-probe:       LRU,   threads=2, status=ok
/tmp/graphbrew-sniper-thread-passive-grasp-probe: GRASP, threads=2, status=ok
```

`sniper_sift_file_thread_smoke` is restored as the reproducible 1/2-thread
email-Eu-core PR smoke. Validated result:

```text
/tmp/graphbrew-sniper-sift-file-thread-smoke-passive-final
rows: 4/4 ok
threads: 1, 2
policies: LRU, GRASP
sniper_omp_wait_policy: passive
```

Treat this as smoke-level multicore validation, not final thread-scaling evidence.

The profile covers:

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

## Overlay Workflow

Tracked GraphBrew Sniper sources live under:

```text
bench/include/sniper_sim/overlays/
bench/include/sniper_sim/configs/graphbrew/
```

The upstream checkout lives under ignored `bench/include/sniper_sim/snipersim/`.
Do not hand-edit tracked policy logic inside the ignored checkout. Instead:

```bash
python3 scripts/setup_sniper.py --skip-build --apply-overlays
```

This copies tracked overlays into Sniper and patches:

- the tracked `graphbrew/graph_sniper.cfg` and `graphbrew/graph_cache_config.cfg`
  configs into the ignored Sniper checkout,
- replacement-policy enum/factory entries for `grasp`, `popt`, and `ecg`,
- insertion-address hooks for graph-aware policies,
- the prefetcher factory entry for `droplet`,
- `.sniper_overlays.json` metadata.

After applying overlays, relink Sniper:

```bash
USE_SDE=1 make -C bench/include/sniper_sim/snipersim/common -j1
USE_SDE=1 make -C bench/include/sniper_sim/snipersim/standalone -j1
```

## Per-Run Sidebands

Runner-launched Sniper jobs set sideband environment variables per output
directory:

```text
SNIPER_GRAPHBREW_CTX
SNIPER_POPT_MATRIX
SNIPER_GRAPHBREW_OUT_EDGES
SNIPER_GRAPHBREW_IN_EDGES
```

Rows record these paths as `sniper_context_path`, `sniper_popt_matrix_path`,
`sniper_out_edges_path`, `sniper_in_edges_path`, and `sniper_sideband_dir`.

Direct native/manual runs still use `/tmp` defaults unless those environment
variables are set explicitly.

GraphBrew Sniper sidebands currently describe virtual benchmark addresses.
`roi_matrix.py --suite sniper` therefore defaults to
`--sniper-address-domain virtual`, which adds
`-g general/translation_enabled=false` so cache-policy and prefetch callbacks see
the same address domain as the sidebands. `--sniper-address-domain translated`
keeps Sniper's baseline MMU path and is reserved for future translated/physical
sideband export work.

`roi_matrix.py --suite sniper` now defaults to the tracked custom Sniper config:

```text
--sniper-root bench/include/sniper_sim/snipersim
--sniper-frontend live
--sniper-omp-wait-policy passive
--sniper-base-config graphbrew/graph_sniper
```

`--sniper-root` can be absolute on Slurm nodes with a prepared Sniper install,
or relative to the GraphBrew repository root for local runs. `--sniper-frontend
sift` inserts Sniper's `--sift` recorder path for bounded frontend experiments;
keep `live` for claim-ready safe profiles until SIFT probes are separately
validated. `--sniper-omp-wait-policy passive` follows Sniper's OpenMP examples
and avoids the SIFT barrier deadlock seen with default active/spinning waits.
The base config
includes the GraphBrew-compatible MimicOS/MMU stack, a private
L1D/private L2/shared NUCA-LLC hierarchy, 64B cache lines, and reduced
`reserve_thp` defaults of 4096 MB memory and 128 MB kernel reservation. The
runner still overrides cache sizes, ways, replacement policy, core count, and
address domain per row so proof and paper sweeps remain explicit in the CSV.

## Thread Smoke

The current thread profile validates runner/config plumbing only. The
`pr_kernel_smoke` workload is single-threaded, so these rows are not final
thread-scaling evidence.

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_thread_scaling \
  --run-dir /tmp/graphbrew-final-sniper-thread-smoke \
  --no-build --force
```

Expected result:

```text
[ok] 05_sniper_thread_scaling_smoke_synthetic_g12_pr: 8 ok rows
```

## Multi-Benchmark Kernel Smoke

Builder-free PR, BFS, and SSSP kernel smokes can be run through one profile:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_kernel_smoke \
  --run-dir /tmp/graphbrew-final-sniper-kernel-smoke \
  --no-build --force
```

This validates Sniper ROI/stat/sideband plumbing across multiple benchmark
shapes and the full final replacement-policy label surface without invoking the
full GraphBrew graph builder.

Expected result:

```text
[status] jobs=3 counts={'ok': 3}
[status] combined_roi_matrix.csv: 27 row(s)
```

## Paper Aggregation

Existing Sniper final-run directories can be aggregated without launching new
simulations:

```bash
python3 scripts/experiments/ecg/paper_pipeline.py \
  --skip-run \
  --input-run-dirs /tmp/graphbrew-final-sniper-kernel-smoke \
                   /tmp/graphbrew-final-sniper-thread-smoke \
                   /tmp/graphbrew-final-sniper-droplet-smoke \
  --run-root /tmp/graphbrew-paper-pipeline-sniper-check-final \
  --no-build
```

The pipeline keeps Sniper `threads` in its relative-metric grouping and writes
Sniper-specific CSVs under `aggregate/`, including:

```text
sniper_relative_metrics.csv
sniper_relative_policy_summary.csv
sniper_thread_scaling_metrics.csv
sniper_cpi_stack_summary.csv
backend_direction_agreement.csv   # when multiple backends are present
```

Validated Sniper-only aggregation after the SIFT/file-backed smokes:

```bash
python3 scripts/experiments/ecg/paper_pipeline.py \
  --skip-run \
  --input-run-dirs /tmp/graphbrew-sniper-all-safe-paths \
                   /tmp/graphbrew-sniper-sift-benchmark-suite-smoke \
                   /tmp/graphbrew-sniper-sift-replacement-smoke \
                   /tmp/graphbrew-sniper-sift-file-smoke \
                   /tmp/graphbrew-sniper-sift-file-replacement-smoke \
  --run-root /tmp/graphbrew-paper-pipeline-sniper-sift-file-check \
  --no-build --force
```

Result after adding file-backed DROPLET: 79 Sniper ROI rows aggregated,
`final_graph=email-Eu-core` present in `roi_matrix_all.csv`, 5 DROPLET ROI rows,
3 Sniper prefetch-quality rows, Sniper relative/CPI/thread/prefetch CSVs written,
tables written, and 12 SVG plus 12 PNG figures emitted after installing the
declared `matplotlib` Python dependency.

## DROPLET Status

`roi_matrix.py --suite sniper --prefetcher DROPLET` attaches the tracked
`droplet` prefetcher to L2 by default, or L1D with `--prefetcher-level l1d`.
The default DROPLET knobs are artifact-informed from the old public Sniper-6.1
tree: `droplet_prefetch_degree=1`, `droplet_indirect_degree=16`, and
`droplet_stride_table_size=64`.

Current safe active smoke:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_droplet_smoke \
  --run-dir /tmp/graphbrew-final-sniper-droplet-smoke \
  --no-build --force
```

Expected result:

```text
[ok] 07_sniper_droplet_kernel_smoke_synthetic_g12_pr: 1 ok rows
[ok] 07_sniper_droplet_kernel_smoke_synthetic_g12_bfs: 1 ok rows
```

Validated behavior in virtual address-domain mode at the profile's 32kB LLC
mechanism-smoke point:

- sideband JSON and edge shadow data load successfully,
- PR and BFS report nonzero edge accesses, indirect requests, Sniper prefetch
  fills, and useful prefetches.

Fresh useful-activity validation at
`/tmp/graphbrew-sniper-droplet-useful-activity-smoke` produced:

```text
PR:  droplet_activity=issued, droplet_useful_activity=useful, pf_issued=1, pf_useful=1
BFS: droplet_activity=issued, droplet_useful_activity=useful, pf_issued=1, pf_useful=1
```

Artifact-default validation at `/tmp/graphbrew-sniper-droplet-artifact-profile`
used the same profile after the 1/16/64 DROPLET default update:

```text
PR:  droplet_prefetch_degree=1, droplet_indirect_degree=16, droplet_stride_table_size=64, edge_accesses=7, indirect_issued=1, pf_issued=1, pf_useful=1, l3_misses=91
BFS: droplet_prefetch_degree=1, droplet_indirect_degree=16, droplet_stride_table_size=64, edge_accesses=3, indirect_issued=4, pf_issued=1, pf_useful=1, l3_misses=448
```

The aggregate pipeline also accepts this run:

```bash
.venv/bin/python3 scripts/experiments/ecg/paper_pipeline.py \
  --skip-run \
  --input-run-dirs /tmp/graphbrew-sniper-droplet-artifact-profile \
  --run-root /tmp/graphbrew-paper-pipeline-droplet-artifact-smoke
```

Validated aggregate: 2/2 ok rows, `prefetcher=DROPLET`, and DROPLET
prefetch-quality figures/tables are emitted.

SSSP now reports edge accesses and indirect requests after the cache-line overlap
fix. The tiny kernel smoke still has zero Sniper prefetch fills and is marked
`active_no_fill`. File-backed rows can also be mechanically active without useful
fills, so claim-oriented analysis must inspect both `droplet_activity` and
`droplet_useful_activity` / `pf_useful`.

The file-backed profile intentionally uses tuned normal-prefetcher knobs rather
than the artifact defaults:

```text
droplet_prefetch_degree=2
droplet_indirect_degree=4
droplet_stride_table_size=16
```

The artifact-default file-backed probe at
`/tmp/graphbrew-sniper-file-droplet-artifact-profile` completed PR/BFS/SSSP, but
BFS and SSSP produced `pf_useful=0` and substantially higher Sniper ticks even
though L3 misses remained low. It is useful for artifact-parameter exploration,
not as final useful-prefetch evidence.

The current tuned revalidation at
`/tmp/graphbrew-sniper-file-droplet-tuned-profile` also completed PR/BFS/SSSP,
but after the current-edge indirect-prefetch change only PR produced useful
prefetch fills:

```text
PR:   pf_issued=2384, pf_useful=1093, l3_misses=4713, sim_ticks=838049347856
BFS:  pf_issued=18,   pf_useful=0,    l3_misses=3495, sim_ticks=11911918329032
SSSP: pf_issued=1,    pf_useful=0,    l3_misses=5954, sim_ticks=11695805881518
```

Treat file-backed DROPLET as active and L3-reducing at this point, but not yet
claim-ready for useful BFS/SSSP prefetches under the current implementation.

## Guardrails

- Do not use the old unbounded live/SDE full-wrapper path as a smoke target. A
  tiny full-PR probe previously left a child process at roughly 53 GiB RSS and
  heavy swap use. Single-thread full-wrapper validation should use the bounded
  SIFT profiles with explicit `--allow-sniper-benchmark-workload` and a memory
  cap.
- Do not run `bench/bin_sniper/sg_kernel` under Sniper/SDE as a smoke target yet.
  Native `.sg` execution is clean, but the Sniper/SDE run repeated the same
  high-memory child-process behavior.
- If you explicitly debug `--allow-sniper-benchmark-workload` or
  `--allow-sniper-sg-kernel-workload`, keep the default
  `--sniper-memory-limit-gb 16` guard or choose another explicit cap. The runner
  wraps these unsafe paths with `prlimit` before launching Sniper.
- Keep the tracked `graphbrew/graph_sniper` config and reduced MimicOS allocator
  defaults unless a run explicitly studies address translation or physical-memory
  allocation behavior.
- Use `pr_kernel_smoke` for fastest Sniper validation. Use bounded SIFT profiles
  for single-thread full-wrapper evidence. Do not claim full-wrapper multicore
  evidence until the Sniper/SIFT OpenMP barrier blocker is fixed.
- Do not compare raw Sniper cycles/ticks directly against gem5 as exact values.
  Use normalized speedups, direction, and policy-rank agreement.
- Keep generated `results/` and ignored `snipersim/` artifacts out of commits.

## External Sniper Usage Check

The ANU Sniper lab, Sniper manual, local `run-sniper` help, and Sniper user-list
threads agree with the core command shape we use:

```text
run-sniper -d <outdir> -c <config> -g section/key=value -- <program> <args>
```

Useful confirmations:

- `-d` is the right way to isolate output directories.
- multiple `-c` and `-g` options are expected and can be mixed,
- `--roi` is the right mode when the benchmark marks ROI start/end; pre-ROI code
  is fast-forwarded and the ROI runs in detailed mode,
- Sniper is interval simulation, so use CPI stacks, normalized speedups, and
  policy-rank agreement rather than treating ticks as gem5-equivalent.

The same references also reinforce the likely mistake in the unsafe path: live
SDE/SIFT frontend execution of full GraphBrew binaries is fragile for our current
environment. For full workloads, the next investigation should prefer bounded
trace/pinball/SIFT reuse or another frontend/run-mode fix instead of repeatedly
live-running full wrappers under SDE.

Our local Sniper checkout's plain `base`, `gainestown`, and `nehalem` configs are
not directly usable because GraphBrew's memory-manager fork expects MimicOS
configuration keys. The runner now uses tracked `graphbrew/graph_sniper`, which
includes the compatible MimicOS/MMU stack and reduces the original baseline's
oversized `reserve_thp` defaults:

```text
perf_model/reserve_thp/memory_size: 131072 MB -> 4096 MB
perf_model/reserve_thp/kernel_size: 32768 MB -> 128 MB
```

These are exposed as `--sniper-mimicos-memory-mb` and
`--sniper-mimicos-kernel-mb`. A `/bin/true` smoke with the reduced allocator
completed with about 198 MiB RSS, while the original baseline advertises a
128 GiB memory model and 32 GiB kernel reservation.

## Native `.sg` Diagnostic

`bench/bin_sniper/sg_kernel` is useful for checking `.sg` parameters and
kernel-only sideband export without Sniper/SDE:

```bash
for benchmark in pr bfs sssp; do
  /usr/bin/time -v timeout 300s bench/bin_sniper/sg_kernel \
    --benchmark "$benchmark" \
    -f results/graphs/email-Eu-core/email-Eu-core.sg \
    -i 1 -r 0 -d 1
done
```

Validated native memory behavior:

```text
tiny.sg PR/BFS/SSSP: under 9 MiB RSS
email-Eu-core.sg PR/BFS/SSSP: under 9 MiB RSS
cit-Patents.sg PR/BFS: about 576 MiB RSS
cit-Patents.sg SSSP: about 718 MiB RSS
```

The same `sg_kernel` target under Sniper/SDE repeated the roughly 50 GiB runaway
child-process issue, so `roi_matrix.py --sniper-workload sg_kernel` is guarded by
default and returns `unsupported` unless `--allow-sniper-sg-kernel-workload` is
passed for tightly bounded frontend debugging.

## Useful Commands

```bash
# Apply overlays and relink
python3 scripts/setup_sniper.py --skip-build --apply-overlays
USE_SDE=1 make -C bench/include/sniper_sim/snipersim/common -j1
USE_SDE=1 make -C bench/include/sniper_sim/snipersim/standalone -j1

# Direct safe replacement smoke
python3 scripts/experiments/ecg/roi_matrix.py \
  --suite sniper \
  --policies LRU SRRIP GRASP POPT_CHARGED POPT ECG:DBG_PRIMARY \
  --l3-sizes 32kB \
  --no-build

# Active PR/BFS DROPLET smoke
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_droplet_smoke \
  --run-dir /tmp/graphbrew-final-sniper-droplet-smoke \
  --no-build --force

# Native-only .sg diagnostic; do not wrap in run-sniper yet
/usr/bin/time -v timeout 300s bench/bin_sniper/sg_kernel \
  --benchmark pr -f results/graphs/email-Eu-core/email-Eu-core.sg -i 1
```