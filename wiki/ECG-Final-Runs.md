# ECG Final Runs

This page records the current ECG / gem5 final-paper run workflow. It is the
place to check before starting long cache-policy runs.

For UVA Slurm runs where each graph/benchmark/policy shard runs independently
and results are aggregated later, see [ECG Slurm runs](ECG-Slurm-Runs.md).
For the current large-graph staging blocker and exact download/convert checklist,
see [the ECG final graph staging plan](../plans/ecg-graph-staging-plan.md).
For the autonomous finish-the-stack backlog, use
[the final-paper completion task](../plans/ecg-autonomous-final-paper-completion.md);
it routes bounded validation to the local workstation and large gem5/Sniper or
final-scale jobs to Slurm shards.

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

## ROI And Sidebands

gem5 wrappers load/build the graph, allocate property arrays, export sideband
metadata, and export the P-OPT matrix before the measured kernel ROI. The
measured region is the graph algorithm loop between `GEM5_WORK_BEGIN` and
`GEM5_WORK_END`.

Runner-launched gem5 jobs set per-run sideband paths:

```text
GEM5_GRAPHBREW_CTX
GEM5_POPT_MATRIX
GEM5_GRAPHBREW_OUT_EDGES
GEM5_GRAPHBREW_IN_EDGES
```

Rows record these as `gem5_context_path`, `gem5_popt_matrix_path`,
`gem5_out_edges_path`, `gem5_in_edges_path`, and `gem5_sideband_dir`. This keeps
gem5 aligned with the Sniper sideband discipline and avoids shared `/tmp`
metadata when jobs are sharded.

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

ECG_PFX cache_sim profile:

```text
LRU + ECG_PFX
GRASP + ECG_PFX
POPT + ECG_PFX
ECG_DBG_ONLY + ECG_PFX
ECG_DBG_PRIMARY + ECG_PFX
ECG_POPT_PRIMARY + ECG_PFX
```

`ECG_PFX` is currently a first-class cache_sim prefetch path. It is exposed in
`roi_matrix.py` as `--prefetcher ECG_PFX` with POPT-ranked defaults matching the
proof-matrix path. gem5 has an experimental x86 pseudo-op hint path behind
`--allow-gem5-ecg-pfx`; Sniper has an experimental `ecg_pfx` prefetcher path
behind tracked overlays and a Sniper common/standalone relink. Neither timing
backend should be used for final performance claims until larger rows show
stable issued/useful prefetch counters.

Current gem5 ECG_PFX smoke status: the x86 instruction-mode tiny profile
completed at `/tmp/graphbrew-gem5-x86-pfx-instruction-smoke` with all three jobs
ok and `timing_valid_for_speedup=0`. PR and SSSP issue ECG_PFX prefetches
(`pfIdentified=pfIssued=4` for PR, `2` for SSSP), while the current BFS tiny
point remains inactive (`pfIssued=0`). Useful prefetches are proven on the
RISC-V/Sniper BFS scale rows below, not by this x86 tiny smoke. Do not use these
rows for performance claims yet.

RISC-V `ecg.extract` scaffold status: the tracked custom-0 decoder now strips
the low 32-bit real vertex ID, decodes the fixed paper DBG/POPT/PFX fields, and
stores the decoded metadata/PFX target in GraphBrew hint storage. A RISC-V
gem5 build now passes and verifies at
`bench/include/gem5_sim/gem5/build/RISCV/gem5.opt`. The local
`riscv64-linux-gnu-gcc/g++` cross toolchain is installed, and a static RISC-V PR
wrapper builds with `make gem5-riscv-m5ops-pr`. A tiny RISC-V instruction-mode
ECG_PFX smoke completed at `/tmp/graphbrew-riscv-ecg-pfx-g6-smoke` with
`pfIdentified=2` and `pfIssued=2` (`pfUseful=0` on that tiny point). Treat this
as instruction-path activation evidence, not a timing claim.

x86 instruction-mode scaffold status: `--ecg-pfx-delivery instruction` now uses
the gem5 x86 pseudo-op encoding (`0F 04 imm16`, `M5OP_WORK_BEGIN=0x5a`) directly
at the ECG_PFX hint site, with the GraphBrew PFX work ID and target vertex
placed in the x86 pseudo-inst ABI argument registers. This removes the extra C
function-call wrapper from the x86 hint path, but it is still a gem5 pseudo-op
prototype and remains timing-invalid for speedup claims.

Small RISC-V vs Sniper ECG_PFX evaluation, 2026-05-26:

```text
/tmp/graphbrew-riscv-ecg-pfx-kernel-proof
/tmp/graphbrew-sniper-ecg-pfx-current-proof
/tmp/graphbrew-ecg-pfx-riscv-sniper-kernel-proof/summary.csv
```

Both legs use synthetic g6, LRU, ECG_PFX at L2, 32kB L3, POPT mode, window 16,
lookahead 4, and hint filter 16. The gem5 leg uses the RISC-V `ecg.extract`
instruction path; the Sniper leg uses the bounded SIFT benchmark wrapper and
explicit `GPFX` hint delivery.

| Kernel | Backend | Status | Counter readout | Interpretation |
|---|---|---|---|---|
| PR | gem5 RISC-V | `ok` | section 1/2: `pfIdentified=1`, `pfIssued=1`, `pfUseful=0` | instruction delivery reaches gem5's prefetch queue, but this tiny PR point has no useful fill |
| PR | Sniper SIFT | `ok` | `hints=4`, `ecg_pfx_issued=4`, `pf_issued=3`, `pf_useful=3` | Sniper hint delivery and cache-fill path are useful on this point |
| BFS | gem5 RISC-V | `ok` | section 1/2: `pfIdentified=1`, `pfIssued=1`, `pfUseful=1` | matched instruction-path useful-prefetch proof |
| BFS | Sniper SIFT | `ok` | `hints=4`, `ecg_pfx_issued=4`, `pf_issued=1`, `pf_useful=1` | matched Sniper useful-prefetch proof |
| SSSP | gem5 RISC-V | `ok` | section 1/2: `pfIdentified=2`, `pfIssued=2`, `pfUseful=1` | RISC-V instruction path has useful fills |
| SSSP | Sniper SIFT | `active_no_fill` | `hints=4`, `ecg_pfx_issued=4`, `pf_issued=0` | Sniper consumes hints and generates requests but does not enqueue fills for this SSSP point |

Use BFS as the current local matched proof point: both backends report nonzero
issued and useful prefetch counters. PR and SSSP remain useful diagnostic rows,
but not complete matched proof points. None of these tiny rows are performance
claims; `timing_valid_for_speedup=0` still applies to ECG_PFX detailed-sim rows.

BFS scale-up smoke, 2026-05-26:

```text
/tmp/graphbrew-ecg-pfx-riscv-sniper-bfs-scale-proof/summary.csv
```

| Scale | Root | Backend | Counter readout | Interpretation |
|---|---:|---|---|---|
| g7 | 1 | gem5 RISC-V | section 1/2: `pfIdentified=2`, `pfIssued=2`, `pfUseful=1` | instruction path remains useful one scale up |
| g7 | 1 | Sniper SIFT | `hints=8`, `ecg_pfx_issued=8`, `pf_issued=1`, `pf_useful=1` | matched Sniper useful-prefetch proof |
| g8 | 9 | gem5 RISC-V | section 1/2: `pfIdentified=9`, `pfIssued=9`, `pfUseful=2` | instruction path remains active and useful at g8 |
| g8 | 9 | Sniper SIFT | `hints=14`, `ecg_pfx_issued=14`, `pf_issued=3`, `pf_useful=3` | matched Sniper useful-prefetch proof |
| g9 | 20 | gem5 RISC-V | section 1/2: `pfIdentified=9`, `pfIssued=9`, `pfUseful=7` | instruction path remains active and more useful at g9 |
| g9 | 20 | Sniper SIFT | `hints=757`, `ecg_pfx_issued=26`, `pf_issued=4`, `pf_useful=2` | matched Sniper useful-prefetch proof |
| g10 | 0 | gem5 RISC-V | section 1/2: `pfIdentified=22`, `pfIssued=22`, `pfUseful=10` | instruction path remains active and useful at g10 |
| g10 | 0 | Sniper SIFT | `hints=3362`, `ecg_pfx_issued=51`, `pf_issued=2`, `pf_useful=2` | matched Sniper useful-prefetch proof |

Root choice matters for BFS: g7 root 0 and g8 roots 0/1 had no ECG_PFX hint
activity, while nearby roots produced useful fills; at g9, roots 0-11 and many
higher roots were active, with root 20 giving the strongest local Sniper useful
signal in the sweep. At g10, root 0 already produced useful fills, but the first
root took long enough that the rest of the local sweep was stopped. Use root
sweeps as a local mechanism-selection step before spending gem5 time on a larger
BFS point, and move beyond g10 to a long local window or Slurm.

Small evaluation recipe:

```bash
make gem5-riscv-m5ops-pr gem5-riscv-m5ops-bfs gem5-riscv-m5ops-sssp PARALLEL=2

root=/tmp/graphbrew-riscv-ecg-pfx-kernel-proof
rm -rf "$root" && mkdir -p "$root"
for bench in pr bfs sssp; do
  case "$bench" in
    pr) opts="-g 6 -k 8 -o 2 -n 1 -i 1" ;;
    bfs) opts="-g 6 -k 8 -o 2 -n 1 -r 0" ;;
    sssp) opts="-g 6 -k 8 -o 2 -n 1 -r 0 -d 1" ;;
  esac
  out="$root/$bench"
  mkdir -p "$out/sidebands"
GEM5_GRAPHBREW_CTX="$out/sidebands/ctx.json" \
GEM5_POPT_MATRIX="$out/sidebands/popt.bin" \
GEM5_GRAPHBREW_OUT_EDGES="$out/sidebands/out_edges.bin" \
GEM5_GRAPHBREW_IN_EDGES="$out/sidebands/in_edges.bin" \
ECG_PREFETCH_MODE=2 \
ECG_PREFETCH_WINDOW=16 \
bench/include/gem5_sim/gem5/build/RISCV/gem5.opt \
  --outdir="$out/m5out" \
  bench/include/gem5_sim/configs/graphbrew/graph_se.py \
  --binary "bench/bin_gem5/${bench}_riscv_m5ops" \
  --options "$opts" \
  --policy LRU \
  --prefetcher ECG_PFX \
  --prefetcher-level l2 \
  --ecg-pfx-lookahead 4 \
  --ecg-pfx-hint-filter 16 \
  --ecg-pfx-delivery instruction \
  --l1d-size 1kB --l2-size 2kB --l3-size 32kB --l3-ways 16
done

python3 scripts/setup_sniper.py --skip-build --apply-overlays
USE_SDE=1 make -C bench/include/sniper_sim/snipersim/common -j1
USE_SDE=1 make -C bench/include/sniper_sim/snipersim/standalone -j1

python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_sift_ecg_pfx_smoke \
  --run-dir /tmp/graphbrew-sniper-ecg-pfx-current-proof \
  --no-build --force --skip-validation-gate
```

Full evaluation guide:

1. Keep the local gate small: PR/BFS/SSSP g6 first, then BFS g7/g8/g9/g10
  root-selected smokes. BFS roots g7/r1, g8/r9, g9/r20, and g10/r0 are the
  current matched proof rows because both backends report useful fills there.
2. Use `cache_sim` to select graph/cache points where PFX has useful fills or
  clear demand-miss reductions before spending gem5/Sniper time.
3. For gem5, run RISC-V instruction delivery on one benchmark/policy at a time;
  do not combine it with large graph sweeps locally.
4. For Sniper, use bounded SIFT full wrappers with `--sniper-memory-limit-gb`
  and require nonzero `ecg_pfx_target_hints_seen`, `ecg_pfx_issued`, and
  eventually `pf_issued`/`pf_useful` before claiming active cache prefetching.
5. Once PR/BFS/SSSP pass local activation on available graphs, generate Slurm
  shards for larger gem5/Sniper/file-backed matrices instead of running them on
  the workstation.
6. Aggregate completed shards with `paper_pipeline.py --skip-run`; verify
  `timing_valid_for_speedup` before using any speedup column.

Local cache_sim diversity screens, 2026-05-26:

```bash
# Small graph, broad kernel surface including PR-SPMV, CC, CC-SV, BC, and TC.
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile local_cache_sim_diversity_smoke \
  --run-dir /tmp/graphbrew-local-cache-diversity-smoke \
  --dry-run --no-build

# Medium graph, safe expanded kernel subset. Use filters for first runs.
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile local_cache_sim_diversity_medium \
  --benchmark pr bfs cc \
  --policy LRU GRASP POPT_CHARGED ECG_DBG_PRIMARY_CHARGED \
  --run-dir /tmp/graphbrew-local-cache-diversity-medium \
  --dry-run --no-build

# Small ECG_PFX screen across extra kernels before spending detailed-sim time.
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile local_cache_sim_pfx_diversity_smoke \
  --run-dir /tmp/graphbrew-local-cache-pfx-diversity-smoke \
  --dry-run --no-build
```

Keep the first actual local runs filtered by `--benchmark` and `--policy`; the
medium profile intentionally excludes BC and TC because those kernels can grow
quickly on cit-Patents.

Summarize completed local cache_sim screens with:

```bash
python3 scripts/experiments/ecg/local_cache_screen_summary.py \
  --input diversity=/tmp/graphbrew-local-cache-diversity-smoke-full/combined_roi_matrix.csv \
  --input pfx=/tmp/graphbrew-local-cache-pfx-diversity-smoke-full/combined_roi_matrix.csv \
  --out /tmp/graphbrew-local-cache-screen-summary.csv
```

The summary ranks policies by L3 misses within each benchmark/prefetcher group,
and L3 size, reports delta versus the local LRU row, and includes ECG_PFX
useful-prefetch rates when present. Reuse the same `label=path` prefix across
inputs when you want charged and uncharged CSVs ranked against the same local LRU row.

Small local screen result, 2026-05-26:

```text
/tmp/graphbrew-local-cache-diversity-smoke-full       8 jobs ok, 72 rows
/tmp/graphbrew-local-cache-pfx-diversity-smoke-full   6 jobs ok, 36 rows
```

On `email-Eu-core`, the best L3-miss reductions versus LRU in the no-prefetch
screen were strongest for BC (`ECG_DBG_PRIMARY`, about 30%), TC (`SRRIP`, about
22%), BFS (`POPT`, about 10%), and PR/PR-SPMV (`SRRIP`, about 7%). CC, CC-SV,
and SSSP were too small at this cache point to differentiate policies. The PFX
screen showed useful cache_sim prefetches for BFS, PR, and SSSP; CC/CC-SV issued
no PFX requests on this graph.

The PR/BFS/BC/PR-SPMV/TC follow-up sections below supersede the raw 4kB
replacement-policy takeaways from this broad smoke for final claims, because
they were refreshed after the 2026-05-27 GRASP/P-OPT faithfulness fix.

Follow-up BC cache-size probe on `email-Eu-core`, refreshed after the
2026-05-27 GRASP/P-OPT faithfulness fix:

```text
/tmp/graphbrew-email-core-bc-cache-size-probe   27 rows ok
/tmp/graphbrew-faithful-email-core-bc-cache-size-refresh   21 rows ok
```

The strong BC result is real at the tiny-cache point but cache-size sensitive.
In the refreshed 4kB rows, `ECG_DBG_ONLY` reduced L3 misses by about 30.9%,
`ECG_DBG_PRIMARY` by about 30.4%, GRASP by about 28.9%, and POPT by about
27.1% versus LRU. At 32kB and 256kB, the graph mostly fits the modeled LLC
working set: all main policies are effectively tied around 315--317 L3 misses.
Treat email-Eu-core BC as a positive tiny-cache mechanism proof, not as a robust
cache-size win by itself.

Follow-up BFS cache-size probe on `email-Eu-core`, refreshed after the
2026-05-27 GRASP/P-OPT faithfulness fix:

```text
/tmp/graphbrew-email-core-bfs-cache-size-probe   27 rows ok
/tmp/graphbrew-faithful-email-core-bfs-cache-size-refresh   21 rows ok
```

BFS shows the same caution pattern with a different mechanism. At 4kB,
the refreshed POPT and `ECG_POPT_PRIMARY` rows reduce L3 misses by about 10.6%
and 10.5%, respectively, versus LRU. At 32kB, that reverses: SRRIP is slightly
best at about 1.7% fewer misses than LRU, while POPT/GRASP/ECG rows miss about
10.6% more than LRU. At 256kB all policies tie at 977 L3 misses. Treat this as
a small-cache P-OPT/ECG-P-OPT mechanism signal and keep the BFS PFX path as a
hint-path activation target, not a proven miss-rate win.

Follow-up BFS ECG_PFX cache-size probe on `email-Eu-core`:

```text
/tmp/graphbrew-email-core-bfs-pfx-cache-size-probe   21 rows ok
```

ECG_PFX activation is consistent but does not materially change cache_sim L3
misses on this small graph. Every row issued 24 PFX requests with 17--18 useful
prefetches, a useful/request rate of about 71--75%. Matching rows versus the
no-PFX BFS cache-size probe were unchanged at 32kB/256kB and moved by less than
0.3% at 4kB. At 4kB the ordering still comes from replacement policy
(`POPT`/`ECG_POPT_PRIMARY` near 10% better than LRU); at 32kB SRRIP remains the
only modest win; at 256kB all rows tie. Treat BFS PFX here as a hint-path
activation proof, not a cache-miss improvement claim.

Follow-up PR cache-size probe on `email-Eu-core`, refreshed after the
2026-05-27 GRASP/P-OPT faithfulness fix:

```text
/tmp/graphbrew-email-core-pr-cache-size-probe   27 rows ok
/tmp/graphbrew-faithful-email-core-pr-cache-size-refresh   21 rows ok
```

This is the closest local screen to the original PR-focused ECG evaluation, and
it is also cache-size sensitive. In the refreshed 4kB rows, SRRIP reduces L3
misses by about 10.3%, POPT by about 10.0%, and `ECG_POPT_PRIMARY` by about
5.3% versus LRU; GRASP and DBG-led rows remain worse than LRU on this graph. At
32kB, LRU is best and graph-aware rows miss more: `ECG_POPT_PRIMARY` is about
6.3% worse, POPT about 4.5% worse, and `ECG_DBG_PRIMARY` about 6.7% worse. At
256kB all policies tie at 2134 L3 misses. Treat email-Eu-core PR as a modest
4kB P-OPT/SRRIP mechanism point, not as a robust graph-aware replacement win.

Follow-up PR-SPMV cache-size probe on `email-Eu-core`, refreshed after the
2026-05-27 GRASP/P-OPT faithfulness fix:

```text
/tmp/graphbrew-faithful-email-core-prspmv-cache-size-refresh   21 rows ok
```

PR-SPMV is also cache-size sensitive, but the refreshed 4kB win is mostly SRRIP
with a small ECG-P-OPT signal. At 4kB, SRRIP reduces L3 misses by about 6.7%
versus LRU and `ECG_POPT_PRIMARY` by about 3.4%; POPT is effectively tied with
LRU, while GRASP and DBG-led rows are much worse. At 32kB, LRU is best and
graph-aware rows miss more; at 256kB all rows tie at 2134 L3 misses. Treat this
as another small-cache SRRIP/ECG-P-OPT control point, not a broad GRASP/POPT
win.

Follow-up TC cache-size probe on `email-Eu-core`, refreshed after the
2026-05-27 GRASP/P-OPT faithfulness fix:

```text
/tmp/graphbrew-faithful-email-core-tc-cache-size-refresh   21 rows ok
```

The TC result remains a strong tiny-cache baseline point, but it is not a
graph-aware replacement win. At 4kB, SRRIP reduces L3 misses by about 22.6%
versus LRU, while POPT is essentially tied with LRU and GRASP/DBG/ECG rows miss
about 34--36% more. At 32kB, POPT and SRRIP are only about 0.6% better than LRU,
while GRASP/DBG/ECG rows are much worse; at 256kB all rows tie at 794 L3
misses. Treat TC as evidence that the local screen can find useful baselines,
not as a current ECG/GRASP claim.

Follow-up PR ECG_PFX cache-size probe on `email-Eu-core`:

```text
/tmp/graphbrew-email-core-pr-pfx-cache-size-probe   21 rows ok
```

PR ECG_PFX issues many hints but should not be claimed as a win in the current
configuration. Each row issued about 36.6k PFX requests. At 4kB, useful counts
were nonzero, but PFX mostly increased L3 misses versus matching no-PFX rows:
POPT worsened by about 15.5%, `ECG_POPT_PRIMARY` by about 8.7%, GRASP by about
9.0%, and LRU by about 0.6%. `ECG_DBG_PRIMARY` improved versus its no-PFX row
by about 9.5%, but it still missed more than LRU and SRRIP. At 32kB, PFX useful
counts were essentially zero except a tiny SRRIP signal and most rows worsened;
at 256kB all policies tied with zero useful PFX. Treat PR ECG_PFX here as a
negative/control tuning point: the hint path is active, but the current target
selection and cache point do not convert that into fewer L3 misses.

Faithfulness correction note, 2026-05-27: the official GRASP implementation uses
hot-line insertion `P_RRIP=1` and hot-line hit promotion `H_RRIP=0`. GraphBrew
previously used `0` for both. The cache_sim/gem5/Sniper implementations were
updated to match upstream GRASP, and gem5/Sniper P-OPT mixed-set behavior was
changed from a far-rereference boost heuristic to upstream Phase 1 non-property
eviction. The PR/BFS/BC/PR-SPMV/TC cache-size probes above have now been
refreshed after the correction. Other older local diversity rows involving
GRASP/DBG-mode ECG, such as CC/CC-SV/SSSP and broad all-kernel summaries, should
still be refreshed before final quantitative claims.

Medium local screen result, 2026-05-26:

```text
/tmp/graphbrew-local-cache-diversity-medium-pr-cc             2 jobs ok, 8 rows
/tmp/graphbrew-local-cache-diversity-medium-pr-cc-uncharged   2 jobs ok, 4 rows
```

On `cit-Patents` with the same 4kB L3 screen, LRU remained best for PR and CC.
Uncharged POPT was still worse than LRU by about 8% on PR and 11% on CC;
uncharged `ECG_DBG_PRIMARY` was worse by about 20% on PR and 23% on CC. Charged
POPT/ECG rows were worse again because their effective L3 capacity dropped to
256B. This is a useful negative result: `cit-Patents` at this cache point should
not be used as a winning ECG claim, but it is valuable for overhead accounting
and for deciding which graph/cache points to move to Slurm.

Follow-up PR cache-size probe on `cit-Patents`:

```text
/tmp/graphbrew-cit-patents-pr-cache-size-probe   6 rows ok
```

Increasing L3 from 4kB to 32kB and 256kB did not flip the local result. At 32kB,
POPT was about 51% worse than LRU and `ECG_DBG_PRIMARY` about 40% worse. At
256kB, `ECG_DBG_PRIMARY` narrowed the gap but still missed about 5.8% more than
LRU, while POPT missed about 17.5% more. This makes `cit-Patents` PR a useful
negative/control point for overhead and cache-size sensitivity, not a priority
candidate for detailed-sim winning-claim runs.

For BFS ECG_PFX scale proofs beyond g10, use the Slurm-ready one-root helper:

```bash
python3 scripts/experiments/ecg/ecg_pfx_scale_proof.py \
  --scale 10 \
  --roots 0 \
  --backend both \
  --out-root /tmp/graphbrew-ecg-pfx-scale-g10-r0

printf '11\t0\tboth\tresults/ecg_experiments/ecg_pfx_scale_proof/g11_r0\n' \
  > results/ecg_experiments/slurm/ecg_pfx_scale_g11.tsv
export SHARDS=results/ecg_experiments/slurm/ecg_pfx_scale_g11.tsv
sbatch --array=0-0 scripts/experiments/ecg/slurm_ecg_pfx_scale_proof.sbatch
```

Validated local smoke outputs:

```text
/tmp/graphbrew-gem5-ecg-pfx-tiny-profile: 6/6 ok rows, PR/BFS/SSSP all pf_issued>0
/tmp/graphbrew-gem5-ecg-pfx-pr-lookahead-smoke:   PR g7,   pf_issued=3, pf_useful=0, pf_late=3 per section
/tmp/graphbrew-gem5-ecg-pfx-bfs-lookahead-smoke:  BFS g6,  pf_issued=2, useful varies on tiny graph reruns
/tmp/graphbrew-gem5-ecg-pfx-sssp-lookahead-smoke: SSSP g6, pf_issued=2, pf_useful=0, pf_late=2 per section
/tmp/graphbrew-sniper-ecg-pfx-pr-smoke:           PR g6 SIFT, ecg_pfx_hints=338, ecg_pfx_issued=1, pf_issued=1, pf_useful=1
/tmp/graphbrew-sniper-ecg-pfx-profile:            PR/BFS/SSSP SIFT, 3/3 ok rows with pf_issued>0
/tmp/graphbrew-sniper-file-ecg-pfx-profile:       email PR/BFS/SSSP SIFT, 3/3 ok rows with pf_issued>0
/tmp/graphbrew-paper-pipeline-sniper-ecg-pfx:     aggregate CSVs plus generic prefetch_* figures
```

Validated available-local cache_sim ECG_PFX output:

```text
/tmp/graphbrew-available-cache-sim-ecg-pfx: 6/6 jobs ok, 36/36 ok rows
/tmp/graphbrew-paper-pipeline-available-ecg-pfx: aggregate CSVs plus generic prefetch_* figures
```

This run covers `email-Eu-core` and `cit-Patents` for PR/BFS/SSSP with
`LRU`, `GRASP`, `POPT`, `ECG_DBG_ONLY`, `ECG_DBG_PRIMARY`, and
`ECG_POPT_PRIMARY`, all under `ECG_PFX` with encoded mode `2` (`popt`), window
`16`, and lookahead `4`. Every row issued runtime ECG_PFX requests.

Interpret the available-local result as mechanism validation, not a universal
win claim. On `cit-Patents`, `ECG_POPT_PRIMARY+ECG_PFX` slightly improves over
`POPT+ECG_PFX` for PR and BFS and nearly ties it for SSSP, but LRU is still
better on cit-Patents PR/BFS at this tiny 4kB L3 point. On `email-Eu-core`,
`ECG_POPT_PRIMARY+ECG_PFX` improves or matches POPT on PR/BFS and SSSP is
saturated at the same small miss count across policies.

Use the repo venv for figure generation; system Python may not have matplotlib.
ECG_PFX smoke aggregation writes generic `prefetch_*` figures, not DROPLET-only
figure names.

Timing interpretation for ECG_PFX:

- cache and prefetch metrics are valid mechanism evidence (`l3_misses`,
  prefetch issued/fill/useful counters, and memory traffic),
- current gem5/Sniper ECG_PFX rows use explicit benchmark-emitted hint delivery
  (`GEM5_ECG_PFX_TARGET` / `SNIPER_ECG_PFX_TARGET`),
- the wrappers now apply a small recent-target filter before crossing the
  simulator magic-call boundary (`--ecg-pfx-hint-filter`, default `16`) so
  obvious duplicate target hints are not over-emitted,
- gem5 also has an opt-in RISC-V custom-instruction delivery path
  (`--ecg-pfx-delivery instruction`, `GEM5_ENABLE_ECG_EXTRACT=1`) that emits
  `ecg.extract` instead of the m5ops PFX target hint when the benchmark is built
  for RISC-V,
- that hint path is a prototype delivery mechanism, not the final
  instruction-carried hardware metadata model,
- therefore `roi_matrix.py` marks gem5/Sniper ECG_PFX rows with
  `timing_model=prototype_explicit_hint_delivery` and
  `timing_valid_for_speedup=0`,
- `paper_pipeline.py` preserves their cache/prefetch metrics but omits
  `speedup_vs_lru` and speedup figures for those rows.

Only use ECG_PFX simulator target-time speedups after the run actually uses an
instruction-carried or otherwise hardware-faithful low-overhead PFX path. The
current X86 gem5/Sniper rows remain cache-performance and precision evidence,
not runtime-speedup evidence.

Preprocessing overhead is measured in two ways. Cache_sim rows already include
`ecg_build_s`, which is the ECG mask/PFX construction time observed inside the
simulator run. For no-simulation overhead analysis, use `bench/bin_sim/ecg_preprocess`:

```bash
make RABBIT_ENABLE=0 bench/bin_sim/ecg_preprocess
ECG_PREFETCH_MODE=2 \
ECG_PREPROCESS_REPEATS=5 \
ECG_PREPROCESS_OUTPUT_JSON=/tmp/graphbrew-ecg-preprocess.json \
OMP_NUM_THREADS=32 \
bench/bin_sim/ecg_preprocess -f results/graphs/email-Eu-core/email-Eu-core.sg -s -o 0 -n 1
```

The JSON separates `graph_load_s` from `degree_scan_s_*`,
`popt_matrix_s_*`, `mask_build_s_*`, and `total_preprocess_s_*`. Use
`total_preprocess_s_*` for end-to-end ECG/PFX metadata cost, and use
`mask_build_s_*` when isolating only compact mask construction. The utility uses
the same OpenMP-enabled degree scan, P-OPT matrix builder, and ECG mask builder
as the cache_sim path, but does not run PR/BFS/SSSP or any cache simulation.

ECG validation gate workflow, 2026-05-27:

```bash
python3 scripts/experiments/ecg/proof_matrix.py \
  --benchmarks pr bfs sssp \
  --graph-path results/graphs/email-Eu-core/email-Eu-core.sg \
  --l3-sizes 4kB \
  --out-dir /tmp/graphbrew-ecg-validation-proof-email-core \
  --timeout-cache 900 \
  --no-build

python3 scripts/experiments/ecg/ecg_validation_gates.py \
  /tmp/graphbrew-ecg-validation-proof-email-core/proof_matrix.csv \
  --out-csv /tmp/graphbrew-ecg-validation-proof-email-core/gates.csv \
  --out-md /tmp/graphbrew-ecg-validation-proof-email-core/gates.md
```

Use file-backed proof gates for quantitative interpretation. Repeated synthetic
`-g 12` PR proof rows changed across separate invocations enough to make policy
comparisons fragile, while `--graph-path` keeps the graph input fixed for every
ablation row.

The refreshed file-backed proof matrix has 42/42 ok cache_sim rows across
PR/BFS/SSSP on `email-Eu-core`. The gate report has 30 verdict rows: 27 pass and
3 fail. Passing gates: GRASP parity (`ECG_DBG_ONLY` equals GRASP), P-OPT parity
(`ECG_POPT_PRIMARY` within 5% of POPT), all PFX demand gates, and the
best-available ECG replacement gate for every benchmark. Failing gates:
`ECG_DBG_POPT` does not beat the stronger prior replacement baseline on PR/BFS,
and `ECG_EMBEDDED` is outside the 10% P-OPT-quality tolerance on PR. The PR
embedded log uses full 7-bit P-OPT hints (`DBG=2 POPT=7`) yet still collapses to
the DBG/GRASP row, so the problem is not bit budget; it is that the stored hint
is too static/averaged compared with dynamic current-vertex POPT.

First embedded-variant result, 2026-05-27:

```text
/tmp/graphbrew-ecg-validation-proof-email-core-epoch   45/45 proof rows ok
```

This run adds `ECG_EPOCH_EMBEDDED`, a cache_sim-only compact current-epoch POPT
hint model. It improves over static embedded but still does not reach dynamic
P-OPT on PR/BFS. On PR, static embedded and DBG-primary both miss 9,332 times,
epoch-embedded misses 6,793 times, combined insertion misses 6,226, and dynamic
`ECG_POPT_PRIMARY` misses 5,248 versus POPT at 5,254. On BFS, epoch-embedded is
1,212 misses versus static embedded at 1,270 and dynamic POPT/`ECG_POPT_PRIMARY`
at 1,186. The design insight is clear: each approach is useful as a point in
the design space, but current-epoch information is necessary and still not
sufficient to fully match dynamic POPT on PR.

Interpret this as mechanism proof plus a clear research gap: ECG parity and PFX
activation are working locally, and at least one ECG replacement mode matches or
beats the strongest prior baseline on each file-backed proof benchmark, but the
specific DBG-primary hybrid and PR embedded replacement story are not yet
paper-level wins.

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

Small-machine final-run smoke:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile rehearsal \
  --run-dir /tmp/graphbrew-local-final-rehearsal-lite \
  --only 02_gem5_replacement_rehearsal \
  --benchmark bfs \
  --policy LRU \
  --no-build \
  --force \
  --skip-validation-gate

.venv/bin/python3 scripts/experiments/ecg/paper_pipeline.py \
  --skip-run \
  --input-run-dirs /tmp/graphbrew-local-final-rehearsal-lite \
  --run-root /tmp/graphbrew-local-final-rehearsal-lite-aggregate-venv
```

Validated local result, 2026-05-26:

```text
/tmp/graphbrew-local-final-rehearsal-lite: 1/1 final-run job ok
combined_roi_matrix.csv: 2 gem5 BFS/LRU ROI rows ok
section 1: sim_ticks=7,266,133,000, l3_misses=41,807
section 2: sim_ticks=7,463,180,000, l3_misses=43,518
/tmp/graphbrew-local-final-rehearsal-lite-aggregate-venv: CSVs, LaTeX tables, SVG figures, and PNG previews generated
```

The full `rehearsal` profile is useful, but it expands to cache_sim proof plus
six gem5 matrix jobs. On a workstation, the first PR replacement matrix alone
can take several minutes per policy, so use the small-machine smoke above for a
quick final-run/pipeline sanity check and reserve the full rehearsal for a long
local window or Slurm.

Run the full rehearsal before any multi-day job when you have enough local time:

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

Run cache_sim ECG_PFX once graph checks pass:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile available_cache_sim_ecg_pfx \
  --run-dir results/ecg_experiments/final_paper_runs/available_cache_sim_ecg_pfx_001

python3 scripts/experiments/ecg/final_paper_run.py \
  --profile final_cache_sim_ecg_pfx \
  --run-dir results/ecg_experiments/final_paper_runs/cache_sim_ecg_pfx_final_001
```

Run the bounded Sniper ECG_PFX synthetic smoke:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_sift_ecg_pfx_smoke \
  --run-dir results/ecg_experiments/final_paper_runs/sniper_ecg_pfx_smoke_001 \
  --no-build --force
```

Run the bounded file-backed email-Eu-core Sniper ECG_PFX smoke:

```bash
python3 scripts/experiments/ecg/final_paper_run.py \
  --profile sniper_sift_file_ecg_pfx_smoke \
  --run-dir results/ecg_experiments/final_paper_runs/sniper_file_ecg_pfx_smoke_001 \
  --no-build --force
```

Run a focused ECG_PFX cache_sim smoke:

```bash
python3 scripts/experiments/ecg/roi_matrix.py \
  --suite cache-sim \
  --benchmark pr \
  --options "-g 12 -k 16 -o 5 -n 1 -i 2" \
  --policies LRU POPT ECG:POPT_PRIMARY \
  --prefetcher ECG_PFX \
  --ecg-pfx-mode popt \
  --ecg-pfx-window 16 \
  --ecg-pfx-lookahead 4 \
  --l3-sizes 4kB \
  --no-build
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
figures/prefetch_l3_miss_reduction_vs_lru.svg  # written for non-DROPLET prefetch rows
figures/prefetch_l3_miss_reduction_by_benchmark.svg
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