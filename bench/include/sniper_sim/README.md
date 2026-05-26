# GraphBrew Sniper Integration

This directory is the Sniper counterpart to `bench/include/gem5_sim/`.

Sniper sits between `cache_sim` and gem5 in the ECG validation stack:

- `cache_sim`: fast mechanism proof and component ablations.
- `sniper_sim`: larger-graph multicore timing and cache/CPI-stack evidence.
- `gem5_sim`: detailed timing and future ISA/hint-path validation.

## Layout

```text
bench/include/sniper_sim/
  README.md
  sniper_harness.h
  configs/graphbrew/
    graph_sniper.cfg
    graph_cache_config.cfg
  overlays/
    common/core/memory_subsystem/cache/
    common/core/memory_subsystem/prefetcher/
  snipersim/                    # upstream clone, ignored by git
```

## Setup

```bash
python3 scripts/setup_sniper.py --dry-run
python3 scripts/setup_sniper.py --skip-build
python3 scripts/setup_sniper.py --jobs 8 --smoke
python3 scripts/setup_sniper.py --build-target configscripts
python3 scripts/setup_sniper.py --skip-build --apply-overlays
python3 scripts/setup_sniper.py --skip-build --graphbrew-smoke
```

The upstream checkout is expected at:

```text
bench/include/sniper_sim/snipersim/
```

This path is ignored by git, like the gem5 checkout.

## Current Status

This is Phase 0/1 scaffolding. Sniper now has an initial `roi_matrix.py --suite
sniper` path for the fast PR-kernel smoke with built-in LRU/SRRIP, full-wrapper
PR/BFS/SSSP smoke paths, graph-aware replacement overlays, DROPLET smoke paths,
and an experimental ECG_PFX prefetch path.

The ECG_PFX path provides `SNIPER_ECG_PFX_TARGET(vertex)` in the benchmark
harness, per-core prefetch-target hint storage in `graph_cache_context_sniper`,
PR/BFS/SSSP wrapper call sites that mirror the gem5 lookahead shape, and a
`setup_sniper.py --apply-overlays` patch that dispatches `SimUser` `GRVT`/`GPFX`
commands in `magic_server.cc`. `EcgPfxPrefetcher` consumes stored target hints,
maps them to the exported property region, deduplicates recent cache lines, and
reports ECG_PFX counters. It is not a final timing claim yet: after applying
overlays, relink Sniper common/standalone and validate small SIFT rows before
using it in paper-scale profiles. The first PR synthetic SIFT smoke consumed
target hints, generated an ECG_PFX request, and reported one useful Sniper cache
prefetch.

The reproducible synthetic profile is `sniper_sift_ecg_pfx_smoke`; it produced
3/3 ok PR/BFS/SSSP rows at `/tmp/graphbrew-sniper-ecg-pfx-profile`. The
file-backed email-Eu-core profile is `sniper_sift_file_ecg_pfx_smoke`; it
produced 3/3 ok rows at `/tmp/graphbrew-sniper-file-ecg-pfx-profile` using
lookahead 4 for PR/BFS and lookahead 0 for SSSP.
Both profiles aggregate through `paper_pipeline.py` at
`/tmp/graphbrew-paper-pipeline-sniper-ecg-pfx` with generic `prefetch_*` output
names.
The next milestones are:

1. Build and smoke-test upstream Sniper on the target host/UVA Slurm node.
2. Add `bench/src_sniper/pr.cc` using `sniper_harness.h` ROI macros.
3. Tune full GraphBrew PR/BFS/SSSP wrapper runs under Sniper/SDE.
4. Port GRASP, P-OPT, ECG, and DROPLET semantics 1:1 from cache_sim/gem5.
5. Broaden `roi_matrix.py --suite sniper --prefetcher ECG_PFX` validation from
  the PR synthetic smoke to BFS/SSSP and file-backed email-Eu-core rows.

See `plans/sniper-sim-integration-plan.md` for the full integration contract.

## Phase 0 Findings

Current upstream checkout used for inspection:

```text
snipersim main @ 56505e42fd98bca863fac181e769bd3c98d2bb33
```

Confirmed ROI API:

```cpp
#include "sim_api.h"
SimRoiStart();
SimRoiEnd();
SimMarker(arg0, arg1);
SimNamedMarker(arg0, str);
SimUser(cmd, arg);
SimGetThreadId();
SimGetNumThreads();
```

Confirmed replacement-policy extension points:

```text
common/core/memory_subsystem/cache/cache_set.h
common/core/memory_subsystem/cache/cache_set_lru.{h,cc}
common/core/memory_subsystem/cache/cache_set_srrip.{h,cc}
common/core/memory_subsystem/cache/cache_base.{h,cc}
common/core/memory_subsystem/cache/cache.cc
```

Sniper already has `CacheSetSRRIP`, so GRASP/POPT/ECG should be implemented as
new `CacheSet` subclasses and wired through `CacheSet::createCacheSet()` plus
the config replacement-policy parser.

Pre-build smoke status:

- `run-sniper` exists immediately after clone, but fails before build because
  `config/sniper.py` has not been generated.
- Sniper's `Makefile` creates `config/sniper.py` and `config/buildconf.*` in the
  `configscripts` target, which depends on the normal dependency setup.
- A full build defaults to SDE on this checkout unless `USE_PIN=1` or another
  frontend is selected.
- Local build attempt reached a host dependency blocker: `sqlite3.h` was missing.
  Install `libsqlite3-dev` on Ubuntu/Debian or `sqlite-devel` on RHEL/Fedora, or
  load a UVA module/toolchain that provides SQLite development headers.

Post-build smoke status:

- After installing `libsqlite3-dev`, `python3 scripts/setup_sniper.py --jobs 1
  --smoke` built Sniper and passed the upstream `/bin/true` smoke.
- `bench/bin_sniper/hello_roi` runs under Sniper and generates `sim.stats`.
- `bench/bin_sniper/pr_kernel_smoke` runs under Sniper ROI mode in about 30s and
  generates parseable L1/L2/NUCA cache counters. This is the current GraphBrew
  Sniper smoke.
- `bench/bin_sniper/pr` builds natively and exports Sniper sideband files, but
  even tiny full GraphBrew PR under Sniper/SDE has high startup/tracing overhead.
  A bounded `--roi --cache-only --no-cache-warming` probe left the child PR
  process at roughly 53 GiB RSS before it was killed. Keep PR as a functional
  native wrapper for now; optimize the Sniper run mode before using it as a fast
  smoke.

Current smoke commands:

```bash
make sniper-hello_roi sniper-pr_kernel_smoke sniper-pr

python3 scripts/setup_sniper.py --skip-build --graphbrew-smoke

bench/include/sniper_sim/snipersim/run-sniper \
  --roi -n 1 \
  -d /tmp/sniper-graphbrew-pr-kernel \
  -caddress_translation_schemes/baseline \
  -- bench/bin_sniper/pr_kernel_smoke

python3 bench/include/sniper_sim/scripts/parse_stats.py \
  /tmp/sniper-graphbrew-pr-kernel
```

Validated automated smoke:

```bash
python3 scripts/setup_sniper.py --skip-build --graphbrew-smoke \
  --graphbrew-smoke-dir /tmp/sniper-graphbrew-pr-kernel-auto
```

This completed successfully and parsed:

```text
instructions: 7846
L1D loads/misses: 2251 / 85
L2 loads/misses: 85 / 85
LLC loads/misses: 99 / 99
```

Initial ROI-matrix smoke:

```bash
python3 scripts/experiments/ecg/roi_matrix.py \
  --suite sniper --policies LRU SRRIP --l3-sizes 32kB \
  --out-dir results/ecg_experiments/roi_matrix/sniper_pr_kernel_smoke

python3 scripts/experiments/ecg/roi_matrix.py \
  --suite sniper --policies GRASP POPT POPT_CHARGED --l3-sizes 32kB \
  --out-dir results/ecg_experiments/roi_matrix/sniper_grasp_pr_kernel_smoke
```

`GRASP` and `POPT` require `scripts/setup_sniper.py --apply-overlays` plus a
Sniper common library/standalone relink. Full `bench/bin_sniper/pr` remains
disabled as a smoke target until the SDE/SIFT run mode is fixed.

Runner-launched Sniper jobs use per-run sidebands under each Sniper output
directory's `graphbrew_sidebands/` folder. Direct native/manual runs still use
the harness defaults under `/tmp` unless the `SNIPER_GRAPHBREW_*` environment
variables are set.

Thread/core smoke sweeps can use:

```bash
python3 scripts/experiments/ecg/roi_matrix.py \
  --suite sniper --policies LRU --threads 1 2 4 --no-build
```

The current `pr_kernel_smoke` workload is single-threaded, so this validates the
runner/config surface only. Real thread-scaling claims need the full benchmark
wrappers after the Sniper/SDE run-mode blocker is fixed.
