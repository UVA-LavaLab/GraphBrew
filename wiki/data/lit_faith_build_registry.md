# Gate 259 — SCons/Make build target registry

Status: **active**

## Totals

- n_backends: 4
- n_canonical_targets: 51
- n_native_kernels: 9
- n_cache_sim_kernels: 9
- n_gem5_kernels: 8
- n_sniper_targets: 8
- n_makefile_harvested_kernels: 26
- n_orphan_sources: 0

## Rules

- **R1** — every (backend,kernel) in CANONICAL_BUILD_TARGETS has a matching source file under SRC_DIRS[backend]
- **R2** — every Makefile KERNELS_<BACKEND> entry is canonical for that backend
- **R3** — every CXXFLAGS_<BACKEND> contains all documented required tokens
- **R4** — every canonical backend maps to the canonical SRC_DIR + BIN_DIR (both must exist in tree and Makefile)
- **R5** — every canonical kernel has a non-empty graph-algorithm family classification
- **R6** — every backend's CXXFLAGS carries the documented optimisation level (-O3 native+sim, -O1 gem5, -O2 sniper)
- **R7** — every .cc under SRC_DIRS[backend] maps to a canonical (backend,kernel) entry — no orphans
- **R8** — every canonical backend has a documented ROI mechanism ∈ {m5ops, sift, sim-callback, none}

## Canonical build targets

| backend | kernel | variant | family | bin |
|---|---|---|---|---|
| `native` | `bc` | `base` | centrality | `bench/bin/bc` |
| `native` | `bfs` | `base` | traversal | `bench/bin/bfs` |
| `native` | `cc` | `base` | connected-component | `bench/bin/cc` |
| `native` | `cc_sv` | `base` | connected-component | `bench/bin/cc_sv` |
| `native` | `pr` | `base` | pagerank | `bench/bin/pr` |
| `native` | `pr_spmv` | `base` | pagerank | `bench/bin/pr_spmv` |
| `native` | `sssp` | `base` | shortest-path | `bench/bin/sssp` |
| `native` | `tc` | `base` | triangle | `bench/bin/tc` |
| `native` | `tc_p` | `base` | triangle | `bench/bin/tc_p` |
| `native` | `converter` | `base` | preprocess | `bench/bin/converter` |
| `cache_sim` | `pr` | `base` | pagerank | `bench/bin_sim/pr` |
| `cache_sim` | `pr_spmv` | `base` | pagerank | `bench/bin_sim/pr_spmv` |
| `cache_sim` | `bfs` | `base` | traversal | `bench/bin_sim/bfs` |
| `cache_sim` | `bc` | `base` | centrality | `bench/bin_sim/bc` |
| `cache_sim` | `cc` | `base` | connected-component | `bench/bin_sim/cc` |
| `cache_sim` | `cc_sv` | `base` | connected-component | `bench/bin_sim/cc_sv` |
| `cache_sim` | `sssp` | `base` | shortest-path | `bench/bin_sim/sssp` |
| `cache_sim` | `tc` | `base` | triangle | `bench/bin_sim/tc` |
| `cache_sim` | `ecg_preprocess` | `base` | preprocess | `bench/bin_sim/ecg_preprocess` |
| `gem5` | `pr` | `base` | pagerank | `bench/bin_gem5/pr` |
| `gem5` | `pr` | `m5ops` | pagerank | `bench/bin_gem5/pr_m5ops` |
| `gem5` | `pr` | `riscv_m5ops` | pagerank | `bench/bin_gem5/pr_riscv_m5ops` |
| `gem5` | `pr_spmv` | `base` | pagerank | `bench/bin_gem5/pr_spmv` |
| `gem5` | `pr_spmv` | `m5ops` | pagerank | `bench/bin_gem5/pr_spmv_m5ops` |
| `gem5` | `pr_spmv` | `riscv_m5ops` | pagerank | `bench/bin_gem5/pr_spmv_riscv_m5ops` |
| `gem5` | `bfs` | `base` | traversal | `bench/bin_gem5/bfs` |
| `gem5` | `bfs` | `m5ops` | traversal | `bench/bin_gem5/bfs_m5ops` |
| `gem5` | `bfs` | `riscv_m5ops` | traversal | `bench/bin_gem5/bfs_riscv_m5ops` |
| `gem5` | `sssp` | `base` | shortest-path | `bench/bin_gem5/sssp` |
| `gem5` | `sssp` | `m5ops` | shortest-path | `bench/bin_gem5/sssp_m5ops` |
| `gem5` | `sssp` | `riscv_m5ops` | shortest-path | `bench/bin_gem5/sssp_riscv_m5ops` |
| `gem5` | `cc` | `base` | connected-component | `bench/bin_gem5/cc` |
| `gem5` | `cc` | `m5ops` | connected-component | `bench/bin_gem5/cc_m5ops` |
| `gem5` | `cc` | `riscv_m5ops` | connected-component | `bench/bin_gem5/cc_riscv_m5ops` |
| `gem5` | `cc_sv` | `base` | connected-component | `bench/bin_gem5/cc_sv` |
| `gem5` | `cc_sv` | `m5ops` | connected-component | `bench/bin_gem5/cc_sv_m5ops` |
| `gem5` | `cc_sv` | `riscv_m5ops` | connected-component | `bench/bin_gem5/cc_sv_riscv_m5ops` |
| `gem5` | `bc` | `base` | centrality | `bench/bin_gem5/bc` |
| `gem5` | `bc` | `m5ops` | centrality | `bench/bin_gem5/bc_m5ops` |
| `gem5` | `bc` | `riscv_m5ops` | centrality | `bench/bin_gem5/bc_riscv_m5ops` |
| `gem5` | `tc` | `base` | triangle | `bench/bin_gem5/tc` |
| `gem5` | `tc` | `m5ops` | triangle | `bench/bin_gem5/tc_m5ops` |
| `gem5` | `tc` | `riscv_m5ops` | triangle | `bench/bin_gem5/tc_riscv_m5ops` |
| `sniper` | `hello_roi` | `base` | smoke | `bench/bin_sniper/hello_roi` |
| `sniper` | `pr_kernel_smoke` | `base` | smoke | `bench/bin_sniper/pr_kernel_smoke` |
| `sniper` | `bfs_kernel_smoke` | `base` | smoke | `bench/bin_sniper/bfs_kernel_smoke` |
| `sniper` | `sssp_kernel_smoke` | `base` | smoke | `bench/bin_sniper/sssp_kernel_smoke` |
| `sniper` | `sg_kernel` | `base` | smoke | `bench/bin_sniper/sg_kernel` |
| `sniper` | `pr` | `base` | pagerank | `bench/bin_sniper/pr` |
| `sniper` | `bfs` | `base` | traversal | `bench/bin_sniper/bfs` |
| `sniper` | `sssp` | `base` | shortest-path | `bench/bin_sniper/sssp` |

## CXXFLAGS

| backend | var_name | opt_level | required_tokens |
|---|---|---|---|
| `native` | `CXXFLAGS_GAP` | `-O3` | `-std=c++17`, `-Wall`, `-fopenmp`, `-DNDEBUG` |
| `cache_sim` | `CXXFLAGS_GAP` | `-O3` | `-std=c++17`, `-Wall`, `-fopenmp`, `-DNDEBUG` |
| `gem5` | `CXXFLAGS_GEM5` | `-O1` | `-std=c++17`, `-Wall`, `-fopenmp`, `-DNDEBUG`, `-DNO_M5OPS` |
| `sniper` | `CXXFLAGS_SNIPER` | `-O2` | `-std=c++17`, `-Wall`, `-fopenmp`, `-DNDEBUG`, `-I$(SNIPER_INCLUDE)` |

## SRC / BIN directories

| backend | src_dir | bin_dir | roi_mechanism |
|---|---|---|---|
| `cache_sim` | `bench/src_sim` | `bench/bin_sim` | `sim-callback` |
| `gem5` | `bench/src_gem5` | `bench/bin_gem5` | `m5ops` |
| `native` | `bench/src` | `bench/bin` | `none` |
| `sniper` | `bench/src_sniper` | `bench/bin_sniper` | `sift` |

## Violations

None.
