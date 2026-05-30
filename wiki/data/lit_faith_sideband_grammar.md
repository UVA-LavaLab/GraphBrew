# Gate 265 — gem5/Sniper sideband filename + env-var grammar

- status: `active`
- registry entries: 8
- tools: gem5, sniper
- roles: context, popt_matrix, out_edges, in_edges
- sideband subdir: `graphbrew_sidebands`
- violations: 0

## Registry

| tool | role | filename | env_var | default_path |
|---|---|---|---|---|
| gem5 | context | `gem5_graphbrew_ctx.json` | `GEM5_GRAPHBREW_CTX` | `/tmp/gem5_graphbrew_ctx.json` |
| gem5 | popt_matrix | `gem5_popt_matrix.bin` | `GEM5_POPT_MATRIX` | `/tmp/gem5_popt_matrix.bin` |
| gem5 | out_edges | `gem5_graphbrew_out_edges.bin` | `GEM5_GRAPHBREW_OUT_EDGES` | `/tmp/gem5_graphbrew_out_edges.bin` |
| gem5 | in_edges | `gem5_graphbrew_in_edges.bin` | `GEM5_GRAPHBREW_IN_EDGES` | `/tmp/gem5_graphbrew_in_edges.bin` |
| sniper | context | `sniper_graphbrew_ctx.json` | `SNIPER_GRAPHBREW_CTX` | `/tmp/sniper_graphbrew_ctx.json` |
| sniper | popt_matrix | `sniper_popt_matrix.bin` | `SNIPER_POPT_MATRIX` | `/tmp/sniper_popt_matrix.bin` |
| sniper | out_edges | `sniper_graphbrew_out_edges.bin` | `SNIPER_GRAPHBREW_OUT_EDGES` | `/tmp/sniper_graphbrew_out_edges.bin` |
| sniper | in_edges | `sniper_graphbrew_in_edges.bin` | `SNIPER_GRAPHBREW_IN_EDGES` | `/tmp/sniper_graphbrew_in_edges.bin` |

## Emit-sites audited

- `bench/include/gem5_sim/gem5_harness.h`
- `bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/cache_set_popt.cc`
- `bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/cache_set_grasp.cc`
- `bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/cache_set_ecg.cc`
- `bench/include/sniper_sim/overlays/common/core/memory_subsystem/parametric_dram_directory_msi/ecg_pfx_prefetcher.cc`
- `bench/include/sniper_sim/overlays/common/core/memory_subsystem/parametric_dram_directory_msi/droplet_prefetcher.cc`

## Parse-sites audited

- `scripts/experiments/ecg/roi_matrix.py`
