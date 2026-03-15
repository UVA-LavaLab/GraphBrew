# gem5 Simulation Infrastructure for GraphBrew

This directory provides **gem5 integration** for validating GraphBrew's graph-aware
cache replacement policies (SRRIP, GRASP, P-OPT, ECG) and the DROPLET indirect
graph prefetcher on a cycle-accurate hardware simulator.

## Directory Structure

```
gem5_sim/
├── .gitignore                 # Ignores cloned gem5/ subdirectory
├── README.md                  # This file
├── overlays/                  # Custom source files copied INTO gem5/src/
│   └── mem/cache/
│       ├── replacement_policies/
│       │   ├── grasp_rp.hh / .cc           GRASP replacement policy
│       │   ├── popt_rp.hh / .cc            P-OPT replacement policy
│       │   ├── ecg_rp.hh / .cc             ECG 3-mode replacement policy
│       │   ├── graph_cache_context_gem5.hh  Adapted GraphCacheContext
│       │   ├── GraphReplacementPolicies.py  SimObject Python bindings
│       │   └── SConscript.patch             Build system patch
│       └── prefetch/
│           ├── droplet.hh / .cc            DROPLET indirect prefetcher
│           ├── GraphPrefetchers.py          SimObject Python bindings
│           └── SConscript.patch             Build system patch
├── overlays/arch/riscv/       # Custom ECG RISC-V instruction
├── configs/graphbrew/         # gem5 Python configuration scripts
│   ├── graph_se.py            SE-mode config for graph benchmarks
│   ├── graph_cache_config.py  Cache hierarchy parameters
│   └── graph_metadata_loader.py  JSON sideband metadata loader
├── scripts/
│   ├── build_gem5.py          Build helper
│   └── parse_stats.py         gem5 stats.txt → CacheResult parser
└── gem5/                      # CLONED gem5 repo (gitignored, ~2GB)
```

## Quick Start

```bash
# 1. Clone and build gem5 with GraphBrew patches
python scripts/setup_gem5.py --isa X86 --jobs $(nproc)

# 2. Run a graph benchmark under gem5 with GRASP policy
gem5_sim/gem5/build/X86/gem5.opt \
    gem5_sim/configs/graphbrew/graph_se.py \
    --binary bench/bin/pr \
    --options "-f results/graphs/soc-pokec/soc-pokec.sg -s -n 1 -o 5" \
    --policy GRASP \
    --graph-metadata results/gem5_metadata/soc-pokec/context.json

# 3. Or use the pipeline integration
python scripts/graphbrew_experiment.py --phase cache --simulator gem5 \
    --graphs soc-pokec --benchmarks pr
```

## Policies Implemented

| Policy | gem5 SimObject | Reference |
|--------|---------------|-----------|
| SRRIP  | `BRRIPRP(btp=0)` | gem5 built-in (no new code) |
| GRASP  | `GraphGraspRP` | Faldu et al., HPCA 2020 |
| P-OPT  | `GraphPoptRP` | Balaji et al., HPCA 2021 |
| ECG    | `GraphEcgRP` | Mughrabi et al., GrAPL 2026 |

## Prefetchers Implemented

| Prefetcher | gem5 SimObject | Reference |
|-----------|---------------|-----------|
| DROPLET   | `GraphDropletPrefetcher` | Basak et al., HPCA 2019 |

## Custom Instructions

- **RISC-V**: `ecg.extract rd, rs1` — custom-0 opcode (0x0B), extracts mask from fat-ID
- **x86**: `m5_ecg_extract()` — gem5 pseudo-instruction fallback

## Context Passing

Static metadata (degree distribution, rereference matrix) passes via JSON sideband file
loaded by gem5 Python config. Dynamic per-access hints pass via custom ECG instruction
→ CSR → cache controller.

## See Also

- `research/gem5/` — Step-by-step integration research documentation
- `bench/include/cache_sim/` — Reference standalone C++ cache simulator
- `scripts/lib/pipeline/gem5.py` — Pipeline integration module
