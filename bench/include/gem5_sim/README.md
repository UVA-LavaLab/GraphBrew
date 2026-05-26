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
| ECG_PFX   | `GraphEcgPfxPrefetcher` | GraphBrew ECG hint path |

## Custom Instructions

- **RISC-V**: `ecg.extract rd, rs1` — custom-0 opcode (`0x0b`) scaffold under `overlays/arch/riscv/`
- **x86**: GraphBrew m5ops work items for current-vertex and ECG_PFX target hints

The RISC-V scaffold decodes the paper-style fixed 64-bit ECG layout: low 32 bits
for the real vertex ID, 8 DBG bits, 8 P-OPT bits, and 16 ECG_PFX bits. It writes
the real vertex ID to `rd`, stores decoded metadata in GraphBrew hint storage,
and forwards the PFX field to the ECG_PFX hint queue, falling back to the real
vertex ID when the PFX field is zero. `build/RISCV/gem5.opt` builds and verifies
after applying overlays. RISC-V benchmark wrappers require the local
`riscv64-linux-gnu-gcc/g++` cross toolchain and can be built with
`make gem5-riscv-m5ops-pr` for the PR smoke path. Current RISC-V ECG_PFX rows are
instruction-path activation evidence until useful-prefetch and timing-validity
checks pass on larger points.

## Context Passing

Static metadata (degree distribution, rereference matrix) passes via JSON sideband file
loaded by gem5 Python config. Dynamic per-access hints pass via custom ECG instruction
→ CSR → cache controller.

## See Also

- `research/gem5/` — Step-by-step integration research documentation
- `bench/include/cache_sim/` — Reference standalone C++ cache simulator
- `scripts/lib/pipeline/gem5.py` — Pipeline integration module
