# gem5 Integration for GraphBrew Cache Policies

This directory documents how each GraphBrew graph-aware caching and prefetching
method integrates into gem5 as a custom SimObject. Each file provides a
step-by-step guide covering the gem5 API mapping, data flow, and validation
against the standalone cache simulator (`bench/include/cache_sim/`).

## Architecture Overview

```
                    ┌─────────────────────────────────────────┐
                    │         GraphBrew Pipeline              │
                    │  graphbrew_experiment.py --simulator gem5│
                    └──────────────┬──────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────┐
                    │   scripts/lib/pipeline/gem5.py          │
                    │   Build gem5 command, parse stats.txt   │
                    └──────────────┬──────────────────────────┘
                                   │
         ┌─────────────────────────▼─────────────────────────────┐
         │                    gem5 SE Mode                        │
         │  ┌─────────┐   ┌──────────┐   ┌─────────────────┐   │
         │  │  CPU     │───│  L1 D/I  │───│  L2 (private)   │   │
         │  │(Timing/  │   │  LRU     │   │  LRU            │   │
         │  │ O3)      │   └──────────┘   └────────┬────────┘   │
         │  └─────────┘                            │             │
         │                              ┌──────────▼──────────┐ │
         │                              │  L3 (shared)        │ │
         │                              │  GRASP / P-OPT / ECG│ │
         │        ┌────────────────────►│  + DROPLET prefetch │ │
         │        │ ECG mask hint       └──────────┬──────────┘ │
         │   ┌────┴─────┐                         │             │
         │   │ Custom    │               ┌─────────▼──────────┐ │
         │   │ ECG Inst  │               │  DRAM (DDR4)       │ │
         │   │ (RISC-V)  │               └────────────────────┘ │
         │   └───────────┘                                      │
         └──────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  JSON Sideband Metadata     │
                    │  (PropertyRegion, Topology, │
                    │   Rereference Matrix, Mask) │
                    └─────────────────────────────┘
```

## Policy-to-SimObject Mapping

| Policy/Prefetcher | Standalone (cache_sim.h) | gem5 SimObject | gem5 Source |
|-------------------|--------------------------|----------------|-------------|
| SRRIP | `EvictionPolicy::SRRIP`, `findVictimSRRIP()` | `BRRIPRP(btp=0)` | gem5 built-in |
| GRASP | `EvictionPolicy::GRASP`, `GRASPState`, `findVictimGRASP()` | `GraphGraspRP` | `grasp_rp.hh/cc` |
| P-OPT | `EvictionPolicy::POPT`, `POPTState`, `findVictimPOPT()` | `GraphPoptRP` | `popt_rp.hh/cc` |
| ECG | `EvictionPolicy::ECG`, `MaskConfig`, `findVictimECG()` | `GraphEcgRP` | `ecg_rp.hh/cc` |
| DROPLET | N/A (not in standalone sim) | `GraphDropletPrefetcher` | `droplet.hh/cc` |

## Context Passing Strategy

**Hybrid approach** (see [context-passing.md](context-passing.md)):
- **Static metadata** (degree distribution, bucket boundaries, rereference matrix):
  JSON sideband file → gem5 Python config → SimObject constructor params
- **Dynamic per-access hints** (ECG fat-ID mask):
  Custom RISC-V instruction → CSR → cache controller reads CSR

## File Index

| File | Topic |
|------|-------|
| [srrip-gem5.md](srrip-gem5.md) | SRRIP → gem5 BRRIP mapping |
| [grasp-gem5.md](grasp-gem5.md) | GRASP degree-bucket replacement |
| [popt-gem5.md](popt-gem5.md) | P-OPT oracle rereference replacement |
| [ecg-gem5.md](ecg-gem5.md) | ECG 3-mode hybrid replacement |
| [droplet-gem5.md](droplet-gem5.md) | DROPLET indirect graph prefetcher |
| [ecg-custom-instruction.md](ecg-custom-instruction.md) | Custom RISC-V instruction for fat-ID |
| [context-passing.md](context-passing.md) | Metadata flow: JSON → gem5 → SimObject |
| [pipeline-integration.md](pipeline-integration.md) | Pipeline integration with graphbrew_experiment.py |

## Quick Start

```bash
# 1. Setup gem5
make setup-gem5

# 2. Run LRU baseline
python scripts/lib/pipeline/gem5.py \
    --graph results/graphs/soc-pokec/soc-pokec.sg \
    --benchmark pr --algorithm 0 --policy LRU

# 3. Run GRASP with DBG reordering
python scripts/lib/pipeline/gem5.py \
    --graph results/graphs/soc-pokec/soc-pokec.sg \
    --benchmark pr --algorithm 5 --policy GRASP

# 4. Run ECG with Leiden+RabbitOrder reordering
python scripts/lib/pipeline/gem5.py \
    --graph results/graphs/soc-pokec/soc-pokec.sg \
    --benchmark pr --algorithm 12 --policy ECG --ecg-mode DBG_PRIMARY
```

## References

- GRASP: Faldu et al., "Domain-Specialized Cache Management for Graph Analytics", HPCA 2020
- P-OPT: Balaji et al., "P-OPT: Practical Optimal Cache Management for Graph Analytics", HPCA 2021
- ECG: Mughrabi et al., "Expressing Locality and Prefetching for Optimal Caching in Graph Structures", GrAPL @ IPDPS 2026
- DROPLET: Basak et al., "Analysis and Optimization of Irregular Graph Applications on GPU", HPCA 2019
- gem5: Binkert et al., "The gem5 Simulator", ACM SIGARCH Computer Architecture News, 2011
