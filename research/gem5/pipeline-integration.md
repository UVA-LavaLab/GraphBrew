# Pipeline Integration: gem5 in graphbrew_experiment.py

## Overview

The gem5 simulation integrates into the existing GraphBrew experiment pipeline
as an alternative backend for Phase 4 (cache simulation). When `--simulator gem5`
is specified, the pipeline routes cache simulation through `scripts/lib/pipeline/gem5.py`
instead of `scripts/lib/pipeline/cache.py`.

## Architecture

```
graphbrew_experiment.py
    │
    ├── Phase 0: Prerequisites (ensure_prerequisites)
    ├── Phase 1a: Download graphs
    ├── Phase 1b: Convert MTX → SG
    ├── Phase 1c: Pre-generate mappings
    ├── Phase 2: Reorder
    ├── Phase 3: Benchmark (native execution)
    │
    ├── Phase 4: Cache Simulation
    │   │
    │   ├── --simulator builtin (default)
    │   │   └── scripts/lib/pipeline/cache.py
    │   │       └── run_cache_simulations()
    │   │       └── bench/bin_sim/{bench} → parse stdout
    │   │
    │   └── --simulator gem5
    │       └── scripts/lib/pipeline/gem5.py
    │           ├── export_gem5_metadata() → JSON sideband
    │           └── run_gem5_simulations()
    │               └── gem5.opt configs/graphbrew/graph_se.py
    │                   → parse m5out/stats.txt
    │
    ├── Phase 5: Analysis
    └── ...
```

## Integration Points

### 1. Pipeline Module (`scripts/lib/pipeline/gem5.py`)

Mirrors `cache.py` interface:

```python
# Same signature as run_cache_simulations()
def run_gem5_simulations(
    graphs: List[GraphInfo],
    algorithms: List[int] = None,
    benchmarks: List[str] = None,
    policy: str = "LRU",
    ecg_mode: str = "DBG_PRIMARY",
    prefetcher: str = "none",
    cpu_type: str = "timing",
) -> List[CacheResult]:
    ...
```

Returns the same `CacheResult` dataclass used by the standalone simulator,
ensuring downstream analysis code works unchanged.

### 2. Metadata Export

Before running gem5, the pipeline exports graph metadata:

```python
# Per-graph metadata
metadata_path = export_gem5_metadata(graph_name, graph_path)
# Creates: results/gem5_metadata/{graph_name}/context.json
```

### 3. gem5 Command Construction

```python
cmd = f"{gem5_binary} --outdir={output_dir} {config_script} " \
      f"--binary={bench_binary} " \
      f'--options="-f {graph_path} -s -o {algo_opt} -n 1" ' \
      f"--policy={policy} --cpu-type={cpu_type}"
```

### 4. Stats Parsing

gem5 outputs statistics to `m5out/stats.txt`. The parser
(`bench/include/gem5_sim/scripts/parse_stats.py`) extracts:

```
system.cpu.dcache.overallMissRate::total    0.131266
system.l2cache.overallMissRate::total       0.079342
system.l3cache.overallMissRate::total       0.015482
system.cpu.ipc                              0.423156
```

These are converted to `CacheResult` fields and stored in the same
`BenchmarkStore` database.

### 5. Experiment Configuration

Add to `scripts/experiments/ecg_config.py`:

```python
GEM5_CONFIG = {
    "GEM5_BINARY": str(PROJECT_ROOT / "bench/include/gem5_sim/gem5/build/X86/gem5.opt"),
    "GEM5_CONFIG_DIR": str(PROJECT_ROOT / "bench/include/gem5_sim/configs/graphbrew"),
    "GEM5_CPU_TYPE": "timing",      # timing | O3 | minor
    "GEM5_ISA": "X86",
    "GEM5_TIMEOUT": 7200,           # 2 hours standard
    "GEM5_TIMEOUT_HEAVY": 14400,    # 4 hours for BC/SSSP
}
```

## Usage

### From Pipeline

```bash
# Full pipeline with gem5 cache simulation
python scripts/graphbrew_experiment.py --full --simulator gem5

# Just cache phase with gem5
python scripts/graphbrew_experiment.py --phase cache --simulator gem5 \
    --graphs soc-pokec --benchmarks pr --algorithms 0,5,8

# gem5 with specific policy
python scripts/graphbrew_experiment.py --phase cache --simulator gem5 \
    --policy ECG --ecg-mode DBG_PRIMARY
```

### Standalone Module

```bash
# Single simulation
python -m scripts.lib.pipeline.gem5 \
    --graph results/graphs/soc-pokec/soc-pokec.sg \
    --benchmark pr --algorithm 5 --policy GRASP

# With ECG + DROPLET
python -m scripts.lib.pipeline.gem5 \
    --graph results/graphs/soc-pokec/soc-pokec.sg \
    --benchmark pr --algorithm 12 \
    --policy ECG --ecg-mode DBG_PRIMARY --prefetcher DROPLET
```

### Makefile

```bash
make setup-gem5         # Clone + build gem5
make clean-gem5         # Remove cloned gem5
```

## Timeouts

gem5 is ~100-1000x slower than native execution:

| Benchmark | Native | Standalone Sim | gem5 (Timing) | gem5 (O3) |
|-----------|--------|----------------|---------------|-----------|
| PR (small graph) | 0.1s | 10s | ~30min | ~5h |
| BFS (small graph) | 0.05s | 5s | ~15min | ~3h |
| BC (medium graph) | 5s | 300s | ~10h | ~days |

**Recommended workflow**:
1. Validate on tiny graphs (soc-karate, 34 vertices) with all policies
2. Use TimingSimpleCPU for policy sweeps
3. Use DerivO3CPU only for final validation of key configurations

## Data Flow

```
results/
├── gem5_metadata/
│   └── {graph_name}/
│       └── context.json          # Graph metadata for gem5
├── gem5_runs/
│   └── {graph}_{bench}_{algo}_{policy}/
│       ├── stats.txt             # gem5 statistics output
│       ├── config.ini            # gem5 configuration dump
│       └── config.json           # gem5 configuration (JSON)
└── data/
    └── benchmarks.json           # Unified results (both simulators)
```

## Cross-Validation

To verify gem5 results match standalone cache_sim:

```bash
# Run same config on both simulators
python scripts/graphbrew_experiment.py --phase cache \
    --simulator builtin --policy LRU --graphs soc-pokec --benchmarks pr

python scripts/graphbrew_experiment.py --phase cache \
    --simulator gem5 --policy LRU --graphs soc-pokec --benchmarks pr

# Compare results (should match within 5-10% for miss rates)
```

Differences arise from:
- gem5 models timing effects (cache contention, bus latency) that standalone
  simulator ignores
- gem5 tracks instruction cache misses (standalone does not)
- gem5 includes cold-start misses from binary loading

For **replacement policy comparison** (GRASP vs SRRIP, ECG vs GRASP), the
**relative** ordering should be identical: if GRASP beats SRRIP in standalone,
it must also beat SRRIP in gem5.
