# Reproducible Experiments

Use the top-level orchestrator for dependency checks, graph preparation,
mapping generation, benchmarks, cache simulation, and verification:

```bash
python3 scripts/graphbrew_experiment.py --help
```

Large graphs, mappings, and result artifacts should live on an external data
partition rather than in the repository.

## Rapid and controlled paths

| Path | Purpose |
|---|---|
| Rapid | dependency checks, parser failures, mapping bugs, and small candidate comparisons |
| Controlled | fixed graph/algorithm policy, repeated trials, verification, affinity, and scheduler checks |

Rapid example:

```bash
python3 scripts/graphbrew_experiment.py \
  --full --quick --size small --trials 1 --skip-cache
```

Inspect a broader plan without executing it:

```bash
python3 scripts/graphbrew_experiment.py \
  --target-graphs 50 --size small --dry-run
```

## Storage

Repository-local outputs are intended for bounded development runs:

```text
results/
├── data/
├── graphs/
├── logs/
└── mappings/
```

Use explicit graph and output roots for large campaigns. Do not commit graph
corpora, mappings, raw timing matrices, or machine-specific logs.

## Measurement contract

Keep these components separate:

```text
representation build
mapping generation
permutation validation
CSR relocation
kernel execution
executed work
verification state
mapping fingerprint
scheduler, nice value, and affinity
```

For repeated use, report:

```text
mapping + reuse x kernel
```

Never infer mapping cost from a kernel run that only loads a pre-generated
mapping.

## Restartability

The generic harness can run one phase at a time:

```bash
python3 scripts/graphbrew_experiment.py --phase reorder --size small
python3 scripts/graphbrew_experiment.py --phase benchmark --size small
python3 scripts/graphbrew_experiment.py --phase cache --size small
```

Specialized campaign runners reuse the same shared download, build, mapping,
verification, and result-store contracts. Their release instructions are
maintained separately from this generic workflow.

## Verification checklist

Before accepting a comparison:

1. confirm graph provenance and dimensions;
2. validate every permutation;
3. ensure each kernel uses the intended mapping fingerprint;
4. bind source, iteration, and weighted-kernel parameters;
5. verify answers or deterministic signatures;
6. confirm trials, threads, affinity, and binary identity;
7. record scheduler and nice state for timing; and
8. preserve failed and timed-out attempts.

## Related pages

- [Benchmark Suite](Benchmark-Suite)
- [Running Benchmarks](Running-Benchmarks)
- [Command-Line Reference](Command-Line-Reference)
- [Troubleshooting](Troubleshooting)
