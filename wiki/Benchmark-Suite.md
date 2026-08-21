# Benchmark Suite

GraphBrew evaluates ordering cost and graph-kernel behavior through one public
entry point:

```bash
python3 scripts/graphbrew_experiment.py --help
```

Use [Running Benchmarks](Running-Benchmarks) for commands and
[Python Scripts](Python-Scripts) for package ownership.

## Common modes

```bash
# Rapid debugging path
python3 scripts/graphbrew_experiment.py --full --size small --quick

# One phase
python3 scripts/graphbrew_experiment.py --phase benchmark --size small

# Inspect a broad plan without executing it
python3 scripts/graphbrew_experiment.py --target-graphs 50 --dry-run

# Frozen publication workflow
python3 scripts/graphbrew_experiment.py --vldb --paper-preview
```

Large graphs, mappings, and campaign results belong under the configured
external graph and artifact roots. Generic observations use
`results/data/benchmarks.json`.

## Measurement boundary

Report these components separately:

```text
mapping generation
permutation validation
CSR relocation
kernel-only time
mapping + reuse x kernel
```

Pre-generated mappings let every kernel use the same permutation without
charging mapping construction inside each kernel trial.

## Amortization

For baseline kernel time \(T_b\), reordered kernel time \(T_r\), mapping cost
\(T_m\), and reuse \(N\):

```text
end-to-end speedup = N x T_b / (T_m + N x T_r)
break-even reuse   = T_m / (T_b - T_r)
```

The analysis modules compute these metrics from raw observations; they do not
replace the raw timing records.

## Results

- generic observations: `results/data/benchmarks.json`
- graph properties: `results/data/graph_properties.json`
- mappings: `results/mappings/`
- frozen campaigns: configured `--paper-artifact-root`

Historical offline-model files may exist under `results/data/`, but the
validated low-reuse rule does not require them.

## Related pages

- [Running Benchmarks](Running-Benchmarks)
- [Reproducible Experiments](Reproducible-Experiments)
- [Command-Line Reference](Command-Line-Reference)
- [Reordering Algorithms](Reordering-Algorithms)
