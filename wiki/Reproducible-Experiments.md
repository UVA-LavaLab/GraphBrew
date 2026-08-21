# Reproducible Experiments

Use the public orchestrator for both rapid checks and frozen campaigns:

```bash
python3 scripts/graphbrew_experiment.py --help
```

Large graphs, mappings, and result artifacts belong under the external data
partition, not in the repository.

## Rapid and final paths

| Path | Purpose | Claim eligible |
|---|---|---|
| Rapid | dependency checks, parser failures, mapping bugs, candidate narrowing | no |
| Final | frozen graph/algorithm policy, repeated mappings and trials, verification, fixed affinity | yes |

Rapid example:

```bash
python3 scripts/graphbrew_experiment.py --vldb 2 --paper-preview \
  --paper-graph-dir /media/NVMeData/00_GraphDatasets/GraphBrew \
  --paper-artifact-root /media/NVMeData/00_GraphDatasets/GraphBrew/artifacts \
  --paper-threads 4 --paper-cpu-list 24-27
```

Final example:

```bash
python3 scripts/graphbrew_experiment.py --vldb \
  --paper-graph-dir /media/NVMeData/00_GraphDatasets/GraphBrew \
  --paper-artifact-root /media/NVMeData/00_GraphDatasets/GraphBrew/artifacts \
  --paper-threads 16 --paper-cpu-list 0-15
```

Use `--dry-run` or `--paper-preview` before broad collection.

## Canonical storage

```text
/media/NVMeData/00_GraphDatasets/GraphBrew/
├── <graph>.sg / <graph>.wsg
└── artifacts/
    ├── vldb_paper/
    ├── vldb_mappings/
    ├── vldb_runs/
    └── INDEX.json
```

Repository-local `results/` is for generic development observations, not
large final campaigns.

## Frozen policy

The campaign policy is owned by:

```text
scripts/experiments/vldb/config.py
scripts/experiments/vldb/sssp_policy.json
scripts/experiments/vldb/sssp_delta_tuning.json
```

It binds:

- graph names and source provenance;
- algorithm IDs, variants, and exact ordered `-o` strings;
- benchmark subset and trial counts;
- thread count and CPU affinity;
- mapping-generation draws;
- source vertices and kernel work policy;
- weighted SSSP conversion, checksum, delta, and answer identity;
- expected exclusions and timeout handling.

The SSSP tuning file in the repository is a compact policy-validation
snapshot. Full tuning trials remain under the external artifact root.

## Measurement contract

Raw observations preserve:

```text
representation build
mapping generation
permutation validation
CSR relocation
kernel time
executed work
verification state
mapping fingerprint
```

Report kernel-only quality and:

```text
mapping + reuse x kernel
```

separately. Never infer mapping cost from a pregenerated MAP kernel run.

## Restartable stages

Long campaigns can run through the versioned stage wrappers:

```bash
python3 scripts/experiments/vldb/stages/01_prep.py --exp 2 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py --exp 2 --preview
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --preview
python3 scripts/experiments/vldb/stages/05_aggregate.py --exp 0
```

Stages reuse content-bound sidecars and skip only work whose identity matches
the requested policy.

Cluster-specific wrappers and environment details are maintained in
[`scripts/experiments/vldb/README.md`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/scripts/experiments/vldb/README.md),
not duplicated in the wiki.

## Outputs

Final artifacts include:

- mapping sidecars and permutation fingerprints;
- one raw JSON record per graph/ordering/kernel attempt;
- verification manifests;
- frozen tables and figures;
- content hashes for source protocols and analyses;
- `INDEX.json` linking campaign outputs.

Failures and timeouts remain part of the record. Aggregation must not reduce
raw attempts to a fastest-run-only database.

## Verification

Before accepting results:

1. confirm graph provenance and dimensions;
2. validate every permutation;
3. ensure each kernel uses the intended mapping fingerprint;
4. bind source/work parameters;
5. verify answers or fingerprints;
6. confirm trials, threads, affinity, and build identity;
7. recompute public claims from frozen evidence.

Public recommendation claims are checked by
`scripts/test/test_documented_recommendations.py`.

## Custom scopes

Use explicit graph, algorithm, benchmark, policy, and artifact-root overrides.
Overrides must remain visible in the generated manifest and must not silently
replace final frozen defaults.

For generic collection outside the frozen campaign:

```bash
python3 scripts/graphbrew_experiment.py \
  --full --quick --size small --trials 1 --skip-cache
```

## Related pages

- [Benchmark Suite](Benchmark-Suite)
- [Running Benchmarks](Running-Benchmarks)
- [Command-Line Reference](Command-Line-Reference)
- [Reordering Algorithms](Reordering-Algorithms)
- [Troubleshooting](Troubleshooting)
