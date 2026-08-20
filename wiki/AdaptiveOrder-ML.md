# AdaptiveOrder

Algorithm 14 loads an offline-produced deterministic selector artifact.
Runtime training and graph-name lookup are not supported.

AdaptiveOrder selects a reordering from graph, kernel, cache-context, and reuse
features using an offline-produced deterministic model. Deployable selection
must improve end-to-end cost, including mapping generation and CSR relocation,
not merely predict the fastest reordered kernel.

## Deployable boundary

The runtime selector may use:

- a perceptron, decision tree, or hybrid artifact produced offline;
- measured graph features;
- kernel identity/access class;
- cache capacities and expected reuse count.

It may not use:

- graph filenames or canonical graph names;
- exact-name benchmark lookup;
- runtime kNN over the benchmark database;
- runtime model training;
- trial execution of multiple reorderers.

Exact-name comparisons remain available only through the explicitly labeled
offline `OracleUpperBound` analysis.

## Current CLI

```bash
# Default: perceptron + fastest-execution criterion
./bench/bin/pr -f graph.el -s -o 14 -n 3

# Model and criterion are independent
./bench/bin/pr -f graph.el -s \
  -o 14::::perceptron:best-endtoend -n 3

# Frozen all-kernel reuse-1/2 rule (reuse is mandatory)
./bench/bin/pr -f graph.el -s \
  -o 14:_:_:_:allkernel-lowreuse-rule:best-endtoend:1 -n 3

```

Current deployable models:

| Model | CLI |
|---|---|
| Perceptron | `perceptron` |
| Frozen PR-only one-use rule | `budgeted-rule` |
| Frozen seven-kernel reuse-1/2 rule | `allkernel-lowreuse-rule` |

Decision-tree and hybrid artifacts remain offline-only until they are
retrained on the Tier-0 schema; the runtime rejects legacy 24-feature models.

Both frozen rules require an explicit positive reuse value and the
`best-endtoend` criterion. The all-kernel rule supports PR, PR-SpMV, BFS, CC,
CC-SV, BC, and SSSP; reuse above 2 or unsupported kernels fall back to Boost
Rabbit (DBG in builds without Boost).

Criteria:

| Criterion | CLI |
|---|---|
| Mapping cost | `fastest-reorder` |
| Kernel time | `fastest-execution` |
| Complete one-use cost | `best-endtoend` |
| Reuse break-even | `best-amortization` |

Unknown models/criteria, graph-name fields, `knn`, and `database` fail closed.

## Offline artifacts

`results/data/adaptive_models.json` is historical model storage and a
load-only artifact. Its perceptron section must contain all five
exact portfolio arms and Tier-0 weights; missing arms fail closed.

`results/data/benchmarks.json` and `graph_properties.json` remain measurement
and training inputs. They are never a deployable runtime oracle.

The deployable perceptron consumes only the ten shared Tier-0 fields from
`adaptive_feature_schema.def`. Legacy 24-feature decision trees/hybrids remain
offline-only until retrained.

## Default portfolio

The selector accepts these exact canonical arms:

```text
0
5
8:csr
12:rabbit:compose:sg_none:comm_identity:intra_hubsort
12:rabbit:compose:sg_super_rabbit:comm_identity:intra_hubsort
```

Artifacts missing any required arm fail closed.
