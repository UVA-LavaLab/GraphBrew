# AdaptiveOrder (research-only)

> Algorithm 14 is research-only and deferred until the independent-ordering
> Pareto gate. It must not be presented as a validated deployment result.

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

```

Current deployable model:

| Model | CLI |
|---|---|
| Perceptron | `perceptron` |

Decision-tree and hybrid artifacts remain offline-only until they are
retrained on the Tier-0 schema; the runtime rejects legacy 24-feature models.

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

## Frozen first portfolio

The first selector study uses exact canonical arms:

```text
0
5
8:csr
12:rabbit:compose:sg_none:comm_identity:intra_hubsort
12:rabbit:compose:sg_super_rabbit:comm_identity:intra_hubsort
```

The public selector protocol will be added here after the
independent-ordering gate passes.
