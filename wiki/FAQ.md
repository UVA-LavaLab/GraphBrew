# FAQ

Short answers to common questions. For full guides see
[Getting Started](Getting-Started),
[Reordering Algorithms](Reordering-Algorithms), and
[Troubleshooting](Troubleshooting).

## Which reordering should I try first?

| Situation | Try |
|---|---|
| Establish a baseline | `-o 0` (ORIGINAL) |
| Very low construction cost | `-o 5` (DBG) |
| Community-oriented baseline | `-o 8:csr` (RabbitOrder CSR) |
| Expensive window-locality reference | `-o 9:csr` (Gorder) |
| Explicit composition | `-o 12:<configuration>` |
| Bandwidth-oriented control | `-o 11:bnf` |

There is no universal winner. Compare complete preprocessing cost, kernel
time, iteration or work changes, and expected mapping reuse on the workload
you intend to run.

## Why can composition help?

Partitioners, block orders, and intra-block layouts target different
properties. Community grouping can reduce cross-region accesses, local
window methods can shorten co-access distance, degree layouts can concentrate
hubs, and bandwidth methods can reduce ID span.

The effect is graph- and kernel-dependent. Hold other stages fixed when
testing one operator.

## Why does reordering sometimes hurt?

1. The input ordering may already have useful structure.
2. The kernel may run too few times to amortize construction.
3. The chosen layout may not match the kernel's access pattern.
4. Vertex IDs may change executed work for propagation or compression
   algorithms.

## Where do benchmark results land?

| Output | Location |
|---|---|
| Standard runs | stdout |
| Explicit C++ self-recording (`-D DIR`) | `DIR/benchmarks.json` |
| Generic pipeline | `results/data/benchmarks.json` |
| Historical offline models | `results/data/adaptive_models.json` |

Use an explicit external output root for large campaigns.

## What is AdaptiveOrder?

Algorithm 14 is an experimental policy-dispatch interface. It resolves to
another ordering and therefore has no intrinsic permutation. Preserve the
complete policy string and resolved mapping fingerprint when using it.

## How do I add an algorithm or benchmark?

See [Contributing](Contributing).

## What graph formats are supported?

`.sg` (GAPBS binary), `.el` (edge list), `.wel` (weighted edge list), and
`.mtx` (Matrix Market). See
[Supported Graph Formats](Supported-Graph-Formats).

## What is the difference between LeidenOrder and GraphBrew-Leiden?

- `-o 15` runs GVE-Leiden followed by one selected post-layout.
- `-o 12:leiden...` uses Leiden as the partitioner inside an explicit
  multi-stage GraphBrew composition.

They are different pipelines and should be identified by their complete
configuration strings.
