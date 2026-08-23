# All-Kernel Low-Reuse Selector

GraphBrew's validated automatic path is a frozen rule for mappings reused once
or twice. It selects either one exact non-Rabbit GraphBrew composition or
Boost Rabbit.

[![GraphBrew low-reuse policy](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-lowreuse-policy.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-lowreuse-policy.svg)

**Figure 1.** The top row is the runtime decision. The bottom row evaluates the
same frozen predicate on two real rule2 holdouts: `amazon0601` selects
FastLeiden-Gorder8 and wins end to end at reuse 2; `soc-Slashdot0811` falls
outside the frozen region and runs the Rabbit baseline.

## Direct answers

### Does it use a previous kernel run?

No. The selector does not run the kernel, Rabbit, or GraphBrew before making
its decision. It does not read earlier benchmark rows or identify the graph
by filename.

It performs one deterministic sampled adjacency pass, combines those
statistics with the selected kernel, machine LLC capacity, and explicit reuse
count, then evaluates a fixed predicate.

### Does it work on a new graph?

Yes, mechanically: a graph unseen during rule derivation can be passed
directly to algorithm 14. The rule uses graph-derived features rather than a
graph-name table.

That is not a universal performance guarantee. Scientific validation currently
covers:

- PR, PR-SpMV, BFS, CC, CC-SV, BC, and SSSP;
- reuse 1 and 2;
- graphs with at least 1000 vertices; and
- the evaluated CPU platform.

The LLC-normalized feature improves portability, but cross-machine
generalization has not yet been established.

### If fallback is Rabbit, are we beating Rabbit?

The answer depends on the claim:

- **Non-Rabbit arm:** yes, on the seven final holdouts where GraphBrew was
  selected, the promoted Leiden-Gorder/BFS composition beat Boost Rabbit at
  reuse 1 and 2.
- **Fallback graph:** no GraphBrew win is claimed. The policy runs Boost
  Rabbit, so its ratio versus always-Boost is exactly 1.0 for that graph.
- **Whole portfolio:** yes, the frozen GraphBrew-or-Rabbit policy beat
  always-Boost in geometric mean because selected graphs contributed wins and
  fallback graphs contributed ties.

The current deployed policy is therefore not Rabbit-free. A paper claim should
distinguish the non-Rabbit arm from the reuse-aware portfolio. Replacing
Rabbit with a non-Rabbit fallback would require a new frozen validation.

## Runtime inputs

The deployed v2 predicate uses only:

| Input | Source | Meaning |
|---|---|---|
| `num_nodes` | exact graph metadata | reject very small graphs |
| `avg_degree` | deterministic sample | sampled mean out-degree |
| `degree_cv` | deterministic sample | sampled degree standard deviation divided by sampled mean |
| `property_wsr_llc` | modeled workload footprint / detected LLC bytes | whether the kernel property state fits near the machine cache |
| kernel | benchmark binary | choose only among the seven validated kernels |
| reuse | explicit CLI field | number of complete kernel invocations sharing the mapping |

The structural sample contains:

```text
max(1024, min(sqrt(num_nodes), 8192))
```

deterministically strided vertices and their adjacency lists. The binary may
log additional diagnostics, but the frozen v2 decision does not use them.

Because the predicate uses degree statistics and a modeled working-set ratio,
the decision is independent of the graph's filename and current vertex
labeling.

## Frozen decision

The candidate is eligible only when:

```text
supported_kernel
&& reuse <= 2
&& num_nodes >= 1000
&& property_wsr_llc <= 3.2
&& (
     (degree_cv <= 2.68
      && (avg_degree <= 60 || property_wsr_llc <= 0.82))
     || degree_cv > 8
   )
```

In plain language:

1. reject unsupported kernels, reuse above 2, small graphs, and property
   footprints above 3.2 times LLC;
2. select GraphBrew for low/moderate degree skew when degree or property
   footprint is bounded; or
3. select GraphBrew for extreme degree skew above 8;
4. use Boost Rabbit for the intermediate-skew region.

The decision is deterministic. If it selects GraphBrew, the resulting mapping
can still vary because the promoted composition uses parallel community
detection. Final studies therefore use repeated mapping draws and record
fingerprints.

## The selected GraphBrew arm

```text
12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:
cd_parallel:sgmb4096:gordf5000:norefine:2:2
```

This means:

- parallel bounded Leiden community detection;
- no additional final super-graph block order;
- largest community blocks first;
- input order for communities of size at most three;
- the relaxed local Gorder heuristic for sizes 4 through 5000;
- BFS in larger communities;
- ordered proposal batches of 4096 internal community super-nodes;
- no Leiden refinement; and
- at most two local-moving iterations and two aggregation passes.

The mechanism and figures for each control are in
[GraphBrewOrder](GraphBrewOrder).

## Deployable interface

```bash
# One complete kernel invocation
./bench/bin/pr -f graph.sg -s \
  -o '14:_:_:_:allkernel-lowreuse-rule:best-endtoend:1' \
  -n 3

# Two complete kernel invocations sharing one mapping
./bench/bin/bfs -f graph.sg -s \
  -o '14:_:_:_:allkernel-lowreuse-rule:best-endtoend:2' \
  -n 3
```

Reuse is not an internal PageRank iteration count. One PR invocation still
performs its configured internal iterations.

## Why fallback can still produce an overall win

For graph \(g\), define:

```text
speedup_g = Boost end-to-end time / chosen-path end-to-end time
```

Then:

```text
GraphBrew selected -> speedup_g > 1 is required by the final gate
Rabbit fallback    -> speedup_g = 1
```

The geometric mean can therefore exceed one without claiming that GraphBrew
wins on every graph.

| Reuse | Seven selected holdouts: Boost/GraphBrew | Lower 95% | Frozen portfolio/always-Boost |
|---:|---:|---:|---:|
| 1 | 1.696x | 1.502x | 1.361x |
| 2 | 1.642x | 1.460x | 1.336x |

## Representative unseen-graph decisions

| Graph | Key features | Frozen decision | What happened |
|---|---|---|---|
| `amazon0601` | avg degree 11.47, CV 0.73, WSR/LLC 0.14 | GraphBrew | 1.778x at reuse 1; 1.745x at reuse 2 |
| `in-2004` | avg degree 34.42, CV 10.76, WSR/LLC 0.48 | GraphBrew | extreme-skew branch; 1.817x and 1.770x |
| `roadNet-TX` | avg degree 2.79, CV 0.36, WSR/LLC 0.48 | GraphBrew | low-degree branch; 1.359x and 1.328x |
| `soc-Slashdot0811` | avg degree 12.95, CV 3.02, WSR/LLC 0.03 | Rabbit | avoided candidate regressions to 0.646x and 0.644x |
| `wiki-topcats` | avg degree 26.71, CV 2.87, WSR/LLC 0.62 | Rabbit | avoided candidate regressions to 0.854x and 0.857x |
| `web-BerkStan` | avg degree 17.78, CV 4.17, WSR/LLC 0.24 | Rabbit | conservative miss: candidate would have reached 1.552x and 1.509x |

The last row is important: the frozen rule is not an oracle. It deliberately
accepts missed headroom to avoid tuning after holdouts are opened.

Final untouched graphs:

- selected: `amazon0601`, `cnr-2000`, `coPapersCiteseer`,
  `coPapersDBLP`, `dblp-2010`, `in-2004`, `roadNet-TX`;
- fallback: `kron_g500-logn18`, `soc-Slashdot0811`, `web-BerkStan`,
  `wiki-Talk`, `wiki-topcats`.

All 30 derivation and validation records are in
[`docs/allkernel-lowreuse-evidence.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/allkernel-lowreuse-evidence.json).

## Evidence accounting

The public selector ratios are portfolio accounting over:

```text
chosen mapping cost + reuse x chosen kernel time
```

The public 30-row evidence file does not store the deployable algorithm-14
feature-extraction time. The binary reports that value as
`Adaptive Feature Time`. A fully deployed timing claim must add it rather than
silently treating selection as free.

This distinction does not affect which arm the frozen predicate selects, but
it matters for final end-to-end reporting.

## Limits

- The policy is validated only for reuse 1 and 2.
- It depends on Boost Rabbit for fallback and is not a Rabbit-free system.
- Graph type alone does not determine the branch; `roadNet-TX` selected
  GraphBrew while larger road/mesh cases helped reject universal static use.
- The decision is deterministic, but `cd_parallel` mappings are
  schedule-sensitive.
- Cross-machine performance remains unvalidated.
- CC-SV can regress even when the seven-kernel aggregate improves.
- Selector feature time must be included in a fully deployed result.

See [AdaptiveOrder](AdaptiveOrder) for the runtime boundary and
[GraphBrewOrder](GraphBrewOrder) for the composition mechanism.
