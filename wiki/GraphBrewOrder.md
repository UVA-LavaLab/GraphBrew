# GraphBrewOrder

GraphBrewOrder (`-o 12`) constructs a vertex permutation from explicit,
independent decisions. It changes vertex IDs and CSR memory placement; it
does not add, remove, or redirect edges.

![GraphBrew relabeling example](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-relabeling-example.svg)

The figure is schematic. The promoted recipe places the largest community
block first, uses Gorder8 inside communities with at most 5000 vertices, and
uses BFS inside larger communities.

## The active composition

The non-Rabbit arm used by the low-reuse policy is:

```text
12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:
cd_parallel:sgmb4096:gordf5000:norefine:2:2
```

A useful plain-English name is:

```text
parallel bounded Leiden -> SizeDesc blocks -> Gorder8-or-BFS vertices
```

This is one fixed experimental treatment. Algorithm 12 does not search for a
better combination at runtime.

## Three layout decisions

| Decision | Active choice | Effect |
|---|---|---|
| Partitioner | bounded GVE-Leiden | discovers communities |
| Block layout | `comm_size_desc` | assigns contiguous ID ranges from largest to smallest community |
| Vertex layout | `intra_gorder:gw8:gordf5000` | uses Gorder8 in small communities and BFS in large communities |

The remaining tokens bound how much work the partitioner and local layout are
allowed to perform.

## Reading every token

| Token | Scope | Meaning |
|---|---|---|
| `12` | dispatcher | execute GraphBrewOrder |
| `leiden` | partitioner | use the GVE-Leiden CSR path |
| `compose` | pipeline | expose block and vertex layout as separate stages |
| `sg_none` | final block layout | do not run an additional Rabbit/RCM order over the final community super-graph |
| `comm_size_desc` | final block layout | place larger community blocks first |
| `intra_gorder` | vertex layout | enable local Gorder |
| `gw8` | vertex layout | use an eight-vertex Gorder sliding window |
| `cd_parallel` | community detection | permit OpenMP execution instead of forcing community detection to one thread |
| `sgmb4096` | internal Leiden aggregation | compute up to 4096 super-node proposals in parallel, then commit them in super-node order |
| `gordf5000` | vertex layout | use BFS when a detected community contains more than 5000 vertices |
| `norefine` | community detection | skip Leiden's constrained refinement phase |
| first `2` | community detection | cap local-moving iterations at two |
| second `2` | community detection | cap aggregation passes at two |

### Why `sg_none` and `sgmb4096` are not contradictory

They refer to different super-graphs:

- `sgmb4096` controls **internal Leiden aggregation**, where current
  communities become super-nodes during detection.
- `sg_none` controls the **final block-layout stage** after detection. It says
  not to apply another coarse ordering to the completed community graph.

## The four cost controls

![GraphBrew cost controls](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-cost-controls.svg)

### `cd_parallel`

The default deterministic community-detection mode temporarily limits the
Leiden section to one OpenMP thread. `cd_parallel` removes that limit.

What it changes:

- local-moving and aggregation work can use the configured thread team;
- community updates can be observed in a different schedule; and
- the final partition and mapping fingerprint can vary across executions.

What it does **not** change:

- the selected algorithm string;
- the frozen policy decision; or
- graph topology.

Final studies therefore record mapping fingerprints and use repeated mapping
draws. A deterministic policy decision does not imply a byte-identical
parallel mapping.

### `sgmb4096`

Leiden repeatedly operates on an aggregated graph whose vertices are current
communities. A fully sequential local-move loop became a preprocessing
bottleneck on that super-graph.

For each batch:

1. up to 4096 community super-nodes compute their best move proposals in
   parallel from the pre-commit state;
2. a single ordered commit applies accepted proposals by community ID; and
3. the next batch observes the updated state.

The number 4096 counts **community super-nodes**, not original vertices,
edges, bytes, or OpenMP threads. `sgmb1` preserves the sequential
proposal/commit semantics. Larger batches reduce serialization but can change
the partition because proposals within a batch do not observe earlier commits
from that same batch.

### `gordf5000`

The threshold is evaluated independently for every detected community:

```text
community size <= 5000  -> Gorder with window 8
community size >  5000  -> hub-rooted BFS
```

It is not a whole-graph cutoff. A graph with millions of vertices can still
run Gorder on thousands of small communities while routing only its largest
blocks through BFS.

The purpose is to bound the expensive tail of local ordering. Removing
`gordf5000` allows Gorder on every community and creates a different,
unvalidated treatment.

### `norefine`

Full Leiden performs:

```text
local moving -> constrained refinement -> aggregation
```

Refinement resets vertices or super-nodes to singletons inside their
pre-refinement community bounds, then moves them again to improve community
connectivity before aggregation.

`norefine` uses:

```text
local moving -----------------------> aggregation
```

This removes an entire move phase. It can reduce preprocessing substantially,
but it can also change community quality and the resulting layout. It is a
deliberate cost-quality trade-off, not a generally preferred Leiden setting.

## Why these controls were combined

The full Leiden-SizeDesc-Gorder8 composition produced strong kernel-only
locality but cost roughly 17-18 times as much to construct as Rabbit in the
initial matrix. Cheap one-pass substitutes reached Rabbit-like construction
cost but lost the quality advantage.

The promoted recipe instead combines:

- parallel community detection;
- ordered batched super-graph moves;
- no refinement;
- two-iteration/two-pass ceilings; and
- Gorder only where the community-size bound permits it.

On the five-graph frozen confirmation set, that exact composition was 10.4%
cheaper to construct than CSR Rabbit and 21.1% cheaper than Boost Rabbit while
retaining a seven-kernel aggregate win. The later eleven-graph static gate
failed on road/mesh and large-graph regimes, which is why deployment uses a
selector rather than applying this composition universally.

## Safe use

```bash
./bench/bin/pr -f graph.sg -s \
  -o '12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:cd_parallel:sgmb4096:gordf5000:norefine:2:2' \
  -n 3
```

Treat any token change as a new algorithm:

| Intended experiment | Change | Consequence |
|---|---|---|
| deterministic community-detection control | replace `cd_parallel` with `cd_serial` | no longer the promoted recipe |
| exact sequential super-move control | replace `sgmb4096` with `sgmb1` | higher serialization; different mapping semantics |
| Gorder all communities | remove `gordf5000` | removes the local-cost bound |
| full Leiden refinement | remove `norefine` | adds refinement work and changes communities |
| larger search budget | change `2:2` | changes partitioning cost and quality |

Do not retune these values on final holdout graphs. The selector evidence is
valid only for the exact frozen string.

## Measurement contract

Report separately:

```text
mapping generation
permutation validation
CSR relocation
kernel-only time
mapping + reuse x kernel
```

Also record graph provenance, build identity, threads, affinity, mapping
fingerprint, trial count, and reuse. Reuse counts complete kernel invocations
sharing one materialized mapping.

Other historical and diagnostic algorithm-12 tokens remain callable, but they
are reference surfaces rather than current recommendations. See
[Command-Line Reference](Command-Line-Reference).

## Related pages

- [All-Kernel Low-Reuse Selector](All-Kernel-Low-Reuse-Selector)
- [AdaptiveOrder](AdaptiveOrder)
- [Reordering Algorithms](Reordering-Algorithms)
- [Reproducible Experiments](Reproducible-Experiments)
