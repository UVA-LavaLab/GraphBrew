# All-Kernel Low-Reuse Selector

GraphBrew's validated automatic path is a frozen decision rule for mappings
reused once or twice. It chooses between a cost-matched GraphBrew composition
and Boost Rabbit.

![GraphBrew architecture](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg)

## Why this design

Rabbit is inexpensive to construct, so a more elaborate layout is useful only
when its kernel savings recover the extra preprocessing cost. Full
Leiden-Gorder composition improved locality but was approximately 17-18 times
more expensive to build than Rabbit. The promoted composition removes the
dominant cost while preserving the useful layout structure:

- two parallel Leiden iterations and passes;
- ordered super-graph proposal batches of 4096;
- no refinement phase;
- communities placed by descending size;
- Gorder8 inside communities of at most 5000 vertices;
- BFS fallback inside larger communities.

Exact configuration:

```text
12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:
cd_parallel:sgmb4096:gordf5000:norefine:2:2
```

On the five-graph cost-matched confirmation set, this composition was 10.4%
cheaper to construct than CSR Rabbit and 21.1% cheaper than Boost Rabbit while
remaining faster across the seven-kernel aggregate.

## What is automatic

The composition above is fixed. The runtime does not search the partitioner,
block layout, or vertex-layout space. Automation is limited to choosing
between that fixed composition and Boost Rabbit.

The deployable interface is:

```text
14:_:_:_:allkernel-lowreuse-rule:best-endtoend:<reuse>
```

where `<reuse>` is `1` or `2`.

## Frozen predicate

The candidate is eligible only when:

- the graph has at least 1000 vertices;
- reuse is at most 2;
- the kernel is PR, PR-SpMV, BFS, CC, CC-SV, BC, or SSSP; and
- the following predicate is true:

```text
property_wsr_llc <= 3.2 && ((degree_cv <= 2.68 && (avg_degree <= 60 || property_wsr_llc <= 0.82)) || degree_cv > 8)
```

Otherwise the selector uses Boost Rabbit. Reduced builds without Boost use
DBG as an explicit fallback.

`property_wsr_llc` is the estimated kernel-property working set divided by
machine last-level-cache capacity. The structural statistics come from the
lightweight graph-analysis pass.

This rule is deterministic. It uses no graph names, runtime training, learned
weights, nearest-neighbor lookup, or trial reorderings.

## Validation protocol

The predicate was derived on 18 graphs. It was frozen before 12 additional
graphs were opened. Mapping construction and all seven kernels were measured
with fixed affinity and repeated trials.

Seven holdouts selected GraphBrew and all seven passed at reuse 1 and 2. Five
holdouts used Boost Rabbit fallback.

| Reuse | Selected holdouts: Boost/GraphBrew | Lower 95% | Full selector/always-Boost |
|---:|---:|---:|---:|
| 1 | 1.696x | 1.502x | 1.361x |
| 2 | 1.642x | 1.460x | 1.336x |

### Final holdout decisions

| Graph | Decision |
|---|---|
| soc-BlogCatalog | GraphBrew; pass |
| soc-LiveMocha | GraphBrew; pass |
| web-Stanford | GraphBrew; pass |
| Amazon0601 | GraphBrew; pass |
| web-Google | GraphBrew; pass |
| web-NotreDame | GraphBrew; pass |
| loc-gowalla_edges | GraphBrew; pass |
| soc-pokec | Boost Rabbit fallback |
| as-skitter | Boost Rabbit fallback |
| cit-Patents | Boost Rabbit fallback |
| citeseer | Boost Rabbit fallback |
| ca-AstroPh | Boost Rabbit fallback |

The full 30-graph derivation and validation rows are in the
[machine-readable evidence file](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/allkernel-lowreuse-evidence.json).

## Interpretation

The result is an end-to-end low-reuse result, not a claim that GraphBrew always
produces faster kernels. The selector wins by balancing mapping cost against
the aggregate kernel savings for the declared reuse.

`reuse=2` means two complete kernel invocations share one materialized
mapping. It does not mean two PageRank iterations; one PageRank invocation
still performs its fixed internal iteration count.

## Limits

- The rule is validated only for reuse 1 and 2.
- It is a fallback policy, not a universal GraphBrew recommendation.
- Road and mesh-like graphs exposed failures of the universal static
  composition and are intentionally routed away by the rule.
- CC-SV can regress even when the seven-kernel end-to-end aggregate improves.
- Mapping construction and kernel time must remain separate in reported
  results.

See [AdaptiveOrder](AdaptiveOrder) for the runtime boundary and
[GraphBrewOrder](GraphBrewOrder) for explicit composition.
