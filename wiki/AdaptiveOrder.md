# AdaptiveOrder

AdaptiveOrder (`-o 14`) is GraphBrew's runtime selection boundary. The
validated path is a frozen deterministic rule, not a machine-learning model.

```bash
./bench/bin/pr -f graph.sg -s \
  -o '14:_:_:_:allkernel-lowreuse-rule:best-endtoend:<reuse>' \
  -n 3
```

`<reuse>` must be `1` or `2`.

## Runtime contract

The rule:

- samples structure from the new graph;
- models the selected kernel's property footprint relative to LLC;
- uses kernel identity and declared reuse;
- evaluates one frozen predicate; and
- chooses the promoted GraphBrew composition or Boost Rabbit.

It does not run candidate orderings first, train at runtime, use graph names,
or query prior benchmark rows. The branch decision is deterministic; a
selected `cd_parallel` GraphBrew mapping can still be schedule-sensitive.

Unsupported kernels, small graphs, and reuse above two use the fallback.

## Timing

Public portfolio evidence accounts for chosen mapping cost plus reused kernel
time. The binary reports `Adaptive Feature Time` separately; fully deployed
timing must include it.

For the predicate, supported kernels, exact arms, graph examples, confidence
intervals, and limitations, use
[All-Kernel Low-Reuse Selector](All-Kernel-Low-Reuse-Selector).

Legacy model modes remain only for offline compatibility.
