# AdaptiveOrder

AdaptiveOrder (`-o 14`) is an experimental runtime-selection compatibility
interface. It has no intrinsic permutation; it dispatches to another ordering.

The repository retains a frozen reuse-1/2 rule for reproducing an earlier
GraphBrew-or-Rabbit portfolio:

```bash
./bench/bin/pr -f graph.sg -s \
  -o '14:_:_:_:allkernel-lowreuse-rule:best-endtoend:1' \
  -n 3
```

That rule is **not a paper contribution**:

- its fallback is Boost Rabbit;
- public portfolio accounting did not include `Adaptive Feature Time`;
- it does not establish a Rabbit-free GraphBrew system; and
- later graph-held-out studies failed to recover enough oracle headroom.

Use Algorithm 14 only for compatibility and historical reproduction. New
paper experiments use explicit Algorithm-12 compositions and always include
ORIGINAL.

See [Historical Low-Reuse Policy](Historical-Low-Reuse-Policy) and
[Evidence and Claims](Evidence-and-Claims).
