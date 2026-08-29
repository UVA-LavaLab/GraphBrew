# Historical Low-Reuse Policy

This archived page records an earlier deterministic reuse-1/2 portfolio. It is
retained for reproducibility, not presented as the GraphBrew paper claim.

## Frozen interface

```text
14:_:_:_:allkernel-lowreuse-rule:best-endtoend:<reuse>
```

The rule used graph statistics, kernel identity, LLC capacity, and explicit
reuse to choose between:

- a bounded parallel GraphBrew composition; and
- Boost Rabbit fallback.

On the seven untouched graphs where it selected GraphBrew, the candidate beat
Boost at reuse one and two. Including five Rabbit fallbacks, the portfolio
also beat always-Boost in geometric mean.

## Why it is not the headline

1. The complete policy is not Rabbit-free.
2. Fallback graphs contribute ties by running the comparator itself.
3. Public accounting omitted Algorithm-14 feature-extraction time.
4. The bounded GraphBrew arm failed as a universal static method.
5. A later explicit reuse-8 graph-held-out study selected ORIGINAL as the
   static action for all 60 cells and found only 3.6% oracle headroom.
6. CART and ridge selectors both regressed relative to ORIGINAL.

No thresholds should be retuned around these results.

## Reproduction only

```bash
./bench/bin/pr -f graph.sg -s \
  -o '14:_:_:_:allkernel-lowreuse-rule:best-endtoend:1' \
  -n 3
```

Fully deployed accounting must include `Adaptive Feature Time`.

Current claims are documented in [Evidence and Claims](Evidence-and-Claims).
