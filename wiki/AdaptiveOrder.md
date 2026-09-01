# AdaptiveOrder

AdaptiveOrder (`-o 14:<policy>`) is an experimental compatibility interface
that dispatches to another ordering. It has no intrinsic permutation.

The selected policy must determine a concrete ordering, after which GraphBrew
records the requested policy, resolved ordering, and mapping identity.

For controlled comparisons, prefer either:

- `-o 12:<configuration>` for an explicit composition; or
- `-o 13:<mapping>` for a pre-generated permutation.

If Algorithm 14 is used, preserve the complete policy string and resolved
mapping fingerprint in the result record. Do not treat the interface itself
as an ordering algorithm.

See [Command-Line Reference](Command-Line-Reference) and
[Reordering Algorithms](Reordering-Algorithms).
