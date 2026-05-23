# TC — Triangle Counting

## Citation

GAP Benchmark Suite — Beamer, Asanović, Patterson (arXiv:1508.03619, 2015).

## Algorithm

Count triangles {u,v,w} where all three edges exist.

```
For each edge (u,v) where u < v:
    count |N(u) ∩ N(v)|    // Set intersection of neighbor lists
```

Sorted neighbor lists → merge-join intersection O(d_u + d_v) per edge. Parallel over edges.

**O(|E| × √|E|)** with degree-based orientation (Chiba-Nishizeki).

**Cache sensitivity**: Very high — each edge reads two neighbor lists. Degree-based orderings (HubSort, HubCluster) place high-degree vertices contiguously, improving cache reuse for the heaviest intersections.

## GraphBrew Integration

- **TC Source**: `bench/src/tc.cc`, `bench/src/tc_p.cc` — Author: Scott Beamer
- **Note**: TC is not in the ECG paper's 7 benchmarks but is in the broader GAP suite
