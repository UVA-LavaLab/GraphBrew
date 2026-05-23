# BC — Betweenness Centrality

## Citations

```bibtex
@article{brandes-2001,
  title   = {A faster algorithm for betweenness centrality},
  author  = {Brandes, Ulrik},
  journal = {Journal of Mathematical Sociology},
  volume  = {25},
  number  = {2},
  pages   = {163--177},
  year    = {2001}
}

@inproceedings{madduri-ipdps09,
  title     = {A faster parallel algorithm and efficient multithreaded implementations for evaluating betweenness centrality on massive datasets},
  author    = {Madduri, Kamesh and Ediger, David and Jiang, Karl and Bader, David A and Chavarria-Miranda, Daniel},
  booktitle = {International Symposium on Parallel \& Distributed Processing (IPDPS)},
  year      = {2009}
}
```

## Algorithm

BC(v) = Σ_{s≠v≠t} σ_st(v) / σ_st — fraction of shortest paths through v.

1. **Forward BFS** from source s: compute distances d[v] and path counts σ[v]
2. **Backward accumulation** (reverse BFS order): δ[w] += (σ[w]/σ[v]) × (1 + δ[v])

**O(V × E)** unweighted. Repeated BFS from many sources — compounding reorder benefits.

## GraphBrew Integration

- **Source**: `bench/src/bc.cc` — Author: Scott Beamer
- **ECG classification**: Traversal-type (very high cache sensitivity)
