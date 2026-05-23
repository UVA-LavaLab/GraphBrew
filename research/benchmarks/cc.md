# CC — Connected Components

## Citations

```bibtex
@inproceedings{afforest-ipdps18,
  title     = {Optimizing Parallel Graph Connectivity Computation via Subgraph Sampling},
  author    = {Sutton, Michael and Ben-Nun, Tal and Barak, Amnon},
  booktitle = {Symposium on Parallel and Distributed Processing (IPDPS)},
  year      = {2018}
}

@article{shiloach-vishkin-82,
  title   = {An O(log n) parallel connectivity algorithm},
  author  = {Shiloach, Yossi and Vishkin, Uzi},
  journal = {Journal of Algorithms},
  volume  = {3},
  number  = {1},
  pages   = {57--67},
  year    = {1982}
}

@inproceedings{bader-icpp05,
  title     = {On the architectural requirements for efficient execution of graph algorithms},
  author    = {Bader, David A and Cong, Guojing and Feo, John},
  booktitle = {International Conference on Parallel Processing (ICPP)},
  year      = {2005}
}
```

## Algorithms

### CC — Afforest (Sutton et al. 2018)
1. **Sample**: Connect each vertex to first 2 neighbors (sparse subgraph sampling)
2. **Identify** large component (most common root)
3. **Process remaining**: Full neighbor scan for non-large-component vertices
4. **Compress**: Path-compress component forest

### CC_SV — Shiloach-Vishkin (1982)
Hook-and-compress: repeatedly hook (union) and compress (path compress) until stable. O(|E| log |V|).

Min-max swap (Kothapalli et al. 2010) handles directed graphs.

## GraphBrew Integration

- **CC Source**: `bench/src/cc.cc` — Authors: Michael Sutton, Scott Beamer
- **CC_SV Source**: `bench/src/cc_sv.cc` — Author: Scott Beamer
- **ECG classification**: Traversal-type (pointer-chasing)
- **P-OPT repo**: Also implements `cc_sv.cc` as test application
