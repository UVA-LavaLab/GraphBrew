# RCM (Reverse Cuthill-McKee) — Algorithm ID: 11

## Citations

```bibtex
@inproceedings{cuthill-mckee-69,
  title     = {Reducing the bandwidth of sparse symmetric matrices},
  author    = {Cuthill, Elizabeth and McKee, James},
  booktitle = {Proceedings of the 1969 24th National Conference of the ACM},
  pages     = {157--172},
  year      = {1969}
}

@article{george-liu-79,
  title   = {An implementation of a pseudoperipheral node finder},
  author  = {George, Alan and Liu, Joseph W. H.},
  journal = {ACM Transactions on Mathematical Software (TOMS)},
  volume  = {5},
  number  = {3},
  year    = {1979}
}

@article{rcmpp-2024,
  title   = {RCM++: Reverse Cuthill-McKee ordering with Bi-Criteria Node Finder},
  author  = {Hou, Yiwei and others},
  journal = {arXiv preprint arXiv:2409.04171},
  year    = {2024}
}

@inproceedings{mlakar-ipdps21,
  title     = {Speculative Parallel Reverse Cuthill-McKee Reordering on Multi- and Many-core Architectures},
  author    = {Mlakar, Daniel and others},
  booktitle = {IEEE IPDPS},
  year      = {2021}
}
```

## Algorithm Description

1. **Starting Node**: George-Liu pseudo-peripheral finder (iterate BFS until eccentricity stabilizes) — or BNF variant (eccentricity + degree criteria)
2. **BFS Ordering**: From starting node, run BFS level-by-level, ordering vertices by ascending degree within each level
3. **Reversal**: Reverse entire ordering (preserves bandwidth, often improves profile)

**Complexity**: O(|V| + |E|) BFS + O(|V| log |V|) degree sorting.
**Best for**: Sparse matrices, FEM meshes, road networks.

## GraphBrew Integration

- **Algorithm ID**: 11 (RCM)
- **C++ Implementation**: `bench/include/graphbrew/reorder/reorder_rcm.h`
- **CLI**: `-o 11`
- **Variants**: `default` (George-Liu), `bnf` (RCM++ Bi-Criteria Node Finder)
