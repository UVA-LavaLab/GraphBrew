# BFS — Direction-Optimizing Breadth-First Search

## Citation

```bibtex
@inproceedings{do-bfs-sc12,
  title     = {Direction-Optimizing Breadth-First Search},
  author    = {Beamer, Scott and Asanovi\'{c}, Krste and Patterson, David},
  booktitle = {International Conference on High Performance Computing, Networking, Storage and Analysis (SC)},
  location  = {Salt Lake City, Utah},
  month     = {November},
  year      = {2012}
}
```

## Algorithm

- **Top-down**: For each frontier vertex u, check neighbors v — if unvisited, add to next frontier
- **Bottom-up**: For each unvisited vertex v, check if any neighbor is in frontier — if yes, claim v (BREAK)
- **Switching heuristic**: `if frontier_edges < unvisited_edges / alpha` → top-down; else → bottom-up (alpha ≈ 15)
- Bottom-up can be **10-20× faster** on scale-free graphs during large-frontier phases

**Why reordering matters**: Bottom-up iterates over ALL unvisited vertices — vertex ordering directly affects cache access patterns.

## GraphBrew Integration

- **Source**: `bench/src/bfs.cc` — Author: Scott Beamer
- **ECG classification**: Traversal-type (irregular access, Section B5)
