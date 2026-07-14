# partition/ — Graph Partitioning

| Path | Method | CLI |
|------|--------|-----|
| `compact_csr.h` | Edge-balanced compact CSR shards with explicit ghosts | `bfs_p -P n -B total` |
| `trust.h` | TRUST partitioning | `-j 1:n:m` |
| `cagra/popt.h` | Cagra/P-OPT (GraphIT-style) partitioning | `-j 0:n:m` |
