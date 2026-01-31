# Cagra / P-OPT Partition Helpers (bench/include/gapbs/cache)

Purpose: Cagra/GraphIT-style CSR slicing and partitioning used by `builder.h` when `-j` specifies type `0`.

Key headers:
- `popt.h`
  - `graphSlicer` — slice CSR by vertex range
  - `MakeCagraPartitionedGraph` — build p_n × p_m partitions (honors `outDegree`, CLI `-z`)

Not cache simulation: For cache sim (L1/L2/L3) headers, see `bench/include/cache/README.md`.
