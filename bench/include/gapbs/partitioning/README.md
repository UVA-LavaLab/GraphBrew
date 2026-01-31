# Cagra / P-OPT Partition Helpers (bench/include/gapbs/partitioning)

Purpose: Cagra/GraphIT-style CSR slicing and partitioning used by `builder.h` when `-j` specifies type `0`.

Key headers:
- `popt.h`
  - `graphSlicer` — slice CSR by vertex range
  - `MakeCagraPartitionedGraph` — build p_n × p_m partitions (honors `outDegree`, CLI `-z`)

Replaces legacy alias:
- `bench/include/gapbs/cache/popt.h` (removed)
