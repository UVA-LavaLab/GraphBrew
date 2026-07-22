# Contributing

How to add a new reordering algorithm or benchmark to GraphBrew.
Both follow the same pattern: edit a small set of files, register
the algorithm/benchmark, rebuild.

## Adding a new reordering algorithm

A reordering produces a permutation `new_ids[old_id] = new_vertex_id`.
You implement it as a single function, register it in three places, and
build.

### 1. Reserve an enum ID

Edit `bench/include/external/gapbs/util.h` and append to
`enum ReorderingAlgo`:

```cpp
enum ReorderingAlgo {
    ORIGINAL = 0,
    // ... existing entries ...
    GoGraphOrder = 16,
    MyNewOrder = 17,    // your algorithm
};
```

Use camelCase. Reserve only one ID per algorithm even if you support
multiple variants (variants go in the parser tokens, e.g. `-o 17:mode`).

### 2. Implement the algorithm

Add a header under `bench/include/graphbrew/reorder/`:

```cpp
// bench/include/graphbrew/reorder/reorder_mynew.h
#pragma once
#include "../../external/gapbs/builder.h"
#include "../../external/gapbs/pvector.h"

template <typename NodeID_, typename DestID_, bool invert>
void GenerateMyNewOrderMapping(
    const CSRGraph<NodeID_, DestID_, invert>& g,
    pvector<NodeID_>& new_ids) {

    NodeID_ N = g.num_nodes();

    // Example: sort by descending degree.
    std::vector<std::pair<int64_t, NodeID_>> deg(N);
    #pragma omp parallel for
    for (NodeID_ v = 0; v < N; ++v) {
        deg[v] = {g.out_degree(v), v};
    }
    std::sort(deg.begin(), deg.end(),
              std::greater<std::pair<int64_t, NodeID_>>());

    #pragma omp parallel for
    for (NodeID_ i = 0; i < N; ++i) {
        new_ids[deg[i].second] = i;
    }
}
```

Conventions:

- Return type is `void`; the result is written into the supplied
  `new_ids` pvector that the caller already sized to `g.num_nodes()`.
- `new_ids[v]` holds the new label for original vertex `v`. The caller
  uses the inverse mapping to relabel the CSR.
- Parallelise with OpenMP where the work is embarrassingly parallel.
- Reuse existing helpers in `reorder_graphbrew.h` (BFS, RCM, scanners)
  rather than re-implementing.

### 3. Wire it into the dispatcher

Edit `bench/include/external/gapbs/builder.h` and add a `case` to
`GenerateMapping`:

```cpp
case MyNewOrder:
    GenerateMyNewOrderMapping(g, new_ids);
    break;
```

If your algorithm produces a permutation that needs MAP support
(i.e. you want it loadable from a precomputed file), also add the
case to the MAP-aware path elsewhere in `builder.h` — search for an
existing entry like `case GOrder:` for the pattern.

### 4. Register the human-readable name

Edit `bench/include/graphbrew/reorder/reorder_types.h`:

- Add a case to `ReorderingAlgoStr()` so logs print
  `"MyNewOrder"` instead of `"17"`.
- Add a case to `intToReorderingAlgo()` so `-o 17` resolves.
- Add an entry to the algorithm-name map at the bottom of the file
  (used by the Python analysis tools).

### 5. (Optional) Add variants

If your algorithm has runtime knobs (resolution, threshold, etc.),
extend `parseGraphBrewConfig()` in `reorder_graphbrew.h` to recognise
the tokens, e.g. `-o 17:fast` or `-o 17:t0.5`. Keep the parser logic
local to your algorithm.

### 6. Rebuild and smoke-test

```bash
make -j8 all all-sim
./bench/bin/pr -f scripts/test/graphs/tiny/tiny.el -s -o 17 -n 3
```

Compare against `-o 0` (ORIGINAL) to confirm your reordering changes
the runtime in a sensible direction.

### Worked example

See `bench/include/graphbrew/reorder/reorder_corder.h` (algorithm 10)
for a compact reference implementation. For a community-aware
algorithm, see `reorder_graphbrew.h` — `orderHybridLeidenRabbit` is
the most heavily commented end-to-end example.

---

## Adding a new benchmark

A benchmark is a complete graph algorithm (PageRank, BFS, etc.)
with its own `main()` under `bench/src/`. The build system already
knows how to compile any new `.cc` file in `bench/src/` against the
GraphBrew headers.

Keep `bench/src/` canonical and stable. Add edge-centric variants under
`bench/src_edge/`, natural GAS variants under `bench/src_gas/`, and cache
simulation drivers under `bench/src_sim/`; register variant binaries in the
corresponding Makefile list.

### 1. Create the source file

```cpp
// bench/src/mynew.cc
#include <iostream>
#include "../include/external/gapbs/benchmark.h"
#include "../include/external/gapbs/builder.h"
#include "../include/external/gapbs/command_line.h"
#include "../include/external/gapbs/graph.h"

template <typename NodeID_, typename DestID_>
pvector<float> MyNewAlgorithm(const CSRGraph<NodeID_, DestID_>& g,
                              int max_iters) {
    NodeID_ N = g.num_nodes();
    pvector<float> result(N, 0.0f);

    for (int iter = 0; iter < max_iters; ++iter) {
        #pragma omp parallel for
        for (NodeID_ v = 0; v < N; ++v) {
            float sum = 0.0f;
            for (NodeID_ u : g.in_neigh(v)) sum += result[u];
            result[v] = sum / std::max<int64_t>(1, g.in_degree(v));
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    CLApp cli(argc, argv, "MyNew algorithm");
    if (!cli.ParseArgs()) return -1;

    Builder<NodeID, NodeID> b(cli);
    CSRGraph<NodeID, NodeID> g = b.MakeGraph();
    auto BenchKernel = [&cli](const CSRGraph<NodeID, NodeID>& g) {
        return MyNewAlgorithm(g, cli.max_iters());
    };
    BenchmarkKernel(cli, g, BenchKernel);
    return 0;
}
```

The `BenchmarkKernel()` helper drives the trial loop, applies any
reordering selected via `-o`, prints timing, and writes JSON output
when `-q` is set.

### 2. (Optional) Add a verifier

If you want correctness checks (`-v` flag), implement a reference
comparison and pass it to `BenchmarkKernel`:

```cpp
auto VerifyKernel = [](const CSRGraph<NodeID,NodeID>& g, const pvector<float>& r) {
    // recompute slowly and compare; return true on match
};
BenchmarkKernel(cli, g, BenchKernel, VerifyKernel);
```

### 3. Register in the Makefile

`KERNELS` is the list of standard binaries built by `make all` and
`KERNELS_SIM` the list built by `make all-sim`. Add your binary name:

```make
KERNELS     = pr pr_spmv bfs bc cc cc_sv sssp tc tc_p mynew
KERNELS_SIM = pr pr_spmv bfs bc cc cc_sv sssp tc        mynew
```

`make all` and `make all-sim` will pick it up automatically from
`bench/src/mynew.cc` (and `bench/src_sim/mynew.cc` if you add the
sim variant; the sim variant typically wraps the standard binary
in cache-sim instrumentation — see `bench/src_sim/pr.cc` for the
pattern).

### 4. Register in the Python pipeline

Edit `scripts/lib/core/utils.py` and add your benchmark to the
`KERNELS` and (if applicable) `BENCHMARKS` lists so that
`graphbrew_experiment.py` knows to run it.

For the VLDB experiment script, also edit
`scripts/experiments/vldb/config.py` if you want it included in the
paper's benchmark list (it has a separate `BENCHMARKS` constant —
keep the paper's 7-benchmark default unless you're extending the
study).

### 5. Build and test

```bash
make -j8 all all-sim
./bench/bin/mynew -f scripts/test/graphs/tiny/tiny.el -s -n 3
```

Sanity-check that running with and without reordering produces
identical results:

```bash
./bench/bin/mynew -f scripts/test/graphs/tiny/tiny.el -s -o 0 -n 1 -v
./bench/bin/mynew -f scripts/test/graphs/tiny/tiny.el -s -o 8 -n 1 -v
```

### Worked example

See `bench/src/pr.cc` for the canonical iterative-algorithm pattern,
or `bench/src/bfs.cc` for traversal. Both are well under 200 lines
including verifier.

---

## Style notes

- Match existing OpenMP patterns: `#pragma omp parallel for schedule(dynamic, 1024)`
  for vertex-parallel loops, `schedule(dynamic, 64)` for community-level.
- Avoid adding new external dependencies. The framework keeps Boost
  optional (only used by `-o 8:boost`); please don't push it back to
  required.
- Avoid duplicating BFS / RCM / scanner code — reuse the helpers
  already in `reorder_graphbrew.h`.
- Run `make all all-sim` and ensure both build clean before
  submitting changes.

## Where to ask

Open an issue on https://github.com/UVA-LavaLab/GraphBrew/issues with
your proposed algorithm or benchmark and the use case. We're happy
to help wire it up.
