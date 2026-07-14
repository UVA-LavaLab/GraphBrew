# FAQ

Short answers to common questions. For full guides see
[Getting-Started](Getting-Started), [Reordering-Algorithms](Reordering-Algorithms),
and [Troubleshooting](Troubleshooting).

## Which reordering should I try first?

| Situation | Try |
|---|---|
| Don't know, want a single safe default | `-o 12:hrab` (best cache, reasonable reorder cost) |
| Iterative algorithm with many trials (PR, SpMV) | `-o 12:hrab` or `-o 12:leiden` |
| Single traversal from one source (BFS, SSSP) | `-o 7` (HUBCLUSTERDBG; cheap reorder) or `-o 12:rabbit` |
| Road / mesh graph | `-o 12:rcm` or `-o 11:bnf` |
| Reorder cost matters most | `-o 8` (RABBITORDER) or `-o 7` (HUBCLUSTERDBG) |
| Reproducing the paper | the experiments in [VLDB-Experiments](VLDB-Experiments) cover the full algo grid |

If you don't care, `-o 12:hrab` is the right default for most graphs as
of May 2026 (see commit `0b9c90c` for the adaptive intra-community fix).

## How much speedup should I expect?

It depends on (a) how cache-unfriendly the original ordering is and
(b) how many iterations your benchmark runs.

Rule of thumb on the paper's evaluation machine (Intel Xeon Silver
4216, 22 MB L3):

| Graph type | Realistic speedup vs ORIGINAL | Reorder cost amortizes after |
|---|---|---|
| social / collaboration / web (community-strong) | 1.3–2× on PR | 2–5 trials |
| road / mesh | 1.5–3× on BFS | 3–5 trials |
| citation (already well-ordered) | 1.0–1.1× on PR | many trials |
| random / synthetic | 1.0× (no community structure to exploit) | never |

The break-even-trials column (`N*`) in the auto-generated paper table
quantifies this per graph.

## Why does reordering sometimes hurt?

Three common reasons:

1. **The original ordering is already good.** Many graphs ship from
   their source in a near-optimal layout (e.g. citation networks
   crawled chronologically). Reordering only adds overhead.
2. **You ran too few iterations to amortize reorder cost.** Reorder
   time is paid once; kernel speedup is paid back per trial. With
   `-n 1` you're seeing reorder + 1 kernel, which is often slower
   than ORIGINAL × 1.
3. **The reordering doesn't match the access pattern.** Degree-only
   methods (DBG, HUBSORT) help iterative power-law workloads but
   crash on community-strong dense graphs (e.g. on hollywood at
   L3=1MB, DBG produced 0.5× speedup — slower than baseline).

## Where do my benchmark results land?

| Output | Location |
|---|---|
| Standard runs | stdout (`Read Time`, `Build Time`, `Average Time`) |
| With `-q out.json` | `out.json` (machine-readable) |
| Pipeline (`graphbrew_experiment.py`) | `results/data/benchmark.json` |
| VLDB script (`vldb_paper_experiments.py`) | `results/vldb_paper/exp{1..8}_*/` |
| Trained adaptive models | `results/data/adaptive_models.json` |

## Where do trained AdaptiveOrder models live?

`results/data/adaptive_models.json`. Models train at runtime inside
the C++ binary on first invocation of `-o 14` with enough data in
`benchmark.json`. See [AdaptiveOrder-ML](AdaptiveOrder-ML).

## How do I add a new algorithm or benchmark?

See [Contributing](Contributing).

## What graph formats are supported?

`.sg` (GAPBS binary, fastest), `.el` (edge list, text), `.wel`
(weighted edge list), `.mtx` (Matrix Market). The first run on a
new graph creates a `.sg` cache alongside the input.
See [Supported-Graph-Formats](Supported-Graph-Formats).

## What's the difference between LeidenOrder (15) and GraphBrew-Leiden (12:leiden)?

- `-o 15` runs the GVE-Leiden reference and uses its community
  membership directly as the vertex order. No post-processing layer.
- `-o 12:leiden` is the full GraphBrew pipeline with Leiden as the
  community detector, BFS within each community for cache locality,
  and hierarchical sort across communities. Strictly more layers,
  usually better cache quality, slightly more reorder cost.

## When should I use DBG vs HUBCLUSTER?

DBG buckets vertices by `log2(degree)`. HUBCLUSTER does a binary
split (hubs vs non-hubs) and only sorts the hub partition.

- **HUBCLUSTER** preserves non-hub spatial structure, so it composes
  well as a layer on top of a community-aware ordering. This is why
  `12:hubcluster` works.
- **DBG** redistributes everything by degree bucket. Use it
  standalone (`-o 5`) for fast experiments, or chained as a refinement
  step (`-o 12:leiden -o 5`) when both community structure and hub
  temporal locality matter.

## How do I cite GraphBrew?

Until the VLDB 2026 paper is published, cite the repository:

```bibtex
@misc{graphbrew,
  title  = {GraphBrew: Multilayered Graph Reordering for Accelerated Graph Processing},
  author = {Mughrabi, Abdullah T. and Baradaran, Morteza and Ibrahim,
            Mohannad M. and Byrd, Gregory T. and Skadron, Kevin},
  year   = 2026,
  howpublished = {\url{https://github.com/UVA-LavaLab/GraphBrew}},
}
```

The full paper draft lives in `research/.../main.tex` (gitignored
working directory).

## Where is the AdaptiveOrder / ML documentation?

[AdaptiveOrder-ML](AdaptiveOrder-ML). Note that AdaptiveOrder is
research-only and **not part of the VLDB 2026 submission** — it's
kept in-tree for future work.

## Common errors

| Error | Fix |
|---|---|
| `fatal error: boost/range/algorithm.hpp` | `sudo apt-get install libboost-all-dev` |
| `-fopenmp not supported` | `sudo apt-get install libomp-dev` |
| `g++ unrecognized command line option '-std=c++17'` | install GCC 7+ |
| `Cannot allocate memory` while building | `make -j2` instead of `-j` |
| `*.sg file not found` after rebuild | re-run with `-f graph.el`; the binary will regenerate `.sg` |
| AdaptiveOrder picks the same algorithm every time | `benchmark.json` is empty or untrained; run the pipeline once: `python3 scripts/graphbrew_experiment.py --train --size small` |

More in [Troubleshooting](Troubleshooting).
