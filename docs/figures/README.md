# Public figures

- `logo.png` — compact project mark used by the README
- `graphbrew-architecture.svg` — GraphBrew infrastructure, validated policy
  context, paper research direction, and the shared six-stage map
- `graphbrew-graph-to-csr.svg` — Stage 1 input graph, tracked CSR row, and profile
- `graphbrew-leiden-transform.svg` — Stage 2 community membership
- `graphbrew-sizedesc-transform.svg` — Stage 3 contiguous block placement
- `graphbrew-gordf5000.svg` — Stage 4 per-community Gorder/BFS dispatch
- `graphbrew-gorder-transform.svg` — Stage 4A relaxed local Gorder
- `graphbrew-bfs-transform.svg` — Stage 4B hub-rooted BFS
- `graphbrew-relabel-emit.svg` — Stage 5 mapping and relabeled CSR
- `graphbrew-locality-outcome.svg` — Stage 6 tracked property-line locality
- `graphbrew-lowreuse-policy.svg` — validated selector with two real holdouts
- `graphbrew-cd-parallel.svg` — serial and parallel community detection
- `graphbrew-sgmb4096.svg` — parallel proposals and ordered commits
- `graphbrew-norefine.svg` — refinement bypass
- `reordering/example-input.svg` — shared measured catalog input
- `reordering/` — one measured output strip per algorithm ID
- `editable/` — generated `.drawio` pages and multi-page bundle

The canonical source is
[`graphbrew-running-example.json`](graphbrew-running-example.json) plus
[`scripts/generate_public_figures.py`](../../scripts/generate_public_figures.py).
[`catalog-capture.json`](catalog-capture.json) binds the measured catalog
orders to the converter hash, tracked edge list, MAP input, commands,
fingerprints, and single-thread environment.
The generator emits every explanatory SVG, every catalog draw.io page, the
multi-page bundle, the catalog wiki page, and
[`public-manifest.json`](public-manifest.json).

Run:

```bash
python3 scripts/generate_public_figures.py --capture-catalog
python3 scripts/generate_public_figures.py
python3 scripts/generate_public_figures.py --check
```

Experimental plots remain in content-bound artifact directories rather than
being committed as duplicate PNG/SVG pairs. Explanatory figures use one stage
map, one tracked object, pastel role cards, charcoal typography, short blue
connectors, invariant footers, and the shared dark-mode remap.

The full algorithm catalog is indexed by
[`reordering/manifest.json`](reordering/manifest.json).
