# Public figures

- `logo.svg` — project mark used by the README
- `graphbrew-architecture.svg` — system architecture and public interfaces
- `graphbrew-lowreuse-policy.svg` — new-graph decision path and fallback claim
- `graphbrew-leiden-transform.svg` — community-membership transformation
- `graphbrew-sizedesc-transform.svg` — SizeDesc block transformation
- `graphbrew-gorder-transform.svg` — Gorder8 local transformation
- `graphbrew-bfs-transform.svg` — BFS local transformation
- `graphbrew-cd-parallel.svg` — serial and parallel community detection
- `graphbrew-sgmb4096.svg` — parallel proposals and ordered commits
- `graphbrew-gordf5000.svg` — community-size Gorder/BFS selection
- `graphbrew-norefine.svg` — refinement bypass
- `reordering/` — one standardized SVG transformation per algorithm ID
- `editable/` — paired `.drawio` sources for Lucidchart/manual editing

The core diagrams are hand-authored SVGs with one source per concept. The
algorithm catalog uses paired publication SVG and editable draw.io sources.
Experimental plots remain in content-bound artifact directories rather than
being committed as duplicate PNG/SVG pairs. The GraphBrewOrder transformation
and control figures reuse the architecture's dashed domains, pastel cards,
charcoal typography, thin blue arrows, and dark-mode behavior.

The full algorithm catalog is indexed by
[`reordering/manifest.json`](reordering/manifest.json).
