# Public figures

All explanatory figures are generated from one checked running example and
one evidence manifest.

## Paper story

- `graphbrew-architecture.svg` — explicit six-stage composition pipeline,
  confirmed quality arm, and Compact-and-Emit
- `graphbrew-evidence-boundary.svg` — confirmed claims versus rejected
  selector and balanced-arm extensions
- `graphbrew-compact-emit.svg` — active-community compaction and direct BFS
  emission

## Running example

- `graphbrew-graph-to-csr.svg`
- `graphbrew-leiden-transform.svg`
- `graphbrew-sizedesc-transform.svg`
- `graphbrew-gorder-transform.svg`
- `graphbrew-bfs-transform.svg`
- `graphbrew-relabel-emit.svg`
- `graphbrew-locality-outcome.svg`

The remaining top-level control figures document callable composition knobs;
`reordering/` contains one measured output strip per algorithm ID, and
`editable/` contains the generated draw.io sources.

Canonical inputs:

- `graphbrew-running-example.json`
- `catalog-capture.json`
- `../recommendation-evidence.json`
- `../../scripts/generate_public_figures.py`

Regenerate and verify:

```bash
python3 scripts/generate_public_figures.py
python3 scripts/generate_public_figures.py --check
```

Experimental plots remain in content-addressed artifact directories rather
than being committed as duplicate image sets.
