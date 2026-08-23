# Editable reordering figures

These files are editable diagrams for the public reordering figure catalog.
Lucidchart officially supports importing draw.io/diagrams.net `.drawio` files:

https://help.lucid.co/hc/en-us/articles/16389149809428-Import-files-into-Lucidchart

## Recommended Lucidchart workflow

1. Download `GraphBrew-reordering-figures.drawio` to import all 17 algorithm
   pages at once, or download one numbered `.drawio` file.
2. In Lucidchart, select **New > Import documents** and choose the file.
3. Edit the individual nodes, edges, labels, memory cells, colors, and
   connectors as Lucidchart shapes.
4. Export the finished page as SVG.
5. Replace the paired file under `docs/figures/reordering/` and keep the
   `.drawio` source synchronized.

## Source policy

- `.drawio` is the editable source for manual polishing.
- `.svg` is the publication artifact embedded by the wiki.
- Each algorithm ID has one same-stem pair, for example:
  `08-rabbitorder.drawio` and `08-rabbitorder.svg`.
- `GraphBrew-reordering-figures.drawio` is a convenience multi-page bundle.
- Do not import the SVG when shape-level editing is required; Lucidchart does
  not document SVG as a native fully editable diagram import.

## Visual contract

- Use a white canvas with the shared light/dark GraphBrew palette.
- Keep the input and output graphs inside separate rounded cards.
- Use circular graph nodes, at least 21 px exported SVG text, 54 px regular
  node diameters, and 60 px emphasized-hub diameters.
- Use dark node outlines and 2.5 px node strokes so IDs remain legible.
- Keep titles, cards, graph geometry, and memory-order rows aligned across all
  17 pages.
- Update the same-stem SVG and draw.io file together, then refresh the
  multi-page bundle.

The catalog manifest is
[`../reordering/manifest.json`](../reordering/manifest.json).
