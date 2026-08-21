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

The catalog manifest is
[`../reordering/manifest.json`](../reordering/manifest.json).
