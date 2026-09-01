# Editable reordering figures

These files are editable diagrams for the public reordering figure catalog.
Lucidchart officially supports importing draw.io/diagrams.net `.drawio` files:

https://help.lucid.co/hc/en-us/articles/16389149809428-Import-files-into-Lucidchart

## Recommended Lucidchart workflow

1. Download `GraphBrew-reordering-figures.drawio` to import the shared input
   plus all 17 algorithm pages, or download one numbered `.drawio` file.
2. In Lucidchart, select **New > Import documents** and choose the file.
3. Use the imported shapes to prototype a visual improvement.
4. Port the accepted change to `scripts/generate_public_figures.py` or
   `graphbrew-running-example.json`.
5. Regenerate every SVG, draw.io page, bundle page, and manifest.

## Source policy

- `graphbrew-running-example.json` and `scripts/generate_public_figures.py`
  are the canonical sources.
- `.drawio` is the generated editable form for visual prototyping.
- `.svg` is the generated rendered artifact embedded by the wiki.
- Each algorithm ID has one same-stem pair, for example:
  `08-rabbitorder.drawio` and `08-rabbitorder.svg`.
- `example-input.drawio` is the shared input page.
- `GraphBrew-reordering-figures.drawio` is the generated 18-page bundle.
- Do not import the SVG when shape-level editing is required; Lucidchart does
  not document SVG as a native fully editable diagram import.

## Visual contract

- Show the shared graph once, then compare output-only algorithm strips.
- Use the shared light/dark palette, stage/algorithm badge, measured output
  array, mechanism card, and invariant footer.
- Highlight exactly one maximally displaced vertex in each output.
- Keep all 17 strips at 1200x360 so their arrays align visually.
- Regenerate the same-stem SVG/draw.io pair and bundle together.

The catalog manifest is
[`../reordering/manifest.json`](../reordering/manifest.json).
