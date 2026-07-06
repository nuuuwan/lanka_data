# TODO — Feature Ideas

Ideas for features to build, grouped by priority. Each item is framed against
the current codebase (the `What / When / Where / How` command grammar, the
dataset registries, and the visual renderers).

## HI PRI

- **Tabular data output formats (`CSV`, `TSV`, `Table`).** `How` currently
  emits only `JSON` or an image. Add data-export renderers so the underlying
  region/value table can be downloaded directly (e.g. `Religion/2024/LK:district/CSV`),
  reusing the same `RegionValueDataset` the plots consume.
- **Vector (`SVG`) output for maps and charts.** Maps/charts render to
  `Image.png` only. Add an `SVG` output (or an `:svg` modifier) for
  print-quality, zoomable graphics from the existing matplotlib visuals.
- **`LineChart` / time-series `How`.** `BumpChart` handles interval ranking, but
  there is no plain multi-year trend line. Add a line/area chart over the census
  and election years for a single region or a small region set.
- **Percentage / per-capita measures.** Values are raw counts (see
  `GIG2Dataset.clean_data_row`). Add share-of-total and per-1,000 variants as
  first-class `What` (or `How` modifiers) so cross-region comparison is fair.

## MID PRI

- **Election analytics: turnout, margin, and swing.** Extend the election
  datasets and `How:Change` coupling to express turnout, winning margin, and
  swing between two elections (e.g. `Presidential/2019-2024/LK:district/Map:Swing`).
- **Cross-measure correlation / cross-tabs.** The `GIG2`/correlation dataset
  scaffolding hints at this. Support comparing two `What` measures across regions
  (scatter or bivariate choropleth), e.g. religion vs. housing quality.
- **"Did you mean" suggestions on errors.** The `command_errors` classes reject
  unknown `What`/`Where`/`How` values. Add fuzzy matching against the registries
  to suggest the nearest valid value in the error payload and console.
- **More chart types.** `StackedBarChart`, `TreeMap`, `Histogram`, and
  `ScatterPlot` renderers under `plot_visual`, wired through `VisualFactory`.
- **Batch / gallery runner.** A workflow that takes many command strings and
  produces a single contact-sheet image or an HTML gallery, extending
  `workflows/single.py` and `examples_build.py`.
- **Animated maps across time.** Render an interval `When` (e.g. `2001-2012-2024`)
  as an animated GIF/MP4 by sequencing the existing map frames.

## WISHLIST

- **Generalise beyond Sri Lanka.** Per `README.philosophy.md`, nothing in the
  grammar is LK-specific. Make the region provider pluggable so the same
  `What/When/Where/How` engine can host other jurisdictions/datasets.
- **Non-place `Where` (a "Who").** `Where` is structured to accept non-place
  identities (people, organisations) in the same position; add a first
  non-geographic entity type.
- **Interactive query-builder UI.** A web front-end over the Vercel app that
  guides users field-by-field with the same autocomplete already in the console
  (`ConsolePromptCompleter`) and live-previews results.
- **Natural-language to command.** Translate a plain-language question into a
  command string, leaning on the grammar's readability described in the
  philosophy doc.
- **Additional data domains.** Economic, health, education, and climate datasets
  registered through the existing `DatasetCommandRegistry` group-provider
  mechanism.
- **Richer cartography.** Dot-density maps, prism/3D extrusion maps, and
  inset/locator maps building on the current `Map`/`Cartogram`/`HexMap` stack.
- **Data provenance & versioning.** Track dataset release versions and let a
  command diff results between two data snapshots, complementing the existing
  `sources` provenance in every response.
