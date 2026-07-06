# TODO — Next Features

Feature ideas for growing Lanka Data without breaking the
`What / When / Where / How` command grammar.

## HI PRI

- **Derived measures as first-class `What`s.** Add built-in computed indicators
  such as turnout rate, winning margin, vote swing, density, growth, and
  per-capita versions of existing measures. The goal is to let users ask for
  derived facts directly instead of inferring them outside the library.

- **Full interval semantics beyond endpoint diffs.** Today most interval queries
  collapse to a start/end diff, while `LineChart` and animations keep the full
  sequence. Extend interval-aware datasets so `JSON`, `CSV`, `Table`, and more
  visuals can return the whole time series for commands like
  `Religion/2001-2012-2024/LK:district/Table`.

- **Bivariate geography for combined measures.** Combined `What`s already open
  the door to cross-measure analysis. Add visuals that make this native:
  bivariate choropleths, quadrant maps, and linked scatter/map outputs for
  commands like `Religion+Ethnicity/2024/LK:district/BivariateMap`.

- **Historical boundary reconciliation.** Comparing data across changing
  geographies is one of the hardest parts of Sri Lankan public data. Add
  explicit crosswalk/apportioning support so interval and multi-year queries can
  compare like with like even when district, DSD, or LG boundaries changed.

## MID PRI

- **Richer export formats.** Extend data-first outputs beyond `JSON`, `CSV`,
  `TSV`, and markdown `Table` to include `GeoJSON`, `Parquet`, and maybe a
  light-weight chart-spec export for downstream apps.

- **Ranking and threshold operators in `Where`.** Add selectors such as “top N
  regions”, “bottom N”, and value-based filters so users can ask for
  `.../LK:district@top10/BarChart`-style queries without manual region lists.

- **Natural-language aliases and discovery.** Keep the strict command grammar,
  but accept friendlier aliases and improve suggestions/autocomplete so users can
  discover valid `What`, `Where`, and `How` values faster.

- **Narrative-ready annotations.** Add optional callouts for maxima, minima,
  biggest changes, and outliers so exported visuals can answer the “what should
  I notice?” question automatically.

## WISHLIST

- **Batch commands and dashboards.** Support a compact way to render a curated
  set of related commands as one shareable bundle for reports or newsroom use.

- **Uncertainty and data-quality overlays.** Surface missingness, provisional
  data, suppressed values, and low-confidence joins directly in the output.

- **Localized labels.** Add Sinhala and Tamil display labels for `What`, `Where`,
  legends, and chart furniture while keeping the command grammar itself stable.
