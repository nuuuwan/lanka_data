# Code Layout (by class)

This document explains the code by walking through its **classes**, in the order
they participate in a query. It complements `README.md` (the API grammar and
examples) and `README.philosophy.md` (the design rationale).

Every query is a single string of the form `<what>/<when>/<where>/<how>`
(for example `Religion/2012-2024/LK:district/Map:Change`). The classes below
turn that string into a JSON payload or a rendered image.

Read top to bottom:

- ommand **fields** compose into a **command**
- a command produces **datasets**,
- datasets plus the command produce a **visual**,
- the supporting, utility, and tooling classes sit around that spine.

The package follows one-class-per-file, with oversized classes split into named
`Mixin` classes that live beside (or in a folder with) the class they extend.

---

## 1. Command field classes

The four fields are the atoms of the API. Each is a frozen dataclass that wraps a
single `value: str`, validates it on construction, and exposes typed accessors so
the rest of the pipeline never has to re-parse the raw string. They live in
`lanka_data.api.fields`.

### `What`

`What` names the indicator being queried (a census table label, an election
label, an election `…Summary` label, or the special value `Empty`). It is a
thin value object: it stores the raw string and exposes `whats` / `is_combined`
for `+`-combined indicators (e.g. `Religion+Ethnicity`). It performs no
enumeration or validation of legal values.

### `When`

`When` is either a single four-digit year (`2024`) or an interval of two
ascending years joined by a dash (`2012-2024`); an empty string is also allowed.
Anything else raises `InvalidWhenError`. Its accessors (`is_interval`, `years`,
`start`, `end`) let downstream code branch on point-in-time vs. interval queries
without string surgery.

### `Where`

`Where` holds the region selector. Construction only enforces the character set
and rejects malformed range syntax; the real interpretation of the selector
happens later in the region layer. It exposes `parent_part`, `child_region_type`
(the part after `:`), `zoom` (the number after `@`), and `top` (the number after
`#`, e.g. `LK:rivers#10` keeps only the top 10 regions by value, surfaced as a
`Top<N>` `region_filter`).

### `How`

`How` describes the output. Its value splits into a `base` (before `:`, e.g.
`Map`, `BarChart`, `JSON`) and an optional `modifier` (after `:`, e.g. `Change`,
`1st`, `Diversity`). Construction validates the base against `BASE_LABELS` and
the modifier against `MODIFIERS`, raising `UnknownHowError` otherwise. It derives
display-facing properties (`base_label`, `modifier_label`, `format()`) and
semantic ones (`rank`, `pct_rank`, `needs_interval`) that later steer dataset and
color selection.

### Support classes for the fields

- **`HowRegistryMixin`** — holds the static tables `BASE_LABELS`, `MODIFIERS`,
  and `INTERVAL_BASES` that define every legal `How`. `How` inherits it, so
  extending the output vocabulary means editing this one class.
- **`HowCategoryMixin`** — adds the `category` / `categories` accessors that
  `How` inherits, used by the color and pair-category visual layers.

---

## 2. The `Command` class

A `Command` is the four validated fields bound together with cross-field rules.
It is assembled from three classes.

### `CommandBase`

`CommandBase` is the dataclass core. Its `__init__` accepts either field objects
or raw strings (via the `*_cmd` keyword aliases), coerces each into its `What` /
`When` / `Where` / `How` object with `_build_field`, and then runs
`_validate_parts` (type checks) and `_validate_coupling`. The coupling rule
rejects a command whose `how` needs an interval (e.g. `BumpChart`, or a `Change`
modifier) when `when` is not an interval — raising `InvalidCommandError`. It
exposes `cmd_id` (the canonical `what/when/where/how` string) plus the
`what_cmd` … `how_cmd` compatibility properties.

### `CommandLoaderMixin`

`CommandLoaderMixin.from_str` is the entry point that parses a raw command
string. It special-cases `Help`, otherwise splits on `/`, requires exactly four
tokens, and constructs the command (raising `InvalidCommandError` on a bad
shape).

### `Command`

`Command` combines `CommandBase` and `CommandLoaderMixin` and adds `copy(...)`,
which returns a new command with selected fields overridden. `copy` is what lets
interval queries be split into their `start` and `end` point-in-time commands.

### Running a command

- **`CommandRunner`** is the orchestrator. `CommandRunner.run(command_str)`
  checks its cache, then either returns `CommandHelp`, or parses the string into
  a `Command`, asks `DatasetFactory` for the datasets, asks `VisualFactory` for
  the visual, and calls `visual.build()`. It returns a dict of the result, the
  merged `DataSource` records, and the query time, and stores it in the cache.
- **`CommandCache`** is an LRU (`OrderedDict`, default 128 entries). Its
  `is_valid` check evicts any cached image result whose file has disappeared.
- **`CommandHelp`** returns the payload served for the `Help` command. The
  registry-driven self-describing index has been deprecated, so it now returns a
  blank payload.
- **Error classes** — `CommandError` (base), `InvalidCommandError`,
  `InvalidWhenError`, `InvalidWhereError`, `UnknownWhatError`, `UnknownHowError`
  — are raised by the fields and command for precise, typed failures.

---

## 3. Dataset classes — constructed from a command

`DatasetFactory` turns a `Command` into one or more `Dataset` objects. A dataset
knows how to fetch its source table, filter it to the requested regions, and
expose a normalized value table.

### The hierarchy

- **`Dataset`** (`ABC`) is the minimal contract: `get_sources()`,
  `get_data_table()`, `has_values()`, and `is_diff()`.
- **`RegionValueDataset`** is the base for every dataset that attaches a map of
  category → value to each region. It stores the resolved region list, indexes
  regions, and — via **`RegionValueDatasetTableMixin`** — collects source rows
  per region (following historical `current_ids`), aggregates and sorts values,
  and `expand_and_clean`s each row into `values`, `total_value`, and
  `pct_values`. Subclasses only implement `get_source_data_table()`,
  `clean_data_row()`, and `get_year()`.
- **`EmptyDataset`** is the geometry-only dataset used by `what = Empty`; it has
  no values and just carries the region list (for blank base maps).
- **`DiffDataset`** wraps two `RegionValueDataset`s (a start and an end) and
  computes per-region differences in values and percentage points. It reports
  `is_diff()` and merges both sources; it powers every interval/`Change` query.

### The concrete source adapters (`dataset/custom/`)

- **`GIG2Dataset`** — abstract base for datasets stored in the external
  `gig-data` repository as TSV. It maps public labels to table IDs from a bundled
  metadata JSON, fetches and cleans the TSV, and applies a local-government ID
  correction. Subclasses supply `metadata_file_path()`, `get_sources()`,
  `get_region_group()`, and `get_year()`.
- **`Census2001Dataset`** (extends `RegionValueDataset`) — 2001 census tables
  from the `lk_census_2001` repository.
- **`Census2012Dataset`** (extends `GIG2Dataset`) — 2012 census via GIG2.
- **`Census2024Dataset`** (extends `RegionValueDataset`) — 2024 census tables
  from the `lk_census_2024` repository.
- **`ElectionDataset`** (extends `GIG2Dataset`) — election results by party from
  the Election Commission region group; adds a `year` and a `supports(label,
  year)` check.
- **`ElectionSummaryDataset`** (extends `ElectionDataset`) — turns a `…Summary`
  label into aggregate turnout metrics (Valid, Rejected, DidNotVote).

### `DatasetFactory`

`DatasetFactory.list_from_command(command)` is the dispatcher:

1. For an **interval** `when`, it `copy`s the command to each of its years,
   builds a dataset per year, and returns a single-element list containing a
   `DiffDataset` of the first and last years (`[diff]`). Both a two-year
   interval such as `2012-2024` and a multi-year interval such as
   `2001-2012-2024` yield exactly one dataset — the diff of the endpoints.
2. For a **single** year it returns one dataset, chosen by `from_command`:
   `Empty` → `EmptyDataset`; otherwise the first census class whose supported
   years and labels match, else the election / election-summary datasets, else
   `UnknownWhatError`.

Region resolution happens here too: `get_region_data_list` calls
`Regions.from_command(command)` (see §5) to expand the `where` selector before
the values are attached.

---

## 4. Visual classes — constructed from a command and its datasets

`VisualFactory` turns `(command, datasets)` into a `Visual` that renders the
final output.

### The hierarchy

- **`Visual`** (`ABC`, dataclass) stores the `command`, `datasets`, and
  `how_cmd`, provides `from_command_and_datasets(...)`, defines the abstract
  `build()`, and merges the datasets' `get_sources()`.
- **`JSONVisual`** returns the last dataset's data table verbatim — the `JSON`
  output type.
- **`PlotVisual`** is the base for all matplotlib charts. It normalizes each
  dataset into subregions, aggregates category labels, and maps categories to
  colors (via `ColorSpecFactory`). `build()` delegates figure composition to the
  `Plot` primitive and each subclass implements `draw(dataset, fig)`.
  - **`MapVisual`** — choropleths, including population-weighted cartograms
    (`Map`, `Cartogram`, `None`); loads geometry through `GeoData`.
  - **`BarChartVisual`** (+ `BarChartDrawMixin`, `BarChartLabelMixin`) — stacked
    bars, with adaptive in-bar label fitting and change-chart handling.
  - **`StackedBarChartVisual`** — 100%-normalized stacked bars, showing each
    region's categorical composition as shares of a full-height bar.
  - **`PieChartVisual`** (+ `PieChartGridMixin`) — a grid of per-region pies;
    falls back to bars for change charts.
  - **`BumpChartVisual`** (+ `BumpChartDataMixin`, `BumpChartDrawMixin`) — a
    rank-change slopegraph between the two years of an interval.
  - **`TreeMapVisual`** (+ `TreeMapData`, `TreeMapDrawMixin`) — a squarified
    treemap of the overall categorical composition across regions.
  - **`HistogramVisual`** (+ `HistogramData`) — the distribution of region
    totals binned into a histogram.
  - **`ScatterPlotVisual`** (+ `ScatterPlotData`, `ScatterPlotPairMixin`,
    `ScatterPlotStats`) — region population against dominant-category share,
    coloured by dominant category; with a combined `What` and a category pair
    in the `How` it plots one category's share against the other with a fitted
    regression line and correlation stats.

### `VisualFactory`

`VisualFactory.from_command_and_datasets` reads `command.how.base` and
instantiates the matching class:

| `how.base`                  | Visual class      |
| --------------------------- | ----------------- |
| `JSON`                      | `JSONVisual`      |
| `Map`, `Cartogram`, `None`  | `MapVisual`       |
| `BarChart`                  | `BarChartVisual`  |
| `StackedBarChart`           | `StackedBarChartVisual` |
| `PieChart`                  | `PieChartVisual`  |
| `BumpChart`                 | `BumpChartVisual` |
| `TreeMap`                   | `TreeMapVisual`   |
| `Histogram`                 | `HistogramVisual` |
| `ScatterPlot`               | `ScatterPlotVisual` |

Keeping this mapping in one class isolates output-type selection from the drawing
code.

---

## 5. Other supporting classes

### Region resolution (`datasets/region/`)

The region classes expand a `Where` selector into concrete region records with
boundaries and history.

- **`Regions`** (extends `Where`, `RegionLoadersMixin`) is the object returned to
  the pipeline: it cleans raw region dicts, builds a human-readable title, and
  caches `region_ids`, `region_type`, and `region_to_current_ids` (the mapping
  that follows historical boundary changes).
- **`Where`** (the region-layer one, distinct from the `Where` field) is a small
  base holding a title and region year.
- **`RegionLoadersMixin`** ties parsing to fetching: `from_command` parses the
  selector, fetches raw data, and returns a `Regions`.
- **`RegionParserMixin`** (+ **`RegionParserRadiusMixin`**) interprets the
  selector grammar: comma lists (`LK-1,LK-2`), ranges (`LK-1...LK-2`), radius
  queries (`id@km`, via Haversine distance), `:child_type` expansion, and the
  `#N` top primitive (`LK:rivers#10`).
- **`RegionRawDataMixin`** (+ **`RegionFetchMixin`**, **`RegionParentMixin`**)
  fetches region definitions from the external `lk_admin_regions` dataset,
  normalizes field names, resolves current vs. historical (`-pre{year}`)
  boundaries, and computes parent/child relationships and full names.
- **`RegionTypeUtils`** classifies a region ID into its type (country, province,
  district, dsd, gnd, ed, pd, lg) and supplies readable names and the prefix maps
  used by the `Where` field.

### Data computations (`api/data/`, `datasets/data/`)

- **`DataSource`** is the provenance record (name + url) attached to every
  result; `merge_datasource_list_of_lists` de-duplicates and sorts them.
- **`Diversity`** computes a normalized religious-diversity index (with an
  optional Pew grouping) and maps it to labelled bands and colors.
- **`FieldNameUtils`** normalizes ethnic/category labels to canonical
  PascalCase names.

### Plot primitives (`visual/plot/`)

These are the low-level drawing helpers used by `PlotVisual` subclasses:

- **`Plot`** composes the whole figure (subfigure per dataset, plus header and
  footer) and saves the PNG.
- **`Header`** / **`Footer`** draw the title bar (built from the `*Formatter`
  classes) and the sources bar.
- **`Font`** installs the bundled TTF into matplotlib.
- **`Text`** is a thin `fig.text` wrapper; **`Label`** and **`LabelFit`** place
  and size region labels inside map polygons; **`Legend`** renders the
  value→color key.
- **Color:** **`ColorSpec`** (+ `ColorSpecCategoryMixin`, `ColorSpecCustomMixin`)
  holds region→color and value→color maps and knows the fixed palettes;
  **`ColorSpecFactory`** (+ `ColorSpecHelpers` / `ColorSpecHelpersMixin` /
  `ClusterColorSpecMixin`, `ColorSpecConstants`) picks the coloring strategy from
  the dataset and `how` modifier (rank, percentage, Change, Diversity, Cluster).
  The `Cluster` / `Cluster-N` modifier (e.g. `Map:Cluster-3`,
  `HexMap:Cluster-3`) groups regions by their total value into `N` clusters using
  in-house 1D k-means (`ClusterData`, default 5) and colors each region by its
  cluster's mean composition: the hue of the most common field in the cluster
  mean, shaded by an alpha equal to that field's share. The legend labels each
  cluster with its two most common fields and their percentages plus `Other`
  (e.g. `Sinhalese (50%), SLMoor (30%), Other (20%)`).
- **Geometry:** **`GeoData`** (+ `GeoDataLoaderMixin`) loads TopoJSON boundaries,
  dissolves and caches them, applies cartogram distortion, and enriches the
  `GeoDataFrame` with the dataset's values.

### Display formatters (`visual/formatters/`)

- **`WhatFormatter`**, **`WhereFormatter`**, **`HowFormatter`** turn each command
  field into the English text shown in chart headers (e.g. the region selector
  into "the Colombo & Gampaha districts").

### Startup wiring

- **`CompatibilityAliases`** registers legacy import paths (e.g.
  `lanka_data.command.*`) as aliases of the current module locations, keeping
  older imports working.

---

## 6. Utility classes (`src/utils_future/`)

`utils_future` is a standalone package of general helpers used throughout
`lanka_data`.

- **`File`** — UTF-8 text file I/O plus metadata (`exists`, `size`,
  `size_human_readable`).
- **`BinaryFile`** (extends `File`) — binary read/write.
- **`JSONFile`** (extends `File`) — read/write JSON with indentation.
- **`WWW`** — HTTP client with on-disk caching; `read_json`, `read_tsv`,
  `download`. This is how the dataset and region classes fetch external data.
- **`Log`** (extends `logging.Logger`) — colored, non-propagating console logs.
- **`Parse`** — `Parse.float` cleans strings (commas, `%`, `pp`, parentheses)
  into floats.
- **`timer`** — decorator that logs slow function calls.
- **`GeoUtils`** — Haversine great-circle distance (used by radius queries).
- **`PolygonUtils`** — Shapely helpers for label placement (largest polygon,
  pole of inaccessibility, interior candidates).
- **`ColorUtils`** — hex/RGB conversion and light/dark detection for contrast.
- **`dcn/`** — the Dougenik–Chrisman–Niemeyer cartogram algorithm:
  **`DCNUtils`** (public façade over **`DCNUtilsRunner`**), with
  **`DCNUtilsAlgorithm`** (force computation and displacement) and
  **`DCNUtilsCompute`** (area/centroid geometry). Used by `GeoData` for
  cartograms.

---

## 7. Tools classes

These classes drive the repository's generated artifacts and interactive use.
They consume the library through `CommandRunner`; they are not part of the query
pipeline itself.

### Examples (`examples/Example/`)

- **`Example`** (+ **`ExampleOutputMixin`**) loads named commands from
  `examples/examples.json`, runs each through `CommandRunner`, and materializes
  the results into `_output/<cmd>/Output.json` (and images). The
  generated outputs are reused by the README.

### README generation (`readme/`)

- **`ReadMe`** assembles `README.md` from its section mixins:
  **`ReadMeSourcesMixin`** (data-source list), **`ReadMeUsageMixin`** (install /
  code / HTTP / CLI usage), **`ReadMeExamplesMixin`** (+
  **`ReadMeExamplesItemMixin`**, each rendered example), and
  **`ReadMeFooterMixin`** (badges). `README.md` is a generated file.

### Console (`console/`)

An interactive REPL for querying the API locally:

- **`ConsoleApp`** — the entry point and read/eval loop.
- **`ConsoleCompleter`** — matches suggestions by prefix (also drives the
  readline fallback).
- **`ConsolePromptCompleter`** — adapts the matches into `prompt_toolkit`
  completions for the drop-down menu.
- **`ConsolePrompt`** — reads input via a `prompt_toolkit` session that shows
  auto-complete suggestions in a drop-down as you type, falling back to
  readline-backed `input` when no interactive terminal is available.
- **`ConsoleLocalCommands`** — built-ins (`help`, `fields`, `examples`,
  `commands`, `clear`, `exit`).
- **`ConsoleRenderer`** — Rich-formatted tables and banners.
- **`ConsoleImageOpener`** — opens rendered images with the platform viewer.

### Workflow entrypoints (`workflows/`)

Thin scripts, not a second application layer: **`single.py`** runs one command,
**`examples_build.py`** regenerates example outputs, **`readme_build.py`**
regenerates `README.md`, **`console.py`** launches `ConsoleApp`, and
`oneoff_lg_correction_map.py` is a one-off maintenance script.

### Deploy surface (`project/`)

The `handler` class in `project/api/index.py` adapts HTTP requests to
`CommandRunner` for the hosted (Vercel) API, validating commands and guarding
image paths. It depends on `lanka_data` but is kept separate so deployment
concerns stay out of the library.

---

## 8. Adding a new dataset class

A dataset is the one place a new *measurement* enters the system. Because the
grammar (§README.philosophy.md) is fixed, adding data never adds an endpoint, a
flag, or a command — it adds a class that satisfies a small contract and
registers itself into two class-level lists. After that, the same
`what/when/where/how` string reaches the new data automatically.

### The contract a dataset must satisfy

Pick a base class by how the source data is shaped:

- Extend **`RegionValueDataset`** when you fetch your own table and attach a
  category → value map to each region (this is the general case, e.g.
  `Census2024Dataset`).
- Extend **`GIG2Dataset`** when the data lives in the external `gig-data`
  repository as TSV keyed by a metadata JSON (e.g. `Census2012Dataset`).
- Extend **`ElectionDataset`** for an election-shaped source with a per-year
  `supports(label, year)` check.

Follow the one-class-per-file rule: the file is named after the class
(`class Foo` → `Foo.py`), lives in `datasets/dataset/custom/`, and any bundled
metadata JSON sits beside it. Then implement the methods the pipeline calls:

- **`get_source_data_table()`** — fetch the raw rows (usually via `WWW`).
- **`clean_data_row(row)`** — normalise one raw row into
  `{"region_id", "values": {category: number}}` (use `FieldNameUtils.normalize`
  for category labels). `RegionValueDataset` handles aggregation, sorting,
  totals, and `pct_values` for you via `RegionValueDatasetTableMixin`.
- **`get_year()`** — the observation year the rows represent.
- **`get_sources()`** — the `DataSource` provenance record(s).

And the three class-level introspection hooks the framework queries to *find*
the dataset:

- **`get_labels()`** — the `What` values this dataset answers (its measurement
  labels).
- **`get_supported_whens()`** — the `When` values it covers.
- **`from_label_and_region_data_list(label, region_data_list)`** — the
  constructor the factory calls once a label matches (`GIG2Dataset` already
  provides this from its metadata JSON).

`has_values()` and `get_data_table()` come from `RegionValueDataset`, so a
typical new census-style class only writes the four instance methods plus the
metadata that backs `get_labels()`.

### How it is seamlessly detected

There is no per-dataset branch anywhere in the pipeline. Detection is *by
protocol*: the framework iterates a list of dataset classes and asks each one,
through `get_labels()` / `get_supported_whens()`, whether it answers the current
command. To wire a new class in, add it to the relevant class-level list:

1. **`DatasetFactory.CENSUS_DATASET_CLASSES`** (or `ELECTION_DATASET_CLASSES`).
   `DatasetFactory.from_command` walks this list and, via
   `_dataset_supports_census`, returns the first class whose
   `get_supported_whens()` contains `command.when_cmd` *and* whose
   `get_labels()` contains `command.what_cmd`. Matching a command to a dataset
   is therefore data-driven — the factory never names your class.

Once the class is in `DatasetFactory`'s list, every orthogonal combination comes
for free without further work: interval queries are split by
`DatasetFactory.list_from_command` and wrapped in a `DiffDataset` (so
`When = 2012-2024` and `How = Map:Change` just work); `Where` is expanded
independently by `Regions`; and every `How` output/modifier is applied later by
the visual layer. Adding one dataset multiplies the answerable query space by
the full `When × Where × How` product — this is the generativity described in
`README.philosophy.md` §3, realised as a registration step rather than new code
paths.

---

## 9. How the philosophy is implemented

`README.philosophy.md` argues for one fixed, minimal, orthogonal grammar. The
class layout above is the direct implementation of that argument. This section
maps each philosophical claim to the code that enforces it.

### One string, no secondary surface (§1, §4)

The entire public interface is `CommandRunner.run(command_str)`. There is no
options object or config: `CommandLoaderMixin.from_str` splits the one string on
`/`, requires exactly four tokens, and builds the `Command`; `CommandBase.cmd_id`
reconstructs the canonical `what/when/where/how` string. Because that string
contains no characters needing escaping, the *same* value is used as a Python
argument (`CommandRunner.run`), a CLI argument (`workflows/single.py`), a URL
path (the `handler` in `project/api/index.py`), and a cache/output file path
(`CommandCache`, `_output/<cmd>/`). One grammar, four hosts — not four
interfaces.

### Four independent fields (§2, §3)

Each axis is its own frozen dataclass — `What`, `When`, `Where`, `How` — that
validates only itself and never inspects another field's value. Orthogonality is
then realised by giving each field its *own* consumer, so the query space is a
Cartesian product rather than an enumerated list:

- `What` selects the **dataset** (`DatasetFactory`, §3/§8).
- `When` selects the **time**, and intervals are split into their years and
  the first/last diffed (`DatasetFactory.list_from_command` → `DiffDataset`).
- `Where` is expanded independently into regions (`Regions.from_command`, §5).
- `How` selects the **visual and modifier** (`VisualFactory` by `how.base`;
  colour/rank/pct/Change/Diversity applied in `ColorSpecFactory`, §4).

No single class knows all four vocabularies at once, which is why a new value on
any one axis composes with every value on the others.

### What independent of How (§2.4)

Datasets emit only normalised values (`values`, `total_value`, `pct_values`);
they never compute a ranking, a difference, or a diversity index. Those are
projections applied downstream by the visual/colour layer (`How.rank`,
`How.pct_rank`, `ColorSpecFactory`, `Diversity`). So the modifier
family (`Map:1st`, `Map:Change`, `Map:DiversityPew`, …) can grow and every
existing measurement acquires the new modifier with no change to the data
vocabulary.

### The single intentional coupling (§2.4)

The one deliberate cross-field constraint — change-style outputs require an
interval — is the only coupling in the code: `CommandBase._validate_coupling`
rejects a command whose `How.needs_interval` is true when `When.is_interval` is
false, raising `InvalidCommandError`. Every other pair of fields is left free.

### Observation time vs. boundary epoch (§2.3)

The philosophy insists these are separate facts. In code, `When` (an interval)
and the `Where` selector's `pre<year>` suffix are parsed and resolved by
different layers: `RegionRawDataMixin` resolves current vs. historical
(`-pre{year}`) boundaries and computes `region_to_current_ids`, which the
datasets follow when aggregating. Measurement year never contaminates boundary
year.

### Empty as the absence of measurement (§2.1)

`What = Empty` is not a dataset with zero rows; it is `EmptyDataset`, whose
`has_values()` returns `False` and which carries only geometry — exactly the
"region with no data bound" the philosophy describes, used for base maps and
boundary inspection.

### Provenance (§4)

Every response is traceable: `CommandRunner.run` returns `sources` (merged
`DataSource` records via `merge_datasource_list_of_lists`) and `query_time_ms`
alongside the result, so even a compact one-string request yields a fully
attributed answer.

### The grammar is domain-agnostic (§0)

The philosophy claims nothing in the grammar is specific to Sri Lanka, censuses,
or elections. The code enforces this: the `lanka_data.api` package (the fields
and the command) must not import any concrete dataset. The `datasets` layer
supplies its census/election values from outside via `DatasetFactory`. The
grammar knows only "there are legal `What` and `When` values"; the domain
supplies them from outside. That separation is what makes §8's "add a dataset,
not an endpoint" possible.

---

## Generated vs. hand-authored files

Some `__init__.py` files are marked auto-generated by `build_inits.py` and only
re-export symbols; the real work is in the sibling module files. Likewise
`README.md` is generated by the `readme/` classes and the `_output/`
tree is generated from `examples/examples.json` — edit the source classes and
JSON, not the generated output.
