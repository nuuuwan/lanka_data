# Code Layout (by class)

This document explains the code by walking through its **classes**, in the order
they participate in a query. It complements `README.md` (the API grammar and
examples) and `README.philosophy.md` (the design rationale).

Every query is a single string of the form `<what>/<when>/<where>/<how>`
(for example `Religion/2012-2024/LK:district/Map:Change`). The classes below
turn that string into a JSON payload or a rendered image. Read top to bottom:
command **fields** compose into a **command**, a command produces **datasets**,
datasets plus the command produce a **visual**, and the supporting, utility, and
tooling classes sit around that spine.

The package follows one-class-per-file, with oversized classes split into named
`Mixin` classes that live beside (or in a folder with) the class they extend.

---

## 1. Command field classes

The four fields are the atoms of the API. Each is a frozen dataclass that wraps a
single `value: str`, validates it on construction, and exposes typed accessors so
the rest of the pipeline never has to re-parse the raw string. They live in
`lanka_data.api.command.fields`.

### `What`

`What` names the indicator being queried (a census table label, an election
label, an election `…Summary` label, or the special value `Empty`). On
construction it rejects any value that is not in `known_values()` and, on
failure, raises `UnknownWhatError` carrying up to five fuzzy `suggestions()`. The
special value `Help` is allowed through untouched. `available_groups()` buckets
the known values into `special`, `census`, `election`, and `election_summary`.

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
(the part after `:`), and `zoom` (the number after `@`).

### `How`

`How` describes the output. Its value splits into a `base` (before `:`, e.g.
`Map`, `BarChart`, `JSON`) and an optional `modifier` (after `:`, e.g. `Change`,
`1st`, `Diversity`). Construction validates the base against `BASE_LABELS` and
the modifier against `MODIFIERS`, raising `UnknownHowError` otherwise. It derives
display-facing properties (`base_label`, `modifier_label`, `format()`) and
semantic ones (`rank`, `pct_rank`, `needs_interval`) that later steer dataset and
color selection.

### Support classes for the fields

- **`WhatIntrospectionMixin`, `WhenIntrospectionMixin`, `WhereIntrospectionMixin`,
  `HowIntrospectionMixin`** — mixins each field inherits that add
  `available_values()` and `describe()` so the field can advertise its valid
  inputs (used by the console and `describe_api`).
- **`HowRegistryMixin`** — holds the static tables `BASE_LABELS`, `MODIFIERS`,
  and `INTERVAL_BASES` that define every legal `How`. `How` inherits it, so
  extending the output vocabulary means editing this one class.
- **`CensusDatasetRegistry`, `ElectionDatasetRegistry`, `RegionTypeRegistry`** —
  tiny holders of class-level state. They exist to break an import cycle: the
  field classes need to know which values are legal, but those values come from
  the concrete dataset/region classes. The registries are populated at startup
  (see `DatasetCommandRegistry` below), and `What`/`When`/`Where` read from them.

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

### `CommandIntrospectionMixin`

`CommandIntrospectionMixin` adds discovery helpers: `field_classes()`,
`available_values()`, `describe_api()`, and the `valid_commands()` /
`valid_what_when_pairs()` generators used to enumerate legal commands (for the
console and README/example generation).

### `Command`

`Command` simply combines the three above (`CommandIntrospectionMixin`,
`CommandBase`, `CommandLoaderMixin`) and adds `copy(...)`, which returns a new
command with selected fields overridden. `copy` is what lets interval queries be
split into their `start` and `end` point-in-time commands.

### Running a command

- **`CommandRunner`** is the orchestrator. `CommandRunner.run(command_str)`
  checks its cache, then either returns `CommandHelp`, or parses the string into
  a `Command`, asks `DatasetFactory` for the datasets, asks `VisualFactory` for
  the visual, and calls `visual.build()`. It returns a dict of the result, the
  merged `DataSource` records, and the query time, and stores it in the cache.
- **`CommandCache`** is an LRU (`OrderedDict`, default 128 entries). Its
  `is_valid` check evicts any cached image result whose file has disappeared.
- **`CommandHelp`** returns the static payload served for the `Help` command.
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

1. For an **interval** `when`, it `copy`s the command to its `start` and `end`
   years, builds a dataset for each, wraps them in a `DiffDataset`, and returns
   all three (`[start, end, diff]`).
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
  - **`PieChartVisual`** (+ `PieChartGridMixin`) — a grid of per-region pies;
    falls back to bars for change charts.
  - **`BumpChartVisual`** (+ `BumpChartDataMixin`, `BumpChartDrawMixin`) — a
    rank-change slopegraph between the two years of an interval.

### `VisualFactory`

`VisualFactory.from_command_and_datasets` reads `command.how.base` and
instantiates the matching class:

| `how.base`                  | Visual class      |
| --------------------------- | ----------------- |
| `JSON`                      | `JSONVisual`      |
| `Map`, `Cartogram`, `None`  | `MapVisual`       |
| `BarChart`                  | `BarChartVisual`  |
| `PieChart`                  | `PieChartVisual`  |
| `BumpChart`                 | `BumpChartVisual` |

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
  queries (`id@km`, via Haversine distance), and `:child_type` expansion.
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
- **`Segregation`** (+ **`SegregationComputeMixin`**) classifies each region as
  segregated or not by comparing its shares to those of its nearby neighbours.
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
  **`ColorSpecFactory`** (+ `ColorSpecHelpers` / `ColorSpecHelpersMixin`,
  `ColorSpecConstants`) picks the coloring strategy from the dataset and
  `how` modifier (rank, percentage, Change, Diversity, Segregation).
- **Geometry:** **`GeoData`** (+ `GeoDataLoaderMixin`) loads TopoJSON boundaries,
  dissolves and caches them, applies cartogram distortion, and enriches the
  `GeoDataFrame` with the dataset's values.

### Display formatters (`visual/formatters/`)

- **`WhatFormatter`**, **`WhereFormatter`**, **`HowFormatter`** turn each command
  field into the English text shown in chart headers (e.g. the region selector
  into "the Colombo & Gampaha districts").

### Startup wiring

- **`DatasetCommandRegistry`** populates the field registries (§1) with the
  concrete census/election dataset classes and the region prefix maps, so the
  field classes can validate against real data without importing it directly.
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
  the results into `examples/outputs/<cmd>/Output.json` (and images). The
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
- **`ConsoleCommandLibrary`** — builds command and field suggestions.
- **`ConsoleCompleter`** — readline tab completion over those suggestions.
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

## Generated vs. hand-authored files

Some `__init__.py` files are marked auto-generated by `build_inits.py` and only
re-export symbols; the real work is in the sibling module files. Likewise
`README.md` is generated by the `readme/` classes and the `examples/outputs/`
tree is generated from `examples/examples.json` — edit the source classes and
JSON, not the generated output.
