# Code Layout

This file maps the repository structure to the code paths that implement it. It is meant to complement the existing README files, not restate the API grammar or the published analyses.

## Top-level layout

- `src/lanka_data/` — the main Python package.
- `src/utils_future/` — shared utility code used by `lanka_data`, including file I/O, HTTP fetch helpers, logging, timers, geometry helpers, and JSON helpers.
- `workflows/` — small entrypoints for running common repository tasks.
- `project/` — a separate deploy wrapper for serving the package over Vercel.
- `examples/` — declarative example commands plus their generated outputs.
- `analysis/` — narrative markdown writeups that consume generated outputs, not core library code.

## Main execution path

Most of the package is organized around one pipeline:

1. `command/` parses the command string into a `Command`.
2. `region/` resolves the `where` token into concrete regions and boundary vintages.
3. `dataset/` selects and loads the underlying data tables for the chosen `what` and `when`.
4. `visual/` turns the resulting dataset into JSON or an image-backed visualization.
5. `CommandRunner` assembles the final response payload with sources and timing.

`/home/runner/work/lanka_data/lanka_data/src/lanka_data/command/CommandRunner.py` is the hub where those steps are wired together.

## `src/lanka_data/`

### `command/`

This layer handles the external command string.

- `CommandBase.py` defines the four stored fields.
- `CommandLoaderMixin.py` parses raw text such as `Religion/2012-2024/LK:district/Map:Change`.
- `Command.py` wraps the parsed command and supports copying with one field changed.
- `CommandRunner.py` is the runtime orchestrator and in-memory cache.
- `CommandHelp.py` provides the special `Help` response.

### `region/`

This layer interprets region syntax and fetches region metadata.

- `RegionParserMixin/` implements selectors such as comma lists, ranges, radius queries, and `:child_type` expansion.
- `RegionRawDataMixin/` fetches raw region definitions from the external `lk_admin_regions` dataset and handles historical boundary lookup.
- `RegionLoadersMixin.py` connects parsing to loading.
- `Regions.py` is the concrete object returned to the rest of the pipeline.
- `RegionTypeUtils.py` and `Where.py` provide lightweight support types.

If you need to change how the `where` field works, this is the first directory to inspect.

### `dataset/`

This layer converts a parsed command plus resolved regions into typed data tables.

- `Dataset.py` defines the abstract dataset interface.
- `DatasetFactory.py` chooses the correct dataset implementation.
- `RegionValueDataset/` contains the shared logic for datasets that attach value maps to regions.
- `DiffDataset.py` combines two point-in-time datasets for interval commands.
- `EmptyDataset.py` supplies geometry-only results.
- `custom/` contains the concrete source adapters:
  - census datasets for 2001, 2012, and 2024
  - election datasets and election summaries
  - JSON metadata files that map public labels to upstream table IDs

The package does not vendor large source tables in-repo; the dataset classes fetch them from external repositories or data endpoints when needed.

### `visual/`

This layer formats output.

- `Visual.py` defines the abstract visualization contract.
- `VisualFactory.py` maps `how` values to concrete visual classes.
- `JSONVisual.py` emits structured JSON.
- `plot_visual/` contains chart and map implementations:
  - `MapVisual.py`
  - `BarChartVisual/`
  - `PieChartVisual/`
  - `BumpChartVisual/`
- `plot/` contains lower-level plotting helpers such as fonts, legends, labels, headers, footers, color rules, and geographic dataframe loaders.
- `formatters/` contains small helpers used to turn command pieces into display text.

This split keeps output-type selection separate from the lower-level drawing code.

### `data/`

Small support objects and computations shared across datasets and visuals live here.

- `DataSource.py` models provenance records.
- `FieldNameUtils.py` normalizes category labels.
- `Diversity.py` and `Segregation/` hold reusable statistical computations.

### `examples/`

This package is not sample text only; it also contains repository build logic.

- `Example/Example.py` reads `examples/examples.json`.
- `ExampleOutputMixin.py` materializes outputs into `examples/outputs/<cmd>/Output.json`.

The generated example outputs are later reused by the README and analysis documents.

### `readme/`

This package generates the main repository README from reusable mixins.

- `ReadMe.py` assembles the document.
- `ReadMeUsageMixin.py`, `ReadMeSourcesMixin.py`, `ReadMeExamplesMixin/`, and `ReadMeFooterMixin.py` each contribute a section.

`workflows/readme_build.py` is just a thin entrypoint into this package.

## Other important directories

### `workflows/`

These are thin scripts rather than a second application layer.

- `single.py` runs one command locally.
- `examples_build.py` regenerates example outputs.
- `readme_build.py` regenerates `README.md`.
- `oneoff_lg_correction_map.py` is a one-off workflow script.

### `project/`

This is the deploy surface for the hosted API.

- `api/index.py` adapts HTTP requests to `CommandRunner`.
- `pyproject.toml` declares a minimal deployment package that depends on `lanka_data`.
- `vercel.json` holds Vercel configuration.

The separation keeps deployment concerns out of the library package itself.

### `examples/`

- `examples/examples.json` is the source of truth for named example commands.
- `examples/outputs/` stores generated outputs, including JSON payloads and rendered images.

### `analysis/`

These markdown files are downstream consumers of the generated outputs. They are useful for understanding how the library is used, but they are not where query parsing, data loading, or rendering logic lives.

## Generated vs hand-authored files

Some package `__init__.py` files are marked as auto-generated by `build_inits.py`. They mostly re-export symbols for convenient imports. The core implementation work happens in the adjacent module files, not in those generated `__init__` files.

Likewise, the main `README.md` is generated by the code under `src/lanka_data/readme/`, and the large `examples/outputs/` tree is generated from `examples/examples.json`.
