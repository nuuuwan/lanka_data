# lanka_data

A unified way to access Sri Lankan public data — population, elections, economy, geography, and more — through a single, predictable address scheme.

Every piece of data answers four questions: **What** is being measured, **When** was it measured, **Where**, and **How** it should be rendered. This is the **WWWH** taxonomy, and it maps directly onto a four-segment URL:

```bash
/<what>/<when>/<where>/<how>
```

A query expressed this way works identically whether issued as an HTTP request.

```bash
https://localhost:8000/<what>/<when>/<where>/<how>
```

Or a Python call.

```python
from lanka_data import Db

print(Db('/<what>/<when>/<where>/<how>'))
```

The underlying data is stored in different formats in multiple data repositories. See [DataRepos.md](DataRepos.md) for details.

## Quick examples

**National population (2012) as JSON:**

```
/Population/2012/LK/JSON
```

```json
{
  "query": "/Population/2012/LK/JSON",
  "source": "Department of Census and Statistics Sri Lanka",
  "source_url": "http://www.statistics.gov.lk/",
  "repo_file": "https://raw.githubusercontent.com/nuuuwan/gig-data/master/gig2/population-total.regions.2012.tsv",
  "total_value": 20357776,
  "n_values": 0,
  "values": {},
  "p_values": null
}
```

**Ethnic composition of Sri Lanka (2024) as JSON:**

```
/Ethnicity/2024/LK/JSON
```

```json
{
  "query": "/Ethnicity/2024/LK/JSON",
  "source": "Department of Census and Statistics Sri Lanka",
  "source_url": "http://www.statistics.gov.lk/",
  "repo_file": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/Population-Preliminary-Report/Population-by-ethnicity/data.tsv",
  "total_value": 21781800,
  "n_values": 10,
  "values": {
    "sinhalese": 16144037,
    "sl_tamil": 2681627,
    "sl_moor": 2283246,
    "indian_tamil": 600360,
    "burgher": 31721,
    "malay": 26650,
    "other": 9160,
    "sl_chetty": 2443,
    "veddha": 1373,
    "bharatha": 1183
  },
  "p_values": { "sinhalese": 74.12, "sl_tamil": 12.31, "sl_moor": 10.48, ... }
}
```

**Same data, rendered as a pie infographic:**

```
/Ethnicity/2024/LK/Pie
```

Returns an SVG.

**Ethnic composition per district (2024) as a choropleth map:**

```
/Ethnicity/2024/LK:Districts/Map
```

**2024 presidential election — national totals as bars:**

```
/Election:Presidential/2024/LK/Bar
```

**Catalog — every measurement available for Sri Lanka in 2024:**

```
/*/2024/LK/JSON
```

```json
{
  "query": "/*/2024/LK/JSON",
  "source": "multiple",
  "source_url": null,
  "repo_file": "multiple",
  "total_value": null,
  "n_values": null,
  "values": {
    "measurements": [
      "AgeGroup", "CookingFuel", "DrinkingWater",
      "Election:Parliamentary", "Election:Presidential",
      "Ethnicity", "Gender", "Households", "Housing",
      "Lighting", "Religion", "Toilet"
    ]
  },
  "p_values": null
}
```

## Specification

### The four axes

Every query has exactly four positional segments, separated by `/`:

1. **What** — the measurement. A single concept (`Population`, `Election`) or a hierarchical path within it, separated by `:` (`Population:Ethnicity`, `Election:Presidential:Summary`).
2. **When** — the time. Accepts year (`2024`), year-month (`2024-09`), or year-month-day (`2024-09-21`). Coarser precision matches all finer-grained values that fall within it.
3. **Where** — the space. Either a specific region by its code, or a region followed by `:` and a level name to enumerate its sub-regions (`LK:Districts`, `EC-01:PollingDivisions`).
4. **How** — the rendering. One of `JSON`, `Bar`, `Pie`, or `Map`.

### Region codes

Regions are identified by their official codes:

- **Country**: `LK`
- **Provinces**: ISO 3166-2 codes — `LK-1` (Western), `LK-2` (Central), … `LK-9` (Sabaragamuwa)
- **Administrative districts**: ISO 3166-2 codes — `LK-11` (Colombo), `LK-12` (Gampaha), …
- **Electoral districts**: `EC-01` (Colombo), `EC-02` (Gampaha), …
- **Polling divisions**: `EC-01A`, `EC-01B`, …
- **DSDs and GNDs**: official Department of Census and Statistics codes

Administrative districts and electoral districts are parallel hierarchies, both nested under provinces.

### Level names

To enumerate the sub-regions of a region, append `:` and a level name:

| Canonical | Aliases |
|---|---|
| `Provinces` | |
| `Districts` | |
| `ElectoralDistricts` | `EDs` |
| `PollingDivisions` | `PDs` |
| `DSDs` | |
| `GNDs` | |

Examples: `LK:Districts` (all 25 districts), `LK-3:Districts` (the 3 districts in Southern Province), `EC-01:PDs` (polling divisions of Colombo electoral district).

### How — rendering modes

The `<how>` segment selects how the data is returned. The renderer dispatches on response shape; not every `<how>` is valid for every query.

| How | Output | Use case |
|---|---|---|
| `JSON` | application/json | Raw data — the canonical machine-readable form |
| `Bar` | SVG | Horizontal bars sorted descending; the workhorse for any `breakdown` response |
| `Pie` | SVG | Proportional slices for share-of-whole framing; caps at ~6 slices with overflow into "other" |
| `Map` | SVG | Choropleth for any `breakdown_by_region` response; geometry layer chosen by the `:Level` suffix |

Omitting `<how>` defaults to `JSON`. Case-insensitive. Every SVG output bakes its query, source, source_url, repo_file, and render timestamp into `<metadata>`, so every infographic is self-attributing and reproducible from its footer alone.

**Valid combinations:**

- `Bar` and `Pie` require a `breakdown` response — `values` must be non-empty for a single region.
- `Map` requires a `breakdown_by_region` response — the `<where>` must include a `:Level` suffix.
- `JSON` is always valid.

Invalid combinations (e.g. `Pie` on a scalar, `Map` on a single region) return an error.

### Wildcards

A `*` in any of the first three positions turns the query into a catalog query. The response is the distinct values for the leftmost wildcarded position, filtered by the non-wildcarded positions. Catalog queries are always returned as JSON regardless of `<how>`.

```
/*/2024/LK/JSON                       → measurements available for LK in 2024
/Election:Presidential/*/LK/JSON      → dates of presidential elections in LK
/Election/*/LK/JSON                   → dates of all elections in LK
/*/*/LK/JSON                          → all measurements available for LK
/*/*/*/JSON                           → all measurements in the system
```

### Composite measurements

Some measurements are composites of several sub-measurements. An election, for example, has both party-level votes and a summary (electors, polled, valid, rejected, turnout, p_value); a parliamentary election additionally has seats.

When no sub-component is specified, the query returns all components together:

```
/Election:Presidential/2024/LK/JSON              → Parties + Summary
/Election:Presidential:Parties/2024/LK/JSON      → just parties
/Election:Presidential:Summary/2024/LK/JSON      → just summary
```

### JSON response format

`JSON` responses are plain JSON. All field names use `snake_case`. Every response includes:

| Field | Description |
|---|---|
| `query` | The path as submitted |
| `source` | Data source name |
| `source_url` | Data source URL |
| `repo_file` | URL of the raw data file |
| `total_value` | Aggregate total — a scalar for a single region, a `{region: total}` dict for sub-region breakdowns |
| `n_values` | Number of breakdown categories (0 if none) |
| `values` | Breakdown values, sorted by value descending |
| `p_values` | Percentage of total for each value in `values` (null if no breakdown) |

Election queries replace `values`/`p_values` with `summary`, `party`, and `p_party`:

| Field | Description |
|---|---|
| `summary` | `{valid, rejected, polled, electors}` — either a flat dict (single region) or keyed by sub-region |
| `party` | Party vote totals (or `{"years": [...]}` for wildcard-when catalog queries) |
| `p_party` | Party vote shares as percentages |

### Case and aliases

Paths are case-insensitive. The canonical form is PascalCase (`Population:Ethnicity`, `LK:Districts`, `Bar`), and queries written in any case resolve to it.

Aliases are recognized for level names (`PDs` ↔ `PollingDivisions`) and may be defined for measurements and rendering modes over time.

### Empty results

A valid query that matches no data returns an empty dict (for `JSON`) or a blank-state placeholder card (for visual modes), with a warning printed to stderr — not an error. Errors are reserved for malformed queries (unknown region codes, syntactically invalid paths, unknown measurements, unknown rendering modes, or invalid `<how>` combinations).

```
/Election:Presidential/2023/LK/JSON
→ {} with warning: "No data for '/Election:Presidential/2023/LK/JSON'."
```

### Ambiguous time matches

When a coarse time query matches multiple events, all are returned, keyed by their precise dates:

```
/Election:General/1960/LK/JSON
→ {
    "1960-03-19": { ... },
    "1960-07-20": { ... }
  }
```

## Status

Work in progress. The notation is the specification; the Python and HTTP implementations are being built against it, with the `JSON` renderer landing first and the visual renderers (`Bar`, `Pie`, `Map`) following.
