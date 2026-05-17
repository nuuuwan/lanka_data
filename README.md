# lanka_data

A unified way to access Sri Lankan public data — population, elections, economy, geography, and more — through a single, predictable address scheme.

Every piece of data answers three questions: **What** is being measured, **When** was it measured, and **Where**. This is the **3W** taxonomy, and it maps directly onto a three-segment URL:

```bash

/<what>/<when>/<where>

```

A query expressed this way works identically whether issued as an HTTP request.

```bash

https://localhost:8000/<what>/<when>/<where>

```

Or a Python call.

```python

from lanka_data import Db

print(Db('/<what>/<when>/<where>'))

```

The underlying data is stored in different formats in multiple data repositories. See [DataRepos.md](DataRepos.md) for details.

## Quick examples

**National population (2012):**

```
/Population/2012/LK
```

```json
{
  "query": "/Population/2012/LK",
  "source": "Department of Census and Statistics Sri Lanka",
  "source_url": "http://www.statistics.gov.lk/",
  "repo_file": "https://raw.githubusercontent.com/nuuuwan/gig-data/master/gig2/population-total.regions.2012.tsv",
  "total_value": 20357776,
  "n_values": 0,
  "values": {},
  "p_values": null
}
```

**Ethnic composition of Sri Lanka (2024):**

```
/Ethnicity/2024/LK
```

```json
{
  "query": "/Ethnicity/2024/LK",
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

**Gender breakdown (2024):**

```
/Gender/2024/LK
```

```json
{
  "query": "/Gender/2024/LK",
  "source": "Department of Census and Statistics Sri Lanka",
  "source_url": "http://www.statistics.gov.lk/",
  "repo_file": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/GN_population_excel/Population-by-Sex/data.tsv",
  "total_value": 21781800,
  "n_values": 2,
  "values": {
    "female": 11269456,
    "male": 10512344
  },
  "p_values": {
    "female": 51.74,
    "male": 48.26
  }
}
```

**Age groups (2024):**

```
/AgeGroup/2024/LK
```

```json
{
  "query": "/AgeGroup/2024/LK",
  "source": "Department of Census and Statistics Sri Lanka",
  "source_url": "http://www.statistics.gov.lk/",
  "repo_file": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/GN_population_excel/Population-by-Age-Group/data.tsv",
  "total_value": 21781800,
  "n_values": 4,
  "values": {
    "age_15_59": 13353837,
    "age_0_14": 4506839,
    "age_65_and_above": 2737814,
    "age_60_64": 1183310
  },
  "p_values": {
    "age_15_59": 61.31,
    "age_0_14": 20.69,
    "age_65_and_above": 12.57,
    "age_60_64": 5.43
  }
}
```

**Religion (2024):**

```
/Religion/2024/LK
```

```json
{
  "query": "/Religion/2024/LK",
  "source": "Department of Census and Statistics Sri Lanka",
  "source_url": "http://www.statistics.gov.lk/",
  "repo_file": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/Population-Preliminary-Report/Population-by-religion/data.tsv",
  "total_value": 21781800,
  "n_values": 6,
  "values": {
    "buddhist": 15199093,
    "hindu": 2734839,
    "islam": 2337379,
    "roman_catholic": 1224348,
    "other_christian": 282185,
    "other": 3956
  },
  "p_values": {
    "buddhist": 69.78,
    "hindu": 12.56,
    "islam": 10.73,
    "roman_catholic": 5.62,
    "other_christian": 1.3,
    "other": 0.02
  }
}
```

**Ethnic composition per district (2024):**

```
/Ethnicity/2024/LK:Districts
```

```json
{
  "query": "/Ethnicity/2024/LK:Districts",
  "source": "Department of Census and Statistics Sri Lanka",
  "source_url": "http://www.statistics.gov.lk/",
  "repo_file": "https://raw.githubusercontent.com/nuuuwan/lk_census_2024/main/data/Population-Preliminary-Report/Population-by-ethnicity/data.tsv",
  "total_value": { "LK-11": 2375415, "LK-12": 2436142, ... },
  "n_values": 10,
  "values": {
    "LK-11": { "sinhalese": 1807945, "sl_moor": 285346, "sl_tamil": 243856, ... },
    "LK-12": { "sinhalese": 2188512, "sl_moor": 123220, "sl_tamil": 97925, ... },
    ...
  },
  "p_values": { "LK-11": { "sinhalese": 76.12, ... }, ... }
}
```

**Population by district (2012):**

```
/Population/2012/LK:Districts
```

```json
{
  "query": "/Population/2012/LK:Districts",
  "source": "Department of Census and Statistics Sri Lanka",
  "source_url": "http://www.statistics.gov.lk/",
  "repo_file": "https://raw.githubusercontent.com/nuuuwan/gig-data/master/gig2/population-total.regions.2012.tsv",
  "total_value": { "LK-11": 2323964, "LK-12": 2304833, "LK-13": 1221948, ... },
  "n_values": 0,
  "values": { "LK-11": {}, "LK-12": {}, ... },
  "p_values": null
}
```

**2024 presidential election — national totals:**

```
/Election:Presidential/2024/LK
```

```json
{
  "query": "/Election:Presidential/2024/LK",
  "source": "Election Commission of Sri Lanka",
  "source_url": "https://elections.gov.lk/",
  "repo_file": "https://raw.githubusercontent.com/nuuuwan/gig-data/master/gig2/government-elections-presidential.regions-ec.2024.tsv",
  "summary": { "valid": 13319616, "rejected": 300300, "polled": 13619916, "electors": 17140354 },
  "total_value": 13319616,
  "n_values": 38,
  "party": { "NPP": 5634915, "SJB": 4363035, "IND16": 2299767, "SLPP": 342781, ... },
  "p_party": { "NPP": 42.31, "SJB": 32.76, "IND16": 17.27, ... }
}
```

**Summary metrics for Colombo electoral district polling divisions:**

```
/Election:Presidential:Summary/2024/EC-01:PDs
```

```json
{
  "query": "/Election:Presidential:Summary/2024/EC-01:PDs",
  "source": "Election Commission of Sri Lanka",
  "source_url": "https://elections.gov.lk/",
  "repo_file": "https://raw.githubusercontent.com/nuuuwan/gig-data/master/gig2/government-elections-presidential.regions-ec.2024.tsv",
  "summary": {
    "EC-01A": { "valid": 69243, "rejected": 2276, "polled": 71519, "electors": 97620 },
    "EC-01B": { "valid": 88982, "rejected": 2522, "polled": 91504, "electors": 118986 },
    "EC-01C": { "valid": 45486, "rejected": 1078, "polled": 46564, "electors": 62623 },
    ...
  },
  "total_value": null,
  "n_values": null,
  "party": {},
  "p_party": null
}
```

**Catalog — every measurement available for Sri Lanka in 2024:**

```
/*/2024/LK
```

```json
{
  "query": "/*/2024/LK",
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

**Catalog — all years for which presidential election data exists:**

```
/Election:Presidential/*/LK
```

```json
{
  "query": "/Election:Presidential/*/LK",
  "source": "Election Commission of Sri Lanka",
  "source_url": "https://elections.gov.lk/",
  "repo_file": "multiple",
  "summary": {},
  "total_value": null,
  "n_values": null,
  "party": { "years": ["1982", "1988", "1994", "1999", "2005", "2010", "2015", "2019", "2024"] },
  "p_party": null
}
```

## Specification

### The three axes

Every query has exactly three positional segments, separated by `/`:

1. **What** — the measurement. A single concept (`Population`, `Election`) or a hierarchical path within it, separated by `:` (`Population:Ethnicity`, `Election:Presidential:Summary`).
2. **When** — the time. Accepts year (`2024`), year-month (`2024-09`), or year-month-day (`2024-09-21`). Coarser precision matches all finer-grained values that fall within it.
3. **Where** — the space. Either a specific region by its code, or a region followed by `:` and a level name to enumerate its sub-regions (`LK:Districts`, `EC-01:PollingDivisions`).

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

### Wildcards

A `*` in any of the three positions turns the query into a catalog query. The response is the distinct values for the leftmost wildcarded position, filtered by the non-wildcarded positions.

```

/*/2024/LK                       → measurements available for LK in 2024
/Election:Presidential/*/LK      → dates of presidential elections in LK
/Election/*/LK                   → dates of all elections in LK
/*/*/LK                          → all measurements available for LK
/*/*/*                           → all measurements in the system

```

### Composite measurements

Some measurements are composites of several sub-measurements. An election, for example, has both party-level votes and a summary (electors, polled, valid, rejected, turnout, p_value); a parliamentary election additionally has seats.

When no sub-component is specified, the query returns all components together:

```

/Election:Presidential/2024/LK              → Parties + Summary
/Election:Presidential:Parties/2024/LK      → just parties
/Election:Presidential:Summary/2024/LK      → just summary

```

### Response format

Responses are plain JSON. All field names use `snake_case`. Every response includes:

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

Paths are case-insensitive. The canonical form is PascalCase (`Population:Ethnicity`, `LK:Districts`), and queries written in any case resolve to it.

Aliases are recognized for level names (`PDs` ↔ `PollingDivisions`) and may be defined for measurements over time.

### Empty results

A valid query that matches no data returns an empty dict with a warning printed to stderr — not an error. Errors are reserved for malformed queries (unknown region codes, syntactically invalid paths, unknown measurements).

```
/Election:Presidential/2023/LK
→ {} with warning: "No data for '/Election:Presidential/2023/LK'."
```

### Ambiguous time matches

When a coarse time query matches multiple events, all are returned, keyed by their precise dates:

```
/Election:General/1960/LK
→ {
    "1960-03-19": { ... },
    "1960-07-20": { ... }
  }
```

## Status

Work in progress. The notation is the specification; the Python and HTTP implementations are being built against it.
