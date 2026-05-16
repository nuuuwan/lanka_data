# lanka_data

A unified way to access Sri Lankan public data — population, elections, economy, geography, and more — through a single, predictable address scheme.

Every piece of data answers three questions: **What** is being measured, **When** was it measured, and **Where**. This is the **3W** taxonomy, and it maps directly onto a three-segment URL:

```

/<what>/<when>/<where>

```

A query expressed this way works identically whether issued as an HTTP request or a Python call.

The underlying data is stored in different formats in multople data repositories. See [DataRepos.md](DataRepos.md) for details.

## Quick examples

```

/Population/2024/LK

```

Sri Lanka's total population in 2024.

```

/Population/2024/LK:Districts

```

Population in 2024, broken down by district.

```

/Population:Ethnicity/2024/LK:Districts

```

Ethnic composition per district in 2024.

```

/Election:Presidential/2024/LK

```

Full results of the 2024 presidential election at the national level — party vote totals plus the summary (electors, polled, valid, rejected, turnout).

```

/Election:Presidential:Summary/2024/EC-01:PDs

```

The summary metrics for each polling division within the Colombo electoral district.

```

/*/2024/LK

```

A catalog: every measurement for which 2024 Sri Lanka data exists.

```

/Election:Presidential/*/LK

```

A catalog: every date on which a presidential election was held.

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

Responses are JSON. The nesting mirrors the breakdown structure of the query:

```json
// /Population:Ethnicity/2024/LK:Districts
{
  "LK-11": {
    "Sinhalese": 1842376,
    "Tamil":     247739,
    "Moor":      257622,
    ...
  },
  "LK-12": { ... },
  ...
}
```

### Case and aliases

Paths are case-insensitive. The canonical form is PascalCase (`Population:Ethnicity`, `LK:Districts`), and queries written in any case resolve to it.

Aliases are recognized for level names (`PDs` ↔ `PollingDivisions`) and may be defined for measurements over time.

### Empty results

A valid query that matches no data returns an empty result with a warning explaining why — not an error. Errors are reserved for malformed queries (unknown region codes, syntactically invalid paths, unknown measurements).

```
/Election:Presidential/2023/LK
→ {} with warning: "No presidential elections in 2023. Nearest: 2019-11-16, 2024-09-21."
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
