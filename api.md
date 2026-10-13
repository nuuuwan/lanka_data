# Lanka Data Static JSON API Guide for AI Agents & Developers

**Lanka Data** provides a static, zero-configuration JSON API designed for direct consumption by AI agents, data analysts, and software developers.

- **Catalog URL**: `https://nuuuwan.github.io/lanka_data/catalog.json`
- **Base Deployment Path**: `/lanka_data`
- **File Name**: `raw.json`

---

## 1. How to Use the Catalog

The fastest and most reliable way for an AI agent or automated script to consume Lanka Data is via the static snapshot catalog:

```
GET https://nuuuwan.github.io/lanka_data/catalog.json
```

### Catalog Structure

```json
{
  "catalogVersion": "1.0",
  "description": "Catalog of published static JSON queries for Lanka Data",
  "generatedAt": "2026-10-13T04:09:48.122Z",
  "count": 3,
  "queries": [
    {
      "query": "Vote/ElectionType=presidential+Time=2024+PD<ED=colombo+Party/Count/JSON",
      "description": "Valid votes by polling division and party where election type is presidential and in 2024 and in colombo electoral district",
      "url": "https://nuuuwan.github.io/lanka_data/Vote/ElectionType=presidential+Time=2024+PD%3CED=colombo+Party/Count/JSON/raw.json",
      "generatedAt": "2026-10-13T04:09:48.122Z"
    }
  ]
}
```

### Recommended Workflow for AI Agents

1. **Fetch `catalog.json`** to discover available datasets, their canonical query strings, and plain-English descriptions.
2. **Match the user's intent** against the `description` or `query` string (e.g., checking entity types, years, regions, or categories).
3. **Fetch the target snapshot `url`** directly with an HTTP GET request. No API key, headers, or client-side execution are required.
4. **Parse and summarize** `datumList` and cite `provenance` in the final answer.

---

## 2. GitHub Pages vs. Arbitrary Queries

> ⚠️ **Important:** GitHub Pages is a static file host. It serves **only published query snapshots** listed in [catalog.json](catalog.json). It does **not** run a dynamic backend server to process arbitrary queries on the fly.

- **Published Snapshots on GitHub Pages**:
  - Generated at build time from [src/nonview/constants/RawJSON.js](src/nonview/constants/RawJSON.js).
  - Stored as static `raw.json` files on GitHub Pages.
  - Returns `application/json` with status `200 OK`.
- **Arbitrary / Unbuilt Queries on GitHub Pages**:
  - Requesting an arbitrary query path (e.g., `/Vote/ElectionType=parliamentary+Time=1994+ED+Party/Count/JSON/raw.json`) returns the single-page application's `404.html` redirect page, **not** JSON.
- **Executing Arbitrary Queries Programmatically**:
  - Run Lanka Data locally:
    ```shell
    git clone https://github.com/nuuuwan/lanka_data.git
    cd lanka_data
    npm install
    npm start
    ```
    The local development server includes middleware that queries upstream sources dynamically:
    ```
    GET http://localhost:3000/lanka_data/{query}/JSON/raw.json
    ```
  - Alternatively, navigate to `https://nuuuwan.github.io/lanka_data/?/{query}/JSON` in a web browser to view and download client-fetched results.

---

## 3. Natural-Language Question → URL Examples

Below are concrete examples mapping natural-language questions to published snapshot URLs in the catalog, as well as an example of an arbitrary query that requires a local environment:

### Example 1: 2024 Presidential Election in Colombo
- **Question**: *"What was the party vote breakdown by polling division in Colombo for the 2024 Presidential Election?"*
- **Query**: `Vote/ElectionType=presidential+Time=2024+PD<ED=colombo+Party/Count/JSON`
- **Published Snapshot URL**:
  ```
  https://nuuuwan.github.io/lanka_data/Vote/ElectionType=presidential+Time=2024+PD%3CED=colombo+Party/Count/JSON/raw.json
  ```

### Example 2: Household Cooking Fuel in Kandy (2012 & 2024)
- **Question**: *"How did household cooking fuel usage in Kandy district compare between the 2012 and 2024 censuses?"*
- **Query**: `House/Time=2012,2024+District=kandy+CookingFuel/Count/JSON`
- **Published Snapshot URL**:
  ```
  https://nuuuwan.github.io/lanka_data/House/Time=2012,2024+District=kandy+CookingFuel/Count/JSON/raw.json
  ```

### Example 3: Provincial Religious Demographics (2012)
- **Question**: *"What was the population breakdown by religion across Sri Lanka's provinces in the 2012 Census?"*
- **Query**: `Person/Time=2012+Province+Religion/Count/JSON`
- **Published Snapshot URL**:
  ```
  https://nuuuwan.github.io/lanka_data/Person/Time=2012+Province+Religion/Count/JSON/raw.json
  ```

### Example 4: Arbitrary / Ad-Hoc Query (Local Server)
- **Question**: *"What were the 1994 parliamentary election results by electoral district and party?"*
- **Query**: `Vote/ElectionType=parliamentary+Time=1994+ED+Party/Count/JSON`
- **Availability**: Not in the default static snapshot catalog; access via the interactive browser interface or run locally at:
  ```
  http://localhost:3000/lanka_data/Vote/ElectionType=parliamentary+Time=1994+ED+Party/Count/JSON/raw.json
  ```

---

## 4. Response Structure and Field Meanings

Every `raw.json` response has the following schema:

```json
{
  "datumList": [
    {
      "query": {
        "queryStr": "Vote/ElectionType=presidential+Time=2024+PD=colombo_north+Party=NPP/Count",
        "aggregate": "Count",
        "dimThingList": [
          { "value": "presidential" },
          { "value": "2024" },
          { "value": "colombo_north" },
          { "value": "NPP" }
        ],
        "subRegionDimThingList": null,
        "parentRegionConstraintList": null
      },
      "answerThing": {
        "value": "17978"
      }
    }
  ],
  "provenance": [
    {
      "source": "The Elections Commission of Sri Lanka",
      "title": "Sri Lanka election results",
      "url": "https://elections.gov.lk"
    }
  ]
}
```

### Root Object
- `datumList`: An array of observation objects (`Datum`).
- `provenance`: An array of data source citations for the dataset.

### Datum Object (`datumList[i]`)
- `query`: The specific dimension coordinate of this observation.
  - `queryStr`: Normalized string representation of this cell (e.g. `Vote/.../Count`).
  - `aggregate`: The aggregation function applied (usually `"Count"`).
  - `dimThingList`: Array of dimension objects in the order specified by the query. Each has a `"value"` string representing that dimension's value.
  - `subRegionDimThingList`: Active sub-regional dimension objects, or `null`.
  - `parentRegionConstraintList`: Parent region constraints applied, or `null`.
- `answerThing`:
  - `value`: The observed numeric quantity as a string (e.g. `"17978"`).

---

## 5. Units & Metric Interpretation

When `aggregate` is `"Count"`:
- **`Person`**: Number of individuals (population count).
- **`House`**: Number of households / housing units.
- **`Vote`**: Number of valid votes cast.
- **`Census`**: Number of census records.

All counts are integer numbers stored as strings. Callers should parse values using `parseInt(datum.answerThing.value, 10)` or `Number(datum.answerThing.value)`.

---

## 6. Geographic Identifiers

Geographic regions use normalized, lower-case `snake_case` identifiers.

### Hierarchy Levels
- **`Country`**: National level (`sri_lanka`).
- **`Province`**: 9 provinces:
  `western`, `central`, `southern`, `northern`, `eastern`, `north_western`, `north_central`, `uva`, `sabaragamuwa`.
- **`District`**: 25 administrative districts (e.g. `colombo`, `gampaha`, `kalutara`, `kandy`, `matale`, `nuwara_eliya`, `galle`, `matara`, `hambantota`, `jaffna`, `ampara`, etc.).
- **`DSD`**: Divisional Secretariat Divisions (256+ administrative divisions).
- **`GND`**: Grama Niladhari Divisions (the smallest administrative subdivisions).
- **`ED`**: Electoral Districts (22 electoral districts used in national elections).
- **`PD`**: Polling Divisions (electoral sub-districts within electoral districts).

### Parent-Scoping Syntax and URL Encoding
In query strings, `<` denotes parent region filtering (e.g. `PD<ED=colombo` selects all Polling Divisions within the Colombo Electoral District).
- In HTTP URLs, `<` must be percent-encoded as `%3C`.
- Example: `PD%3CED=colombo`

---

## 7. Missing-Value and Grouping Conventions

- **`excluded_small` ("Others")**:
  In high-cardinality categorical fields (such as political parties with minor vote shares), small or infrequent values are grouped into `"excluded_small"`. In user interfaces, this is presented as "Others".
- **`none`**:
  Where a measured attribute is absent or not applicable (for instance, a household reporting no cooking fuel), the value is `"none"` or `"None"`.
- **Sparse / Absent Combinations**:
  If a particular combination of dimensions had zero observations or was unrecorded, that combination is omitted from `datumList` rather than included with a zero or null value.

---

## 8. How to Cite Provenance

Every response includes one or more provenance objects:

```json
"provenance": [
  {
    "source": "The Elections Commission of Sri Lanka",
    "title": "Sri Lanka election results",
    "url": "https://elections.gov.lk"
  }
]
```

When displaying data or generating answers, AI agents and researchers should cite the data source and link to the official URL:

> **Data Source:** [Sri Lanka election results](https://elections.gov.lk), The Elections Commission of Sri Lanka (via Lanka Data).
