# Lanka Data: finding data from live metadata

Start at the [discovery guide](https://nuuuwan.github.io/lanka_data/data/index.html)
or [source registry](https://nuuuwan.github.io/lanka_data/sources.json). Both are
linked from the homepage without requiring JavaScript. The registry publishes
only metadata endpoints and base URLs from the same configuration used by the
app. It contains no observations, copied metadata or fixed query catalog.

## Infer available data

1. Identify the subject, year, geographic level, place and breakdown requested.
2. Fetch the live metadata for each relevant source. Match a metadata key's
   entity, dimension-name set and aggregate to the question.
3. Read the relative file paths associated with that key. Append each path to
   that source's base URL with a slash, and fetch the relevant files.
4. Confirm the year and coordinate values in their contents. Filter to the
   requested observations before calculating and cite the files actually read.

No authentication or API key is required. A ChatGPT session needs web access or
an HTTP-capable tool. If metadata lacks the requested combination, or retrieval
fails, explain the gap rather than inferring a number or substituting another
year or geographic level.

## The same metadata sources as the app

| Source | Live discovery | Official attribution |
| --- | --- | --- |
| `Census2024` | [Metadata](https://raw.githubusercontent.com/nuuuwan/lk_census_2024/refs/heads/main/metadata/lanka_data.metadata.json), [repository](https://github.com/nuuuwan/lk_census_2024). Census 2024 population and housing topics, plus historical and administrative tables in the census report. | [Department of Census and Statistics](https://www.statistics.gov.lk), Census of Population and Housing 2024 |
| `GIG` | [Metadata](https://raw.githubusercontent.com/nuuuwan/gig-data/refs/heads/master/lanka_data/metadata.json), [repository](https://github.com/nuuuwan/gig-data). Census 2012 population and household topics and historical elections. | [Department of Census and Statistics](https://www.statistics.gov.lk), Census of Population and Housing 2012; for `Vote`, [Elections Commission](https://elections.gov.lk), Sri Lanka election results |

The public repositories are intermediaries, not official agency-hosted APIs.
Coverage varies by topic, year and geographic level. Metadata and file contents
are authoritative for availability; the gallery and the UI's category lists do
not define every possible cross-tabulation. Some upstream metadata tables are
not supported by the app's query engine but can still be read directly.

`sources.json` has `schemaVersion`, `description` and a `sources` array. Each
source provides `id`, `description`, `baseURL` and `metadataURL`. It is generated
from [DataSources.json](https://github.com/nuuuwan/lanka_data/blob/main/src/nonview/constants/DataSources.json),
which also supplies the app's data-source URLs. It does not enumerate datasets
or freeze metadata at build time.

### Interpret a metadata index

Each live index is an object mapping a key to one or more relative paths:

```json
{
  "House/Time+District+CookingFuel/Count": [
    "data/House-CookingFuel/lanka_data.json"
  ]
}
```

This entry in `Census2024` describes an available combination of dimensions,
not a specific district or year. Its file URL is the source's `baseURL` plus `/`
plus `data/House-CookingFuel/lanka_data.json`.

Match `House`, the set `{Time, District, CookingFuel}`, and `Count`. Dimension
order can vary between metadata keys. The app checks for an exact key first,
then compares keys after sorting dimension names. It fetches the associated
files and filters their observations to the query's coordinates. A metadata key
can list several files, and a file can support several metadata keys.

For election results, a question about votes by party in a polling division
maps to `Vote/ElectionType+Time+PD+Party/Count`. Inspect the listed files for the
requested election type, year and polling division. Do not manufacture file
paths from years or assume a filename establishes coverage. Avoid fetching
unrelated tables. Do not combine separate religion and ethnicity tables into
an unsupported joint distribution.

## Read upstream JSON

Source files are nested objects. Paths identify the entity, dimension values
and aggregate, ending in a typed scalar. An illustrative path is:

```text
House
  Time:2024
    District:kandy
      CookingFuel:fire_wood
        Count: "Int:123"
```

Here `123` is an illustrative value, not an observation. Actual values must be
read from the metadata-linked file. Split dimension keys and typed values at
the first `:`. Preserve types such as `Int`, `Float`, `Percent` and `Bool` and
aggregates such as `Count` and `IsIn`; not every table is a count.

A single file can contain several years and several overlapping geographic
levels. Select the dimension-name set and coordinate values you want before
summing. Preserve explicit zeros; absent rows do not establish zero. Metadata
and source files can change independently of the website deployment.

### Working Python example: Kandy cooking fuel in 2024

Uses the standard library. This discovers matching files from live metadata,
then filters their actual observations. It does not depend on a prebuilt query
or a data snapshot on the website. Raw GitHub JSON may be served as text, so
parse the response body rather than requiring a JSON content type.

```python
import json
from urllib.request import urlopen

BASE = "https://nuuuwan.github.io/lanka_data"

def get_json(url):
    with urlopen(url, timeout=60) as response:
        return json.load(response)  # HTTP errors raise; HTML fails JSON parsing

def leaves(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, path + (key,))
    else:
        yield path, node

registry = get_json(BASE + "/sources.json")
source = next(s for s in registry["sources"] if s["id"] == "Census2024")
metadata = get_json(source["metadataURL"])
wanted_dimensions = {"Time", "District", "CookingFuel"}
paths = set()
for key, file_paths in metadata.items():
    entity, dimensions, aggregate = key.split("/")
    if (entity == "House" and aggregate == "Count"
            and set(dimensions.split("+")) == wanted_dimensions):
        paths.update(file_paths)
if not paths:
    raise ValueError("This dimension combination is unavailable in metadata")

observations = {}
for relative_path in sorted(paths):
    url = source["baseURL"] + "/" + relative_path
    for path, value in leaves(get_json(url)):
        if path[0] != "House" or path[-1] != "Count":
            continue
        coordinates = dict(token.split(":", 1) for token in path[1:-1])
        if (set(coordinates) != wanted_dimensions
                or coordinates["Time"] != "2024"
                or coordinates["District"] != "kandy"):
            continue
        value_type, number = value.split(":", 1)
        if value_type != "Int":
            raise ValueError(f"Unexpected count type: {value_type}")
        fuel = coordinates["CookingFuel"]
        if fuel in observations:
            raise ValueError(f"Duplicate coordinate requires review: {fuel}")
        observations[fuel] = (int(number), url)
if not observations:
    raise ValueError("No observations found for Kandy in 2024; this is not zero")
for fuel, (count, url) in sorted(observations.items()):
    print(f"2024, Kandy District, {fuel}: {count} households; data: {url}")
print("Source: Department of Census and Statistics, Census 2024, "
      "https://www.statistics.gov.lk (via nuuuwan/lk_census_2024)")
```

For a different question, choose the source, entity, dimension-name set,
aggregate and coordinate filters using its metadata. The example is a traversal
pattern, not a universal assumption that all tables use `House` or integer counts.

## Query syntax and geography

The general interactive form is `Entity/Dimension+Dimension=value/Count/Visual`.

- `Person`, `House` and `Vote` select the subject. `Count` selects the metric.
- `+` separates dimensions; an unfiltered dimension selects its available values.
- `=` filters a value; `,` selects several, e.g. `Time=2012,2024`.
- `<` scopes a child geography to a parent, e.g. `PD<ED=colombo`.
- `JSON`, `Table` or a chart/map type selects the presentation.

Region identifiers are normalized lower-case names such as `sri_lanka`,
`colombo` and `nuwara_eliya`; party identifiers such as `NPP` remain case-sensitive.
Use identifiers found in the returned data. Administrative geography
(`Country`, `Province`, `District`, `DSD`, `GND`) and electoral geography
(`ED`, `PD`) are distinct. A District and an ED with the same name are not
interchangeable. GND exists as a concept in the code but that alone does not
establish query or source coverage. Consult the metadata for each topic.

The app resolves geography using
[lk_admin_regions](https://github.com/nuuuwan/lk_admin_regions/tree/main/data/ents).
For custom parent filtering, use its region records and parent IDs rather than
substring matching names or assuming an electoral/administrative equivalence.

Use metadata file paths unchanged. In constructed query URLs, encode `<` as `%3C`; keep `/`
as path separators and `+` as dimension separators, not spaces. The interactive
link can use `https://nuuuwan.github.io/lanka_data/?/` followed by the query.

## Units, missing data and calculations

- `Person/.../Count` measures people. Household examples under `House` measure
  households; inspect the topic's universe before describing it as buildings,
  occupied housing units or all dwellings.
- `Vote/...+Party/Count` measures votes attributed to a party/category. It does
  not by itself establish turnout, registered electors or rejected ballots.
  For presidential data, do not relabel party identifiers as candidate names
  without a source mapping.
- `Census`, `District` and `AdministrativeEntity` also occur in upstream metadata.
  Read their aggregate and typed values; do not label them all “census records”.
- Preserve explicit zeros. An absent row has no universally defined meaning;
  the loader does not fill missing combinations or certify that they are zero.
- `none` is a source category, not automatically missing data. `excluded_small`
  is displayed as “Others” where present; do not assume raw source files collapse
  small parties or that all category lists are exhaustive.
- Before totals, select one observation year, geographic level, metric and
  non-overlapping category set. Check for total rows, overlapping files and
  duplicate coordinates. The query engine concatenates matching source rows;
  it does not guarantee deduplication or reconciliation.
- For shares, state the denominator and whether it covers all categories. For
  changes, distinguish absolute differences, percentage change and percentage
  points; check that definitions and geographic boundaries are comparable.

## Cite the result

State the value, unit, observation year, geography and breakdown. Link to the
exact upstream file you fetched and the official source in the attribution
table above, identifying the intermediary repository. Label calculated totals
or percentages and state the denominator. A retrieval or deployment date is
not the observation year.

For results downloaded from the app or the local endpoint, cite the source
objects in `provenance` and the query used. Agency URLs can be homepages rather
than original tables; do not claim to have checked an official report unless
you actually retrieved it.

## Interactive and local query results

The interactive JSON view executes supported queries in the browser and offers
a download of the fetched data. Its response is different from upstream nested
JSON: `datumList` contains observations and `provenance` lists contributing
sources. Each row has `query.queryStr` (entity, named dimensions and aggregate)
and `answerThing.value` (a string). Read dimension names from that row's
`queryStr`, not from the request's dimension order. JavaScript class names are
not included in the serialized dimension objects. A multi-year result may have
several dataset-level source citations.

To run the same query engine locally:

```shell
git clone https://github.com/nuuuwan/lanka_data.git
cd lanka_data
npm install
npm start
```

Use `http://localhost:3000/lanka_data/{entity}/{dimensions}/Count/JSON/raw.json`.
For example:

```text
http://localhost:3000/lanka_data/House/Time=2024+District=kandy+CookingFuel/Count/JSON/raw.json
```

The local middleware executes queries on demand and returns JSON attachments;
failures return a non-success JSON error. Unsupported dimension combinations
can return an empty `datumList`; an empty result is not a zero count.

GitHub Pages serves the app and discovery instructions. It does not execute
queries or publish `/JSON/raw.json` files. Such a URL can return an HTML 404
redirect, not data. Use live metadata and upstream files for HTTP retrieval
without a local server, or use the browser's JSON download.

`npm run build` compiles the app and publishes only local configuration and
instructions: the discovery page, `sources.json`, `llms.txt` and this guide.
It never fetches metadata or raw data, executes queries, or builds data snapshots.
