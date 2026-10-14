# Lanka Data: finding and retrieving data

Start at the [data directory](https://nuuuwan.github.io/lanka_data/data/index.html).
It contains ordinary HTML links and tables, so a browsing agent can start from
`https://nuuuwan.github.io/lanka_data/` without executing React. The same data
powers the interactive charts and maps.

## Answer a question from the site

1. Identify the **subject, year, geographic level, place and breakdown**. For
   example, “2024 presidential votes by party in Colombo's polling divisions”
   means `Vote`, `ElectionType=presidential`, `Time=2024`, `PD<ED=colombo`, `Party`.
2. Read [catalog.json](https://nuuuwan.github.io/lanka_data/catalog.json).
   Match the whole question to a query, or find a broader snapshot you can filter
   without changing its year, unit or geography. Fetch the entry's exact `url`.
   If your browsing tool cannot read JSON, follow `pageURL` to its HTML table.
3. Read the observations, filter to the requested coordinates and calculate only
   over compatible rows. Cite the fetched data URL and its `provenance`.
4. If no snapshot fits, use the
   [complete upstream directory](https://nuuuwan.github.io/lanka_data/data/sources.html)
   or [sources.json](https://nuuuwan.github.io/lanka_data/sources.json). These index
   all tables advertised by the app's upstream metadata, not just the examples.
   Follow the exact listed file URLs and use the nested-file instructions below.
5. If neither route supplies the requested data, say what is unavailable. Do not
   invent a number, treat missing data as zero, or silently substitute another
   census year or geographic level. A browser-capable agent can also open the
   interactive query; a coding agent can run the local endpoint.

Web access is required. Publishing these links makes the site discoverable but
cannot force a particular ChatGPT session to browse, index the site, or support
large JSON downloads. No login or API key is needed.

## What is available

The app reads two public data repositories. Official source attribution is
included in query responses; these repositories are intermediaries, not official
agency-hosted APIs.

| Source | Coverage and discovery | Official attribution |
| --- | --- | --- |
| `Census2024` | [Live metadata](https://raw.githubusercontent.com/nuuuwan/lk_census_2024/refs/heads/main/metadata/lanka_data.metadata.json), [repository](https://github.com/nuuuwan/lk_census_2024). Census 2024 population and housing topics, plus historical and administrative tables in the census report. | [Department of Census and Statistics](https://www.statistics.gov.lk), Census of Population and Housing 2024 |
| `GIG` | [Live metadata](https://raw.githubusercontent.com/nuuuwan/gig-data/refs/heads/master/lanka_data/metadata.json), [repository](https://github.com/nuuuwan/gig-data). Census 2012 population and household topics and historical election results. | [Department of Census and Statistics](https://www.statistics.gov.lk), Census of Population and Housing 2012; for `Vote`, [Elections Commission](https://elections.gov.lk), Sri Lanka election results |

Coverage varies by table. Population topics include religion and ethnicity;
household topics include cooking fuel, lighting and drinking water. Elections
include presidential, parliamentary and local government results. The live
metadata and file contents determine supported years and combinations; a
category's existence in the UI does not establish that every cross-tab exists.
Some upstream metadata tables are not supported by the app's query engine but
can still be fetched directly.

### Published snapshots

The default query and every homepage example are built as static JSON and HTML
on each successful deployment. The catalog is generated from
[RawJSON.js](https://github.com/nuuuwan/lanka_data/blob/main/src/nonview/constants/RawJSON.js)
and [ExampleQueries.js](https://github.com/nuuuwan/lanka_data/blob/main/src/nonview/constants/ExampleQueries.js).
Use the catalog as the authoritative list rather than constructing URLs.

| Question | Canonical JSON query to look up in the catalog |
| --- | --- |
| 2024 presidential votes by polling division and party in Colombo? | `Vote/ElectionType=presidential+Time=2024+PD<ED=colombo+Party/Count/JSON` |
| How did Kandy household cooking fuel change between 2012 and 2024? | `House/Time=2012,2024+District=kandy+CookingFuel/Count/JSON` |
| Household lighting across Sri Lanka in 2012? | `House/Time=2012+Country+Lighting/Count/JSON` |
| 1994 parliamentary results by electoral district and party? | `Vote/ElectionType=parliamentary+Time=1994+ED+Party/Count/JSON` |
| 2020 parliamentary results in Colombo's polling divisions? | `Vote/ElectionType=parliamentary+Time=2020+PD<ED=colombo+Party/Count/JSON` |
| 2005 presidential results by electoral district and party? | `Vote/ElectionType=presidential+Time=2005+ED+Party/Count/JSON` |
| 2025 local government results by electoral district and party? | `Vote/ElectionType=local_government+Time=2025+ED+Party/Count/JSON` |
| Population by province and religion in 2012? | `Person/Time=2012+Province+Religion/Count/JSON` |
| Drinking water by polling division within Galle district in 2012? | `House/Time=2012+PD<District=Galle+SourceOfDrinkingWater/Count/JSON` |
| Population by province and ethnicity in 2024? | `Person/Time=2024+Province+Ethnicity/Count/JSON` |

### Catalog schema

`catalog.json` has `catalogVersion` (currently `"1.1"`), `description`,
`generatedAt` (UTC build timestamp), `count`, `guideURL`, `sourcesURL`, and
`queries`. Each query entry has:

- `query`: the exact published query, ending in `/JSON`.
- `description`: a readable description of its filters and breakdowns.
- `url`: absolute URL of the static `raw.json` response.
- `pageURL`: absolute URL of an HTML table of the same observations and provenance.
- `visualURL`: interactive JSON view, requiring JavaScript.
- `rowCount`: number of observations, not a population or vote total.
- `generatedAt`: snapshot build timestamp, **not the census/election date**.

The build fails if a snapshot is empty or generation fails. Upstream directory
metadata is captured at build time, but its linked source files remain live.
Do not assume those files have the same retrieval date as a published snapshot.

## Read a snapshot response

A response contains `datumList` and `provenance`. This is one real observation
from the 2024 presidential Colombo snapshot (the response includes other rows):

```json
{
  "datumList": [{
    "query": {
      "dimThingList": [
        {"value": "presidential"}, {"value": "2024"},
        {"value": "colombo_north"}, {"value": "NPP"}
      ],
      "aggregate": "Count",
      "queryStr": "Vote/ElectionType=presidential+Time=2024+PD=colombo_north+Party=NPP/Count",
      "subRegionDimThingList": null,
      "parentRegionConstraintList": null
    },
    "answerThing": {"value": "17978"}
  }],
  "provenance": [{
    "source": "The Elections Commission of Sri Lanka",
    "title": "Sri Lanka election results",
    "url": "https://elections.gov.lk"
  }]
}
```

`queryStr` identifies the row's entity, named dimensions and aggregate. The
serialized objects omit JavaScript class names, so **do not infer dimension
names from the requested query's order**. Read each row's `queryStr`; upstream
files can use another order. `dimThingList` contains values in that row's order.
The two constraint fields are internal query details, not additional observations.

`answerThing.value` is a string. For these `Count` snapshots, parse it as an
integer and retain full precision. Do not take rounded chart labels as data.
`provenance` is a dataset-level list of contributing sources; a multi-year query
can contain multiple citations, rather than a separate citation for each row.

### Working Python example

Uses the standard library. It discovers the URL, verifies the response, and
prints the 2024 presidential observations for Colombo North.

```python
import json
from urllib.request import urlopen

BASE = "https://nuuuwan.github.io/lanka_data"

def get_json(url):
    with urlopen(url, timeout=60) as response:
        if response.headers.get_content_type() != "application/json":
            raise ValueError(f"Expected JSON at {url}")
        return json.load(response)

catalog = get_json(BASE + "/catalog.json")
query = "Vote/ElectionType=presidential+Time=2024+PD<ED=colombo+Party/Count/JSON"
entry = next(item for item in catalog["queries"] if item["query"] == query)
data = get_json(entry["url"])
for row in data["datumList"]:
    entity, dimensions, aggregate = row["query"]["queryStr"].split("/")
    coordinates = dict(token.split("=", 1) for token in dimensions.split("+"))
    if coordinates.get("PD") == "colombo_north":
        print(coordinates["Party"], int(row["answerThing"]["value"]))
print("Data:", entry["url"])
print("Sources:", data["provenance"])
```

## Find data beyond the snapshots

`sources.json` contains `catalogVersion`, `generatedAt`, `description`, and
`sources`. Each source has `id`, `baseURL`, `metadataURL`, and `datasets`.
Each dataset entry has a `metadataKey` and absolute `urls`.

A metadata key such as `House/Time+District+CookingFuel/Count` describes an
**available combination of dimensions**, not a filter for a specific district
or year. The original live metadata maps each such key to relative file paths;
append a path to that source's `baseURL`, or use the expanded URLs in
`sources.json`. Inspect all relevant files for the requested year and values.
Dimension order can vary; match dimension names as a set when searching keys.
Do not combine separate religion and ethnicity tables into a joint distribution.

Upstream JSON differs from the snapshot response. It is a nested object whose
paths name the entity, dimensions and aggregate, ending in a typed scalar:

```text
House
  Time:2024
    Country:sri_lanka
      CookingFuel:fire_wood
        Count: "Int:3381781"
```

That example is from the
[2024 cooking fuel source file](https://raw.githubusercontent.com/nuuuwan/lk_census_2024/refs/heads/main/data/House-CookingFuel/lanka_data.json).
A file can include Country, Province, District and other levels together; summing
all leaves would double-count. Preserve types such as `Int`, `Float`, `Percent`
and `Bool` and aggregates such as `Count` and `IsIn`; not every table is a count.

This extension to the Python example retrieves district-level Kandy cooking fuel
observations from the upstream 2024 census source. Raw GitHub JSON may be served
as text, so this reader parses the body without requiring a JSON content type.

```python
sources = get_json(BASE + "/sources.json")
source = next(s for s in sources["sources"] if s["id"] == "Census2024")
dataset = next(d for d in source["datasets"]
               if d["metadataKey"] == "House/Time+District+CookingFuel/Count")

def leaves(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, path + (key,))
    else:
        yield path, node

for url in dict.fromkeys(dataset["urls"]):
    with urlopen(url, timeout=60) as response:
        raw = json.load(response)
    for path, value in leaves(raw):
        if path[0] != "House" or path[-1] != "Count":
            continue
        coordinates = dict(token.split(":", 1) for token in path[1:-1])
        if (set(coordinates) == {"Time", "District", "CookingFuel"}
                and coordinates["Time"] == "2024"
                and coordinates["District"] == "kandy"):
            value_type, number = value.split(":", 1)
            if value_type != "Int":
                raise ValueError(f"Unexpected count type: {value_type}")
            print(coordinates["CookingFuel"], int(number), url)
```

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

Use catalog URLs unchanged. In constructed URLs, encode `<` as `%3C`; keep `/`
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
  is displayed as “Others” where present; do not assume raw snapshots collapse
  small parties or that all category lists are exhaustive.
- Before totals, select one observation year, geographic level, metric and
  non-overlapping category set. Check for total rows, overlapping files and
  duplicate coordinates. The query engine concatenates matching source rows;
  it does not guarantee deduplication or reconciliation.
- For shares, state the denominator and whether it covers all categories. For
  changes, distinguish absolute differences, percentage change and percentage
  points; check that definitions and geographic boundaries are comparable.

## Cite the result

Give the value, unit, observation year, geography and breakdown. Link to the
actual snapshot/table or upstream file you read, plus the official source in
`provenance`. For upstream files, use the source attribution table above and
identify the intermediary repository. Report calculations as calculations.

Example: “Colombo North recorded 17,978 NPP votes in the 2024 presidential
results ([Lanka Data snapshot](https://nuuuwan.github.io/lanka_data/Vote/ElectionType=presidential+Time=2024+PD%3CED=colombo+Party/Count/JSON/raw.json);
source: [Elections Commission of Sri Lanka](https://elections.gov.lk)).”

Provenance URLs can be agency homepages rather than the original table. Do not
claim to have checked an official report unless you actually retrieved it.

## Hosting and local access

GitHub Pages serves static files. Only catalog entries have published
`/JSON/raw.json` endpoints. A fabricated path can return a 404 HTML redirect;
it is not a dynamic query API. Check HTTP success and parseability before using
a response. The browser's JSON view runs JavaScript and can download results for
other queries supported by the upstream sources.

To run the same query engine locally:

```shell
git clone https://github.com/nuuuwan/lanka_data.git
cd lanka_data
npm install
npm start
```

Use `http://localhost:3000/lanka_data/{entity}/{dimensions}/Count/JSON/raw.json`.
The local middleware executes supported queries and returns JSON attachments;
failures return a non-success JSON error. Unsupported dimension combinations can
return an empty `datumList`; an empty result is not a zero count.

`npm run build` generates the catalog, snapshots, HTML directory/tables,
`sources.json` and a hosted copy of this guide in `build/`. `public/llms.txt` and
the homepage discovery links are copied into the deployment. Generated catalogs
are not checked into `public/`, where stale entries could outlive their files.
The generated directory is available in the production build, not a fresh
`npm start` session. Source changes take effect on the live site after deployment.
