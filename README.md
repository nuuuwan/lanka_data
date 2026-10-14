# Lanka Data

**Lanka Data** brings Sri Lankan census and election data into one tool for
visual exploration and programmatic access.

🚀 **Try it live:** [nuuuwan.github.io/lanka_data](https://nuuuwan.github.io/lanka_data)

## Philosophy: one query, one tool

I've wanted a single place to explore Sri Lanka's data, with a unified
visualization tool and programmatic API. Lanka Data is a step toward that goal,
starting with census and election data—not yet all data about Sri Lanka.

My aim is simplicity: **one query, one tool—for people and code**. Exploring data
visually and using it in a program should not require two separate workflows.

The query describes the data and its facets; the final segment chooses how to
present it. Use a chart or map to explore, `Table` to inspect rows, or `JSON` to
access the underlying data. The visual tool and programmatic API share the same
query logic, with source provenance included alongside the results.

## Explore visually

The query menu always includes example queries below the recent queries,
regardless of how many recent queries are saved.

Opening `/lanka_data` or `/lanka_data/` without a query shows the home page,
with nine example cards in a 3-by-3 desktop gallery, adapting to smaller screens.
Each card shows a saved screenshot containing the visual title, chart or map,
and source footer, with no separate text beneath it;
choose one to open its live visualization.
The example list contains only query strings; accessible link labels and
filesystem-safe preview filenames are derived from each query.
The AppBar and bottom query navigator remain visible on the home page. You can
also enter a query or use Query Wizard there once the data options are ready.
PNG download stays disabled until a visual is available.
Query URLs continue to open their visuals directly.
On screens narrower than 600px, a dismissible warning explains that Lanka Data
is optimized for desktop but also works on mobile. It appears once per browser
tab session, including across query navigation and reloads.
Click **Lanka Data** in the AppBar to return to the parameter-free home URL
without clearing local storage.

1. Choose a home-page example or open **Query Wizard** from the bottom query bar.
2. Choose the data, display, and facets together in a single view, with compact
   inline examples alongside the controls.
3. Close the wizard, then use **Update visualization** or press **Enter** in the
   query bar to apply your changes.

Available visuals include bar, stacked bar, pie, Marimekko, TreeMap, and maps.
Examples span the 2012 and 2024 censuses, the 1994 and 2020 parliamentary
elections, the 2005 presidential election, and the 2025 local government
election. They include election maps, religious composition on a UnitSquareMap
or UnitHexMap, and province populations and religious shares in a Marimekko chart.
Marimekko percentage axes always run from 0% to 100%, with ticks every 10%.
Use **Table** for tabular results; the Blocks visual has been retired.

### Rebuild example screenshots

After editing `src/nonview/constants/ExampleQueries.js`, start the app with
`npm start`. In another terminal, run:

```shell
npx playwright install chromium # Required once after installing dependencies
npm run screenshots:examples
```

The script reads the current query list, downloads each complete visual with
its title and source footer, and captures the desktop home page. It updates
`public/example-images`, deriving preview filenames from the queries. Existing
images are preserved, and captures are staged before replacing the current set.
Run `npm run build` afterward to include the refreshed images in deployment.
For a different running server, use
`npm run screenshots:examples -- http://localhost:3001`.

### Query Wizard tips

- All parameters are shown together; there are no steps or Back/Next buttons.
  Short examples appear inline in a small font.
- Use the plus to add a facet and the cross to remove one. `Count` stays in the
  query and URL without a separate wizard control.
- Edits appear immediately in the query bar but are not applied until you close
  the wizard and update the visualization. Enter in a wizard field does not
  apply the query or close the wizard.
- Reopening preserves your edits. The bottom query bar and its
  buttons are disabled while the wizard is open.

### Share, download, and refresh

- **Copy share link** in the bottom bar shares the current query URL.
  Copy is disabled when the query is empty or contains only whitespace.
- **Update visualization** is disabled when the query is empty or unchanged.
- **Share with QR** in the **...** menu opens a QR code for the current visual.
- **Download visual as PNG** saves the title, all facets, and source footer,
  including content below the viewport. App/query bars are excluded; a rendered
  visual must be available. The filename comes from the current query URL,
  excluding the host and `/lanka_data` prefix, with unsafe characters replaced
  by hyphens and its length capped at 200 characters.
- **Download JSON** in the `JSON` view saves data for use in your own code.
- **Access data Programmatically** in the **...** menu opens the API guide below.
- **Refresh** in the **...** menu clears all local storage, including cached data
  and recent queries, then reloads `/lanka_data` with no query path, URL parameters,
  or fragment. The home page shows the example links at the base URL.
- Version information is at the bottom of the **...** menu.

### Layout

The visual comes first, with query controls fixed at the bottom. Single visuals
fit their title, panel, and source footer within 95% of the viewport width and
height where possible, updating as the viewport or content changes. Scaling is
limited to preserve readable text; taller visuals may extend below the viewport.
Multiple charts or maps use natural grid sizes. Dense bar charts scroll
horizontally on narrow screens instead of compressing category labels.
Value and map labels have maximum font sizes; values too small to label legibly
remain available in tooltips. Tiny map regions retain their hover details rather
than displaying unreadably small labels. Shape maps abbreviate long region names
when needed; the full names remain in hover details.
Visuals remain horizontally centered in their panels when scaled; charts measure
their unscaled layout size to avoid shrinking twice.

Visual panels use a plain white background. UI controls are borderless, using
background shades and focus highlights to distinguish controls. Map boundaries
and chart data outlines remain visible.

## Use the same query from code

The interactive app and local endpoint share the same query logic. Change a
query's final visual segment to `JSON` to inspect or download its result, which
contains `datumList` and `provenance`. On a local instance, append `/raw.json`
for an HTTP response. GitHub Pages does not serve query results as raw endpoints.

For direct access without running the app, discover upstream data from the
**same live metadata indexes used by the app**. See the **[API guide](api.md)**
for metadata matching, JSON traversal, units and citations.

### For AI agents

You can ask a web-enabled ChatGPT session, for example:

> Find the 2024 presidential votes by party for Colombo North from
> https://nuuuwan.github.io/lanka_data/. Give the exact counts and cite the data.

The homepage links to a **[discovery guide](https://nuuuwan.github.io/lanka_data/data/index.html)**,
**[llms.txt](https://nuuuwan.github.io/lanka_data/llms.txt)** and a
**[source registry](https://nuuuwan.github.io/lanka_data/sources.json)**, readable
without JavaScript. The registry contains metadata endpoints and base URLs from
[the app's source configuration](src/nonview/constants/DataSources.json), not a
cached dataset catalog or a list of preselected queries.

1. **Identify the question's coordinates.** Determine the subject, year, place,
   geographic level and breakdown. Population, households and votes have
   different units. An administrative `District` is not interchangeable with an
   electoral district (`ED`).
2. **Read the live metadata used by the app.** Fetch the registry's `metadataURL`
   for each relevant source. The indexes map keys such as
   `House/Time+District+CookingFuel/Count` to relative data-file paths. Match the
   entity, set of dimension names and aggregate. Dimension order can vary; a
   combination must actually exist in metadata, not merely as separate fields.
3. **Retrieve the relevant upstream files.** Join the source's `baseURL`, `/`,
   and each matching metadata path. Fetch those files and confirm the requested
   year, region and categories in their contents. Metadata describes available
   dimension combinations; filenames alone do not establish all available values.
4. **Read and calculate carefully.** Upstream JSON nests entity, dimension:value
   keys and aggregate, ending in a typed value such as `Int:123`. It does not use
   the app's `datumList` response format. Select the requested coordinates and
   geographic level before calculating. Preserve units and explicit zeros;
   missing rows are not proof of zero. Check overlapping totals and duplicate
   observations before summing.
5. **Answer with evidence.** State the observation year, geography and unit.
   Cite the exact upstream file and the official agency, identifying the
   intermediary repository. Label calculations and their denominator. Explain
   unavailable combinations or retrieval failures instead of inventing a result.

The two live indexes are:

| App source | Metadata | Base URL for the listed relative paths |
| --- | --- | --- |
| `Census2024` | [Census metadata](https://raw.githubusercontent.com/nuuuwan/lk_census_2024/refs/heads/main/metadata/lanka_data.metadata.json) | `https://raw.githubusercontent.com/nuuuwan/lk_census_2024/refs/heads/main` |
| `GIG` | [Census and election metadata](https://raw.githubusercontent.com/nuuuwan/gig-data/refs/heads/master/lanka_data/metadata.json) | `https://raw.githubusercontent.com/nuuuwan/gig-data/refs/heads/master` |

For example, infer household cooking fuel availability by looking for
`House/Time+District+CookingFuel/Count`; infer party election results by looking
for `Vote/ElectionType+Time+PD+Party/Count`. Then inspect the listed files to find
Kandy in 2024 or Colombo North in the 2024 presidential election. The gallery is
an illustration of queries, not a definition of available data.

No login or API key is needed. ChatGPT needs web access or another HTTP-capable
tool; these instructions cannot force a session to browse or a search engine to
index the site. The **[API guide](api.md)** includes a complete Python example
that reads live metadata, selects files, filters observations and cites sources.

### Hosted and local availability

| Access | Behavior |
| --- | --- |
| Direct upstream files | Discover available data from live metadata, then fetch the listed URLs |
| Browser `JSON` view and download | Executes supported queries against upstream sources, with local caching |
| Local `/JSON/raw.json` endpoint | Executes supported queries against upstream sources on demand |
| GitHub Pages `/JSON/raw.json` | Not supported; Pages does not run a query API or host query snapshots |

`npm run build` compiles the app and publishes discovery instructions, a hosted
copy of `api.md`, and `sources.json` from local configuration. **It does not fetch
metadata, download raw data, execute queries or generate data snapshots.** Agents
read the live metadata at retrieval time, just as the app does. The generated
discovery page becomes available after building and deploying; direct metadata
links also work independently of the deployment.

The deployment workflow transfers the build between jobs as a tar archive.
Deploy jobs share a concurrency group so only one job pushes to `gh-pages`
at a time; a running deployment is allowed to finish.

## Run locally

Run `npm install`, then `npm start`, and open
[localhost:3000/lanka_data](http://localhost:3000/lanka_data).

For programmatic queries, use the same `/lanka_data/.../JSON/raw.json` path with
`http://localhost:3000`. Local responses are served as attachments; query failures
return a non-success status with a JSON error. Allow more than 10 seconds for large
queries. Restart the development server after server middleware changes.

## Social sharing

Use [the short social post](README.social.md) with any of the nine saved previews
in [public/example-images](public/example-images/). These same images appear on
the home page. They are snapshots, not live data; recapture the corresponding
visual when updating them. The separate `README.social.images` folder is retired.
