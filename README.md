# Lanka Data

**Lanka Data** brings Sri Lankan census and election data into one tool for
visual exploration and programmatic access.

🚀 **Try it live:** [nuuuwan.github.io/lanka_data](https://nuuuwan.github.io/lanka_data)

## Philosophy: one query, one tool

My aim is simplicity: **one query, one tool—for people and code**. Exploring data
visually and using it in a program should not require two separate workflows.

The query describes the data and its facets; the final segment chooses how to
present it. Use a chart or map to explore, `Table` to inspect rows, or `JSON` to
access the underlying data. The visual tool and programmatic API share the same
query logic, with source provenance included alongside the results.

## Explore visually

1. Open **Query Wizard** from the bottom query bar.
2. Follow **Data → Facets → Visual** to choose what to explore, how to group or
   filter it, and how to display it.
3. Close the wizard, then use **Update visualization** or press **Enter** in the
   query bar to apply your changes.

Available visuals include bar, stacked bar, pie, Marimekko, TreeMap, and maps.
Use **Table** for tabular results; the Blocks visual has been retired.

### Query Wizard tips

- Move with Back/Next or click a step title. Each step includes dismissible help
  and examples; help returns when the wizard reopens.
- Use the plus to add a facet and the cross to remove one. `Count` stays in the
  query and URL without a separate wizard control.
- Edits appear immediately in the query bar but are not applied until you close
  the wizard and update the visualization. Enter in a wizard field only advances
  to the next step.
- Reopening starts at Data with edits preserved. The bottom query bar and its
  buttons are disabled while the wizard is open.

### Share, download, and refresh

- **Copy share link** in the bottom bar shares the current query URL.
- **Share with QR** in the **...** menu opens a QR code for the current visual.
- **Download visual as PNG** saves the title, all facets, and source footer,
  including content below the viewport. App/query bars are excluded; a rendered
  visual must be available.
- **Download JSON** in the `JSON` view saves data for use in your own code.
- **Refresh** in the **...** menu clears all local storage, including cached data
  and recent queries, then reloads `/lanka_data` with no query path, URL parameters,
  or fragment. The default visual loads without changing the base URL.
- Version information is at the bottom of the **...** menu.

### Layout

The visual comes first, with query controls fixed at the bottom. Single visuals
fit their title, panel, and source footer within 95% of the viewport width and
height, updating as the viewport or content changes. Multiple charts or maps use
natural grid sizes and may extend below the viewport.

UI components are borderless, using background shades and focus highlights to
distinguish controls. Map boundaries and chart data outlines remain visible.

## Use the same query from code

Change a query's final visual segment to `JSON` for the browser view; append
`/raw.json` for direct programmatic access where the endpoint is available.
The response contains `datumList` and `provenance`.

### Published API example

The published query below is available directly from GitHub Pages as
`application/json`, without running the React app:

```python
import requests

url = "https://nuuuwan.github.io/lanka_data/Vote/ElectionType=presidential+Time=2024+PD%3CED=colombo+Party/Count/JSON/raw.json"
response = requests.get(url, timeout=60)
response.raise_for_status()
data = response.json()
print(data)
```

### Hosted and local availability

| Access | Supported queries | Data freshness |
| --- | --- | --- |
| GitHub Pages `/JSON/raw.json` | Queries listed in [RawJSON.js](src/nonview/constants/RawJSON.js) | Snapshot from the latest successful build |
| Browser `JSON` view and download | Queries supported by the app's data sources | Loaded from upstream sources, with local caching |
| Local `/JSON/raw.json` endpoint | Queries supported by the app's data sources | Loaded from upstream sources |

GitHub Pages is static-only: it does not run an API server for arbitrary queries.
`npm run build` fetches upstream data and generates the listed static JSON files
for deployment. A failed fetch or generation fails the build.

## Run locally

Run `npm install`, then `npm start`, and open
[localhost:3000/lanka_data](http://localhost:3000/lanka_data).

For programmatic queries, use the same `/lanka_data/.../JSON/raw.json` path with
`http://localhost:3000`. Local responses are served as attachments; query failures
return a non-success status with a JSON error. Allow more than 10 seconds for large
queries. Restart the development server after server middleware changes.

## Social sharing

Use [the short social post](README.social.md) with the four screenshots in
[README.social.images](README.social.images/).
