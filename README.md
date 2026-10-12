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

Opening `/lanka_data` or `/lanka_data/` without a query shows the home page,
with a responsive gallery of example cards. Each card has a saved preview and
a prominent title without a separate description; choose one to open its live visualization.
The AppBar and bottom query navigator remain visible on the home page. You can
also enter a query or use Query Wizard there once the data options are ready.
PNG download stays disabled until a visual is available.
Query URLs continue to open their visuals directly.
Click **Lanka Data** in the AppBar to return to the parameter-free home URL
without clearing local storage.

1. Choose a home-page example or open **Query Wizard** from the bottom query bar.
2. Follow **Data → Facets → Visual** to choose what to explore, how to group or
   filter it, and how to display it.
3. Close the wizard, then use **Update visualization** or press **Enter** in the
   query bar to apply your changes.

Available visuals include bar, stacked bar, pie, Marimekko, TreeMap, and maps.
Examples include presidential results on a HexMap, religious composition on a
UnitSquareMap, and province populations and religious shares in a Marimekko chart.
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
height, updating as the viewport or content changes. Multiple charts or maps use
natural grid sizes and may extend below the viewport.

UI components are borderless, using background shades and focus highlights to
distinguish controls. Map boundaries and chart data outlines remain visible.

## Use the same query from code

Change a query's final visual segment to `JSON` for the browser view; append
`/raw.json` for direct programmatic access where the endpoint is available.
The response contains `datumList` and `provenance`.

### Published API examples

The published query below is available directly from GitHub Pages as
`application/json`, without running the React app. All three examples request
the same data and check for HTTP errors.

#### Python

Requires the `requests` package.

```python
import requests

url = "https://nuuuwan.github.io/lanka_data/Vote/ElectionType=presidential+Time=2024+PD%3CED=colombo+Party/Count/JSON/raw.json"
response = requests.get(url, timeout=60)
response.raise_for_status()
data = response.json()
print(data)
```

#### JavaScript (Node.js 18+)

Uses built-in `fetch`; save as an `.mjs` file to use top-level `await`.

```javascript
const url = "https://nuuuwan.github.io/lanka_data/Vote/ElectionType=presidential+Time=2024+PD%3CED=colombo+Party/Count/JSON/raw.json";
const response = await fetch(url, { signal: AbortSignal.timeout(60_000) });
if (!response.ok) {
  throw new Error(`HTTP ${response.status}: ${response.statusText}`);
}
const data = await response.json();
console.log(data);
```

#### Go

Uses only the standard library.

```go
package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "time"
)

func main() {
    url := "https://nuuuwan.github.io/lanka_data/Vote/ElectionType=presidential+Time=2024+PD%3CED=colombo+Party/Count/JSON/raw.json"
    client := &http.Client{Timeout: 60 * time.Second}
    response, err := client.Get(url)
    if err != nil {
        log.Fatal(err)
    }
    defer response.Body.Close()
    if response.StatusCode != http.StatusOK {
        log.Fatalf("HTTP %s", response.Status)
    }
    var data map[string]json.RawMessage
    if err := json.NewDecoder(response.Body).Decode(&data); err != nil {
        log.Fatal(err)
    }
    fmt.Println(string(data["datumList"]))
    fmt.Println(string(data["provenance"]))
}
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

Use [the short social post](README.social.md) with any of the nine saved previews
in [public/example-images](public/example-images/). These same images appear on
the home page. They are snapshots, not live data; recapture the corresponding
visual when updating them. The separate `README.social.images` folder is retired.
