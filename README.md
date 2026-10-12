# Lanka Data

**Lanka Data** is a React app for exploring and visualizing Sri Lankan data — from demographics and census figures to elections and more. Build custom queries, then view the results as beautiful, responsive charts including bar, stacked bar, pie, Marimekko, and TreeMap charts.

🚀 **Try it live:** [nuuuwan.github.io/lanka_data](https://nuuuwan.github.io/lanka_data)

A single visual's title and light-grey Paper panel span 95% of the viewport width.
Single visuals shrink proportionally when needed so the entire visual, including its
panel padding, title header, and source footer, fits within 95% of the viewport
height. The fit updates when the
viewport or visual content changes. Multiple charts or maps use their natural
grid sizes without viewport fitting and may extend below the viewport.

The visual is shown first, with the query and its update and share
controls in a fixed bottom bar. Open **Query Wizard** in a drawer above the compact
query bar. The top AppBar and bottom query bar share the same responsive height.
Follow **Data → Facets → Visual**, using the back and forward
arrow buttons or clicking a step title to navigate directly. Each step has an info alert with indented examples
on separate lines and a short navigation tip. Facets
use one neutral background with white inputs to distinguish borderless controls.
Field labels sit above their inputs rather than floating inside them.
The plus adds a facet and the cross removes one. Count has no wizard control,
but remains in the query and URL. Back/Next are at the top and Close is at the
bottom. Help can be dismissed for each step and returns when the wizard reopens.
Examples illustrate only some of the available options. Wizard edits update the
query bar immediately without applying it. Close the wizard, then use Refresh
or Enter in the query bar to apply changes. Enter in a wizard field only advances
to the next step. Reopening starts at Data with edits preserved.
While the wizard is open, the query bar and all its buttons are disabled.
Wizard edits still update the displayed query; closing restores the controls
subject to their normal loading and visual-availability rules.

Version information is at the bottom of the **...** menu. Choose **Share with QR**
in that menu to open a QR code for the current visual. The old footer QR is retired.
Choose **Refresh** in the same menu to clear all local storage (including cached
data and recent queries) and reload `/lanka_data` without query paths, URL
parameters, or a fragment. The default visual loads while the URL stays at the
app's base path.

UI components are borderless, with background shades and focus highlights
distinguishing controls. Map boundaries and chart data outlines remain visible.

Use the download icon in the bottom query bar to save a PNG with its title and source
footer. All facets are included, even below the viewport; app/query bars and the
download button are excluded. Download is enabled only when a rendered visual is available.

## Querying JSON from code

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

The response contains `datumList` and `provenance`. `npm run build` fetches data
from upstream sources and writes static `/JSON/raw.json` files for the queries
listed in `src/nonview/constants/RawJSON.js`. The deployment includes these files;
their data is a snapshot from the latest successful build. Builds fail if fetching
or generating the published data fails.

GitHub Pages supports only those listed queries, not arbitrary dynamic queries.
Other queries still support **Download JSON** in the `/JSON` browser view.
For arbitrary programmatic queries, use the same path with
`http://localhost:3000` while running the local development server (`npm start`).
Local responses are served as attachments; query failures return a non-success
status with a JSON error. Allow more than 10 seconds for large local queries and
restart the development server after server middleware changes.

The Blocks visual has been retired; use Table for tabular results.
