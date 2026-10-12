# Lanka Data

**Lanka Data** is a React app for exploring and visualizing Sri Lankan data — from demographics and census figures to elections and more. Build custom queries, then view the results as beautiful, responsive charts including bar, stacked bar, pie, Marimekko, and TreeMap charts.

🚀 **Try it live:** [nuuuwan.github.io/lanka_data](https://nuuuwan.github.io/lanka_data)

The visual title and light-grey Paper panel span 95% of the viewport width.
Visuals shrink proportionally when needed so the entire visual, including its
panel padding, title header, and source footer, fits within 95% of the viewport
height. The fit updates when the
viewport or visual content changes.

The visual is shown first, with the query and its update, random-query, and share
controls in a fixed bottom bar. Open **Query Wizard** in a drawer above the compact
grey query bar. Follow **Data → Facets → Visual**, using the back and forward
arrow buttons to navigate. Each step has an info alert with examples. Facet rows
use alternating background shades and white inputs to distinguish borderless controls.
The plus adds a facet and the cross removes one. Count has no wizard control,
but remains in the query and URL. Enter advances
from a text field; only the refresh button on the Visual step applies the
query and closes the wizard. Closing without updating retains edits in the query
bar. Reopening starts at Data with those edits preserved.

Version information is at the bottom of the **...** menu. Choose **Share with QR**
in that menu to open a QR code for the current visual. The old footer QR is retired.

UI components are borderless, with background shades and focus highlights
distinguishing controls. Map boundaries and chart data outlines remain visible.
