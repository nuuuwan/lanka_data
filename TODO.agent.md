# TODO

## 1. Share Stacked Chart Legends Across Facets

Update `StackedBarChart` and `MarimekkoChart` so a multi-facet visualization renders one deduplicated category legend below `MultiChartLayout`, matching the existing shared PieChart legend behavior. Derive stable `{ id, label, color }` items from all facet data in `ChartVisual`, remove each chart's local legend, and preserve a legend for single-facet views too. Do not change map legends.

## 2. Use a Common Bar-Chart Value Domain Per View

Make bar and stacked-bar facet comparisons honest by calculating the highest value across all facets in `ChartVisual` and passing it to the chart visual. Configure Nivo `valueScale` with that shared maximum for `BarChart` and `StackedBarChart`; retain zero as the minimum and preserve current humanized axis ticks. Keep Marimekko on its percentage scale.

## 3. Add Value-Sorted Pie Slices

Sort each PieChart's input data by descending numeric value before passing it to `ResponsivePie`, with a deterministic alphabetical tie-breaker on `id`. Ensure the shared legend uses the same ordered category list where possible, does not mutate facet data, and keeps the current value-only in-slice labels and disabled arc-link labels.

## 4. Add Pie Hover Focus

Improve `PieChart` interaction with Nivo's `activeOuterRadiusOffset` and `activeInnerRadiusOffset` so the hovered slice has a clear, restrained focus state. Use constants rather than magic numbers, preserve the current color palette and tooltips, and ensure the focus offsets do not make slices overlap their container at small sizes.

## 5. Add Accessible Pattern Fills for Pie and Stacked Categories

Add optional Nivo SVG pattern fills to PieChart and StackedBarChart so categories remain distinguishable without color alone. Use a small deterministic set of built-in Nivo patterns selected from category identity, retain the existing semantic colors as the base, and ensure the legend swatches represent the same fills or remain understandable without them. Keep the feature scoped to these two visuals.

## 6. Improve TreeMap Label Fitting and Contrast

Replace TreeMap's fixed `labelSkipSize` and darkened label color with a custom Nivo label component that fits or shortens labels based on each node rectangle's dimensions. Reuse the project's `String.shorten` and `FormatUtils.isLightColor` logic, show the full label and humanized value in the tooltip, and avoid rendering labels that cannot meet the existing minimum readable font size.

## 7. Add Interactive TreeMap Focus Zoom

Use `ResponsiveTreeMap` click handling to let a user focus on a selected top-level node, showing that node's contents in the available TreeMap area. Add a compact breadcrumb/back control owned by an organism or page, reset focus when the input data changes, and preserve the normal all-data view by default. Do not alter visual-query routing or source data.

## 8. Add Area-Bump Start and End Labels

Enhance `AreaBump` with Nivo start/end labels that show each series name at the first and last rank. Reuse the current series colors, suppress labels that would overlap or fall below a sensible viewport width, and retain existing tooltip behavior and custom ordering logic. Keep labels concise and rely on tooltips for full details when space is insufficient.

## 9. Add Shared Rich Chart Tooltips

Create a reusable chart tooltip molecule for BarChart, StackedBarChart, PieChart, MarimekkoChart, AreaBump, and TreeMap. It should show a color swatch, category/series label, a humanized value, and where applicable the percentage of the current facet total. Replace each chart's local ad hoc tooltip while preserving correct Nivo tooltip prop shapes and not adding state to atoms or moles.

## 10. Add Nivo Animation Preference

Add a persistent user preference for chart animation, defaulting to the current no-animation behavior. Provide a compact control in the visualization UI, store the setting with the repository's existing client-side persistence pattern, and pass it to all Nivo chart components (`BarChart`, `StackedBarChart`, `MarimekkoChart`, `PieChart`, `AreaBump`, and `TreeMap`). Use a restrained Nivo motion configuration and respect reduced-motion preferences.
