# TODOs by Agent

## 1. Add a Recent Queries Menu

Add a compact recent-queries menu that stores the last five successfully loaded visual queries in `localStorage`. Let users reopen or clear entries, avoid duplicates, and handle unavailable storage gracefully. Keep persistence logic in `nonview/base`, put stateful UI in an organism, and add focused tests for ordering, deduplication, clearing, and navigation.

## 2. Add a Copy Share Link Action

Add a clearly labelled action beside the query form that copies the current visualization URL to the clipboard and briefly confirms success. Preserve the full query route and the `/lanka_data` basename, provide a fallback when the Clipboard API is unavailable, make the control keyboard-accessible, and test both success and failure states.

## 3. Add CSV Data Export

Add a CSV visual option alongside the existing JSON visual so users can preview and download the current datum set as `lanka-data.csv`. Create a generic CSV serializer in `nonview/base` that correctly escapes commas, quotes, and newlines; register the visual in the existing visual options; and add serializer and component tests.

## 4. Add a Table Visual

Add a responsive, accessible table visual for inspecting query results without a chart. Display query dimensions and aggregate values as columns, include sortable column headers, format values consistently with existing visuals, register the new visual with `VisualFactory`, and cover empty, single-row, and sorting behavior with tests.

## 5. Make Query Fields Searchable

Improve the layperson query form by replacing long entity, field, and categorical-value selectors with searchable MUI autocomplete controls. Preserve unknown values from expert-mode queries, keep keyboard navigation and accessible labels working, and update the existing form tests to cover filtering, selection, and mode switching.

## 6. Show Inline Query Validation

Validate each layperson query part before submission and show specific inline guidance for missing entities, aggregates, visuals, fields, operators, or filter values. Disable Update only when the query is incomplete, leave expert-mode parsing behavior unchanged, centralize reusable validation outside the view layer, and add unit and interaction tests for valid and invalid combinations.

## 7. Display Data Provenance

Show a small “About this data” panel beneath each successful visualization with the contributing data source, dataset title, and source URL when available. Extend the data-source result metadata without changing existing query results, omit unavailable fields cleanly, use safe external-link attributes, and test single-source and mixed-source responses.

## 8. Add an Example Query Gallery

Add a collapsible gallery of six curated example queries covering census, election, chart, and map use cases. Selecting an example should navigate through the existing query route and work on mobile and keyboard-only navigation. Store examples in `nonview/constants`, build the stateful gallery as an organism, and test labels and route changes.

## 9. Add a Data Loading Retry Action

Improve failed data loads by adding a Retry button that repeats the current request without changing the URL. Keep parse errors distinct from network/data-source errors, prevent duplicate requests while retrying, preserve stale-request cancellation, and extend `VisualQueryPage` tests to cover failure, retry success, and repeated clicks.

## 10. Add a Color-Blind-Friendly Palette Toggle

Add a user-selectable color-blind-friendly palette that applies consistently to charts, legends, and maps and persists across visits. Define palette values in `nonview/constants`, manage the preference through a custom context hook, retain the current palette as the default, and add tests for toggling, persistence, and representative visual colors.

## 11. Add Insight-Led Visual Headings

Add a reusable visual header that presents a plain-language takeaway above every chart, followed by a subtitle containing the metric, unit, geography, date range, and essential caveats. Keep heading metadata with the visual query so shared views retain their context, provide a factual fallback when no authored takeaway exists, and test complete, partial, and missing metadata.

## 12. Add Focused Series Highlighting

Let each chart designate one or more focus series while rendering the remaining series in neutral grey. Store semantic series colors in a single constants module, keep the same category color across every visual type, preserve sufficient contrast, and test focus changes, unknown categories, and legend consistency.

## 13. Add Small-Multiple Comparison Views

Add a small-multiple visual that splits a multi-series result into repeated charts with shared scales and consistent axes instead of layering every series in one plot. Make the grid responsive, keep ordering deterministic, limit each panel to one comparison idea, and test scale sharing, panel order, empty series, and mobile layout.

## 14. Move Advanced Controls Behind Disclosure

Keep the default visual view focused by placing expert controls such as scale type, date range, series selection, and downloads in an accessible expandable panel. Preserve control choices in the URL, expose an accurate expanded state to assistive technology, retain sensible defaults, and test keyboard operation, deep links, and reset behavior.

## 15. Add Direct Links to Individual Visuals

Give every rendered visual a stable permalink that restores its query, visual type, filters, and presentation settings without relying on session state. Add a copy-link action near the visual, preserve the `/lanka_data` basename, handle obsolete query parameters safely, and add route round-trip and backward-compatibility tests.

## 16. Add Chart Annotations

Support concise annotations that connect an explanatory note to a specific datum, category, or date so important changes are understandable without inspecting every mark. Keep annotation definitions outside presentation components, reposition or collapse them on narrow screens, include their text in accessible output, and test missing targets, overlap handling, and responsive placement.

## 17. Add a Guided Story View

Add an optional story view that presents an authored sequence of text steps and synchronizes each step with a defined visual state. Keep the normal exploratory view available, make step order and chart states data-driven, support keyboard and reduced-motion navigation, and test forward, backward, direct-link, and mobile behavior.

## 18. Render Honest Loading Previews

Replace blank visual containers during data loading with a lightweight preview that reserves the final dimensions, clearly says what is loading, and becomes interactive only after the data is ready. Avoid showing fabricated values, distinguish initial loading from refreshes, preserve the last successful visual during background updates, and test slow, failed, stale, and successful requests.

## 19. Add Screenshot-Safe Visual Context

Ensure an exported or screenshotted visual includes its insight-led title, subtitle, legend, source, last-updated date, and Lanka Data attribution within the visual boundary. Keep the on-screen layout uncluttered, omit unavailable metadata cleanly, use the same context for image export and embeds, and add snapshot tests for complete and partial metadata.

## 20. Add a Methodology and Citation Drawer

Add an accessible drawer for each visual that exposes source links, methodology notes, update dates, data definitions, a generated citation, and concise reuse guidance. Derive these details from query result metadata, clearly label missing or mixed provenance, provide copy-citation feedback, and test single-source, multi-source, and incomplete metadata cases.
