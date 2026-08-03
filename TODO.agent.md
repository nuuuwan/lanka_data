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
