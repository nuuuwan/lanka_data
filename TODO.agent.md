# TODO

## Add an answer-first visual header

Generate a plain-language finding from the active query and display it above the visual as the page headline. Keep the encoded query and datum count as secondary details so readers encounter the result before the tool.

## Make every visual self-explanatory

Add a generated title and subtitle to each visual, covering the measure, population, geography, time period, units, and active filters. Ensure this context remains visible with every chart or map without requiring the query form.

## Hide advanced query controls by default

Move the current layperson/expert controls into a collapsed “Change this view” section after the initial visual loads. Preserve the current query URL and editing behavior while giving first-time visitors a clean, result-first view.

## Focus multi-facet results on one view

Show the first facet as the primary chart and let readers select other facets from a compact control instead of immediately rendering a dense grid. Include the active facet in the visual title so each state communicates one idea.

## Add a share and reuse toolbar

Provide actions beside the visual to copy its permanent URL and download the currently displayed data as CSV. Include the generated title and query context in the export so the result remains understandable when reused.

## Establish an editorial typography layout

Replace the all-purpose monospace presentation with a readable sans/serif type hierarchy, constrain explanatory text to roughly 65 characters per line, and reduce page width while increasing whitespace around the visual.

## Use colour only to direct attention

Introduce a central semantic palette for recurring concepts and make neutral grey the fallback for unhighlighted marks. Apply category colours consistently across chart types, legends, and query choices.

## Add a guided example narrative

Create a short “Start exploring” sequence using three curated Sri Lanka data questions. Each step should pair one sentence of interpretation with a link that loads the matching query and visual state.

## Replace the blocking loading dialog

Render an inline placeholder in the visual area with a clear loading message and the existing progress steps, leaving the page context visible. Keep the last successful visual on screen while a changed query loads where possible.

## Surface provenance with every result

Add a source block below the visual showing the dataset name, source link, last-updated value when available, active query, and a copyable citation. Populate it from data-source metadata rather than relying on the site footer.
