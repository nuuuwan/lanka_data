# UI/UX Design Principles

## Lead with the answer, not the tool

OWID's Energy page opens with a plain-language claim ("The world lacks a safe, low-carbon, cheap energy infrastructure") before any chart. The insight is the headline; the visualisation is evidence.

## Every chart is a sentence

Titles state findings, not variables. "Why did renewables become so cheap so fast?" beats "LCOE by technology, 2010-2023." Subtitles carry units, scope and caveats so the chart survives being screenshotted.

## One idea per view

Each OWID grapher shows a single metric. Complexity is handled by adding more small charts, not more layers on one chart.

## Progressive disclosure

Default view is simple. Country selectors, log scales, time sliders, "download data" and source notes sit one click deep. Novices see a clean chart; experts can drill.

## Charts are objects, not decorations

Every OWID chart has its own URL, embed code, PNG, CSV and citation. Design assumes the chart will be reused elsewhere, so it carries its own context.

## Typography and whitespace do the styling

A single serif/sans pairing, generous line-height, narrow measure (~65 chars), lots of white. Almost no borders, shadows, or panel chrome.

## Colour is semantic and scarce

Grey is the default; colour is spent only on the series being discussed. Categorical palettes stay stable across the whole site, so "solar" is the same colour everywhere.

## Narrative scaffolding around the data

The Pudding uses scrollytelling: text and chart state advance together, so the reader never faces an unexplained visual. Order is authored, not left to exploration.

## Fast, and honest about loading

Static thumbnails and server-rendered images first, interactivity hydrated after. The Pudding's map ships a visible "Loading Map..." state rather than a blank box.

## Provenance is part of the UI

Sources, methodology links, last-updated dates, "Cite this work" and "Reuse this work" are surfaced in the layout, not buried in a footer. Trust is treated as a design requirement.

The common thread: these sites optimise for *comprehension per second*, and treat the dashboard aesthetic (dense grids, KPI tiles, chart junk) as a failure mode.
