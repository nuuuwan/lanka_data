# TODOs

## Unbreak Now

## High Pri

## Mid Pri

- FEATURE: add local government data.
- FEATURE: Directly access JSON. Also, CSV, GeoJSON etc.

## Low Pri

### Add CSV Data Export

Add a CSV visual option alongside the existing JSON visual so users can preview and download the current datum set as `lanka-data.csv`. Create a generic CSV serializer in `nonview/base` that correctly escapes commas, quotes, and newlines; register the visual in the existing visual options; and add serializer and component tests.

## Wishlist

### Show Inline Query Validation

Validate each layperson query part before submission and show specific inline guidance for missing entities, aggregates, visuals, fields, operators, or filter values. Disable Update only when the query is incomplete, leave expert-mode parsing behavior unchanged, centralize reusable validation outside the view layer, and add unit and interaction tests for valid and invalid combinations.
