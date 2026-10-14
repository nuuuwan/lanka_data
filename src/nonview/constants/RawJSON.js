import { DEFAULT_VISUAL_QUERY_STR } from "./APP.js";
import { EXAMPLE_QUERIES } from "./ExampleQueries.js";

// Keep every homepage example accessible without executing the React app.
export const PUBLISHED_RAW_JSON_QUERIES = [
  ...new Set(
    [DEFAULT_VISUAL_QUERY_STR, ...EXAMPLE_QUERIES].map((query) =>
      query.replace(/\/[^/]+$/, "/JSON"),
    ),
  ),
];
