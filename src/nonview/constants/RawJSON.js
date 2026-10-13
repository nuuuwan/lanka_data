import { DEFAULT_VISUAL_QUERY_STR } from "./APP.js";

export const PUBLISHED_RAW_JSON_QUERIES = [
  DEFAULT_VISUAL_QUERY_STR.replace(/\/[^/]+$/, "/JSON"),
  "House/Time=2012,2024+District=kandy+CookingFuel/Count/JSON",
  "Person/Time=2012+Province+Religion/Count/JSON",
];
