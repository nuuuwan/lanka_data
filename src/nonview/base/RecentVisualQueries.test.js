import {
  RECENT_VISUAL_QUERIES_LIMIT,
  RECENT_VISUAL_QUERIES_STORAGE_KEY,
} from "../constants/APP.js";
import RecentVisualQueries from "./RecentVisualQueries.js";

beforeEach(() => {
  localStorage.clear();
});

test("keeps the five most recently added queries in newest-first order", () => {
  const queries = Array.from(
    { length: RECENT_VISUAL_QUERIES_LIMIT + 1 },
    (_value, index) => `query-${index}`,
  );

  queries.forEach((query) => RecentVisualQueries.add(query));

  expect(RecentVisualQueries.read()).toEqual(queries.slice(1).reverse());
});

test("moves a duplicate query to the front instead of adding another copy", () => {
  RecentVisualQueries.add("query-a");
  RecentVisualQueries.add("query-b");

  expect(RecentVisualQueries.add("query-a")).toEqual(["query-a", "query-b"]);
});

test("clears only the recent queries entry", () => {
  localStorage.setItem("other-data", "keep");
  RecentVisualQueries.add("query-a");

  expect(RecentVisualQueries.clear()).toEqual([]);
  expect(RecentVisualQueries.read()).toEqual([]);
  expect(localStorage.getItem("other-data")).toBe("keep");
});

test("handles unavailable and malformed storage without throwing", () => {
  const unavailableStorage = {
    getItem: () => {
      throw new Error("Storage unavailable");
    },
    setItem: () => {
      throw new Error("Storage unavailable");
    },
    removeItem: () => {
      throw new Error("Storage unavailable");
    },
  };
  localStorage.setItem(RECENT_VISUAL_QUERIES_STORAGE_KEY, "{bad json");

  expect(RecentVisualQueries.read()).toEqual([]);
  expect(RecentVisualQueries.add("query-a", unavailableStorage)).toEqual([
    "query-a",
  ]);
  expect(RecentVisualQueries.clear(unavailableStorage)).toEqual([]);
});
