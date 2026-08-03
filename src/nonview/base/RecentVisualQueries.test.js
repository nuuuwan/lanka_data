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

  expect(RecentVisualQueries.read().map(({ query }) => query)).toEqual(
    queries.slice(1).reverse(),
  );
});

test("moves a duplicate query to the front instead of adding another copy", () => {
  RecentVisualQueries.add("query-a", undefined, 1);
  RecentVisualQueries.add("query-b", undefined, 2);

  expect(RecentVisualQueries.add("query-a", undefined, 3)).toEqual([
    { query: "query-a", timestamp: 3 },
    { query: "query-b", timestamp: 2 },
  ]);
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
  expect(RecentVisualQueries.add("query-a", unavailableStorage, 1)).toEqual([
    { query: "query-a", timestamp: 1 },
  ]);
  expect(RecentVisualQueries.clear(unavailableStorage)).toEqual([]);
});

test("reads legacy string entries without losing saved queries", () => {
  localStorage.setItem(
    RECENT_VISUAL_QUERIES_STORAGE_KEY,
    JSON.stringify(["query-a"]),
  );

  expect(RecentVisualQueries.read()).toEqual([
    { query: "query-a", timestamp: null },
  ]);
});
