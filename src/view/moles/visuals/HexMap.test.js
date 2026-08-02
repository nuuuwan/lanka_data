import { getHexagonCount } from "./HexMap.js";

test("scales hexagon counts to the largest region", () => {
  expect(getHexagonCount(100, 100)).toBe(80);
  expect(getHexagonCount(50, 100)).toBe(40);
  expect(getHexagonCount(0, 100)).toBe(0);
});

test("keeps a non-zero region visible", () => {
  expect(getHexagonCount(1, 100)).toBe(1);
});
