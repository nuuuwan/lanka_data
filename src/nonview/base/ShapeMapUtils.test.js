import {
  assignShapes,
  buildHexGrid,
  getHexPoints,
  getShapeCounts,
  getValuePerShape,
} from "./ShapeMapUtils.js";

test("finds proportional shape counts within the configured error", () => {
  const valuePerShape = getValuePerShape([100, 50]);

  expect(getShapeCounts({ large: 100, small: 50 }, valuePerShape)).toEqual({
    large: 2,
    small: 1,
  });
});

test("keeps regions visible when their weight is zero", () => {
  expect(getShapeCounts({ empty: 0 })).toEqual({ empty: 1 });
});

test("builds enough pointy-top hexagons for every shape", () => {
  const { centers, radius } = buildHexGrid([0, 0, 100, 100], 20);

  expect(centers.length).toBeGreaterThanOrEqual(20);
  expect(getHexPoints(centers[0], radius)).toHaveLength(6);
  expect(getHexPoints(centers[0], radius)[0][0]).toBeCloseTo(centers[0][0]);
});

test("minimizes total placement cost instead of assigning greedily", () => {
  const regions = [
    { id: "flexible", centroid: [0, 0], count: 1 },
    { id: "fixed", centroid: [1, 0], count: 1 },
  ];

  expect(
    assignShapes(regions, [
      [1, 0],
      [-10, 0],
    ]),
  ).toEqual([
    { id: "flexible", center: [-10, 0] },
    { id: "fixed", center: [1, 0] },
  ]);
});
