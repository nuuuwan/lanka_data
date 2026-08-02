import {
  assignShapes,
  buildHexGrid,
  getBestHexLabelFit,
  getHexBoundaryEdges,
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

test("fits labels to the longest contiguous run of hexagons", () => {
  const radius = 10;
  const points = [
    [0, 0],
    [Math.sqrt(3) * radius, 0],
    [2 * Math.sqrt(3) * radius, 0],
  ];

  const fit = getBestHexLabelFit(points, radius);

  expect(fit).toMatchObject({
    center: [Math.sqrt(3) * radius, 0],
    angle: 0,
  });
  expect(fit.width).toBeCloseTo(3 * Math.sqrt(3) * radius);
});

test("omits shared edges inside a region boundary", () => {
  const radius = 10;
  const shapes = [
    { id: "same-region", center: [0, 0] },
    { id: "same-region", center: [Math.sqrt(3) * radius, 0] },
  ];

  expect(getHexBoundaryEdges(shapes, radius)).toHaveLength(10);
});
