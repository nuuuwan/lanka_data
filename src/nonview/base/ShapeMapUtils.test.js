import {
  assignShapes,
  buildHexGrid,
  buildSquareGrid,
  getBestHexLabelFit,
  getBestSquareLabelFit,
  getHexBoundaryEdges,
  getHexPoints,
  getSquareBoundaryEdges,
  getSquarePoints,
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

test("caps pathological shape counts for interactive rendering", () => {
  const weights = [1_000_000, 1];
  const valuePerShape = getValuePerShape(weights, 400);
  const counts = getShapeCounts(
    { dominant: weights[0], smallest: weights[1] },
    valuePerShape,
  );

  expect(counts.dominant + counts.smallest).toBeLessThanOrEqual(400);
  expect(counts.smallest).toBe(1);
});

test("builds enough pointy-top hexagons for every shape", () => {
  const { centers, radius } = buildHexGrid([0, 0, 100, 100], 20);

  expect(centers.length).toBeGreaterThanOrEqual(20);
  expect(getHexPoints(centers[0], radius)).toHaveLength(6);
  expect(getHexPoints(centers[0], radius)[0][0]).toBeCloseTo(centers[0][0]);
});

test("builds enough squares for every shape", () => {
  const { centers, size } = buildSquareGrid([0, 0, 100, 100], 20);

  expect(centers.length).toBeGreaterThanOrEqual(20);
  expect(getSquarePoints(centers[0], size)).toHaveLength(4);
  expect(getSquarePoints(centers[0], size)[0][0]).toBeCloseTo(
    centers[0][0] - size / 2,
  );
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

  test("fits labels to the longest contiguous run of squares", () => {
    const size = 10;
    const fit = getBestSquareLabelFit(
      [
        [0, 0],
        [size, 0],
        [2 * size, 0],
      ],
      size,
    );

    expect(fit).toMatchObject({ center: [size, 0], angle: 0 });
    expect(fit.width).toBe(3 * size);
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

test("omits shared square edges inside a region boundary", () => {
  const size = 10;
  const shapes = [
    { id: "same-region", center: [0, 0] },
    { id: "same-region", center: [size, 0] },
  ];

  expect(getSquareBoundaryEdges(shapes, size)).toHaveLength(6);
});
