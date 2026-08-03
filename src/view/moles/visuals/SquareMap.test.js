import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
import {
  areSquareCentersAdjacent,
  assignShapes,
  orderSquareCenters,
} from "../../../nonview/base/ShapeMapUtils.js";
import { buildSquareMapLayout, shareSquareMapScale } from "./SquareMap.js";

afterEach(() => {
  jest.restoreAllMocks();
});

test("warps regions before assigning their proportional squares", () => {
  const computeSpy = jest
    .spyOn(CartogramUtils, "compute")
    .mockImplementation(() => {});
  const feature = {
    type: "Feature",
    properties: { id: "region-a", name: "Region A" },
    geometry: {
      type: "Polygon",
      coordinates: [
        [
          [79, 7],
          [80, 7],
          [80, 8],
          [79, 8],
          [79, 7],
        ],
      ],
    },
  };

  const layout = buildSquareMapLayout(
    {
      facetKey: "",
      regions: [
        {
          id: "region-a",
          feature,
          weight: 100,
          display: { color: "#123456", label: "Value", value: 100 },
        },
      ],
    },
    20,
  );

  expect(computeSpy).toHaveBeenCalledWith(expect.any(Array), {
    "region-a": 100,
  });
  expect(computeSpy.mock.calls[0][0][0]).not.toBe(feature);
  expect(layout.squares).toHaveLength(5);
  expect(layout.squares.every(({ regionId }) => regionId === "region-a")).toBe(
    true,
  );
  expect(layout.squares.every(({ points }) => points.length === 4)).toBe(true);
});

test("shares the scale across faceted square maps", () => {
  const maps = shareSquareMapScale([
    { facetKey: "first", shapeValueMin: 10, shapeValueMax: 12 },
    { facetKey: "second", shapeValueMin: 20, shapeValueMax: 24 },
  ]);

  expect(maps).toEqual([
    { facetKey: "first", shapeValueMin: 10, shapeValueMax: 24 },
    { facetKey: "second", shapeValueMin: 10, shapeValueMax: 24 },
  ]);
});

test("assigns each region through edge-connected squares", () => {
  const size = 10;
  const centers = Array.from({ length: 3 }, (_, row) =>
    Array.from({ length: 3 }, (_, column) => [column * size, row * size]),
  ).flat();
  const assignments = assignShapes(
    [
      { id: "region-a", centroid: [10, 10], count: 4 },
      { id: "region-b", centroid: [0, 0], count: 3 },
    ],
    centers,
    orderSquareCenters,
  );

  for (const regionId of ["region-a", "region-b"]) {
    const regionCenters = assignments
      .filter(({ id }) => id === regionId)
      .map(({ center }) => center);
    const connectedCenters = [regionCenters[0]];
    while (connectedCenters.length < regionCenters.length) {
      const nextCenter = regionCenters.find(
        (center) =>
          !connectedCenters.includes(center) &&
          connectedCenters.some((connectedCenter) =>
            areSquareCentersAdjacent(connectedCenter, center, size),
          ),
      );
      expect(nextCenter).toBeDefined();
      connectedCenters.push(nextCenter);
    }
  }
});
