import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
import { buildSquareMapLayout } from "./SquareMap.js";
import UnitSquareMap from "./UnitSquareMap.js";
import VisualFactory from "./VisualFactory.js";

afterEach(() => {
  jest.restoreAllMocks();
});

test("represents every region with exactly one square", () => {
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
  const regions = [
    {
      id: "region-a",
      feature,
      weight: 100,
      display: { color: "#123456", label: "Value", value: 100 },
    },
    {
      id: "region-b",
      feature: {
        ...feature,
        properties: { id: "region-b", name: "Region B" },
      },
      weight: 1,
      display: { color: "#654321", label: "Other", value: 1 },
    },
  ];

  const layout = buildSquareMapLayout(
    { facetKey: "", regions },
    undefined,
    true,
  );

  expect(layout.squares).toHaveLength(regions.length);
  expect(layout.squares.map(({ regionId }) => regionId).sort()).toEqual([
    "region-a",
    "region-b",
  ]);
  expect(computeSpy).not.toHaveBeenCalled();
});

test("registers UnitSquareMap as a non-chart visual", () => {
  expect(VisualFactory.get("UnitSquareMap")).toBe(UnitSquareMap);
  expect(UnitSquareMap.IS_CHART).toBe(false);
});
