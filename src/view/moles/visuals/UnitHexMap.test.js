import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
import { buildHexMapLayout } from "./HexMap.js";
import UnitHexMap from "./UnitHexMap.js";
import VisualFactory from "./VisualFactory.js";

afterEach(() => {
  jest.restoreAllMocks();
});

test("represents every region with exactly one hexagon", () => {
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

  const layout = buildHexMapLayout({ facetKey: "", regions }, undefined, true);

  expect(layout.hexagons).toHaveLength(regions.length);
  expect(layout.hexagons.map(({ regionId }) => regionId).sort()).toEqual([
    "region-a",
    "region-b",
  ]);
  expect(computeSpy).not.toHaveBeenCalled();
});

test("registers UnitHexMap as a non-chart visual", () => {
  expect(VisualFactory.get("UnitHexMap")).toBe(UnitHexMap);
  expect(UnitHexMap.IS_CHART).toBe(false);
});
