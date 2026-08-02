import CartogramUtils from "../../../nonview/core/cartogram/CartogramUtils.js";
import { buildHexMapLayout } from "./HexMap.js";

afterEach(() => {
  jest.restoreAllMocks();
});

test("warps regions before assigning their proportional hexagons", () => {
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

  const layout = buildHexMapLayout(
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
  expect(layout.hexagons).toHaveLength(5);
  expect(layout.hexagons.every(({ regionId }) => regionId === "region-a")).toBe(
    true,
  );
  expect(layout).not.toHaveProperty("features");
});
