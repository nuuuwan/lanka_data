import { buildRegionLabels } from "./GeoVisualUtils.js";

test("builds a centered label for each displayed region", () => {
  const features = [
    {
      id: "EC-01",
      fill: "#ffffff",
      properties: { name: "Colombo North" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [10, 20],
            [30, 20],
            [30, 40],
            [10, 40],
            [10, 20],
          ],
        ],
      },
    },
  ];

  expect(buildRegionLabels(features, (coordinate) => coordinate)).toEqual([
    {
      backgroundColor: "#ffffff",
      id: "EC-01",
      name: "Colombo North",
      position: [20, 30],
    },
  ]);
});
