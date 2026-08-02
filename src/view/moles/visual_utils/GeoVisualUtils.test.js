import { buildRegionLabels, getFittedLabelFontSize } from "./GeoVisualUtils.js";

test("fits label font size within the available width and height", () => {
  expect(getFittedLabelFontSize("Four", 24, 20)).toBe(10);
  expect(getFittedLabelFontSize("Four", 48, 5)).toBe(5);
});

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
      fontSize: 20 / (13 * 0.6),
      id: "EC-01",
      name: "Colombo North",
      position: [20, 30],
    },
  ]);
});
