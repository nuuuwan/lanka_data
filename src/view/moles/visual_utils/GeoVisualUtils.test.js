import { buildRegionLabels, getFittedLabelFontSize } from "./GeoVisualUtils.js";

test("fits label font size within the available width and height", () => {
  expect(getFittedLabelFontSize("Four", 24, 20)).toBe(10);
  expect(getFittedLabelFontSize("Four", 48, 5)).toBe(5);
});

test("builds a fitted label for each displayed region", () => {
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

  const [label] = buildRegionLabels(features, (coordinate) => coordinate);

  expect(label).toMatchObject({
    backgroundColor: "#ffffff",
    id: "EC-01",
    name: "Colombo North",
  });
  expect(label.angle).toEqual(expect.any(Number));
  expect(label.position[0]).toBeGreaterThan(10);
  expect(label.position[0]).toBeLessThan(30);
  expect(label.position[1]).toBeGreaterThan(20);
  expect(label.position[1]).toBeLessThan(40);
  expect(label.fontSize).toBeGreaterThan(0);
});

test("rotates a label to fit a narrow region", () => {
  const features = [
    {
      id: "vertical",
      fill: "#000000",
      properties: { name: "Tall Region" },
      geometry: {
        type: "Polygon",
        coordinates: [
          [
            [0, 0],
            [10, 0],
            [10, 100],
            [0, 100],
            [0, 0],
          ],
        ],
      },
    },
  ];

  const [label] = buildRegionLabels(features, (coordinate) => coordinate);

  expect(label.angle).toBe(90);
  expect(label.fontSize).toBeGreaterThan(7.5);
});
