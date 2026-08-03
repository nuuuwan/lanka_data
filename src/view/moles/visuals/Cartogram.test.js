import { getGlobalAreaProjectionScales } from "./Cartogram.js";

test("scales cartogram area in proportion to facet totals", () => {
  const scales = getGlobalAreaProjectionScales([
    { projectionScale: 100, total: 50 },
    { projectionScale: 100, total: 100 },
  ]);

  expect(scales[0] ** 2 / scales[1] ** 2).toBeCloseTo(0.5);
});

test("uses one global scale while keeping every cartogram fitted", () => {
  const cartograms = [
    { projectionScale: 80, total: 25 },
    { projectionScale: 100, total: 100 },
  ];
  const scales = getGlobalAreaProjectionScales(cartograms);

  expect(scales[0] ** 2 / scales[1] ** 2).toBeCloseTo(0.25);
  scales.forEach((scale, index) => {
    expect(scale).toBeLessThanOrEqual(cartograms[index].projectionScale);
  });
});

test("returns zero scales when every facet total is zero", () => {
  expect(
    getGlobalAreaProjectionScales([
      { projectionScale: 80, total: 0 },
      { projectionScale: 100, total: 0 },
    ]),
  ).toEqual([0, 0]);
});
