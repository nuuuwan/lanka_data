import { buildRegionIdToWeight } from "./Cartogram.js";

test("weights a hyphenated feature by its total vote count", () => {
  const features = [{ properties: { id: "EC-06", name: "Nuwara-Eliya" } }];
  const dataMap = new Map([
    ["nuwara_eliya", [{ value: 99550 }, { value: 250428 }, { value: 5897 }]],
  ]);

  expect(buildRegionIdToWeight(features, dataMap)).toEqual({
    "EC-06": 355875,
  });
});
