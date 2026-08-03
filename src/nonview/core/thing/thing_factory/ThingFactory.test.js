import RegionFactory from "../concept/category_concept/region/RegionFactory.js";
import ThingFactory from "./ThingFactory.js";

test("provides stable class names for production serialization", () => {
  expect(ThingFactory.Vote.getClassName()).toBe("Vote");
  expect(ThingFactory.ElectionType.getClassName()).toBe("ElectionType");
  expect(ThingFactory.PD.getClassName()).toBe("PD");
});

test("builds stable region identifiers and URLs", () => {
  expect(
    RegionFactory.list().map((RegionClass) => RegionClass.regionClassId()),
  ).toEqual(["country", "province", "district", "dsd", "ed", "pd"]);
  expect(ThingFactory.PD.getGeoURL()).toMatch(/\/pds\.topojson$/);
});
