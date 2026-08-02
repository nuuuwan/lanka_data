import { groupDatumListByFacet } from "../visual_utils/GeoVisualUtils.js";
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

test("groups cartogram data into separate time facets", () => {
  const datumList = [2015, 2019].flatMap((year) =>
    ["colombo_north", "colombo_central"].map((region) => ({
      query: {
        dimThingList: [
          {
            value: String(year),
            getHumanReadableValue: () => `Time=${year}`,
          },
          { value: region },
          { value: "npp" },
        ],
      },
    })),
  );

  expect(groupDatumListByFacet(datumList, [0])).toEqual([
    {
      facetKey: "Time=2015",
      facetDatumList: datumList.slice(0, 2),
    },
    {
      facetKey: "Time=2019",
      facetDatumList: datumList.slice(2),
    },
  ]);
});
