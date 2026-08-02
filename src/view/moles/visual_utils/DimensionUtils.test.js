import Time from "../../../nonview/core/thing/concept/atoms/Time.js";
import DimensionUtils from "./DimensionUtils.js";

function createDatum(year) {
  return { query: { dimThingList: [Time.fromValue(year)] } };
}

test("sorts time facets chronologically instead of by total", () => {
  const datumList = [createDatum(2019), createDatum(2015), createDatum(2024)];
  const facets = [
    { facetKey: "Time=2019", total: 30 },
    { facetKey: "Time=2015", total: 20 },
    { facetKey: "Time=2024", total: 10 },
  ];

  expect(
    DimensionUtils.sortFacets(
      facets,
      datumList,
      [0],
      (a, b) => b.total - a.total,
    ).map(({ facetKey }) => facetKey),
  ).toEqual(["Time=2015", "Time=2019", "Time=2024"]);
});

test("sorts time-axis data chronologically", () => {
  const datumList = [createDatum(2019), createDatum(2015), createDatum(2024)];
  const data = [{ id: "2019" }, { id: "2015" }, { id: "2024" }];

  expect(
    DimensionUtils.sortDataByTime(data, datumList, 0).map(({ id }) => id),
  ).toEqual(["2015", "2019", "2024"]);
});
