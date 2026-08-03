import VisualMetadata from "./VisualMetadata.js";
import Int from "./thing/concept/atoms/Int.js";
import Percent from "./thing/concept/atoms/Percent.js";
import Time from "./thing/concept/atoms/Time.js";
import Religion from "./thing/concept/category_concept/Religion.js";
import District from "./thing/concept/category_concept/region/District.js";
import Province from "./thing/concept/category_concept/region/Province.js";
import Person from "./thing/entity/Person.js";

function getQuery({
  aggregate = "Count",
  dimensions,
  constraints = null,
  subRegions = null,
}) {
  return {
    aggregate,
    dimThingList: dimensions,
    entityClass: Person,
    parentRegionConstraintList: constraints,
    subRegionDimThingList: subRegions,
  };
}

test("describes measure, population, geography, time, units, and filters", () => {
  const time = Time.fromValue("2024");
  const district = new District("colombo");
  const religion = Religion.fromValue("buddhist");
  const query = getQuery({ dimensions: [time, district, religion] });
  const metadata = VisualMetadata.from(query, {
    datumList: [{ answerThing: new Int(100) }],
  });

  expect(metadata.title).toBe("Count of people");
  expect(metadata.subtitle).toBe(
    "Population: people • Geography: colombo district • Time period: 2024 • Units: people • Filters: time: 2024; district: colombo; religion: buddhist",
  );
});

test("states varying context and reports when there are no active filters", () => {
  const query = getQuery({
    aggregate: "Share",
    dimensions: [
      Time.fromValue("*"),
      Province.fromValue("*"),
      Religion.fromValue("*"),
    ],
  });
  const metadata = VisualMetadata.from(query, {
    datumList: [{ answerThing: new Percent(50) }],
  });

  expect(metadata.subtitle).toContain("Geography: all available province");
  expect(metadata.subtitle).toContain("Time period: all available periods");
  expect(metadata.subtitle).toContain("Units: percent");
  expect(metadata.subtitle).toContain("Filters: none");
});
