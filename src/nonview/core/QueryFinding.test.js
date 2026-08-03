import getQueryFinding from "./QueryFinding.js";
import Thing from "./thing/Thing.js";

function makeThing(className, value, label = value) {
  return {
    value,
    getLabel: () => label,
    constructor: {
      getClassName: () => className,
      fromValue: (nextValue) =>
        makeThing(className, nextValue, nextValue === value ? label : nextValue),
    },
  };
}

test("describes query groupings and constraints in plain language", () => {
  const query = {
    aggregate: "Count",
    entityClass: { getClassName: () => "Person" },
    dimThingList: [
      makeThing("Time", "2024"),
      makeThing("District", "colombo", "Colombo"),
      makeThing("Religion", Thing.WILDCARD),
    ],
  };

  expect(getQueryFinding(query)).toBe(
    "Count of people by religion in 2024 and in Colombo district",
  );
});

test("humanizes unfamiliar entity and dimension names", () => {
  const query = {
    aggregate: "AverageValue",
    entityClass: { getClassName: () => "CensusOfficer" },
    dimThingList: [makeThing("HighestEducationLevel3", Thing.WILDCARD)],
  };

  expect(getQueryFinding(query)).toBe(
    "Average value of census officers by highest education level 3",
  );
});
