import DataSourceFactory from "./DataSourceFactory.js";

test("infers query choices from data metadata", () => {
  const queryOptions = DataSourceFactory.getQueryOptionsFromMetadata([
    {
      "Vote/Time+Party/Count": ["votes.json"],
      "Person/Time+Province+Religion/Count": ["people.json"],
    },
    {
      "House/Time+District+CookingFuel/Count": ["houses.json"],
      "Person/Time+District+Religion/Count": ["more-people.json"],
    },
  ]);

  expect(queryOptions.entities).toEqual(["House", "Person", "Vote"]);
  expect(queryOptions.dimensionsByEntity.Person).toEqual([
    "District",
    "Province",
    "Religion",
    "Time",
  ]);
});
