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

test("adds provenance for the single source that returns data", async () => {
  const datum = { id: "election-result" };
  class ElectionDataSource {
    static async getDatumListForQuery() {
      return [datum];
    }

    static getProvenanceForQuery() {
      return {
        source: "The Elections Commission of Sri Lanka",
        title: "Sri Lanka election results",
        url: "https://elections.gov.lk",
      };
    }
  }
  class EmptyDataSource {
    static async getDatumListForQuery() {
      return [];
    }

    static getProvenanceForQuery() {
      return { source: "Unused source" };
    }
  }
  jest
    .spyOn(DataSourceFactory, "getDataSourceClasses")
    .mockReturnValue([ElectionDataSource, EmptyDataSource]);

  const datumSet = await DataSourceFactory.getDatumSetForQuery("query");

  expect(datumSet.datumList).toEqual([datum]);
  expect(datumSet.provenance).toEqual([
    {
      source: "The Elections Commission of Sri Lanka",
      title: "Sri Lanka election results",
      url: "https://elections.gov.lk",
    },
  ]);
});

test("keeps mixed-source datum results and metadata in source order", async () => {
  const firstDatum = { id: "2012" };
  const secondDatum = { id: "2024" };
  const createDataSource = (datum, title) =>
    class {
      static async getDatumListForQuery() {
        return [datum];
      }

      static getProvenanceForQuery() {
        return {
          source: "Department of Census and Statistics of Sri Lanka",
          title,
          url: "https://www.statistics.gov.lk",
        };
      }
    };
  jest
    .spyOn(DataSourceFactory, "getDataSourceClasses")
    .mockReturnValue([
      createDataSource(firstDatum, "Census 2012"),
      createDataSource(secondDatum, "Census 2024"),
    ]);

  const datumSet = await DataSourceFactory.getDatumSetForQuery("query");

  expect(datumSet.datumList).toEqual([firstDatum, secondDatum]);
  expect(datumSet.provenance.map(({ title }) => title)).toEqual([
    "Census 2012",
    "Census 2024",
  ]);
});
