import Query from "../Query.js";
import AbstractDataSource from "./AbstractDataSource.js";

class TestDataSource extends AbstractDataSource {
  static async getMetadata() {
    return {
      "Vote/ElectionType+Time+Party/Count": ["votes.json"],
    };
  }
}

test("finds metadata regardless of dimension order", async () => {
  const query = await Query.fromString(
    "Vote/Time=2024+Party+ElectionType=presidential/Count",
  );

  await expect(TestDataSource.getMetadataForQuery(query)).resolves.toEqual([
    "votes.json",
  ]);
});
