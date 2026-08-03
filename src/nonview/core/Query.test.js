import Query from "./Query.js";

test("matches comma-separated dimension values", async () => {
  const query = await Query.fromString(
    "Vote/ElectionType=presidential,parliamentary+Time=2005,2015+Party/Count",
  );

  const matchingQuery = Query.fromKeyValueList([
    "Vote",
    "ElectionType:parliamentary",
    "Time:2015",
    "Party:united_national_party",
    "Count",
  ]);
  const wrongElectionTypeQuery = Query.fromKeyValueList([
    "Vote",
    "ElectionType:local_government",
    "Time:2015",
    "Party:united_national_party",
    "Count",
  ]);
  const wrongTimeQuery = Query.fromKeyValueList([
    "Vote",
    "ElectionType:presidential",
    "Time:2024",
    "Party:united_national_party",
    "Count",
  ]);

  expect(matchingQuery.isSubsetOf(query)).toBe(true);
  expect(wrongElectionTypeQuery.isSubsetOf(query)).toBe(false);
  expect(wrongTimeQuery.isSubsetOf(query)).toBe(false);
  expect(query.getMetadataKey()).toBe("Vote/ElectionType+Time+Party/Count");
});

test("preserves existing single-value matching", async () => {
  const query = await Query.fromString(
    "Vote/ElectionType=presidential+Time+Party/Count",
  );
  const matchingQuery = Query.fromKeyValueList([
    "Vote",
    "ElectionType:presidential",
    "Time:2024",
    "Party:united_national_party",
    "Count",
  ]);

  expect(matchingQuery.isSubsetOf(query)).toBe(true);
});

test("matches dimensions regardless of order", async () => {
  const query = await Query.fromString(
    "Vote/Time=2024+PD<ED=colombo+Party+ ElectionType=presidential/Count",
  );
  const matchingQuery = Query.fromKeyValueList([
    "Vote",
    "ElectionType:presidential",
    "Time:2024",
    "PD:colombo_north",
    "Party:united_national_party",
    "Count",
  ]);

  expect(matchingQuery.isSubsetOf(query)).toBe(true);
  expect(Query.normalizeMetadataKey(query.getMetadataKey())).toBe(
    Query.normalizeMetadataKey("Vote/ElectionType+Time+PD+Party/Count"),
  );
});
