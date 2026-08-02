import Query from "./Query.js";

describe("Query", () => {
  test("parses wildcard query string", () => {
    const query = Query.fromString("Person/Time+Religion+Sex/Count");
    expect(query.entityClass.name).toBe("Person");
    expect(query.dimThingList.length).toBe(3);
    expect(query.dimThingList[0].constructor.name).toBe("Time");
    expect(query.dimThingList[0].value).toBe("*");
    expect(query.aggregate).toBe("Count");
  });

  test("parses query string with explicit values", () => {
    const query = Query.fromString("Person/Time=2012+Religion+Sex/Count");
    expect(query.entityClass.name).toBe("Person");
    expect(query.dimThingList[0].value).toBe("2012");
    expect(query.dimThingList[1].value).toBe("*");
    expect(query.dimThingList[2].value).toBe("*");
    expect(query.aggregate).toBe("Count");
  });

  test("round-trips query string from parts", () => {
    const query = Query.fromString("Person/Time=2012+Religion+Sex/Count");
    const rebuilt = Query.getQueryStringFromParts(
      query.entityClass,
      query.dimThingList,
      query.aggregate,
    );
    expect(rebuilt).toBe("Person/Time=2012+Religion+Sex/Count");
  });

  test("fromKeyValueList builds query with explicit values", () => {
    const keyValueList = [
      "Person",
      "Time:2012",
      "Religion:buddhist",
      "Sex:female",
      "Count",
    ];
    const query = Query.fromKeyValueList(keyValueList);
    expect(query.toString()).toBe(
      "Person/Time=2012+Religion=buddhist+Sex=female/Count",
    );
  });

  test("isSubsetOf matches explicit query values", () => {
    const dataQuery = Query.fromKeyValueList([
      "Person",
      "Time:2012",
      "Religion:buddhist",
      "Sex:female",
      "Count",
    ]);
    const userQuery = Query.fromString("Person/Time=2012+Religion+Sex/Count");
    expect(dataQuery.isSubsetOf(userQuery)).toBe(true);
  });

  test("isSubsetOf rejects mismatched explicit values", () => {
    const dataQuery = Query.fromKeyValueList([
      "Person",
      "Time:2012",
      "Religion:buddhist",
      "Sex:female",
      "Count",
    ]);
    const userQuery = Query.fromString("Person/Time=2011+Religion+Sex/Count");
    expect(dataQuery.isSubsetOf(userQuery)).toBe(false);
  });
});
