/**
 * @jest-environment jsdom
 */
import Query from "./Query.js";
import Time from "./thing/concept/atoms/Time.js";
import District from "./thing/concept/category_concept/region/District.js";
import Province from "./thing/concept/category_concept/region/Province.js";
import Region from "./thing/concept/category_concept/region/region/Region.js";
import Religion from "./thing/concept/category_concept/Religion.js";
import Person from "./thing/entity/Person.js";

describe("Query", () => {
  beforeAll(() => {
    Region.load({
      province: [
        { id: "LK-1", name: "Western" },
        { id: "LK-3", name: "Southern" },
      ],
      district: [
        { id: "LK-11", name: "Colombo", province_id: "LK-1" },
        { id: "LK-12", name: "Gampaha", province_id: "LK-1" },
      ],
    });
  });

  test("fromString parses a normal query", async () => {
    const query = await Query.fromString(
      "Person/Time=2024+Province+Religion/Count",
    );
    expect(query.entityClass).toBe(Person);
    expect(query.aggregate).toBe("Count");
    expect(query.dimThingList.length).toBe(3);
    expect(query.dimThingList[0]).toBeInstanceOf(Time);
    expect(query.dimThingList[1]).toBeInstanceOf(Province);
    expect(query.dimThingList[2]).toBeInstanceOf(Religion);
    expect(query.subRegionDimThingList).toBeNull();
  });

  test("fromString expands a sub-region syntax into its parent region", async () => {
    const query = await Query.fromString(
      "Person/Time=2024+District=Colombo+Religion/Count",
    );
    expect(query.entityClass).toBe(Person);
    expect(query.aggregate).toBe("Count");
    expect(query.dimThingList.length).toBe(3);
    expect(query.dimThingList[0]).toBeInstanceOf(Time);
    expect(query.dimThingList[1]).toBeInstanceOf(Province);
    expect(query.dimThingList[1].value).toBe("western");
    expect(query.dimThingList[2]).toBeInstanceOf(Religion);
    expect(query.subRegionDimThingList).toHaveLength(1);
    expect(query.subRegionDimThingList[0]).toBeInstanceOf(District);
    expect(query.subRegionDimThingList[0].value).toBe("colombo");
  });

  test("getMetadataKey uses the expanded parent region class", async () => {
    const query = await Query.fromString(
      "Person/Time=2024+District=Colombo+Religion/Count",
    );
    expect(query.getMetadataKey()).toBe("Person/Time+Province+Religion/Count");
  });

  test("getSubRegionFilter is null for non-sub-region queries", async () => {
    const query = await Query.fromString(
      "Person/Time=2024+Province+Religion/Count",
    );
    expect(query.getSubRegionFilter()).toBeNull();
  });
});
