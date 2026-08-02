import Thing from "./Thing.js";
import ThingFactory from "./thing_factory/ThingFactory.js";

describe("Thing layer", () => {
  test("stores value as a string", () => {
    const thing = new Thing(42);
    expect(thing.value).toBe("42");
  });

  test("ThingFactory round-trips key/value pairs", () => {
    const testCases = [
      "Int:12345",
      "Religion:buddhist",
      "Sex:female",
      "Time:2012",
      "ElectionType:presidential",
    ];
    for (const keyValue of testCases) {
      const thing = ThingFactory.fromKeyValue(keyValue);
      expect(thing.toKeyValue()).toBe(keyValue);
    }
  });

  test("Int humanizes large numbers", () => {
    const int = ThingFactory.fromKeyValue("Int:12345");
    expect(int.getHumanReadableValue()).toBe("12K");
  });

  test("CategoryConcept fromValue normalizes aliases", () => {
    const sex = ThingFactory.fromKeyValue("Sex:both sexes");
    expect(sex.value).toBe("both_sexes");
  });

  test("colors are available for known concepts", () => {
    const religion = ThingFactory.fromKeyValue("Religion:buddhist");
    expect(religion.getColor()).toBe("#FFBE29");
  });
});
