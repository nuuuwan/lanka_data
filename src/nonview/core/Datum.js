import ThingFactory from "./thing/thing_factory/ThingFactory.js";
import ShallowDict from "../base/ShallowDict.js";
import Query from "./Query.js";

export default class Datum {
  constructor(query, answerThing) {
    this.query = query;
    this.answerThing = answerThing;
  }

  static fromShallowDictEntry(entry) {
    const [keyValueList, cellKeyValue] = entry;

    const query = Query.fromKeyValueList(keyValueList);

    const cellThing = ThingFactory.fromKeyValue(cellKeyValue);

    return new Datum(query, cellThing);
  }

  static listFromLankaData(lankaData) {
    const shallowDict = ShallowDict.fromDeep(lankaData);
    return Array.from(shallowDict.entries()).map((entry) =>
      Datum.fromShallowDictEntry(entry),
    );
  }
}
