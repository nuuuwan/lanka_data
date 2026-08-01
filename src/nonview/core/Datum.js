import ThingFactory from "./thing/thing_factory/ThingFactory.js";

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
}
