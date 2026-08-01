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

    const [cellClassName, cellValue] = cellKeyValue.split(
      Query.DELIM_KEY_VALUE,
    );
    const CellClass = ThingFactory[cellClassName];
    if (!CellClass) {
      throw new Error(`CellClass "${cellClassName}" not found in ThingFactory`);
    }
    const cellThing = new CellClass(cellValue);

    return new Datum(query, cellThing);
  }
}
