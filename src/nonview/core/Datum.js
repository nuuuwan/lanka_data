import ThingFactory from "./thing/ThingFactory.js";

export default class Datum {
  constructor(entityClass, dimThingList, aggregate, cellThing) {
    this.entityClass = entityClass;
    this.dimThingList = dimThingList;
    this.aggregate = aggregate;
    this.cellThing = cellThing;
  }

  static fromShallowDictEntry(entry) {
    const [keyValueList, cellKeyValue] = entry;
    const entityClassName = keyValueList[0];
    const entityClass = ThingFactory[entityClassName];
    if (!entityClass) {
      throw new Error(
        `Entity class "${entityClassName}" not found in ThingFactory`,
      );
    }

    const DELIM_KEY_VALUE = ":";
    const dimThingList = keyValueList.slice(1, -1).map((keyValue) => {
      const [dimClassName, dimValue] = keyValue.split(DELIM_KEY_VALUE);
      const DimClass = ThingFactory[dimClassName];
      if (!DimClass) {
        throw new Error(`DimClass "${dimClassName}" not found in ThingFactory`);
      }
      return new DimClass(dimValue);
    });

    const aggregate = keyValueList[keyValueList.length - 1];

    const [cellClassName, cellValue] = cellKeyValue.split(DELIM_KEY_VALUE);
    const CellClass = ThingFactory[cellClassName];
    if (!CellClass) {
      throw new Error(`CellClass "${cellClassName}" not found in ThingFactory`);
    }
    const cellThing = new CellClass(cellValue);
    return new Datum(entityClass, dimThingList, aggregate, cellThing);
  }
}
