import ThingFactory from "./thing/thing_factory/ThingFactory.js";

export default class Query {
  static DELIM_KEY_VALUE = ":";
  constructor(entityClass, dimThingList, aggregate) {
    this.entityClass = entityClass;
    this.dimThingList = dimThingList;
    this.aggregate = aggregate;
  }

  static fromKeyValueList(keyValueList) {
    const entityClassName = keyValueList[0];
    const entityClass = ThingFactory[entityClassName];
    if (!entityClass) {
      throw new Error(
        `Entity class "${entityClassName}" not found in ThingFactory`,
      );
    }

    const dimThingList = keyValueList.slice(1, -1).map((keyValue) => {
      const [dimClassName, dimValue] = keyValue.split(Query.DELIM_KEY_VALUE);
      const DimClass = ThingFactory[dimClassName];
      if (!DimClass) {
        throw new Error(`DimClass "${dimClassName}" not found in ThingFactory`);
      }
      return new DimClass(dimValue);
    });

    const aggregate = keyValueList[keyValueList.length - 1];

    return new Query(entityClass, dimThingList, aggregate);
  }
}
