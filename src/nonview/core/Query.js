import ThingFactory from "./thing/thing_factory/ThingFactory.js";
import KeyValue from "./KeyValue.js";

export default class Query {
  constructor(entityClass, dimThingList, aggregate) {
    this.entityClass = entityClass;
    this.dimThingList = dimThingList;
    this.aggregate = aggregate;
  }

  static fromKeyValueList(keyValueList) {
    const entityClass = ThingFactory.fromKey(keyValueList[0]);

    const dimThingList = keyValueList.slice(1, -1).map((keyValue) => {
      return ThingFactory.fromKeyValue(keyValue);
    });

    const aggregate = keyValueList[keyValueList.length - 1];
    return new Query(entityClass, dimThingList, aggregate);
  }
}
