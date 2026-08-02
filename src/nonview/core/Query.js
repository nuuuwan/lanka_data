import ThingFactory from "./thing/thing_factory/ThingFactory.js";
import Thing from "./thing/Thing.js";

export default class Query {
  static DELIM_TOKEN = "/";
  static DELIM_DIM = "+";
  static DELIM_EQ = "=";

  constructor(entityClass, dimThingList, aggregate, queryStr) {
    this.entityClass = entityClass;
    this.dimThingList = dimThingList;
    this.aggregate = aggregate;
    this.queryStr = queryStr;
  }

  toString() {
    return this.queryStr;
  }

  isSubsetOf(otherQuery) {
    if (this.entityClass !== otherQuery.entityClass) {
      return false;
    }
    if (this.aggregate !== otherQuery.aggregate) {
      return false;
    }
    if (this.dimThingList.length !== otherQuery.dimThingList.length) {
      return false;
    }
    for (let i = 0; i < this.dimThingList.length; i++) {
      const thisDimThing = this.dimThingList[i];
      const otherDimThing = otherQuery.dimThingList[i];
      if (thisDimThing.constructor !== otherDimThing.constructor) {
        return false;
      }
      if (
        otherDimThing.value !== Thing.WILDCARD &&
        thisDimThing.value !== otherDimThing.value
      ) {
        return false;
      }
    }
    return true;
  }

  static fromString(queryStr) {
    const tokens = queryStr.split(Query.DELIM_TOKEN);
    const entityClassName = tokens[0];
    const entityClass = ThingFactory.fromKey(entityClassName);

    const dimToken = tokens[1];
    const dimThingList = dimToken.split(Query.DELIM_DIM).map((token) => {
      return ThingFactory.fromKeyValue(token);
    });
    const aggregate = tokens[tokens.length - 1];
    return new Query(entityClass, dimThingList, aggregate, queryStr);
  }

  static getQueryStringFromParts(entityClass, dimThingList, aggregate) {
    const entityClassName = entityClass.name;
    const dimInnerTokens = dimThingList.map((dimThing) => {
      if (dimThing.value === Thing.WILDCARD) {
        return dimThing.constructor.name;
      }
      return [dimThing.constructor.name, Query.DELIM_EQ, dimThing.value].join(
        "",
      );
    });
    const dimToken = dimInnerTokens.join(Query.DELIM_DIM);
    const aggregateToken = aggregate;
    return [entityClassName, dimToken, aggregateToken].join(Query.DELIM_TOKEN);
  }

  static fromKeyValueList(keyValueList) {
    const entityClass = ThingFactory.fromKey(keyValueList[0]);

    const dimInnerTokens = keyValueList.slice(1, -1);
    const dimThingList = dimInnerTokens.map((keyValue) => {
      return ThingFactory.fromKeyValue(keyValue);
    });

    const aggregate = keyValueList[keyValueList.length - 1];
    const queryStr = Query.getQueryStringFromParts(
      entityClass,
      dimThingList,
      aggregate,
    );
    return new Query(entityClass, dimThingList, aggregate, queryStr);
  }
}
