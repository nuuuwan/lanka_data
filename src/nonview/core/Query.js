import ThingFactory from "./thing/thing_factory/ThingFactory.js";
import Thing from "./thing/Thing.js";

export default class Query {
  static DELIM_TOKEN = "/";
  static DELIM_DIM = "+";
  static DELIM_EQ = "=";

  constructor(
    entityClass,
    dimThingList,
    aggregate,
    queryStr,
    subRegionDimThingList = null,
  ) {
    this.entityClass = entityClass;
    this.dimThingList = dimThingList;
    this.aggregate = aggregate;
    this.queryStr = queryStr;
    this.subRegionDimThingList = subRegionDimThingList;
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

  static async fromString(queryStr) {
    const tokens = queryStr.split(Query.DELIM_TOKEN).filter(Boolean);
    const entityClassName = tokens[0];
    const entityClass = ThingFactory.fromKey(entityClassName);

    const dimToken = tokens[1];
    const dimThingList = dimToken.split(Query.DELIM_DIM).map((token) => {
      return ThingFactory.fromKeyValue(token);
    });
    const aggregate = tokens[tokens.length - 1];

    const subRegionDimThingList = dimThingList.filter(
      (dimThing) => dimThing.constructor.SUB_REGION_OF,
    );
    const expandedDimThingList =
      await Query.expandSubRegionDimThingList(dimThingList);
    const expandedQueryStr = Query.getQueryStringFromParts(
      entityClass,
      expandedDimThingList,
      aggregate,
    );

    return new Query(
      entityClass,
      expandedDimThingList,
      aggregate,
      subRegionDimThingList.length > 0 ? expandedQueryStr : queryStr,
      subRegionDimThingList.length > 0 ? subRegionDimThingList : null,
    );
  }

  static async expandSubRegionDimThingList(dimThingList) {
    const expandedDimThingList = [];
    for (const dimThing of dimThingList) {
      if (dimThing.constructor.SUB_REGION_OF) {
        const subRegionEnt = dimThing.getEnt();
        const parentRegionId =
          subRegionEnt[dimThing.constructor.SUB_REGION_ID_KEY];
        const parentRegionClass = dimThing.constructor.SUB_REGION_OF;
        await parentRegionClass.init();
        const parentRegion = parentRegionClass.fromRegionId(parentRegionId);
        expandedDimThingList.push(parentRegion);
        continue;
      }
      expandedDimThingList.push(dimThing);
    }
    return expandedDimThingList;
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

  getMetadataKey() {
    const dimToken = this.dimThingList
      .map((dimThing) => dimThing.constructor.name)
      .join(Query.DELIM_DIM);
    return [this.entityClass.name, dimToken, this.aggregate].join(
      Query.DELIM_TOKEN,
    );
  }

  static getMetadataKeyFromParts(entityClass, dimThingList, aggregate) {
    const dimToken = dimThingList
      .map((dimThing) => dimThing.constructor.name)
      .join(Query.DELIM_DIM);
    return [entityClass.name, dimToken, aggregate].join(Query.DELIM_TOKEN);
  }

  getSubRegionFilter() {
    if (!this.subRegionDimThingList) {
      return null;
    }
    return (datum) => {
      return this.subRegionDimThingList.every((subRegionThing) => {
        const parentRegionClass = subRegionThing.constructor.SUB_REGION_OF;
        const parentRegionDimIndex = datum.query.dimThingList.findIndex(
          (dimThing) => dimThing.constructor === parentRegionClass,
        );
        if (parentRegionDimIndex === -1) {
          return false;
        }
        const parentRegionThing =
          datum.query.dimThingList[parentRegionDimIndex];
        return (
          parentRegionThing.getEnt().id ===
          subRegionThing.getEnt()[subRegionThing.constructor.SUB_REGION_ID_KEY]
        );
      });
    };
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
