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
    parentRegionConstraintList = null,
  ) {
    this.entityClass = entityClass;
    this.dimThingList = dimThingList;
    this.aggregate = aggregate;
    this.queryStr = queryStr;
    this.subRegionDimThingList = subRegionDimThingList;
    this.parentRegionConstraintList = parentRegionConstraintList;
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
      if (otherDimThing.value === Thing.WILDCARD) {
        if (
          otherQuery.parentRegionConstraintList &&
          otherQuery.parentRegionConstraintList.length > 0
        ) {
          const constraint = otherQuery.parentRegionConstraintList.find(
            (c) => c.childClass === otherDimThing.constructor,
          );
          if (
            constraint &&
            !constraint.childValues.includes(thisDimThing.value)
          ) {
            return false;
          }
        }
        continue;
      }
      if (thisDimThing.value !== otherDimThing.value) {
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
    const dimThingList = [];
    const parentRegionConstraintList = [];
    for (const token of dimToken.split(Query.DELIM_DIM)) {
      const parentConstraintIndex = token.indexOf("<");
      if (parentConstraintIndex === -1) {
        dimThingList.push(ThingFactory.fromKeyValue(token));
        continue;
      }
      const childClassName = token.slice(0, parentConstraintIndex);
      const parentKeyValue = token.slice(parentConstraintIndex + 1);
      const ChildClass = ThingFactory.fromKey(childClassName);
      await ChildClass.init();
      const parentRegion = ThingFactory.fromKeyValue(parentKeyValue);
      const childRegions = ChildClass.getChildRegions(parentRegion, ChildClass);
      const childValues = childRegions.map((region) => region.value);
      dimThingList.push(ChildClass.fromValue(Thing.WILDCARD));
      parentRegionConstraintList.push({
        childClass: ChildClass,
        childValues,
      });
    }
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
      parentRegionConstraintList.length > 0 ? expandedQueryStr : queryStr,
      subRegionDimThingList.length > 0 ? subRegionDimThingList : null,
      parentRegionConstraintList.length > 0 ? parentRegionConstraintList : null,
    );
  }

  static async expandSubRegionDimThingList(dimThingList) {
    const expandedDimThingList = [];
    for (const dimThing of dimThingList) {
      if (
        dimThing.value !== Thing.WILDCARD &&
        dimThing.constructor.SUB_REGION_OF
      ) {
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

  getParentRegionFilter() {
    if (!this.parentRegionConstraintList) {
      return null;
    }
    return (datum) => {
      return this.parentRegionConstraintList.every((constraint) => {
        const dimIndex = datum.query.dimThingList.findIndex(
          (dimThing) => dimThing.constructor === constraint.childClass,
        );
        if (dimIndex === -1) {
          return false;
        }
        const dimThing = datum.query.dimThingList[dimIndex];
        return constraint.childValues.includes(dimThing.value);
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
