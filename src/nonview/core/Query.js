import ThingFactory from "./thing/thing_factory/ThingFactory.js";
import Thing from "./thing/Thing.js";

export default class Query {
  static DELIM_TOKEN = "/";
  static DELIM_DIM = "+";
  static DELIM_EQ = "=";
  static DELIM_VALUE = ",";

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
    const otherDimThingByClass = new Map(
      otherQuery.dimThingList.map((dimThing) => [
        dimThing.constructor,
        dimThing,
      ]),
    );
    for (const thisDimThing of this.dimThingList) {
      const otherDimThing = otherDimThingByClass.get(thisDimThing.constructor);
      if (!otherDimThing) {
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
      const thisValues = Query.getThingValues(thisDimThing);
      const otherValues = Query.getThingValues(otherDimThing);
      if (!thisValues.every((value) => otherValues.includes(value))) {
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
    for (const rawToken of dimToken.split(Query.DELIM_DIM)) {
      const token = rawToken.trim();
      const parentConstraintIndex = token.indexOf("<");
      if (parentConstraintIndex === -1) {
        dimThingList.push(Query.getThingFromToken(token));
        continue;
      }
      const childClassName = token.slice(0, parentConstraintIndex);
      const parentKeyValue = token.slice(parentConstraintIndex + 1);
      const ChildClass = ThingFactory.fromKey(childClassName);
      const parentRegion = Query.getThingFromToken(parentKeyValue);
      const childRegions = Query.getThingValues(parentRegion).flatMap((value) =>
        ChildClass.getChildRegions(
          parentRegion.constructor.fromValue(value),
          ChildClass,
        ),
      );
      const childValues = childRegions.map((region) => region.value);
      dimThingList.push(ChildClass.fromValue(Thing.WILDCARD));
      parentRegionConstraintList.push({
        childClass: ChildClass,
        childValues,
      });
    }
    const aggregate = tokens[tokens.length - 1];

    const subRegionDimThingList = dimThingList.filter((dimThing) =>
      dimThing.constructor.getParentRegionInfo?.(),
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

  static getThingFromToken(token) {
    const delimIndex = token.search(/[:=]/);
    if (delimIndex === -1) {
      return ThingFactory.fromKeyValue(token);
    }
    const ThingClass = ThingFactory.fromKey(token.slice(0, delimIndex));
    const values = token
      .slice(delimIndex + 1)
      .split(Query.DELIM_VALUE)
      .map((value) => ThingClass.fromValue(value).value);
    const thing = ThingClass.fromValue(values[0]);
    if (values.length > 1) {
      thing.valueList = values;
    }
    return thing;
  }

  static getThingValues(thing) {
    return thing.valueList || [thing.value];
  }

  static async expandSubRegionDimThingList(dimThingList) {
    const expandedDimThingList = [];
    for (const dimThing of dimThingList) {
      const parentRegionInfo = dimThing.constructor.getParentRegionInfo?.();
      if (dimThing.value !== Thing.WILDCARD && parentRegionInfo) {
        const parentRegionClass = ThingFactory.fromKey(
          parentRegionInfo.parentClassName,
        );
        const parentRegionValues = [
          ...new Set(
            Query.getThingValues(dimThing).map((value) => {
              const subRegionEnt = dimThing.constructor
                .fromValue(value)
                .getEnt();
              const parentRegionId = subRegionEnt[parentRegionInfo.parentIdKey];
              return parentRegionClass.fromRegionId(parentRegionId).value;
            }),
          ),
        ];
        const parentRegion = parentRegionClass.fromValue(parentRegionValues[0]);
        if (parentRegionValues.length > 1) {
          parentRegion.valueList = parentRegionValues;
        }
        expandedDimThingList.push(parentRegion);
        continue;
      }
      expandedDimThingList.push(dimThing);
    }
    return expandedDimThingList;
  }

  static getQueryStringFromParts(entityClass, dimThingList, aggregate) {
    const entityClassName = entityClass.getClassName();
    const dimInnerTokens = dimThingList.map((dimThing) => {
      if (dimThing.value === Thing.WILDCARD) {
        return dimThing.constructor.getClassName();
      }
      return [
        dimThing.constructor.getClassName(),
        Query.DELIM_EQ,
        Query.getThingValues(dimThing).join(Query.DELIM_VALUE),
      ].join("");
    });
    const dimToken = dimInnerTokens.join(Query.DELIM_DIM);
    const aggregateToken = aggregate;
    return [entityClassName, dimToken, aggregateToken].join(Query.DELIM_TOKEN);
  }

  getMetadataKey() {
    const dimToken = this.dimThingList
      .map((dimThing) => dimThing.constructor.getClassName())
      .join(Query.DELIM_DIM);
    return [this.entityClass.getClassName(), dimToken, this.aggregate].join(
      Query.DELIM_TOKEN,
    );
  }

  static getMetadataKeyFromParts(entityClass, dimThingList, aggregate) {
    const dimToken = dimThingList
      .map((dimThing) => dimThing.constructor.getClassName())
      .join(Query.DELIM_DIM);
    return [entityClass.getClassName(), dimToken, aggregate].join(
      Query.DELIM_TOKEN,
    );
  }

  static normalizeMetadataKey(metadataKey) {
    const [entityClassName, dimToken, aggregate] = metadataKey.split(
      Query.DELIM_TOKEN,
    );
    const normalizedDimToken = dimToken
      .split(Query.DELIM_DIM)
      .filter(Boolean)
      .sort()
      .join(Query.DELIM_DIM);
    return [entityClassName, normalizedDimToken, aggregate].join(
      Query.DELIM_TOKEN,
    );
  }

  getSubRegionFilter() {
    if (!this.subRegionDimThingList) {
      return null;
    }
    return (datum) => {
      return this.subRegionDimThingList.every((subRegionThing) => {
        const parentRegionInfo =
          subRegionThing.constructor.getParentRegionInfo();
        const parentRegionClass = ThingFactory.fromKey(
          parentRegionInfo.parentClassName,
        );
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
          subRegionThing.getEnt()[parentRegionInfo.parentIdKey]
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
