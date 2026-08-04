import ThingFactory from "../thing/thing_factory/ThingFactory.js";
import Thing from "../thing/Thing.js";
import {
  expandSubRegionDimThingList,
  getQueryStringFromParts,
  getThingFromToken,
  getThingValues,
} from "./QueryTokenUtils.js";

export async function parseQueryString(queryString) {
  const tokens = queryString.split("/").filter(Boolean);
  const entityClass = ThingFactory.fromKey(tokens[0]);
  const dimThingList = [];
  const parentRegionConstraintList = [];
  for (const rawToken of tokens[1].split("+")) {
    const token = rawToken.trim();
    const parentConstraintIndex = token.indexOf("<");
    if (parentConstraintIndex === -1) {
      dimThingList.push(getThingFromToken(token));
      continue;
    }
    const ChildClass = ThingFactory.fromKey(
      token.slice(0, parentConstraintIndex),
    );
    const parentRegion = getThingFromToken(
      token.slice(parentConstraintIndex + 1),
    );
    const childValues = getThingValues(parentRegion).flatMap((value) =>
      ChildClass.getChildRegions(
        parentRegion.constructor.fromValue(value),
        ChildClass,
      ).map((region) => region.value),
    );
    dimThingList.push(ChildClass.fromValue(Thing.WILDCARD));
    parentRegionConstraintList.push({
      childClass: ChildClass,
      childValues,
      parentRegion,
    });
  }
  const aggregate = tokens.at(-1);
  const subRegionDimThingList = dimThingList.filter((dimThing) =>
    dimThing.constructor.getParentRegionInfo?.(),
  );
  const expandedDimThingList = await expandSubRegionDimThingList(dimThingList);
  const expandedQueryString = getQueryStringFromParts(
    entityClass,
    expandedDimThingList,
    aggregate,
  );
  return {
    entityClass,
    dimThingList: expandedDimThingList,
    aggregate,
    queryString: parentRegionConstraintList.length
      ? expandedQueryString
      : queryString,
    subRegionDimThingList: subRegionDimThingList.length
      ? subRegionDimThingList
      : null,
    parentRegionConstraintList: parentRegionConstraintList.length
      ? parentRegionConstraintList
      : null,
  };
}

export function parseKeyValueList(keyValueList) {
  const entityClass = ThingFactory.fromKey(keyValueList[0]);
  const dimThingList = keyValueList
    .slice(1, -1)
    .map((keyValue) => ThingFactory.fromKeyValue(keyValue));
  const aggregate = keyValueList.at(-1);
  return {
    entityClass,
    dimThingList,
    aggregate,
    queryString: getQueryStringFromParts(entityClass, dimThingList, aggregate),
  };
}
