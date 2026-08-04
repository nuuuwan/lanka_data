import ThingFactory from "../thing/thing_factory/ThingFactory.js";
import Thing from "../thing/Thing.js";

const DELIM_DIMENSION = "+";
const DELIM_EQUALS = "=";
const DELIM_VALUE = ",";

export function getThingValues(thing) {
  return thing.valueList || [thing.value];
}

export function getThingFromToken(token) {
  const delimiterIndex = token.search(/[:=]/);
  if (delimiterIndex === -1) {
    return ThingFactory.fromKeyValue(token);
  }
  const ThingClass = ThingFactory.fromKey(token.slice(0, delimiterIndex));
  const values = token
    .slice(delimiterIndex + 1)
    .split(DELIM_VALUE)
    .map((value) => ThingClass.fromValue(value).value);
  const thing = ThingClass.fromValue(values[0]);
  if (values.length > 1) {
    thing.valueList = values;
  }
  return thing;
}

export async function expandSubRegionDimThingList(dimThingList) {
  const expandedDimThingList = [];
  for (const dimThing of dimThingList) {
    const parentRegionInfo = dimThing.constructor.getParentRegionInfo?.();
    if (dimThing.value !== Thing.WILDCARD && parentRegionInfo) {
      const parentRegionClass = ThingFactory.fromKey(
        parentRegionInfo.parentClassName,
      );
      const parentRegionValues = [
        ...new Set(
          getThingValues(dimThing).map((value) => {
            const subRegion = dimThing.constructor.fromValue(value).getEnt();
            return parentRegionClass.fromRegionId(
              subRegion[parentRegionInfo.parentIdKey],
            ).value;
          }),
        ),
      ];
      const parentRegion = parentRegionClass.fromValue(parentRegionValues[0]);
      if (parentRegionValues.length > 1) {
        parentRegion.valueList = parentRegionValues;
      }
      expandedDimThingList.push(parentRegion);
    } else {
      expandedDimThingList.push(dimThing);
    }
  }
  return expandedDimThingList;
}

export function getQueryStringFromParts(entityClass, dimThingList, aggregate) {
  const dimensionToken = dimThingList
    .map((dimThing) => {
      if (dimThing.value === Thing.WILDCARD) {
        return dimThing.constructor.getClassName();
      }
      return [
        dimThing.constructor.getClassName(),
        DELIM_EQUALS,
        getThingValues(dimThing).join(DELIM_VALUE),
      ].join("");
    })
    .join(DELIM_DIMENSION);
  return [entityClass.getClassName(), dimensionToken, aggregate].join("/");
}
