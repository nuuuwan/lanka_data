import ThingFactory from "./thing/thing_factory/ThingFactory.js";
import Thing from "./thing/Thing.js";

export const QUERY_DELIMITERS = {
  token: "/",
  dimension: "+",
  equal: "=",
  value: ",",
};

export function getThingFromToken(token) {
  const delimIndex = token.search(/[:=]/);
  if (delimIndex === -1) return ThingFactory.fromKeyValue(token);
  const ThingClass = ThingFactory.fromKey(token.slice(0, delimIndex));
  const values = token
    .slice(delimIndex + 1)
    .split(QUERY_DELIMITERS.value)
    .map((value) => ThingClass.fromValue(value).value);
  const thing = ThingClass.fromValue(values[0]);
  if (values.length > 1) thing.valueList = values;
  return thing;
}

export function getThingValues(thing) {
  return thing.valueList || [thing.value];
}

export async function expandSubRegionDimensions(dimensions) {
  const expanded = [];
  for (const thing of dimensions) {
    const parentInfo = thing.constructor.getParentRegionInfo?.();
    if (thing.value !== Thing.WILDCARD && parentInfo) {
      const ParentClass = ThingFactory.fromKey(parentInfo.parentClassName);
      const values = [
        ...new Set(
          getThingValues(thing).map((value) => {
            const entity = thing.constructor.fromValue(value).getEnt();
            return ParentClass.fromRegionId(entity[parentInfo.parentIdKey])
              .value;
          }),
        ),
      ];
      const parent = ParentClass.fromValue(values[0]);
      if (values.length > 1) parent.valueList = values;
      expanded.push(parent);
    } else {
      expanded.push(thing);
    }
  }
  return expanded;
}

export function getQueryString(entityClass, dimensions, aggregate) {
  const dimensionToken = dimensions
    .map((thing) =>
      thing.value === Thing.WILDCARD
        ? thing.constructor.getClassName()
        : [
            thing.constructor.getClassName(),
            QUERY_DELIMITERS.equal,
            getThingValues(thing).join(QUERY_DELIMITERS.value),
          ].join(""),
    )
    .join(QUERY_DELIMITERS.dimension);
  return [entityClass.getClassName(), dimensionToken, aggregate].join(
    QUERY_DELIMITERS.token,
  );
}
