import ThingFactory from "./thing/thing_factory/ThingFactory.js";
import Thing from "./thing/Thing.js";
import {
  expandSubRegionDimensions,
  getQueryString,
  getThingFromToken,
  getThingValues,
  QUERY_DELIMITERS,
} from "./QueryTokens.js";

export async function parseQuery(QueryClass, queryString) {
  const tokens = queryString.split(QUERY_DELIMITERS.token).filter(Boolean);
  const entityClass = ThingFactory.fromKey(tokens[0]);
  const dimensions = [];
  const parentConstraints = [];
  for (const rawToken of tokens[1].split(QUERY_DELIMITERS.dimension)) {
    const token = rawToken.trim();
    const constraintIndex = token.indexOf("<");
    if (constraintIndex === -1) {
      dimensions.push(getThingFromToken(token));
      continue;
    }
    const ChildClass = ThingFactory.fromKey(token.slice(0, constraintIndex));
    const parentRegion = getThingFromToken(token.slice(constraintIndex + 1));
    const childValues = getThingValues(parentRegion)
      .flatMap((value) =>
        ChildClass.getChildRegions(
          parentRegion.constructor.fromValue(value),
          ChildClass,
        ),
      )
      .map((region) => region.value);
    dimensions.push(ChildClass.fromValue(Thing.WILDCARD));
    parentConstraints.push({
      childClass: ChildClass,
      childValues,
      parentRegion,
    });
  }
  const aggregate = tokens.at(-1);
  const subRegions = dimensions.filter((thing) =>
    thing.constructor.getParentRegionInfo?.(),
  );
  const expanded = await expandSubRegionDimensions(dimensions);
  const expandedString = getQueryString(entityClass, expanded, aggregate);
  return new QueryClass(
    entityClass,
    expanded,
    aggregate,
    parentConstraints.length ? expandedString : queryString,
    subRegions.length ? subRegions : null,
    parentConstraints.length ? parentConstraints : null,
  );
}

export function queryFromKeyValues(QueryClass, keyValues) {
  const entityClass = ThingFactory.fromKey(keyValues[0]);
  const dimensions = keyValues
    .slice(1, -1)
    .map((value) => ThingFactory.fromKeyValue(value));
  const aggregate = keyValues.at(-1);
  return new QueryClass(
    entityClass,
    dimensions,
    aggregate,
    getQueryString(entityClass, dimensions, aggregate),
  );
}
