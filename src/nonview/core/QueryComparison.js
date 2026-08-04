import Thing from "./thing/Thing.js";
import { getThingValues } from "./QueryTokens.js";

export function isQuerySubset(query, otherQuery) {
  if (
    query.entityClass !== otherQuery.entityClass ||
    query.aggregate !== otherQuery.aggregate ||
    query.dimThingList.length !== otherQuery.dimThingList.length
  ) {
    return false;
  }
  const otherByClass = new Map(
    otherQuery.dimThingList.map((thing) => [thing.constructor, thing]),
  );
  return query.dimThingList.every((thing) => {
    const other = otherByClass.get(thing.constructor);
    if (!other) return false;
    if (other.value === Thing.WILDCARD) {
      const constraint = otherQuery.parentRegionConstraintList?.find(
        (item) => item.childClass === other.constructor,
      );
      return !constraint || constraint.childValues.includes(thing.value);
    }
    const otherValues = getThingValues(other);
    return getThingValues(thing).every((value) => otherValues.includes(value));
  });
}
