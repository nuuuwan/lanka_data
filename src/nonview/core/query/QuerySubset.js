import Thing from "../thing/Thing.js";
import { getThingValues } from "./QueryTokenUtils.js";

export default function isQuerySubset(query, otherQuery) {
  if (
    query.entityClass !== otherQuery.entityClass ||
    query.aggregate !== otherQuery.aggregate ||
    query.dimThingList.length !== otherQuery.dimThingList.length
  ) {
    return false;
  }
  const otherDimensions = new Map(
    otherQuery.dimThingList.map((dimThing) => [dimThing.constructor, dimThing]),
  );
  return query.dimThingList.every((dimThing) => {
    const otherDimThing = otherDimensions.get(dimThing.constructor);
    if (!otherDimThing) {
      return false;
    }
    if (otherDimThing.value === Thing.WILDCARD) {
      const constraint = otherQuery.parentRegionConstraintList?.find(
        (item) => item.childClass === otherDimThing.constructor,
      );
      return !constraint || constraint.childValues.includes(dimThing.value);
    }
    const otherValues = getThingValues(otherDimThing);
    return getThingValues(dimThing).every((value) =>
      otherValues.includes(value),
    );
  });
}
