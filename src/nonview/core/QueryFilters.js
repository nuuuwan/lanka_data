import ThingFactory from "./thing/thing_factory/ThingFactory.js";

export function getSubRegionFilter(query) {
  if (!query.subRegionDimThingList) return null;
  return (datum) =>
    query.subRegionDimThingList.every((subRegionThing) => {
      const parentInfo = subRegionThing.constructor.getParentRegionInfo();
      const ParentClass = ThingFactory.fromKey(parentInfo.parentClassName);
      const parentIndex = datum.query.dimThingList.findIndex(
        (thing) => thing.constructor === ParentClass,
      );
      if (parentIndex === -1) return false;
      return (
        datum.query.dimThingList[parentIndex].getEnt().id ===
        subRegionThing.getEnt()[parentInfo.parentIdKey]
      );
    });
}

export function getParentRegionFilter(query) {
  if (!query.parentRegionConstraintList) return null;
  return (datum) =>
    query.parentRegionConstraintList.every((constraint) => {
      const index = datum.query.dimThingList.findIndex(
        (thing) => thing.constructor === constraint.childClass,
      );
      return (
        index !== -1 &&
        constraint.childValues.includes(datum.query.dimThingList[index].value)
      );
    });
}
