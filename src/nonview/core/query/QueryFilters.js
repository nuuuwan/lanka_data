import ThingFactory from "../thing/thing_factory/ThingFactory.js";

export function getSubRegionFilter(query) {
  if (!query.subRegionDimThingList) {
    return null;
  }
  return (datum) =>
    query.subRegionDimThingList.every((subRegionThing) => {
      const parentRegionInfo = subRegionThing.constructor.getParentRegionInfo();
      const parentRegionClass = ThingFactory.fromKey(
        parentRegionInfo.parentClassName,
      );
      const parentRegionDimIndex = datum.query.dimThingList.findIndex(
        (dimThing) => dimThing.constructor === parentRegionClass,
      );
      if (parentRegionDimIndex === -1) {
        return false;
      }
      const parentRegionThing = datum.query.dimThingList[parentRegionDimIndex];
      return (
        parentRegionThing.getEnt().id ===
        subRegionThing.getEnt()[parentRegionInfo.parentIdKey]
      );
    });
}

export function getParentRegionFilter(query) {
  if (!query.parentRegionConstraintList) {
    return null;
  }
  return (datum) =>
    query.parentRegionConstraintList.every((constraint) => {
      const dimIndex = datum.query.dimThingList.findIndex(
        (dimThing) => dimThing.constructor === constraint.childClass,
      );
      return (
        dimIndex !== -1 &&
        constraint.childValues.includes(
          datum.query.dimThingList[dimIndex].value,
        )
      );
    });
}
